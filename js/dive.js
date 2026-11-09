'use strict';
/* ================================================================
 * Dive — the dive model: zoom state machine, view edits, base-
 * frame submission.
 *
 * - owns diveState (the one mutable dive model); UI wiring lives
 *   in ui.js, the render loop and boot in app.js, shared names in
 *   common.js — classic scripts, one global lexical scope, no
 *   exports
 * - ui.js calls startDive / togglePause; app.js calls
 *   stepZoom / submitRender / fitCanvasToViewport and reads
 *   diveState at runtime
 *
 * Rendering is single-flight AND window-cached (overrender zoom
 * cache), with the render/presentation split (docs/ARCHITECTURE.md):
 * - at most one base frame render runs at any time; each settled
 *   base serves a full BASE_OVERRENDER_FACTOR× zoom window
 * - the next base is requested only at the window's end (crop
 *   fraction 1/BASE_OVERRENDER_FACTOR), so no window range is
 *   discarded (or immediately on cache invalidation: view jump,
 *   click-zoom, resize) — renders never stack; the gap until the
 *   next base settles magnifies the old base past 1:1 (soft)
 * - between settles the zoom is presenter.js's CSS transform on
 *   the canvas (setZoom every tick the window is open) — zero
 *   WebGPU work, so a slow base render can never stall the zoom
 *   (the single-queue FIFO stall, docs/ARCHITECTURE.md)
 * - engine.commit() blits the full settled base into the canvas
 *   once per settle; the transform is a pure function of dive
 *   state (windowStartScale / viewportScale), so a new base lands
 *   already correctly cropped — no swap choreography, only the
 *   soft→sharp pop (as before)
 * ================================================================ */

/* --- constants --- */
const CLICK_ZOOM_FACTOR = 2;       // scale divisor per canvas click — one octave per click

/* --- adaptive zoom rate: auto-slow when the GPU can't keep up ---
 * The zoom itself can never stall (CSS transform), so "not keeping
 * up" means one thing: the next base's flight (request→settle)
 * outlasts its zoom window (ln(BOF)/rate). Each settled flight
 * retunes the rate so the window fits the flight with headroom,
 * and comfortable flights climb back toward the user's setting. */
const ADAPTIVE_FLIGHT_HEADROOM = 1.25; // window duration must exceed flight time by this factor before a slowdown lifts
const ADAPTIVE_RECOVERY_STEP = 1.15;   // per-comfortable-flight rate climb back toward the user's speed

/* Hidden-tab guard (the sample's clock poison):
 * - hidden tabs clamp inter-band setTimeout sleeps to ≥1 s → a
 *   flight settling while hidden reports wall time far past its
 *   GPU cost (would pin the rate to SPEED_MIN)
 * - flights spanning a hide are skipped: the stamp (hide
 *   transitions + the hidden bit) must match request → settle;
 *   requests only ever leave rAF, which never fires hidden
 * - hiding auto-pauses a running dive: rAF never fires hidden, so
 *   the wall-clock zoom law would bank the hidden span into one
 *   giant step at the return tick (floored to MIN_S)
 * - showing undoes exactly that pause — a user's pause must not
 *   resume on show */
let tabHiddenEventCount = 0; // bumped on every transition to hidden
let tabAutoPaused = false;    // the hide (not the user) paused the dive — the show resumes it
if (typeof document !== 'undefined') {
    document.addEventListener('visibilitychange', () => {
        if (document.hidden) {
            tabHiddenEventCount++;
            tabAutoPaused = diveState.isRunning;
            if (tabAutoPaused) togglePause(); // the resume path resets lastZoomStepTime — no hidden-span jump later
        } else if (tabAutoPaused) {
            tabAutoPaused = false;
            togglePause(); // resumes the dive where it left off (refuses at the depth floor, like any resume)
        }
    });
}
const currentTabHideStamp = () => tabHiddenEventCount + (document.hidden ? 1 : 0);

/* --- state: the one mutable dive model (ui.js reads it at runtime) --- */
const diveState = {
    engine: null,                    // WebGPU engine instance (null until init succeeds)
    isRunning: false,                // zoom stepping active
    hasReachedMaxDepth: false,       // hit MIN_S — only Reset proceeds from here
    needsRepaint: true,              // repaint needed while paused (resize)
    centerXFixed: 0n,                  // dive center, real part — BigInt fixed-point at CENTER_FIXED_POINT_BITS (fixed for the whole dive)
    centerYFixed: 0n,                  // dive center, imag part — same scale
    userZoomRatePerSecond: DEFAULT_SPEED, // the user's speed setting (slider/hotkeys) — the auto-slowdown's ceiling and recovery target
    zoomRatePerSecond: DEFAULT_SPEED,// zoom factor per second actually stepped (auto-slowed below the user rate when flights overrun their window)
    isAutoSpeedEnabled: true,        // the footer Auto checkbox — off = no auto-slowdown (the recovery climb toward the user's speed still runs)
    viewportScale: 0,                // current scale: complex units per pixel
    baseWindowStartScale: 0,         // viewport scale the active base frame's window started at (0 = no usable base — render ASAP); setZoom's magnification is this / viewportScale
    baseRenderGeneration: 0,         // bumps on every base-cache invalidation (view jump, click-zoom, resize) — a render settling after an invalidation never promotes its stale window
    lastZoomStepTime: 0,             // performance.now() of the last zoom step (ms)
    isRenderInFlight: false,         // a frame submission is running
    lastFrameGpuMs: 0,               // GPU-wait time of the last settled frame (HUD readout; see engine_frame.js)
    lastFrameBands: 0,               // band count of the last settled frame (HUD readout; see engine_frame.js)
    fps: 0,                          // rAF ticks per second over the last FPS_SAMPLE_WINDOW_MS (HUD readout; sampled in app.js)
};

/* --- readouts (consumed by the HUD in ui.js) --- */

/**
 * Returns the zoom factor relative to dive start (×1 at start) —
 * the HUD depth number.
 * @returns {number} BASE_SCALE divided by the current viewport height in complex units.
 */
function currentDepthFactor() {
    return BASE_SCALE / (diveState.viewportScale * diveState.engine.h);
}

/* --- base-frame cache (the overrender zoom window bookkeeping) --- */

/**
 * Drops the active base frame — the next render loop tick issues
 * a fresh base render at the current view. Called on every view
 * edit (goToView, click-zoom) and resize: the settled base
 * texture no longer matches the view it must crop from.
 * - generation bump disowns any in-flight render: on settle its
 *   captured generation no longer matches → no commit, window
 *   never opens
 * - presentation holds by itself: the window is closed so setZoom
 *   stops being called — the canvas keeps the last committed base
 *   at its frozen transform until the fresh base lands
 * @returns {void}
 */
function invalidateBaseFrame() {
    diveState.baseRenderGeneration++;
    diveState.baseWindowStartScale = 0;
}

/* --- dive control --- */

/**
 * Starts a fresh dive at the given target: drops the old reference
 * orbit, parks the whole set in view and waits paused — Resume /
 * Space starts the zoom.
 * @param {{name: string, x: string, y: string}} target - DIVE_TARGETS entry (plain-decimal coordinates) to dive into.
 * @returns {void}
 */
function startDive(target) {
    goToView(target.x, target.y, BASE_SCALE);
}

/**
 * Recenters the dive on arbitrary coordinates and zooms so the
 * given complex-plane height fills the canvas.
 * startDive and the clipboard go-to both route through here; same
 * contract — fresh reference orbit, paused, awaiting repaint.
 * @param {string} xText - Center real part, plain-decimal string (any digit count — the canonical path into the BigInt center).
 * @param {string} yText - Center imag part, plain-decimal string.
 * @param {number} viewHeight - Complex-plane viewport height; clamped to the dive's zoom range (MIN_S .. whole set).
 * @returns {void}
 */
function goToView(xText, yText, viewHeight) {
    diveState.centerXFixed = decimalStringToFixedPoint(xText, CENTER_FIXED_POINT_BITS);
    diveState.centerYFixed = decimalStringToFixedPoint(yText, CENTER_FIXED_POINT_BITS);
    diveState.engine.resetRef(); // never reuse the previous target's reference orbit
    invalidateBaseFrame();       // the settled base frame crops the wrong view — render a fresh one
    diveState.viewportScale = Math.min(BASE_SCALE / diveState.engine.h,
                                        Math.max(MIN_S, viewHeight / diveState.engine.h));
    diveState.lastZoomStepTime = performance.now();
    diveState.isRunning = false; // jumps land paused — the default state
    diveState.hasReachedMaxDepth = false;
    diveState.needsRepaint = true;
    updatePauseButton();
}

/**
 * Pauses or resumes zooming. Resuming refuses at the depth floor —
 * only Reset (a new startDive) proceeds from there.
 * @returns {void}
 */
function togglePause() {
    if (diveState.isRunning) {
        diveState.isRunning = false;
    } else if (!diveState.hasReachedMaxDepth) {
        diveState.lastZoomStepTime = performance.now(); // no time jump across the pause
        diveState.isRunning = true;
    }
    updatePauseButton();
}

/**
 * Recenters the dive on the clicked canvas point and zooms one
 * CLICK_ZOOM_FACTOR step (paused or running; floor: recenter
 * only). Mapping mirrors buildFixedPointConfig (y up, centers
 * at ±½ step); the click lands in the UNtransformed box — rect
 * is post-transform (scale(m)), so rebuild it from
 * clientWidth/Height around the shared center. The offset joins
 * the BigInt center float64-multiplied, one rounding at
 * CENTER_FIXED_POINT_BITS.
 * @param {PointerEvent} event - The canvas click event.
 * @returns {void}
 */
function clickToZoom(event) {
    if (!diveState.engine) return; // boot race — engine not up yet (or failed)
    const engine = diveState.engine;
    const canvas = engine.canvas;
    const rect = canvas.getBoundingClientRect(); // POST-transform: the presenter's scale(m) stretches it
    const cssLeft = rect.left + (rect.width - canvas.clientWidth) / 2; // the center-origin scale keeps the rect center —
    const cssTop = rect.top + (rect.height - canvas.clientHeight) / 2; // clientWidth/Height are layout (transform-free)
    const clickXpx = ((event.clientX - cssLeft) * engine.w) / canvas.clientWidth;  // CSS px → device px
    const clickYpx = ((event.clientY - cssTop) * engine.h) / canvas.clientHeight;
    const centerScale = Number(1n << BigInt(CENTER_FIXED_POINT_BITS)); // complex units → center fixed-point multiplier
    diveState.centerXFixed += BigInt(Math.round((clickXpx - (engine.w / 2 - 0.5)) * diveState.viewportScale * centerScale));
    diveState.centerYFixed += BigInt(Math.round(((engine.h / 2 - 0.5) - clickYpx) * diveState.viewportScale * centerScale));
    diveState.viewportScale = Math.max(MIN_S, diveState.viewportScale / CLICK_ZOOM_FACTOR);
    invalidateBaseFrame();  // the base frame crops around the OLD center — a fresh one must render
    diveState.needsRepaint = true; // applies on the next frame (instant while paused too)
}

/* --- adaptive zoom rate --- */

/**
 * Retunes the zoom rate from one settled base flight (called in
 * submitRender's settle path):
 * - window overrun (flight × headroom > window): slow to the
 *   rate whose window fits the flight, clamped to the user rate
 *   / SPEED_MIN — Auto checkbox off skips the slowdown
 * - comfortable flights climb back toward the user's rate
 *   (ADAPTIVE_RECOVERY_STEP per settle) — always, checkbox or not
 * @param {number} flightMs - Wall-clock ms from render request to settle.
 * @returns {void}
 */
function adaptZoomRate(flightMs) {
    const windowMs = (Math.log(BASE_OVERRENDER_FACTOR) / diveState.zoomRatePerSecond) * MS_PER_SECOND;
    if (diveState.isAutoSpeedEnabled && flightMs * ADAPTIVE_FLIGHT_HEADROOM > windowMs) {
        const slowedRate = Math.log(BASE_OVERRENDER_FACTOR)
            / ((flightMs / MS_PER_SECOND) * ADAPTIVE_FLIGHT_HEADROOM);
        diveState.zoomRatePerSecond = Math.min(diveState.userZoomRatePerSecond,
                                                Math.max(SPEED_MIN, slowedRate));
    } else if (diveState.zoomRatePerSecond < diveState.userZoomRatePerSecond) {
        diveState.zoomRatePerSecond = Math.min(diveState.userZoomRatePerSecond,
                                                diveState.zoomRatePerSecond * ADAPTIVE_RECOVERY_STEP);
    }
}

/* --- render --- */

/**
 * Submits one single-flight BASE frame render and commits it on
 * settle:
 * - latches the request-time scale — setZoom's magnification is
 *   correct the moment the base lands; commit() blits the full
 *   base, the transform crops
 * - pitch AND budget at the window-END scale (BASE_OVERRENDER_FACTOR
 *   resolution — detail must exist for the deepest view served)
 * - the settle-time generation check drops renders orphaned by
 *   an invalidation (view jump / click / resize mid-grind)
 * @returns {void}
 */
function submitRender() {
    diveState.isRenderInFlight = true;
    const requestScale = diveState.viewportScale;
    const requestGeneration = diveState.baseRenderGeneration;
    const requestTimeMs = performance.now();
    const requestHideStamp = currentTabHideStamp(); // a flight spanning a hidden stretch is no timing sample (clamped inter-band sleeps)
    const wasRunning = diveState.isRunning; // paused-time flights (e.g. the post-goToView reference build) have no deadline — not adaptation samples
    // view contract: exact BigInt center (engine rescales to frame
    // precision) + float64 scale s — a ratio, where float64 error is
    // invisible. s is the BASE pitch (window-start view at 2× res).
    const basePitchScale = requestScale / BASE_OVERRENDER_FACTOR;
    const view = {
        centerXFixed: diveState.centerXFixed,
        centerYFixed: diveState.centerYFixed,
        centerFractionBits: CENTER_FIXED_POINT_BITS,
        s: basePitchScale,
    };
    diveState.engine.render(view, iterationCountForScale(basePitchScale))
        .catch((error) => { console.error(error); showFailureNotice(error.message); })
        .finally(() => {
            diveState.isRenderInFlight = false;
            if (requestGeneration === diveState.baseRenderGeneration) {
                diveState.baseWindowStartScale = requestScale; // the window opens — setZoom starts cropping from this base
                diveState.engine.commit(); // full base into the canvas — the queue is empty here (single flight), so the blit never waits behind a render
            } // else: an invalidation orphaned this settle — no commit; the canvas holds the last frame at its frozen transform
            diveState.lastFrameGpuMs = diveState.engine.lastFrameTimingsMs?.gpuMs ?? 0; // shown by updateHud next to the depth
            diveState.lastFrameBands = diveState.engine.lastFrameBandCount ?? 0;          // same HUD line, next to the GPU time
            if (wasRunning && !document.hidden && currentTabHideStamp() === requestHideStamp) { // hidden-tab flights are poisoned by clamped inter-band sleeps — no sample
                adaptZoomRate(performance.now() - requestTimeMs);                         // keep the zoom window ahead of the GPU
            }
        });
}

