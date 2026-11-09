'use strict';
/* ================================================================
 * App — the entry point: render loop, canvas sizing, boot.
 *
 * - steps the dive model (dive.js) once per rAF tick and drives
 *   the presenter's CSS zoom; UI wiring lives in ui.js, shared
 *   names in common.js — classic scripts, one global lexical
 *   scope, no exports
 *
 * Zoom law: s ← s·e^(−r·Δt) stepped every frame — the exponential
 * makes steps multiply out exactly, so any frame rate yields the
 * same zoom over time. Pause stops stepping; a speed change
 * applies at the next step.
 *
 * Boot has no fallback: WebGPU or a thrown error → failure
 * notice.
 * ================================================================ */

/* --- constants --- */
const MAX_DEVICE_PIXEL_RATIO = 2; // DPR cap — keeps hi-dpi phones from rendering 4× the pixels
const MIN_CANVAS_SIZE_PX = 64;    // canvas never smaller than this, even on tiny windows
const FPS_SAMPLE_WINDOW_MS = 500; // HUD fps averaging window — rAF ticks counted over this span
const BASE_WINDOW_END_CROP_FRACTION = 1 / BASE_OVERRENDER_FACTOR; // crop fraction at the 1:1 window end — the settled base is fully consumed, so the next base is requested here (while it renders, the transform magnifies the old base past 1:1)

/* --- state: fps sampling (onAnimationFrame's HUD readout) --- */
let fpsTickCounter = 0;   // rAF ticks in the current averaging window
let fpsWindowStartMs = 0; // performance.now() the window opened (0 = none yet)

/* --- render loop --- */

/**
 * Advances the zoom by the elapsed wall time (exponential law — see
 * file header) and latches the max-depth floor at MIN_S.
 * @param {number} currentTimeMs - performance.now() timestamp of this frame.
 * @returns {void}
 */
function stepZoom(currentTimeMs) {
    if (!diveState.isRunning) return;

    const elapsedMs = currentTimeMs - diveState.lastZoomStepTime;
    diveState.viewportScale *= Math.exp((-diveState.zoomRatePerSecond * elapsedMs) / MS_PER_SECOND);
    diveState.lastZoomStepTime = currentTimeMs;

    if (diveState.viewportScale > MIN_S) return;
    diveState.viewportScale = MIN_S;
    diveState.isRunning = false;
    diveState.hasReachedMaxDepth = true;
    updatePauseButton();
}

/**
 * Handles one rAF tick, in order:
 * - stop the loop on device loss
 * - step the zoom; drive the CSS zoom transform (no WebGPU work)
 * - submit the next base render once the previous settled and
 *   the window asks for one
 * - refresh the HUD (fps sample lives here too); reschedule
 * @returns {void}
 */
function onAnimationFrame() {
    if (!diveState.engine.available) {
        showFailureNotice('device lost');
        return; // loop ends — nothing left to render into
    }

    const nowMs = performance.now();
    stepZoom(nowMs);
    const hasOpenBaseWindow = diveState.baseWindowStartScale > 0;
    if (hasOpenBaseWindow) {
        setZoom(diveState.baseWindowStartScale / diveState.viewportScale); // the zoom IS this transform — pure CSS, the queue stays render-only
    }

    const isBaseFrameDue = diveState.needsRepaint
        || (diveState.isRunning && (!hasOpenBaseWindow // no window: always due — recovery
            || diveState.viewportScale / diveState.baseWindowStartScale <= BASE_WINDOW_END_CROP_FRACTION)); // open window: due at its end
    if (diveState.engine.w && !diveState.isRenderInFlight && isBaseFrameDue) {
        diveState.needsRepaint = false;
        submitRender();
    }

    fpsTickCounter++; // the HUD's fps sample — ticks per window, scaled to per-second
    if (fpsWindowStartMs === 0) fpsWindowStartMs = nowMs;
    const fpsElapsedMs = nowMs - fpsWindowStartMs;
    if (fpsElapsedMs >= FPS_SAMPLE_WINDOW_MS) {
        diveState.fps = (fpsTickCounter * MS_PER_SECOND) / fpsElapsedMs;
        fpsWindowStartMs = nowMs;
        fpsTickCounter = 0;
    }

    updateHud();
    requestAnimationFrame(onAnimationFrame);
}

/* --- wiring / lifecycle --- */

/**
 * Resizes the engine canvas to the window at the capped device
 * pixel ratio, and marks the view dirty so a paused dive repaints.
 * @returns {void}
 */
function fitCanvasToViewport() {
    if (!diveState.engine) return; // boot race — engine not up yet (or failed); the post-init call sizes it
    const devicePixelRatio = Math.min(window.devicePixelRatio || 1, MAX_DEVICE_PIXEL_RATIO);
    const canvasWidthPx = Math.max(MIN_CANVAS_SIZE_PX, Math.round(window.innerWidth * devicePixelRatio));
    const canvasHeightPx = Math.max(MIN_CANVAS_SIZE_PX, Math.round(window.innerHeight * devicePixelRatio));
    const sizeChanged = diveState.engine.w !== canvasWidthPx || diveState.engine.h !== canvasHeightPx;
    diveState.engine.resize(canvasWidthPx, canvasHeightPx);
    diveState.needsRepaint = true; // paused view must repaint after resize
    if (sizeChanged) invalidateBaseFrame(); // the engine's 2× base texture died with the resize — render a fresh one at the new size
}

/**
 * Creates the engine and initializes it on the canvas. No
 * fallback: WebGPU or a thrown error → failure notice.
 * @returns {Promise<object>} The initialized engine.
 */
async function createEngine() {
    const engine = new WebGPUPerturbationEngine();
    await engine.init(getElementById('cv'));
    return engine;
}

/**
 * Boots the app: wire the footer controls, bring up the engine,
 * size the canvas, start the first dive and enter the render loop.
 * @returns {Promise<void>}
 */
async function initApp() {
    wireControls();
    initPresenter(getElementById('cv')); // the canvas's CSS transform is the zoom — bind it before the first tick

    try {
        diveState.engine = await createEngine();
    } catch (error) {
        console.warn('engine init failed:', error.message);
        showFailureNotice(error.message);
        return;
    }

    fitCanvasToViewport();
    startDive(DIVE_TARGETS[0]); // page load = Reset: whole set in view at the first target
    requestAnimationFrame(onAnimationFrame);
}

if (typeof document !== 'undefined' && typeof FRACTALDIVE_NO_AUTOBOOT === 'undefined') initApp(); // auto-boot in the browser; skipped under a bare Bun import or when a test page sets FRACTALDIVE_NO_AUTOBOOT
