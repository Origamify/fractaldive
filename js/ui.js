'use strict';
/* ================================================================
 * UI — footer-bar controls, HUD, failure notice.
 *
 * Owns everything in the footer: pause/reset buttons, target /
 * speed / auto-speed inputs, the HUD readout, and the WebGPU-failure
 * notice. Speed constants live in common.js. Loaded after the engine
 * scripts and before dive.js/app.js — app.js initApp() calls
 * wireControls(); dive state (diveState in dive.js) is only touched
 * here at runtime, never at load time.
 * ================================================================ */

/* --- constants --- */
const SPEED_HOTKEY_STEP = 0.05; // zoom-rate change per +/- key press
const SPEED_SLIDER_MAX = 100;   // slider top end — must match max="" on #speed in index.html

const STATUS_FLASH_MS = 1600;                       // toast lifetime for copy/go-to feedback
const VIEW_DECIMAL_TEXT_PATTERN = /^[+-]?(?:\d+(?:\.\d*)?|\.\d+)$/; // plain-decimal coordinate strings the view JSON accepts (no exponent part)
const AUTO_SLOWED_FACTOR = 0.995;                   // effective rate below this fraction of the user's = auto-slowed (footer slider + HUD marker)

/* --- state --- */
let statusFlashTimerId = 0; // pending toast fade-out timeout (0 = none)
let syncedSpeedRate = 0;    // rate the slider+label currently show (0 = none yet — forces the first sync)

/* --- speed --- */

/**
 * Maps a slider position to a zoom rate on a log scale spanning
 * SPEED_MIN..SPEED_MAX.
 * @param {number} sliderValue - Slider position, 0..SPEED_SLIDER_MAX.
 * @returns {number} Zoom factor per second.
 */
const sliderValueToZoomRate = (sliderValue) =>
    SPEED_MIN * (SPEED_MAX / SPEED_MIN) ** (sliderValue / SPEED_SLIDER_MAX);

/**
 * Maps a zoom rate back to the nearest slider position.
 * @param {number} zoomRate - Zoom factor per second, SPEED_MIN..SPEED_MAX.
 * @returns {number} Rounded slider position, 0..SPEED_SLIDER_MAX.
 */
const zoomRateToSliderValue = (zoomRate) =>
    Math.round((SPEED_SLIDER_MAX * Math.log(zoomRate / SPEED_MIN)) / Math.log(SPEED_MAX / SPEED_MIN));

/**
 * Sets the dive zoom rate and syncs the footer speed control
 * (slider + label) to it.
 * @param {number} zoomRate - Zoom factor per second.
 * @returns {void}
 */
function setZoomRate(zoomRate) {
    diveState.userZoomRatePerSecond = zoomRate; // the user's setting — ceiling/recovery target for the auto-slowdown
    diveState.zoomRatePerSecond = zoomRate;    // a manual change resets any auto-slowdown
    syncSpeedControls();                       // slider + label show the new rate
}

/**
 * Reads the speed slider and applies the mapped zoom rate.
 * @returns {void}
 */
function onSpeedInput() {
    setZoomRate(sliderValueToZoomRate(Number(getElementById('speed').value)));
}

/**
 * Nudges the zoom rate by speedDelta (the +/- hotkeys) and parks the
 * slider at the nearest step — the exact rate holds until the slider
 * is used.
 * @param {number} speedDelta - Signed zoom-rate change.
 * @returns {void}
 */
function nudgeZoomRate(speedDelta) {
    const clampedRate = Math.min(SPEED_MAX, Math.max(SPEED_MIN, diveState.userZoomRatePerSecond + speedDelta));
    setZoomRate(clampedRate); // syncSpeedControls parks the slider at the nearest step
}

/* --- auto speed (the adaptive zoom-rate slowdown's footer face) --- */

/**
 * Tells the auto-slowdown state: the dive is stepping slower than
 * the user's setting (the GPU overran its zoom window).
 * @returns {boolean} True while the adaptive rate holds the dive below the user's speed.
 */
function isAutoSlowed() {
    return diveState.zoomRatePerSecond < diveState.userZoomRatePerSecond * AUTO_SLOWED_FACTOR;
}

/**
 * Syncs the footer speed slider and label to the rate actually being
 * stepped: while the auto-slowdown is active the control nudges down
 * to the effective rate and climbs back on recovery, so it always
 * shows the live speed; otherwise it parks at the user's rate.
 * Skips unchanged rates — never fights an in-progress drag.
 * @returns {void}
 */
function syncSpeedControls() {
    const rate = isAutoSlowed() ? diveState.zoomRatePerSecond : diveState.userZoomRatePerSecond;
    if (rate === syncedSpeedRate) return;
    syncedSpeedRate = rate;
    getElementById('speed').value = zoomRateToSliderValue(rate);
    getElementById('speedLbl').textContent = `×${rate.toFixed(2)}/s`;
}

/**
 * Handles the Auto checkbox: off stops the auto-slowdown (any
 * slowed rate recovers via the per-flight climb, not instantly);
 * on resumes full adaptation at the next settled flight.
 * @returns {void}
 */
function onAutoSpeedToggle() {
    diveState.isAutoSpeedEnabled = getElementById('autoSpeedChk').checked;
}

/* --- readouts --- */

/**
 * Refreshes the pause button label/disabled state and the HUD
 * max-depth styling from the current dive state.
 * @returns {void}
 */
function updatePauseButton() {
    const shouldShowPause = diveState.isRunning || diveState.hasReachedMaxDepth;
    getElementById('pauseBtn').textContent = shouldShowPause ? '⏸ Pause' : '▶ Resume';
    getElementById('pauseBtn').disabled = diveState.hasReachedMaxDepth;
    getElementById('hud').classList.toggle('max', diveState.hasReachedMaxDepth);
}

/**
 * Refreshes the HUD depth readout (and the max-depth banner line),
 * and keeps the footer speed control tracking the live rate.
 * @returns {void}
 */
function updateHud() {
    const hudParts = [`depth ×${formatEngineeringNotation(currentDepthFactor())}`];
    if (isAutoSlowed()) hudParts.push('auto-slow'); // GPU-bound: the dive slowed below the user's setting — the footer control shows the rate
    if (diveState.lastFrameGpuMs > 0) hudParts.push(`gpu ${diveState.lastFrameGpuMs.toFixed(1)} ms`);
    if (diveState.fps > 0) hudParts.push(`${diveState.fps.toFixed(0)} fps`);
    if (diveState.lastFrameBands > 0) hudParts.push(`${diveState.lastFrameBands} bands`);
    if (diveState.hasReachedMaxDepth) hudParts.unshift('max depth reached — Reset');
    getElementById('hud').textContent = hudParts.join(' · ');
    syncSpeedControls(); // the auto-slowdown's slider nudge — the rate may have changed on a settle since the last frame
}

/**
 * Shows the WebGPU failure banner with the reason.
 * @param {string} reason - Short failure description.
 * @returns {void}
 */
function showFailureNotice(reason) {
    document.body.classList.add('failed');
    getElementById('notice').textContent =
        `WebGPU unavailable — ${reason}. Open in a WebGPU-capable browser (e.g. Chromium) to dive.`;
}

/**
 * Flashes a transient toast above the footer — the copy/go-to
 * feedback channel. The HUD can't host it: the render loop
 * rewrites it every frame.
 * @param {string} message - Short status line to show for STATUS_FLASH_MS.
 * @returns {void}
 */
function flashStatus(message) {
    const toast = getElementById('toast');
    toast.textContent = message;
    toast.classList.add('show');
    clearTimeout(statusFlashTimerId);
    statusFlashTimerId = setTimeout(() => toast.classList.remove('show'), STATUS_FLASH_MS);
}

/* --- view clipboard (JSON) --- */

/**
 * Builds the current view as a JSON-serializable object:
 * - dive center as plain-decimal strings at full
 *   CENTER_FIXED_POINT_BITS precision (float64 never touches it)
 * - viewport height in complex-plane units — survives window
 *   size changes (unlike pixels-per-complex-unit)
 * @returns {{x: string, y: string, viewHeight: number}} The current center and zoom.
 */
function buildCurrentViewJson() {
    return {
        x: fixedPointToDecimalString(diveState.centerXFixed, CENTER_FIXED_POINT_BITS, VIEW_JSON_FRACTION_DIGITS),
        y: fixedPointToDecimalString(diveState.centerYFixed, CENTER_FIXED_POINT_BITS, VIEW_JSON_FRACTION_DIGITS),
        viewHeight: diveState.viewportScale * diveState.engine.h,
    };
}

/**
 * Validates one x/y field of a parsed view JSON and normalizes it:
 * a finite number or a plain-decimal string, returned as text.
 * @param {number|string} value - The parsed x or y field.
 * @returns {string|null} The coordinate as decimal text, or null if invalid.
 */
function normalizeViewCoordinateText(value) {
    if (typeof value === 'number') return isFinite(value) ? String(value) : null;
    if (typeof value === 'string' && VIEW_DECIMAL_TEXT_PATTERN.test(value.trim())) return value.trim();
    return null;
}

/**
 * Parses a clipboard view JSON (as written by buildCurrentViewJson).
 * @param {string} jsonText - Raw clipboard text.
 * @returns {{xText: string, yText: string, viewHeight: number}|null} The validated view, or null if unusable.
 */
function parseViewJson(jsonText) {
    let parsed;
    try {
        parsed = JSON.parse(jsonText);
    } catch (error) {
        return null;
    }
    if (parsed == null || typeof parsed !== 'object') return null;
    const xText = normalizeViewCoordinateText(parsed.x);
    const yText = normalizeViewCoordinateText(parsed.y);
    const viewHeight = Number(parsed.viewHeight);
    if (xText === null || yText === null || !isFinite(viewHeight) || viewHeight <= 0) return null;
    return { xText, yText, viewHeight };
}

/**
 * Legacy copy fallback for when the async clipboard API refuses
 * (permission denied, odd origins) — select the text in an
 * off-screen textarea, run deprecated execCommand('copy').
 * @param {string} text - Text to copy.
 * @returns {boolean} True if the copy command reported success.
 */
function copyTextWithHiddenTextarea(text) {
    const textarea = document.createElement('textarea');
    textarea.value = text;
    textarea.style.position = 'fixed';
    textarea.style.opacity = '0';
    document.body.appendChild(textarea);
    textarea.select();
    const copied = document.execCommand('copy');
    textarea.remove();
    return copied;
}

/**
 * Copy button handler: puts the current view JSON on the clipboard.
 * @returns {Promise<void>}
 */
async function copyViewToClipboard() {
    if (!diveState.engine) return; // boot race — engine not up yet (or failed)
    const jsonText = JSON.stringify(buildCurrentViewJson(), null, 2);
    let copied = false;
    try {
        await navigator.clipboard.writeText(jsonText);
        copied = true;
    } catch (error) {
        copied = copyTextWithHiddenTextarea(jsonText);
    }
    flashStatus(copied ? 'view copied' : 'copy failed — clipboard blocked');
}

/**
 * Go-to button handler: reads a view JSON from the clipboard (a
 * paste prompt if clipboard reading is blocked, e.g. on file://)
 * and jumps the dive there, paused.
 * @returns {Promise<void>}
 */
async function goToClipboardView() {
    if (!diveState.engine) return; // boot race — engine not up yet (or failed)
    let jsonText = '';
    try {
        jsonText = await navigator.clipboard.readText();
    } catch (error) {
        jsonText = prompt('Paste FractalDive view JSON:') || ''; // clipboard read blocked
    }
    const view = parseViewJson(jsonText);
    if (!view) {
        flashStatus('invalid view JSON on clipboard');
        return;
    }
    goToView(view.xText, view.yText, view.viewHeight);
    flashStatus('view applied');
}

/* --- wiring --- */

/**
 * Returns the dive target currently selected in the dropdown.
 * @returns {{name: string, x: string, y: string}} The selected DIVE_TARGETS entry (plain-decimal coordinates).
 */
function getSelectedTarget() {
    return DIVE_TARGETS[getElementById('targetSel').selectedIndex];
}

/**
 * Handles hotkeys regardless of focus: Space = pause/resume,
 * +/- = speed ±SPEED_HOTKEY_STEP. preventDefault keeps focused
 * footer controls (buttons, select) from double-firing the key.
 * @param {KeyboardEvent} event - The keydown event.
 * @returns {void}
 */
function onKeydown(event) {
    if (event.code === 'Space') {
        if (event.repeat) return;
        event.preventDefault(); // page scroll + focused-button activation
        togglePause();
    } else if (event.key === '+' || event.key === '=') {
        event.preventDefault();
        nudgeZoomRate(+SPEED_HOTKEY_STEP); // repeat allowed — hold to ramp
    } else if (event.key === '-' || event.key === '_') {
        event.preventDefault();
        nudgeZoomRate(-SPEED_HOTKEY_STEP);
    }
}
/**
 * Populates the target dropdown and wires every footer
 * control plus the canvas click, window resize/keydown listeners.
 * Called once from app.js initApp().
 * @returns {void}
 */
function wireControls() {
    for (const target of DIVE_TARGETS) getElementById('targetSel').add(new Option(target.name, target.name));

    const speedSlider = getElementById('speed');
    speedSlider.addEventListener('input', onSpeedInput);
    setZoomRate(DEFAULT_SPEED); // exact default — syncSpeedControls parks the slider at the nearest step

    const autoSpeedCheckbox = getElementById('autoSpeedChk');
    autoSpeedCheckbox.checked = diveState.isAutoSpeedEnabled; // diveState default: on
    autoSpeedCheckbox.addEventListener('change', onAutoSpeedToggle);

    getElementById('pauseBtn').addEventListener('click', togglePause);
    getElementById('resetBtn').addEventListener('click', () => startDive(getSelectedTarget()));
    getElementById('copyBtn').addEventListener('click', copyViewToClipboard);
    getElementById('gotoBtn').addEventListener('click', goToClipboardView);
    getElementById('targetSel').addEventListener('change', () => startDive(getSelectedTarget()));
    getElementById('cv').addEventListener('click', clickToZoom);
    window.addEventListener('resize', fitCanvasToViewport);
    window.addEventListener('keydown', onKeydown);
}
