'use strict';
/* ================================================================
 * Presenter — the zoom's presentation layer (CSS transform).
 *
 * - one zoom process for the whole dive: the canvas bitmap holds
 *   the full 2× base frame (engine.commit blits it at settle) and
 *   the visible crop is this module's center-origin transform —
 *   the browser compositor re-samples the bitmap every frame,
 *   the WebGPU queue is never involved (the queue-stall fix,
 *   docs/ARCHITECTURE.md)
 * - stateless besides the change gate: app.js computes the
 *   magnification (windowStartScale / viewportScale) and calls
 *   setZoom every tick the window is open; when it stops calling
 *   (pause hold, invalidation) the last transform simply holds
 * - never touches WebGPU or dive state — magnification arrives
 *   as a number
 * ================================================================ */

/* --- state: the one DOM handle + the change gate --- */
let presenterCanvas = null; // the dive's canvas (#cv) — its CSS transform IS the zoom
let presenterLastScale = 0; // last written scale — style writes skip when unchanged

/**
 * Binds the presenter to the canvas element. Once at boot, before
 * the first setZoom.
 * @param {HTMLCanvasElement} canvas - The dive's canvas (#cv — engine.commit renders into it).
 * @returns {void}
 */
function initPresenter(canvas) {
    presenterCanvas = canvas;
}

/**
 * Shows the given magnification of the canvas bitmap: transform
 * scale(m) about the center — the old crop-blit's center sub-rect
 * (m = windowStartScale / viewportScale: 1 at window start with a
 * 2:1 compositor downsample, 2 at the 1:1 window end, soft
 * upscale past it).
 * @param {number} magnification - Bitmap scale factor (≥ 1 across a window).
 * @returns {void}
 */
function setZoom(magnification) {
    if (!presenterCanvas || magnification === presenterLastScale) return;
    presenterCanvas.style.transform = `scale(${magnification})`;
    presenterLastScale = magnification;
}
