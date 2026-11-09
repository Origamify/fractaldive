'use strict';
/* ================================================================
 * Fixed point — BigInt fixed-point frame math for the perturbation
 * engine.
 *
 * Pure math, values in / values out — no DOM, no GPU, no dive
 * state — and testable from the browser test pages. Load after
 * common.js, before engine.js (its every-frame caller).
 *
 * Two number formats meet here:
 * - FRAME precision — BigInt fraction bits growing with zoom
 *   depth; quantizes the pixel grid, feeds the exact center C
 * - DELTA constants — f32 plane-edge offsets + steps the render
 *   kernel assembles δ from (tolerance-rounded)
 * The reference limb width (REFERENCE_LIMB_COUNT × 32, LSB 2^-148)
 * covers the frame precision at every dive depth — the shift
 * between them is always an exact left shift.
 * ================================================================ */

/* --- constants: frame precision (module-local) --- */
const FIXED_POINT_HEADROOM_BITS = 8;    // fractional bits beyond bits(1/s) — sub-pixel margin so pixel centers stay distinct
const FIXED_POINT_MIN_BITS = 16;        // shallow-zoom floor (~5 decimal digits)
const FIXED_POINT_MAX_BITS = 1000;      // ~300 decimal digits — far past MIN_S = 1e-35, and keeps 2^bits finite in float64
const LIMB_BITS = 32;                   // u32 limbs — WGSL's only comfortable ALU width
const SUB_STEP_COLUMN_SHIFT_BITS = Math.log2(SAMPLE_COLUMNS_PER_PIXEL) + 1; // fixed-point shift for the x sub-step: step >> shift = step/(2·columns), the x lattice spacing
const SUB_STEP_ROW_SHIFT_BITS = Math.log2(SAMPLE_ROWS_PER_PIXEL) + 1;       // y sub-step shift — step/(2·rows); supersampling offsets are odd multiples of the sub-steps

/* --- constants: the reference limb format (module-local) ---
 * The per-frame Z table is computed exactly on the GPU in signed
 * u32-limb fixed point at LSB 2^-REFERENCE_FRACTION_BITS (a
 * constant width at every depth), then stored as df32 pairs for
 * the f32 delta kernel. Sizing, bottom up:
 *   - LSB 2^-148: at the dive floor (MIN_S = 1e-35/px) the frame
 *     grid carries ~125 fractional bits → 23 bits below the grid:
 *     exact at any depth the app dives
 *   - integer headroom to 2^44: |Z| ≤ 6.3 for every stored row
 *     (rows stop at |Z|² > 4 + final update) — margin far beyond
 *     anything the dive produces
 *   - 148 + 44 = 192 bits = 6 u32 limbs exactly
 * The delta half (δ, ε) rides plain f32 — no constants of its
 * own: the 1e-35 floor keeps δ within ~2× of f32's min normal
 * (~1.2e-38); a flushed subnormal δ renders that sub-sample as
 * the reference (sub-pixel error — the tolerance regime). */
const REFERENCE_FRACTION_BITS = 148;        // fractional bits of the reference limb format (LSB 2^-148)
const REFERENCE_INTEGER_HEADROOM_BITS = 44; // integer bits above the point (range ±2^43)
const REFERENCE_LIMB_COUNT = Math.ceil((REFERENCE_FRACTION_BITS + REFERENCE_INTEGER_HEADROOM_BITS) / LIMB_BITS); // = 6 — tier of the reference kernel (the delta kernel is f32)
const REFERENCE_STOP_MAGNITUDE_SQUARED = 4; // the reference table stops (after storing) the first row with |Z|² > this — rebasing takes over from there

/* --- helpers (pure math — testable from the test pages) --- */

/**
 * Returns the fixed-point fractional-bit count for a viewport scale.
 * Why: one bit per power of two of zoom + headroom, clamped to sane
 * bounds. Sizes the FRAME precision, not the GPU's constant limb width.
 * @param {number} viewportScale - Complex units per pixel (view.s).
 * @returns {number} Fractional bits, FIXED_POINT_MIN_BITS..FIXED_POINT_MAX_BITS.
 */
function fractionBitsForScale(viewportScale) {
    const bitsForScale = Math.ceil(Math.log2(1 / viewportScale));
    return Math.min(FIXED_POINT_MAX_BITS,
        Math.max(FIXED_POINT_MIN_BITS, bitsForScale + FIXED_POINT_HEADROOM_BITS));
}

/**
 * Builds the frame configuration for one frame.
 * - reference half (exact): center C rescaled to limb width + stop threshold
 * - delta half (f32): plane-edge offsets + pixel/sub-pixel steps
 * - every delta term is step-scaled (reference = center) → the
 *   ~2⁻²⁴·step rounding sits far below the step/4 lattice
 * @param {object} view - Exact dive view: BigInt center at centerFractionBits; s = units/pixel.
 * @param {number} canvasWidthPx - Frame width in device px.
 * @param {number} canvasHeightPx - Frame height in device px.
 * @returns {object} { referenceStopThreshold, centerReal, centerImag } (BigInt, limb width) + { planeLeftF32, planeTopF32, pixelStepF32, subStepXF32, subStepYF32 } (f32).
 * @throws {Error} Frame precision above the limb format (unreachable above MIN_S).
 */
function buildFixedPointConfig(view, canvasWidthPx, canvasHeightPx) {
    const fractionBits = fractionBitsForScale(view.s);
    if (fractionBits > REFERENCE_FRACTION_BITS) {
        throw new Error(`frame precision ${fractionBits} bits exceeds the reference limb format's ${REFERENCE_FRACTION_BITS}`);
    }
    const floatScale = Number(1n << BigInt(fractionBits)); // float64 → fixed-point multiplier for the scale-derived offsets
    // rescale the center to frame precision:
    // - left shift = exact growth; right shift drops only bits
    //   shallower than this frame can use
    const centerShift = BigInt(fractionBits - view.centerFractionBits);
    const centerXFixed = centerShift >= 0n ? view.centerXFixed << centerShift : view.centerXFixed >> -centerShift;
    const centerYFixed = centerShift >= 0n ? view.centerYFixed << centerShift : view.centerYFixed >> -centerShift;
    const pixelStepFixed = BigInt(Math.max(1, Math.round(view.s * floatScale))); // ≥1 or adjacent pixels collapse
    const limbShift = BigInt(REFERENCE_FRACTION_BITS - fractionBits); // always ≥ 0 here — exact left shift into the limb format
    const toLimbWidth = (frameValue) => frameValue << limbShift;
    // one reference-limb-width BigInt → its f32 rounding (the delta kernel's frame constants)
    const toF32 = (deltaValue) => Math.fround(Number(deltaValue) * Math.pow(2, -REFERENCE_FRACTION_BITS));
    // sample pixel centers the way the exact engine did: center pixel at +½ step (span in complex units, then fixed)
    const planeSpanReal = BigInt(Math.round((canvasWidthPx / 2 - 0.5) * view.s * floatScale));
    const planeSpanImag = BigInt(Math.round((canvasHeightPx / 2 - 0.5) * view.s * floatScale));
    return {
        // reference pass inputs (exact): the frame center at limb width + the |Z|² stop threshold
        referenceStopThreshold: BigInt(REFERENCE_STOP_MAGNITUDE_SQUARED) << BigInt(2 * REFERENCE_FRACTION_BITS),
        centerReal: toLimbWidth(centerXFixed),
        centerImag: toLimbWidth(centerYFixed),
        // delta pass inputs (f32): the plane edges' offsets from the reference center —
        // the reference IS the center, so every δ term is step-scaled
        planeLeftF32: toF32(toLimbWidth(-planeSpanReal)),
        planeTopF32: toF32(toLimbWidth(planeSpanImag)),
        pixelStepF32: toF32(toLimbWidth(pixelStepFixed)),
        // supersampling sub-steps: odd multiples place the lattice
        // around each pixel center (±step/4 for the 2×2 lattice)
        subStepXF32: toF32(toLimbWidth(pixelStepFixed >> BigInt(SUB_STEP_COLUMN_SHIFT_BITS))),
        subStepYF32: toF32(toLimbWidth(pixelStepFixed >> BigInt(SUB_STEP_ROW_SHIFT_BITS))),
    };
}

/**
 * Packs a BigInt into little-endian two's-complement u32 limbs,
 * truncating to width — the CPU side of filling the limb buffers.
 * @param {bigint} value - Signed BigInt fixed-point value.
 * @param {number} limbCount - Output width in limbs.
 * @returns {number[]} Limbs, value mod 2^(32·limbCount).
 */
function bigIntToLimbs(value, limbCount) {
    const wrappedValue = BigInt.asUintN(LIMB_BITS * limbCount, value);
    const limbs = new Array(limbCount);
    for (let limbIndex = 0; limbIndex < limbCount; limbIndex++) {
        limbs[limbIndex] = Number((wrappedValue >> BigInt(LIMB_BITS * limbIndex)) & 0xFFFFFFFFn);
    }
    return limbs;
}
