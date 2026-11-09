'use strict';
/* ================================================================
 * Engine kernel WGSL — bindings chunk (RETIRED exact engine,
 * oracle only — loaded by tests/perturb_frame_test.html, never
 * by the app): the kernel WGSL preamble
 * (header doc, buffer-layout / lattice / palette constants,
 * KernelParams struct, the two @group bindings) plus the input
 * helpers loadLimbs and magnitudeLog2Approx (the limbs→f32 log2
 * bridge feeding the palette). First chunk spliced after
 * LIMB_ARRAY_MATH_WGSL — WGSL requires declaration before use.
 * The lattice constants are spliced from common.js at script load.
 * ================================================================ */

/* --- kernel source chunk (WGSL, not JavaScript) --- */
const ENGINE_KERNEL_WGSL_BINDINGS = `
// ================================================================
// Engine kernel — WGSL port of the u32-limb bigint Mandelbrot orbit.
//
// One fragment invocation per pixel — the vs/fs entries below
// draw a fullscreen triangle, so fs runs once per render-target
// pixel, supersampling it (sampleColorPacked × the lattice above);
// mirrors the fixed-point semantics of fixed_point.js's frame
// math (escape check before budget check, zi before zr) so GPU
// frames match the engine contract exactly. The limb count is a
// splice point: the wrapper substitutes __LIMB_COUNT__ per
// precision tier (limbCountForBits) before compiling — WGSL array
// lengths must be compile-time constants, one variant per tier.
// The limb-array primitives above (limb_array_math.js) use only
// u32 wrap arithmetic (WGSL has no u64 ALU): wide products split
// into 16-bit halves, carries are wrap-detected (sum < operand),
// and mulWord32's high word is kept below 2^32 by construction
// (see its comment for the bound).
//
// Bindings: (0) uniforms — frame scalars plus the palette (three
// f32 cosine phases, packed interior rgb) — (1) read-only limb
// inputs in one array — threshold(2N) then planeLeft(N), planeTop(N),
// pixelStep(N), subStepX(N), subStepY(N). pixelColorPacked returns
// the pixel's averaged rgba8-packed
// color, r | g<<8 | b<<16 | 255<<24 (little-endian byte order);
// fs unpacks it into its vec4 output, and the 8-bit unorm store
// maps k/255 straight back to k — exact. The palette cycle runs
// in-shader off magnitudeLog2Approx's log2(|z|²) — the engine's
// inner term is log2(|z|²)/2, so it feeds through directly
// (halved); the f32 log2/cos pipeline is the only deviation from
// the retired f64 CPU paint — sub-palette-cosine.
// ================================================================

// --- constants: buffer layout ---
const THRESHOLD_INPUT_OFFSET: u32 = 0u;                                           // escape threshold, 2N limbs — first limbInputs entry
const PLANE_LEFT_INPUT_OFFSET: u32 = THRESHOLD_INPUT_OFFSET + PRODUCT_LIMB_COUNT; // plane left edge, N limbs
const PLANE_TOP_INPUT_OFFSET: u32 = PLANE_LEFT_INPUT_OFFSET + LIMB_COUNT;         // plane top edge, N limbs
const PIXEL_STEP_INPUT_OFFSET: u32 = PLANE_TOP_INPUT_OFFSET + LIMB_COUNT;         // complex units per pixel, N limbs
const SUB_STEP_X_INPUT_OFFSET: u32 = PIXEL_STEP_INPUT_OFFSET + LIMB_COUNT;        // sub-pixel x step, N limbs
const SUB_STEP_Y_INPUT_OFFSET: u32 = SUB_STEP_X_INPUT_OFFSET + LIMB_COUNT;        // sub-pixel y step, N limbs

// --- constants: supersampling lattice (spliced from common.js at script load) ---
const SAMPLE_COLUMNS_PER_PIXEL: u32 = ${SAMPLE_COLUMNS_PER_PIXEL}u; // x sub-positions per pixel — offsets at odd multiples of subStepX
const SAMPLE_ROWS_PER_PIXEL: u32 = ${SAMPLE_ROWS_PER_PIXEL}u;       // y sub-positions — offsets at odd multiples of subStepY; columns × rows orbits per pixel

// --- constants: palette (f32 — the kernel colors pixels itself) ---
const CHANNEL_MAX: u32 = 255u;                          // 8-bit channel ceiling, also the packed alpha
const FULL_TURN_RADIANS: f32 = 6.283185307179586;      // palette cosine cycles once across the full escape range
const PALETTE_CYCLE_ITERATIONS: f32 = 16.0;            // one palette cycle per 16 smooth iterations — sqrt warp
const SMOOTH_LOG_FLOOR: f32 = 1e-4;                    // floor on log2(log2|z|²/2) so near-threshold escapes can't push mu past +∞
const MIN_MAGNITUDE_SQUARED_LOG2: f32 = 0.000144262291; // log2(1.0001) — |z|² ≤ 1.0001 (complex units) is treated as un-escaped

struct KernelParams {
  fractionBits: u32,     // fixed-point fractional width of this frame
  iterationBudget: u32,  // max orbit iterations (from the dive schedule)
  widthPx: u32,          // canvas width in pixels
  heightPx: u32,         // canvas height in pixels
  phaseR: f32,           // palette cosine phase per channel (radians, PALETTE contract from common.js)
  phaseG: f32,
  phaseB: f32,
  interiorPacked: u32,   // interior color, r | g<<8 | b<<16 (little-endian byte order)
};
@group(0) @binding(0) var<uniform> params: KernelParams;
@group(0) @binding(1) var<storage, read> limbInputs: array<u32>;

// --- orbit kernel helpers (mirrors engine.js iterateMandelbrotOrbit) ---

// a = input[base .. base+N)
// Returns the input limbs at baseOffset (threshold, plane edges, or
// pixel step) packed as an N-limb fixed-point value.
fn loadLimbs(baseOffset: u32) -> limb_array_int {
  var loadedLimbs: limb_array_int;
  for (var limbIndex = 0u; limbIndex < LIMB_COUNT; limbIndex++) {
    loadedLimbs[limbIndex] = limbInputs[baseOffset + limbIndex];
  }
  return loadedLimbs;
}

// log2(|z|²)
// Returns log2(|z|² / 2^(2·fractionBits)) from the top nonzero limb
// — the one float the in-shader palette math needs (f32 log2 vs
// the retired f64 paint is a sub-palette-cosine deviation).
fn magnitudeLog2Approx(magnitudeLimbs: limb_array_product) -> f32 {
  var topIndex = PRODUCT_LIMB_COUNT - 1u;
  loop {
    if (topIndex == 0u || magnitudeLimbs[topIndex] != 0u) { break; }
    topIndex = topIndex - 1u;
  }
  let log2Magnitude = f32(32u * topIndex) + log2(f32(max(magnitudeLimbs[topIndex], 1u)));
  return log2Magnitude - f32(2u * params.fractionBits);
}
`;
