'use strict';
/* ================================================================
 * Perturb kernel WGSL — the f32 delta render kernel's full
 * source: five chunks (bindings, palette, delta, supersample,
 * render) assembled at script load into PERTURB_KERNEL_WGSL, in
 * that order below.
 *
 * PURE f32 — no limb library, no __LIMB_COUNT__ splice: exact limb
 * math lives in the reference compute pass only; the delta side's
 * width is f32's own 24 bits at every zoom depth.
 * - palette + render chunks stay named constants — the
 *   fractaldive_test oracle splices them into the retired exact
 *   engine's kernel (tests/oracle/*)
 * - the oracle chunks are not loaded by the app; they remain the
 *   frozen oracle the test checks this kernel against
 * ================================================================ */

/* --- kernel source chunk (WGSL, not JavaScript) — bindings: preamble --- */
const PERTURB_KERNEL_WGSL_BINDINGS = `
// ================================================================
// Perturbation kernel — plain-f32 delta iteration for one
// sub-pixel sample against the df32 reference table.
// - update: ε ← 2·(Z_hi·ε + Z_lo·ε) + ε² + δ; escape checked on
//   the reconstructed z = Z + ε
// - Zhuoran rebase when |z| < |ε| or the table runs out
// - the FIRST lattice sample also tracks D = dz/dc (D ← 2·z·D + 1)
//   for the adaptive-supersampling distance test — delta chunk
// - tolerance contract: frames match the exact engine modulo
//   boundary pixels (stored-ε rounding 2⁻²³·|ε|/step surfaces
//   as occasional ±1 smooth counts)
//
// Why f32 is enough:
// - Z rides as a df32 pair (~48 bits; its rounding error enters
//   scaled by |ε|)
// - rebase re-anchors ε at fresh f32 precision whenever it grows
// - dive floor 1e-35 keeps every δ within ~2× of f32's minimum
//   normal (~1.2e-38); a flushed subnormal δ renders that
//   sub-sample as the reference — an error below its δ
// - magnitudes stay tiny (|Z| ≤ 6.3, |ε| ≤ ~7e4, |z|² ≤ 2¹⁸) —
//   no wrap analysis, saturation never reached
//
// Bindings:
// - (0) uniforms — frame scalars + palette, the SAME 8-word
//   layout the reference pass reads (one buffer, both passes);
//   this kernel touches iterationBudget + palette fields
// - (1) read-only frame inputs — five f32 bit patterns: plane
//   left/top δ offsets, pixel step, two lattice sub-steps
// - (2) the df32 reference table built this frame — one vec4u per
//   row (reHi | reLo | imHi | imLo) after a one-element header
//   carrying rowCount in .x
//
// Colors: palette cycle runs in-shader off f32 log2(|z|²) — the
// retired exact kernel's formula, native instead of
// limb-approximated. Output rgba8 packed r | g<<8 | b<<16 | 255<<24;
// the 8-bit unorm store maps k/255 straight back to k — exact.
// ================================================================

// --- constants: buffer layout ---
const FRAME_INPUT_FIELD_WORDS: u32 = 1u; // every frame input is one f32 word — offsets chain off this
const PLANE_LEFT_INPUT_OFFSET: u32 = 0u; // f32: plane left edge − reference center (δ real part of the corner pixel)
const PLANE_TOP_INPUT_OFFSET: u32 = PLANE_LEFT_INPUT_OFFSET + FRAME_INPUT_FIELD_WORDS; // f32: plane top edge − reference center (δ imag part)
const PIXEL_STEP_INPUT_OFFSET: u32 = PLANE_TOP_INPUT_OFFSET + FRAME_INPUT_FIELD_WORDS; // f32: complex units per pixel
const SUB_STEP_X_INPUT_OFFSET: u32 = PIXEL_STEP_INPUT_OFFSET + FRAME_INPUT_FIELD_WORDS; // f32: lattice x sub-step (step / 2·columns)
const SUB_STEP_Y_INPUT_OFFSET: u32 = SUB_STEP_X_INPUT_OFFSET + FRAME_INPUT_FIELD_WORDS; // f32: lattice y sub-step (step / 2·rows)

// --- constants: reference table layout (mirrors reference_orbit_wgsl.js) ---
const REFERENCE_TABLE_HEADER_ELEMENTS: u32 = 1u; // element 0 = (rowCount, 0, 0, 0) — rows start after it
const REFERENCE_TABLE_STRIDE: u32 = 1u;          // one vec4u element per row: reHi | reLo | imHi | imLo

// --- constants: escape ---
const ESCAPE_MAGNITUDE_SQUARED: f32 = 65536.0; // |z|² bailout — the same constant the exact engine's threshold encoded

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
  referenceFractionBits: u32,  // 148 — the reference limb format's LSB (unused here — shared-layout field)
  iterationBudget: u32,    // max orbit iterations (from the dive schedule)
  widthPx: u32,            // canvas width in pixels (unused by the math — shared-layout field)
  heightPx: u32,           // canvas height in pixels (unused by the math — shared-layout field)
  phaseR: f32,             // palette cosine phase per channel (radians, PALETTE contract from common.js)
  phaseG: f32,
  phaseB: f32,
  interiorPacked: u32,     // interior color, r | g<<8 | b<<16 (little-endian byte order)
};
@group(0) @binding(0) var<uniform> params: KernelParams;
@group(0) @binding(1) var<storage, read> frameInputs: array<u32>;
@group(0) @binding(2) var<storage, read> referenceTable: array<vec4u>;

// --- kernel helpers ---

// input[offset] → f32
// Returns one frame constant — the buffer stores f32 bit patterns.
fn loadFrameF32(offset: u32) -> f32 {
  return bitcast<f32>(frameInputs[offset]);
}
`;

/* --- kernel source chunk (WGSL, not JavaScript) — palette (shared with the test oracle) --- */
const PERTURB_KERNEL_WGSL_PALETTE = `
// --- palette math (f32 — mirrors the retired engine.js CPU paint) ---

// ch(t) = round(1/2 + 1/2·cos(2π·t + φ))
// Returns one palette channel: cosine wave over the palette
// position, own phase offset (round ties away from zero — same as
// Math.round on these non-negative 0..255 values).
fn paletteChannel(palettePosition: f32, phaseRadians: f32) -> u32 {
  let half = f32(CHANNEL_MAX) / 2.0;
  return u32(round(half + half * cos(FULL_TURN_RADIANS * palettePosition + phaseRadians)));
}

// t = frac(sqrt(n_smooth / 16))
// Returns the palette position in [0, 1) for an escape.
// - smooth (fractional) iteration count from log2(|z|²) in
//   complex units, sqrt-warped, one cycle per PALETTE_CYCLE_ITERATIONS
fn paletteCycleFraction(iterationsUsed: u32, log2MagnitudeSquared: f32) -> f32 {
  var smoothIterations = f32(iterationsUsed);
  if (log2MagnitudeSquared > MIN_MAGNITUDE_SQUARED_LOG2) {
    smoothIterations = smoothIterations - log2(max(SMOOTH_LOG_FLOOR, log2MagnitudeSquared / 2.0));
  }
  let cycle = sqrt(max(smoothIterations, 0.0)) / sqrt(PALETTE_CYCLE_ITERATIONS);
  return cycle - floor(cycle);
}

// rgba8(t) = r | g<<8 | b<<16 | 255<<24
// Returns an escaped pixel's packed rgba8 color: cosine channels,
// opaque alpha, little-endian byte order (r, g, b, a).
fn escapeColorPacked(palettePosition: f32) -> u32 {
  return paletteChannel(palettePosition, params.phaseR)
    | (paletteChannel(palettePosition, params.phaseG) << 8u)
    | (paletteChannel(palettePosition, params.phaseB) << 16u)
    | (CHANNEL_MAX << 24u);
}
`;

/* --- kernel source chunk (WGSL, not JavaScript) — delta: the perturbation orbit --- */
const PERTURB_KERNEL_WGSL_DELTA = `
// row m → (reHi, reLo, imHi, imLo)
// Returns reference table row m as its df32 quadruple — Z_m, the
// full-precision half of the perturbation pair (built this frame
// by reference_orbit_wgsl.js).
// - one 16-byte vec4u load per iteration instead of four scalar
//   loads — the table read is the inner loop's memory traffic
fn loadReferenceRow(referenceIndex: u32) -> vec4<f32> {
  let row = referenceTable[REFERENCE_TABLE_HEADER_ELEMENTS + referenceIndex * REFERENCE_TABLE_STRIDE];
  return vec4<f32>(bitcast<f32>(row.x), bitcast<f32>(row.y), bitcast<f32>(row.z), bitcast<f32>(row.w));
}

// rowCount → u32
// Returns the number of reference rows stored this frame (the
// table header's .x — ≥ 2 by the build contract: Z₀ = 0 and Z₁ = C
// always).
fn referenceTableLength() -> u32 {
  return referenceTable[0].x;
}

// --- constants: adaptive supersampling (distance estimation — first lattice sample only) ---
// D = dz/dc rides ONE orbit per pixel: D ← 2·z·D + 1 per
// iteration (z already reconstructed for the escape check; D is
// a property of z, not ε — rebase re-expresses ε only, so D
// needs no rebase handling). At escape the exterior distance
// estimate DE ≈ |z|·ln|z|/|D| bounds the distance to the set and
// the iteration-band density (|∇smooth| ≈ 1/(ln2·DE)):
// - DE > SSAA_FLAT_DISTANCE_FACTOR·pixelStep ⇒ the pixel's color
//   cannot change within the pixel ⇒ the remaining lattice
//   orbits are skippable
// - |D| grows ~2ⁿ, overflowing f32 past n ≈ 128 → D renormalizes
//   by exact 2⁻³² factors under an i32 exponent; the recurrence's
//   +1 rides its own 2⁻³²ᵏ scale (derivativeScale)
// - the DE test compares in log2 space, reusing the escape's
//   log2(|z|²)
const DERIVATIVE_RENORMALIZE_THRESHOLD: f32 = 1073741824.0;       // 2³⁰ per component — one 2·|z|·|D| step (|z| ≤ 256) still fits f32 with margin
const DERIVATIVE_RENORMALIZE_SCALE: f32 = 2.3283064365386963e-10; // 2⁻³² — exact power of two, renormalization rounds nothing
const DERIVATIVE_MAGNITUDE_SQUARED_FLOOR: f32 = 1e-30;            // log2 guard — a pathological D cancellation cannot reach −∞
const NATURAL_LOG_2: f32 = 0.6931471805599453;
const SSAA_FLAT_DISTANCE_FACTOR: f32 = 2.0;                       // pixel-step clearance DE must show before a pixel renders single-sample (raise past any screen's DE range to disable adaptivity)

struct SampleResult {
  color: u32,     // packed rgba8 word — interior word or palette escape color
  flat: bool,     // tracked escapes only: DE > factor·step — color provably constant across the pixel, the lattice is skippable
}

// sample(δ, trackDistance) → SampleResult { color, flat }
// Perturbation orbit for ONE sub-pixel sample, plain f32.
// - ε ← 2·(Z_hi·ε + Z_lo·ε) + ε² + δ replaying the df32 table;
//   escape on z = Z + ε, checked BEFORE budget (the exact
//   engine's order)
// - Zhuoran rebase when |z| < |ε| or the last row: ε ← z against
//   Z₀ = 0, table consumed again from row 0 — with rounded ε the
//   rebase is also the precision reset that keeps 24 bits alive
// - trackDistance also steps D = dz/dc (constants above) and
//   classifies the escape's flatness — the verdict is returned
//   explicitly, never guessed from a color word
fn sampleColorPacked(deltaReal: f32, deltaImag: f32, trackDistance: bool) -> SampleResult {
  let tableLength = referenceTableLength();
  var epsilonReal = 0.0;
  var epsilonImag = 0.0;
  var referenceIndex = 0u;
  var iterationsUsed = 0u;
  var derivativeReal = 0.0;    // D = dz/dc, tracked samples only — renormalized mantissa
  var derivativeImag = 0.0;
  var derivativeExponent = 0;  // i32 — true D = stored D · 2^exponent (D₀ = 0)
  var derivativeScale = 1.0;   // the recurrence's +1 term at stored scale — 2^-exponent (exact power of two)
  var result: SampleResult = SampleResult(params.interiorPacked | (CHANNEL_MAX << 24u), false); // budget-exhaustion result — the loop breaks with it unchanged
  loop {
    var referenceRow = loadReferenceRow(referenceIndex);
    let zReal = (referenceRow.x + referenceRow.y) + epsilonReal;
    let zImag = (referenceRow.z + referenceRow.w) + epsilonImag;
    let magnitudeSquared = zReal * zReal + zImag * zImag;
    if (magnitudeSquared > ESCAPE_MAGNITUDE_SQUARED) {
      let log2MagnitudeSquared = log2(magnitudeSquared);
      result.color = escapeColorPacked(paletteCycleFraction(iterationsUsed, log2MagnitudeSquared));
      if (trackDistance) {
        // flat ⇔ DE > factor·step, compared in log2 space:
        // log2(|z|·ln|z|) − (exponent + ½·log2|D|²) − log2(step) > log2(factor)
        let naturalLogMagnitude = log2MagnitudeSquared * 0.5 * NATURAL_LOG_2;
        let log2Derivative = f32(derivativeExponent)
          + 0.5 * log2(max(DERIVATIVE_MAGNITUDE_SQUARED_FLOOR,
              derivativeReal * derivativeReal + derivativeImag * derivativeImag));
        let log2DistancePerPixelStep = 0.5 * log2MagnitudeSquared + log2(naturalLogMagnitude)
          - log2Derivative - log2(loadFrameF32(PIXEL_STEP_INPUT_OFFSET));
        result.flat = log2DistancePerPixelStep > log2(SSAA_FLAT_DISTANCE_FACTOR);
      }
      break;
    }
    if (iterationsUsed >= params.iterationBudget) {
      break;
    }
    if (referenceIndex == tableLength - 1u || magnitudeSquared < epsilonReal * epsilonReal + epsilonImag * epsilonImag) {
      // rebase: re-express the pixel value against Z₀ = 0 and
      // restart the table; row-0 update degenerates to ε² + δ
      // (2Z₀ = 0) — same algebra, re-based
      epsilonReal = zReal;
      epsilonImag = zImag;
      referenceIndex = 0u;
      referenceRow = loadReferenceRow(0u);
    }
    // ε ← 2·(Z·ε) + ε² + δ — Z as its df32 pair:
    // - hi products carry the term, lo products patch it to
    //   ~2⁻⁴⁵ relative (product-of-parts, no error-free transforms —
    //   nothing for a fast-math compiler to rewrite)
    let linearReal = referenceRow.x * epsilonReal - referenceRow.z * epsilonImag
      + referenceRow.y * epsilonReal - referenceRow.w * epsilonImag;
    let linearImag = referenceRow.x * epsilonImag + referenceRow.z * epsilonReal
      + referenceRow.y * epsilonImag + referenceRow.w * epsilonReal;
    let squaredReal = epsilonReal * epsilonReal - epsilonImag * epsilonImag;
    let squaredImag = 2.0 * epsilonReal * epsilonImag;
    epsilonReal = 2.0 * linearReal + squaredReal + deltaReal;
    epsilonImag = 2.0 * linearImag + squaredImag + deltaImag;
    if (trackDistance) {
      // D ← 2·z·D + 1 — this iteration's z (the escape check's
      // reconstruction); D belongs to z, and rebase re-expresses
      // ε only, so D carries across rebases unchanged. The +1
      // rides derivativeScale: after k rescales the stored
      // recurrence is 2·z·D′ + 2⁻³²ᵏ — an unscaled +1 next to a
      // rescaled D′ would swamp it
      let nextDerivativeReal = 2.0 * (zReal * derivativeReal - zImag * derivativeImag) + derivativeScale;
      derivativeImag = 2.0 * (zReal * derivativeImag + zImag * derivativeReal);
      derivativeReal = nextDerivativeReal;
      if (max(abs(derivativeReal), abs(derivativeImag)) > DERIVATIVE_RENORMALIZE_THRESHOLD) {
        // exact 2⁻³² rescale — the exponent carries what the
        // mantissa dropped; components re-enter [0.25, 2³⁹)
        derivativeReal = derivativeReal * DERIVATIVE_RENORMALIZE_SCALE;
        derivativeImag = derivativeImag * DERIVATIVE_RENORMALIZE_SCALE;
        derivativeScale = derivativeScale * DERIVATIVE_RENORMALIZE_SCALE;
        derivativeExponent = derivativeExponent + 32;
      }
    }
    iterationsUsed++;
    referenceIndex++;
  }
  return result;
}
`;

/* --- kernel source chunk (WGSL, not JavaScript) — supersample: the adaptive lattice wrapper --- */
const PERTURB_KERNEL_WGSL_SUPERSAMPLE = `
// color(x, y) → packed rgba8 word, adaptively supersampled
// Runs sampleColorPacked at the pixel's sub-pixel lattice.
// - base δ = planeLeft + x·step (real), planeTop − y·step (imag)
//   — complex y up / screen y down; sub-positions at odd
//   multiples of the sub-steps (±step/4 per axis at 2×2)
// - the FIRST sample runs tracked (distance estimator): a flat
//   verdict ⇒ its color IS the pixel — no averaging, remaining
//   orbits skipped; otherwise the full lattice runs (samples
//   1..N untracked — the verdict is spent) and averages
//   bit-identically to the fixed-lattice engine; interior first
//   samples never early out — one interior sub-sample proves
//   nothing about siblings
// - flat pixels the fixed engine would have averaged carry at
//   most sub-band color drift — same divergence class as a
//   boundary flip, inside the tolerance contract
// - channel average = (Σ + sampleCount/2) / sampleCount — round
//   half up, integer-exact; alpha 255 everywhere stays 255
// - real pixels only — the target is exactly the frame, no bounds check
fn pixelColorPacked(pixelX: u32, pixelY: u32) -> u32 {
  let pixelStep = loadFrameF32(PIXEL_STEP_INPUT_OFFSET);
  let baseDeltaReal = loadFrameF32(PLANE_LEFT_INPUT_OFFSET) + f32(pixelX) * pixelStep;
  let baseDeltaImag = loadFrameF32(PLANE_TOP_INPUT_OFFSET) - f32(pixelY) * pixelStep;
  let subStepX = loadFrameF32(SUB_STEP_X_INPUT_OFFSET);
  let subStepY = loadFrameF32(SUB_STEP_Y_INPUT_OFFSET);
  var redSum: u32 = 0u;
  var greenSum: u32 = 0u;
  var blueSum: u32 = 0u;
  // the FIRST lattice sample runs tracked (distance estimator —
  // delta chunk): (1 − columns, 1 − rows) sub-steps, the same
  // position the fixed lattice visited first (−step/4, −step/4
  // for 2×2)
  let firstDeltaReal = baseDeltaReal + f32(1 - i32(SAMPLE_COLUMNS_PER_PIXEL)) * subStepX;
  let firstDeltaImag = baseDeltaImag - f32(1 - i32(SAMPLE_ROWS_PER_PIXEL)) * subStepY;
  let first = sampleColorPacked(firstDeltaReal, firstDeltaImag, true);
  if (first.flat) {
    // distance-estimator early-out — see the chunk header: the
    // color is provably constant across the pixel
    return first.color;
  }
  redSum = redSum + (first.color & 0xFFu);
  greenSum = greenSum + ((first.color >> 8u) & 0xFFu);
  blueSum = blueSum + ((first.color >> 16u) & 0xFFu);
  for (var rowIndex = 0u; rowIndex < SAMPLE_ROWS_PER_PIXEL; rowIndex++) {
    let rowOffset = 2 * i32(rowIndex) + 1 - i32(SAMPLE_ROWS_PER_PIXEL); // odd multiples: −3, −1, +1, +3 for 4 rows
    // f32(rowOffset) is an exact small integer, so the signed multiply is bit-identical to the ±|offset|·subStep form (and −(−b) = +b exactly in IEEE)
    let sampleDeltaImag = baseDeltaImag - f32(rowOffset) * subStepY; // −rowOffset·subStepY — complex y grows up (screen y down)
    for (var columnIndex = 0u; columnIndex < SAMPLE_COLUMNS_PER_PIXEL; columnIndex++) {
      if (rowIndex == 0u && columnIndex == 0u) {
        continue; // the tracked first sample is already accumulated
      }
      let columnOffset = 2 * i32(columnIndex) + 1 - i32(SAMPLE_COLUMNS_PER_PIXEL); // odd multiples: −3, −1, +1, +3 for 4 columns
      let sampleDeltaReal = baseDeltaReal + f32(columnOffset) * subStepX; // +columnOffset·subStepX
      let sample = sampleColorPacked(sampleDeltaReal, sampleDeltaImag, false);
      redSum = redSum + (sample.color & 0xFFu);
      greenSum = greenSum + ((sample.color >> 8u) & 0xFFu);
      blueSum = blueSum + ((sample.color >> 16u) & 0xFFu);
    }
  }
  let sampleCount = SAMPLE_COLUMNS_PER_PIXEL * SAMPLE_ROWS_PER_PIXEL;
  let roundingHalf = sampleCount / 2u; // round half up on the channel average
  return ((redSum + roundingHalf) / sampleCount)
    | (((greenSum + roundingHalf) / sampleCount) << 8u)
    | (((blueSum + roundingHalf) / sampleCount) << 16u)
    | (CHANNEL_MAX << 24u);
}
`;

/* --- kernel source chunk (WGSL, not JavaScript) — render: the vs/fs entries (shared with the test oracle) --- */
const PERTURB_KERNEL_WGSL_RENDER = `
// --- render entries: one fullscreen triangle, one fragment per pixel ---

// vs(i) → clip-space position
// Returns one of three hardcoded positions covering the viewport.
// - NDC (−1,−1) = top-left: WebGPU render passes share the
//   kernel's top-left, y-down convention, so no flip
@vertex
fn vs(@builtin(vertex_index) vertexIndex: u32) -> @builtin(position) vec4f {
  var clipPositions = array<vec2f, 3>(vec2f(-1.0, -1.0), vec2f(3.0, -1.0), vec2f(-1.0, 3.0));
  return vec4f(clipPositions[vertexIndex], 0.0, 1.0);
}

// fs(p) → rgba color
// Colors one pixel (supersampled — see pixelColorPacked).
// - position is the pixel center (+0.5) — truncation gives the
//   integer pixel index
// - fragment output components map to the target format's
//   channels by name → same shader correct on rgba8unorm and
//   bgra8unorm alike; unorm store maps k/255 back to k exactly
@fragment
fn fs(@builtin(position) fragmentPosition: vec4f) -> @location(0) vec4f {
  let packedColor = pixelColorPacked(u32(fragmentPosition.x), u32(fragmentPosition.y));
  return vec4f(
    f32(packedColor & 0xFFu) / 255.0,
    f32((packedColor >> 8u) & 0xFFu) / 255.0,
    f32((packedColor >> 16u) & 0xFFu) / 255.0,
    1.0);
}
`;

/* --- assembled kernel (WGSL, not JavaScript) — chunks spliced in order --- */
const PERTURB_KERNEL_WGSL = PERTURB_KERNEL_WGSL_BINDINGS
  + PERTURB_KERNEL_WGSL_PALETTE
  + PERTURB_KERNEL_WGSL_DELTA
  + PERTURB_KERNEL_WGSL_SUPERSAMPLE
  + PERTURB_KERNEL_WGSL_RENDER;
