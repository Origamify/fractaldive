'use strict';
/* ================================================================
 * Reference orbit WGSL — the per-frame Z-table builder (compute
 * shader). Assembles the u32-limb math library ahead of the
 * reference compute source at script load; engine.js splices
 * __LIMB_COUNT__ = REFERENCE_LIMB_COUNT before compiling (a
 * constant tier — the exact half of perturbation never varies
 * with zoom depth).
 *
 * Computes the reference orbit Z ← Z² + C for the frame center C,
 * once per frame, EXACTLY in u32-limb wrap arithmetic.
 * - rows stored as df32 pairs (reHi, reLo, imHi, imLo — ~48
 *   significant bits), the format the f32 delta kernel multiplies
 *   against (table error enters scaled by ε → per-step linear-term
 *   error ~2⁻⁴⁵·|ε|)
 * - table lives entirely on the GPU (binding 2: read_write here,
 *   read-only in the render kernel) — nothing returns to the CPU
 *
 * Layout: one vec4u per row after a one-element header carrying
 * rowCount in .x (header pads to 16 bytes — an array<vec4u>
 * binding's size must be a multiple of its element stride).
 * - rows 0 and 1 (Z₀ = 0, Z₁ = C) always stored — rebasing needs both
 * - first row with |Z|² > REFERENCE_STOP_MAGNITUDE_SQUARED stored
 *   too (the cap-crosser); table ends there or at the budget
 * ================================================================ */

/* --- reference compute source (WGSL, not JavaScript) --- */
const REFERENCE_COMPUTE_WGSL = `
// ================================================================
// Reference orbit — exact Z-table build for the perturbation
// kernel (one invocation per frame; engine_frame.js dispatches 1
// workgroup of 1).
//
// Bindings:
// - (0) uniforms — the SAME 8-word KernelParams layout the render
//   kernel reads (one buffer, both passes); this shader touches
//   referenceFractionBits and iterationBudget only
// - (1) read-only limb inputs — centerReal (N), centerImag (N),
//   stop threshold (2N, = 4 << 2·referenceFractionBits)
// - (2) the reference table, read_write here
//
// Mirrors the retired exact kernel's z ← z² + c step exactly
// (escape check before update, three-squares cross identity,
// arithmetic shift by referenceFractionBits) — one orbit instead
// of ~10⁶, amortized to nothing.
// ================================================================

// --- constants: input layout ---
const REFERENCE_REAL_INPUT_OFFSET: u32 = 0u;                    // center real part, N limbs at reference limb width
const REFERENCE_IMAG_INPUT_OFFSET: u32 = REFERENCE_REAL_INPUT_OFFSET + LIMB_COUNT; // center imag part
const REFERENCE_STOP_INPUT_OFFSET: u32 = REFERENCE_IMAG_INPUT_OFFSET + LIMB_COUNT; // stop threshold, 2N limbs at product width

// --- constants: table layout (shared with the render kernel's bindings chunk — offsets derive, never magic) ---
const REFERENCE_TABLE_HEADER_ELEMENTS: u32 = 1u; // element 0 = (rowCount, 0, 0, 0); rows start after it
const REFERENCE_TABLE_STRIDE: u32 = 1u;          // one vec4u element per row: reHi | reLo | imHi | imLo — one 16-byte vector load on the read side

struct KernelParams {
  referenceFractionBits: u32,     // 148 — value scale of the limb format (LSB 2^-148, fixed_point.js)
  iterationBudget: u32,       // max orbit iterations (from the dive schedule) — also the row cap
  widthPx: u32,               // canvas width in pixels (unused here — shared-layout field)
  heightPx: u32,              // canvas height in pixels (unused here — shared-layout field)
  phaseR: f32,                // palette cosine phases (unused here — shared-layout fields)
  phaseG: f32,
  phaseB: f32,
  interiorPacked: u32,
};
@group(0) @binding(0) var<uniform> params: KernelParams;
@group(0) @binding(1) var<storage, read> referenceInputs: array<u32>;
@group(0) @binding(2) var<storage, read_write> referenceTable: array<vec4u>;

// a = input[base .. base+N)
// Returns N input limbs packed as one reference-limb-width fixed-point value.
fn loadLimbs(baseOffset: u32) -> limb_array_int {
  var loadedLimbs: limb_array_int;
  for (var limbIndex = 0u; limbIndex < LIMB_COUNT; limbIndex++) {
    loadedLimbs[limbIndex] = referenceInputs[baseOffset + limbIndex];
  }
  return loadedLimbs;
}

// --- df32 split: an exact reference-limb-format value → (hi, lo) f32 pair ---

// |v|
// Returns the magnitude (the shared lib has no abs — negate wraps only the sign).
fn limbMagnitude(v: limb_array_int) -> limb_array_int {
  var magnitude = v;
  if (limbIsNegative(v)) { magnitude = limbNegate(v); }
  return magnitude;
}

// bitLength(|m|)
// Returns the significant bit count of a non-negative limb array
// (0 for zero) — the split's normalization input.
fn limbBitLength(m: limb_array_int) -> u32 {
  var topIndex = LIMB_COUNT - 1u;
  loop {
    if (topIndex == 0u || m[topIndex] != 0u) { break; }
    topIndex = topIndex - 1u;
  }
  if (m[topIndex] == 0u) { return 0u; }
  return 32u * topIndex + (32u - countLeadingZeros(m[topIndex]));
}

// (n, shift): m ≈ n·2^shift
// Rounds m to its top 24 significant bits, round-half-even — n in
// [2^23, 2^24] (or m exactly, when shorter than 24 bits).
// - rounding rule mirrored bit-for-bit by the BigInt oracle
//   (tests/fractaldive_test.js)
fn top24Bits(m: limb_array_int) -> vec2u {
  let bitLength = limbBitLength(m);
  if (bitLength == 0u) { return vec2u(0u, 0u); }
  let shift = max(bitLength, 24u) - 24u;
  let limbIndex = shift / 32u;
  let bitIndex = shift % 32u;
  var window: u32;
  if (bitIndex == 0u) {
    window = m[limbIndex];
  } else {
    let lowPart = m[limbIndex] >> bitIndex;
    var highPart = 0u;
    if (limbIndex + 1u < LIMB_COUNT) { highPart = m[limbIndex + 1u] << (32u - bitIndex); }
    window = lowPart | highPart;
  }
  let truncated = window & 0xFFFFFFu;
  if (shift == 0u) { return vec2u(truncated, 0u); } // fewer than 24 significant bits — exact
  let roundLimb = (shift - 1u) / 32u;
  let roundBit = (shift - 1u) % 32u;
  let roundBitSet = (m[roundLimb] >> roundBit) & 1u;
  var sticky: u32 = 0u; // any nonzero bit strictly below the round bit
  if (roundBit > 0u) { sticky = m[roundLimb] & ((1u << roundBit) - 1u); }
  for (var lowerLimb: u32 = 0u; lowerLimb < roundLimb; lowerLimb++) {
    sticky = sticky | m[lowerLimb];
  }
  var rounded = truncated;
  if (roundBitSet == 1u && (sticky != 0u || (truncated & 1u) == 1u)) { rounded = truncated + 1u; }
  return vec2u(rounded, shift);
}

// n·2^shift scattered into limbs
// Places the ≤24-bit n at bit offset shift — n ≤ 2^24 spans at
// most two limbs; the exact subtrahend for the residual.
fn scatter24(n: u32, shift: u32) -> limb_array_int {
  var scattered: limb_array_int = limb_array_int();
  let limbIndex = shift / 32u;
  let bitIndex = shift % 32u;
  if (bitIndex == 0u) {
    scattered[limbIndex] = n;
  } else {
    scattered[limbIndex] = n << bitIndex;
    if (limbIndex + 1u < LIMB_COUNT) { scattered[limbIndex + 1u] = n >> (32u - bitIndex); }
  }
  return scattered;
}

// n·2^(shift−referenceFractionBits) → f32
// The scaled integer as an f32.
// - exact for every normal result (n ≤ 2^24, power-of-two scaling)
// - subnormal results forced to 0 explicitly → bit-identical on
//   FTZ and non-FTZ stacks alike (dropped |Z| < 2^-126 rows sit
//   ~26 bits below the deepest δ scale — tolerance-irrelevant)
fn scaledF32(n: u32, shift: u32) -> f32 {
  let exponent = i32(shift) - i32(params.referenceFractionBits);
  if (n == 0u || exponent < -126) { return 0.0; }
  return exp2(f32(exponent)) * f32(n);
}

// v → (hi, lo)
// The df32 pair of an exact signed value: hi = top 24 bits of v,
// lo = the exact residual rounded to ITS top 24 bits (~48 bits
// total).
// - pure integer + power-of-two arithmetic — no two-sum, nothing
//   for a fast-math compiler to break
fn limbToF32Pair(v: limb_array_int) -> vec2<f32> {
  let negative = limbIsNegative(v);
  let magnitude = limbMagnitude(v);
  let hiSplit = top24Bits(magnitude);
  let hi = scaledF32(hiSplit.x, hiSplit.y);
  var residual: limb_array_int;
  if (negative) {
    residual = limbAdd(limbNegate(magnitude), scatter24(hiSplit.x, hiSplit.y)); // −|v| + hiN·2^shift
  } else {
    residual = limbSub(magnitude, scatter24(hiSplit.x, hiSplit.y));             // |v| − hiN·2^shift
  }
  let residualNegative = limbIsNegative(residual); // set only when rounding went up past |v|
  let loSplit = top24Bits(limbMagnitude(residual));
  var lo = scaledF32(loSplit.x, loSplit.y);
  if (residualNegative) { lo = -lo; }
  if (negative) { return vec2<f32>(-hi, lo); }
  return vec2<f32>(hi, lo);
}

// row m ← (Zr, Zi)
// Stores one reference row as its df32 quadruple — the only table
// writer; single invocation, so no barriers.
fn storeReferenceRow(rowIndex: u32, realLimbs: limb_array_int, imagLimbs: limb_array_int) {
  let rowBase = REFERENCE_TABLE_HEADER_ELEMENTS + rowIndex * REFERENCE_TABLE_STRIDE;
  let realPair = limbToF32Pair(realLimbs);
  let imagPair = limbToF32Pair(imagLimbs);
  referenceTable[rowBase] = vec4u(
    bitcast<u32>(realPair.x), bitcast<u32>(realPair.y),
    bitcast<u32>(imagPair.x), bitcast<u32>(imagPair.y));
}

@compute @workgroup_size(1)
fn main() {
  // single invocation by contract — the dispatch is workgroups(1)
  let centerRealLimbs = loadLimbs(REFERENCE_REAL_INPUT_OFFSET);
  let centerImagLimbs = loadLimbs(REFERENCE_IMAG_INPUT_OFFSET);
  var stopThresholdLimbs: limb_array_product; // |Z|² cap at 2N width — the escape-compare width
  for (var limbIndex = 0u; limbIndex < PRODUCT_LIMB_COUNT; limbIndex++) {
    stopThresholdLimbs[limbIndex] = referenceInputs[REFERENCE_STOP_INPUT_OFFSET + limbIndex];
  }

  var zReal: limb_array_int = limb_array_int(); // Z₀ = 0 — the algebra that makes rebasing exact
  var zImag: limb_array_int = limb_array_int();
  storeReferenceRow(0u, zReal, zImag);
  var rowCount = 1u; // rows stored so far — table always carries Z₀ and Z₁

  loop {
    // stop check on the CURRENT (already stored) row: a row past
    // |Z|² > 4 is the cap-crosser — the table ends with it
    let zRealSquared = limbSquare(zReal);
    let zImagSquared = limbSquare(zImag);
    let magnitudeSquared = limbAddWide(zRealSquared, zImagSquared);
    if (limbUnsignedGreaterWide(magnitudeSquared, stopThresholdLimbs)) { break; }
    if (rowCount > params.iterationBudget) { break; } // rows 0..budget stored — the render loop never reads past that

    // Z ← Z² + C; cross term via the three-squares identity
    // 2·Zr·Zi = (Zr+Zi)² − Zr² − Zi² — same step the exact engine ran
    let crossDoubled = limbSubWide(
      limbSubWide(limbSquare(limbAdd(zReal, zImag)), zRealSquared),
      zImagSquared);
    zImag = limbAdd(
      limbTruncateLow(limbShiftRightArithmeticWide(crossDoubled, params.referenceFractionBits)),
      centerImagLimbs);
    zReal = limbAdd(
      limbTruncateLow(limbShiftRightArithmeticWide(limbSubWide(zRealSquared, zImagSquared), params.referenceFractionBits)),
      centerRealLimbs);
    storeReferenceRow(rowCount, zReal, zImag);
    rowCount++;
  }
  referenceTable[0] = vec4u(rowCount, 0u, 0u, 0u);
}
`;

/* --- assembled reference kernel (WGSL, not JavaScript) --- */
const REFERENCE_KERNEL_WGSL = LIMB_ARRAY_MATH_WGSL + REFERENCE_COMPUTE_WGSL;
