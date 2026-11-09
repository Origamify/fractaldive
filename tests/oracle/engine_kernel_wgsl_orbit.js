'use strict';
/* ================================================================
 * Engine kernel WGSL — orbit chunk (RETIRED exact engine,
 * oracle only — loaded by tests/perturb_frame_test.html, never
 * by the app): the exact fixed-point
 * Mandelbrot orbit per sub-pixel sample (sampleColorPacked) and
 * its supersampling wrapper (pixelColorPacked). Spliced after
 * the palette chunk.
 * ================================================================ */

/* --- kernel source chunk (WGSL, not JavaScript) --- */
const ENGINE_KERNEL_WGSL_ORBIT = `
// color(cr, ci) → packed rgba8 word
// z ← z² + c for ONE sub-pixel sample at the exact fixed-point c
// (until |z|² > threshold or budget spent): the cosine palette for
// escapes, the palette's flat color for interior. The supersampling
// wrapper below averages columns × rows of these per pixel.
fn sampleColorPacked(realLimbs: limb_array_int, imagLimbs: limb_array_int) -> u32 {
  var thresholdLimbs: limb_array_product; // escape threshold at 2N width
  for (var limbIndex = 0u; limbIndex < PRODUCT_LIMB_COUNT; limbIndex++) {
    thresholdLimbs[limbIndex] = limbInputs[THRESHOLD_INPUT_OFFSET + limbIndex];
  }
  var zr: limb_array_int = limb_array_int();
  var zi: limb_array_int = limb_array_int();
  var zrSquared: limb_array_product;
  var ziSquared: limb_array_product;
  var iterationsUsed = 0u;
  var colorWord = params.interiorPacked | (CHANNEL_MAX << 24u); // budget-exhaustion result — the loop breaks with it unchanged
  loop {
    zrSquared = limbSquare(zr);
    ziSquared = limbSquare(zi);
    let magnitudeSquared = limbAddWide(zrSquared, ziSquared);
    if (limbUnsignedGreaterWide(magnitudeSquared, thresholdLimbs)) {
      colorWord = escapeColorPacked(paletteCycleFraction(iterationsUsed, magnitudeLog2Approx(magnitudeSquared)));
      break;
    }
    if (iterationsUsed >= params.iterationBudget) {
      break;
    }
    // z ← z² + c; the cross term uses the three-squares identity
    // 2·zr·zi = (zr+zi)² − zr² − zi² — one more square replaces the
    // signed multiply, the value is born doubled so the shift is the
    // plain fractionBits (same as the zr line)
    let crossDoubled = limbSubWide(
      limbSubWide(limbSquare(limbAdd(zr, zi)), zrSquared),
      ziSquared);
    zi = limbAdd(
      limbTruncateLow(limbShiftRightArithmeticWide(crossDoubled, params.fractionBits)),
      imagLimbs);
    zr = limbAdd(
      limbTruncateLow(limbShiftRightArithmeticWide(limbSubWide(zrSquared, ziSquared), params.fractionBits)),
      realLimbs);
    iterationsUsed++;
  }
  return colorWord;
}

// color(x, y) → packed rgba8 word, supersampled
// Runs sampleColorPacked at the pixel's sub-pixel lattice: base c
// = planeLeft + x·step + i·(planeTop − y·step), then odd multiples
// of the sub-steps (±step/4 on each axis for the 2×2 lattice;
// screen y grows down). Channel
// average = (Σ + sampleCount/2) / sampleCount — round half up,
// integer-exact; alpha is 255 in every sample so it stays 255.
// Called only for real pixels — the render target is exactly the
// frame, so no bounds check.
fn pixelColorPacked(pixelX: u32, pixelY: u32) -> u32 {
  let baseRealLimbs = limbAdd(
    limbMultiplySmall(loadLimbs(PIXEL_STEP_INPUT_OFFSET), pixelX),
    loadLimbs(PLANE_LEFT_INPUT_OFFSET));
  let baseImagLimbs = limbSub(
    loadLimbs(PLANE_TOP_INPUT_OFFSET),
    limbMultiplySmall(loadLimbs(PIXEL_STEP_INPUT_OFFSET), pixelY));
  let subStepXLimbs = loadLimbs(SUB_STEP_X_INPUT_OFFSET);
  let subStepYLimbs = loadLimbs(SUB_STEP_Y_INPUT_OFFSET);
  var redSum: u32 = 0u;
  var greenSum: u32 = 0u;
  var blueSum: u32 = 0u;
  for (var rowIndex = 0u; rowIndex < SAMPLE_ROWS_PER_PIXEL; rowIndex++) {
    let rowOffset = 2 * i32(rowIndex) + 1 - i32(SAMPLE_ROWS_PER_PIXEL); // odd multiples: −3, −1, +1, +3 for 4 rows
    let rowStepScaled = limbMultiplySmall(subStepYLimbs, u32(abs(rowOffset)));
    var sampleImagLimbs = limbSub(baseImagLimbs, rowStepScaled); // −rowOffset·subStepY — complex y grows up (screen y down); select() rejects arrays — branch
    if (rowOffset < 0) { sampleImagLimbs = limbAdd(baseImagLimbs, rowStepScaled); }
    for (var columnIndex = 0u; columnIndex < SAMPLE_COLUMNS_PER_PIXEL; columnIndex++) {
      let columnOffset = 2 * i32(columnIndex) + 1 - i32(SAMPLE_COLUMNS_PER_PIXEL); // odd multiples: −3, −1, +1, +3 for 4 columns
      let columnStepScaled = limbMultiplySmall(subStepXLimbs, u32(abs(columnOffset)));
      var sampleRealLimbs = limbAdd(baseRealLimbs, columnStepScaled); // +columnOffset·subStepX; select() rejects arrays — branch
      if (columnOffset < 0) { sampleRealLimbs = limbSub(baseRealLimbs, columnStepScaled); }
      let sampleColor = sampleColorPacked(sampleRealLimbs, sampleImagLimbs);
      redSum = redSum + (sampleColor & 0xFFu);
      greenSum = greenSum + ((sampleColor >> 8u) & 0xFFu);
      blueSum = blueSum + ((sampleColor >> 16u) & 0xFFu);
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
