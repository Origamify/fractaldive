'use strict';
/* ================================================================
 * FractalDive engine & app tests — one page, five sections:
 *
 *   1. reference orbit — df32 Z-table vs a BigInt twin
 *      (bit-for-bit, every row + row count)
 *   2. perturb frames — the shipped f32 engine's frames vs the
 *      retired exact per-pixel kernel (tests/oracle/*) within the
 *      documented tolerance (includes the 1e-35 floor view)
 *   3. banded render — multi-band frames BIT-IDENTICAL to single-
 *      pass frames through the real engine + band partition unit
 *      check
 *   4. present blit — the commit blit's shader exact at crop 1
 *      (the only crop the app presents; the CSS transform owns
 *      the zoom between settles)
 *   5. render/presentation split — engine commit lifecycle + dive
 *      window/generation state machine + presenter transform, on
 *      mocks (no GPU work)
 *      - 5a engine render→settle→commit lifecycle; 5b the dive
 *      window/generation state machine + presenter transform;
 *      5c the adaptive zoom rate (slowdown/clamps/recovery/sample gates)
 *
 * Sections run in this order on purpose: the GPU sections first,
 * the mock section last (it shadows performance.now and rAF for
 * determinism — after that the real clock is gone for the page).
 *
 * Sections 1, 2 and 4 share one WebGPU device and one reference-
 * orbit compute pipeline; section 3 boots the real engine (its
 * own device); section 5 injects a mock device.
 *
 * Owns nothing but this page; never touches the app's state
 * (dive.js and app.js are loaded under FRACTALDIVE_NO_AUTOBOOT).
 * ================================================================ */

/* --- constants --- */
const FRAME_ITERATION_BUDGET = 512; // shared by sections 2 and 3
const ESCAPE_RADIUS_SQUARED = 65536; // |z|² bailout — the delta kernel's ESCAPE_MAGNITUDE_SQUARED (the smooth-color palette is tuned for it)

// The shared view sweep (sections 2 and 3): shallow → floor plus
// the rebase path. expectStructure only where a small window
// provably catches the set boundary (deep views may sit wholly
// interior); allowance widens section 2's tolerance where
// knife-edge orbits concentrate (c ≈ −2 chaos, not bugs).
const FRAME_SWEEP = [
  { label: 'shallow', centerXText: '-0.75', centerYText: '0', viewportScale: 0.02, expectStructure: true },
  { label: 'mid seahorse', centerXText: '-0.743643887037158704752191506114774', centerYText: '0.131825904205311970493132056385139', viewportScale: 1e-12 },
  { label: 'deep snail', centerXText: '-0.743643798682766355449481514223428920788260491591077596960934100751435664684724', centerYText: '0.131825855119617217414972223459643983679919674841621934993603034335655825404623', viewportScale: 1e-25 },
  { label: 'floor', centerXText: '-0.743643798682766355449481514223428920788260491591077596960934100751435664684724', centerYText: '0.131825855119617217414972223459643983679919674841621934993603034335655825404623', viewportScale: 1e-35 }, // MIN_S — the stretched floor; tolerance-contract depth, past the 2⁻¹⁰⁰ external proof
  { label: 'rebase path', centerXText: '-2.05', centerYText: '0', viewportScale: 0.05, expectStructure: true },
];

/* --- state --- */
let passCount = 0;
let failCount = 0;

/* --- reporting --- */

/**
 * Records one invariant verdict as a log line.
 * @param {boolean} condition - The verdict.
 * @param {string} label - Invariant description.
 * @returns {void}
 */
function check(condition, label) {
  if (condition) { passCount++; logLine('pass', label); return; }
  failCount++;
  logLine('fail', label);
}

/**
 * Writes the final summary line (the Selenium probe waits for this).
 * @param {string} text - Final verdict.
 * @returns {void}
 */
function finishTests(text) {
  const summary = document.getElementById('test-summary');
  summary.textContent = text;
  summary.className = text.startsWith('PASS') ? 'pass' : 'fail';
}

/* --- shared GPU helpers --- */

/**
 * Returns a compiled shader module, throwing with the compiler's
 * own line messages on error (getCompilationInfo surfaces the
 * naga errors the pipeline calls would only swallow).
 * @param {GPUDevice} device - Active device.
 * @param {string} code - WGSL source.
 * @param {string} label - Shader label for the log lines.
 * @returns {Promise<GPUShaderModule>}
 */
async function compileShaderModule(device, code, label) {
  const shaderModule = device.createShaderModule({ code, label });
  const compilationInfo = await shaderModule.getCompilationInfo();
  const errors = compilationInfo.messages.filter((message) => message.type === 'error');
  if (errors.length > 0) {
    for (const message of errors) logLine('fail', `${label}: line ${message.lineNum}: ${message.message}`);
    throw new Error(`${label} did not compile`);
  }
  return shaderModule;
}

/**
 * Returns the reference-orbit compute pipeline at the app's
 * constant limb tier — shared by sections 1 and 2.
 * @param {GPUDevice} device - Active device.
 * @returns {Promise<GPUComputePipeline>}
 */
async function buildReferencePipeline(device) {
  const shaderModule = await compileShaderModule(
    device,
    REFERENCE_KERNEL_WGSL.replaceAll('__LIMB_COUNT__', String(REFERENCE_LIMB_COUNT)),
    'reference-orbit',
  );
  return device.createComputePipeline({ layout: 'auto', compute: { module: shaderModule, entryPoint: 'main' } });
}

/**
 * Packs the shared kernel uniform layout: fractionBits, maxIter,
 * width, height, palette phases, interior rgb word. The reference
 * compute kernel reads only the first two words.
 * @param {number} fractionBits - Fixed-point fractional bits.
 * @param {number} maxIter - Iteration budget.
 * @param {number} widthPx - Frame width (0 for compute-only passes).
 * @param {number} heightPx - Frame height (0 for compute-only passes).
 * @returns {Uint32Array} 8 uniform words.
 */
function packKernelUniforms(fractionBits, maxIter, widthPx, heightPx) {
  const uniformWords = new Uint32Array(8);
  uniformWords.set([fractionBits, maxIter, widthPx, heightPx]);
  new Float32Array(uniformWords.buffer).set(PALETTE.ph, 4);
  uniformWords[7] = PALETTE.interior[0] | (PALETTE.interior[1] << 8) | (PALETTE.interior[2] << 16);
  return uniformWords;
}

/**
 * Returns a FRAME_SWEEP case as an exact dive view for the
 * engine / fixed-point frame math.
 * @param {object} frameCase - The sweep case.
 * @returns {object} { centerXFixed, centerYFixed, centerFractionBits, s }.
 */
function makeView(frameCase) {
  return {
    centerXFixed: decimalStringToFixedPoint(frameCase.centerXText, CENTER_FIXED_POINT_BITS),
    centerYFixed: decimalStringToFixedPoint(frameCase.centerYText, CENTER_FIXED_POINT_BITS),
    centerFractionBits: CENTER_FIXED_POINT_BITS,
    s: frameCase.viewportScale,
  };
}

/* ================================================================
 * Section 1 — reference orbit: df32 Z-table vs BigInt oracle
 * ================================================================
 * One dispatch per case, the whole table copied back, every row's
 * df32 pair (re_hi, re_lo, im_hi, im_lo) compared bit-for-bit
 * against a BigInt twin that mirrors the shader exactly — the
 * orbit arithmetic (adds wrap at 192 bits, squares/subs stay exact
 * at 384, the rescale is an arithmetic >> 148 then a 192-bit
 * truncating add) AND the df32 split (top-24-bits round-half-even,
 * exact limb residual, second 24-bit round, explicit subnormal→0
 * at the f32 scale).
 *
 * Cases: a deep boundary center (long bounded orbit), the interior
 * (c = 0), an escaping center just outside the set (cap-crosser
 * row stored mid-table), a fast cap-crosser (|Z₁|² > 4 — table is
 * exactly Z₀ and Z₁), and the dendrite c = i at a small budget.
 * ================================================================ */

/**
 * Runs the reference-orbit section.
 * @param {GPUDevice} device - Shared device.
 * @param {GPUComputePipeline} referencePipeline - Compiled reference pipeline.
 * @returns {Promise<void>}
 */
async function referenceOrbitTests(device, referencePipeline) {
  const WIDE_LIMB_COUNT = 2 * REFERENCE_LIMB_COUNT; // product width of the format
  const VALUE_BITS = BigInt(32 * REFERENCE_LIMB_COUNT); // BigInt.asIntN width of one limb-format value
  const STOP_MAGNITUDE_SQUARED = BigInt(REFERENCE_STOP_MAGNITUDE_SQUARED); // mirrors fixed_point.js
  const REFERENCE_TABLE_STRIDE = 4;        // df32 pair words per row: reHi | reLo | imHi | imLo (one vec4u element)
  const REFERENCE_TABLE_HEADER_WORDS = 4;  // one vec4u header element: rowCount in .x — padded to the array<vec4u> element stride
  const INPUT_WORD_COUNT = 2 * REFERENCE_LIMB_COUNT + WIDE_LIMB_COUNT; // centerReal + centerImag + stop threshold

  /** Returns value wrapped to the format width — the mirror of limbAdd's dropped carry / limbTruncateLow. */
  const wrapToFormatWidth = (value) => signedAtWidth(value, VALUE_BITS);

  /**
   * Returns the oracle's Z-table for a center and budget — the
   * same loop the shader runs: store row, stop when |Z|² > 4 (the
   * row is kept), else update via the three-squares identity with
   * the same wrap/truncate points.
   */
  const buildOracleTable = (centerReal, centerImag, iterationBudget) => {
    const stopThreshold = STOP_MAGNITUDE_SQUARED << BigInt(2 * REFERENCE_FRACTION_BITS);
    let zReal = 0n;
    let zImag = 0n;
    const rows = [zReal, zImag];
    let rowCount = 1;
    for (;;) {
      const zRealSquared = zReal * zReal; // exact — squares live at wide width in the shader
      const zImagSquared = zImag * zImag;
      if (zRealSquared + zImagSquared > stopThreshold) break; // current row is the cap-crosser — kept
      if (rowCount > iterationBudget) break;                  // rows 0..budget stored
      const crossDoubled = wrapToFormatWidth(zReal + zImag) ** 2n - zRealSquared - zImagSquared; // limbAdd wraps the sum, then exact square
      zImag = wrapToFormatWidth((crossDoubled >> BigInt(REFERENCE_FRACTION_BITS)) + centerImag);
      zReal = wrapToFormatWidth(((zRealSquared - zImagSquared) >> BigInt(REFERENCE_FRACTION_BITS)) + centerReal);
      rows.push(zReal, zImag);
      rowCount++;
    }
    return rows;
  };

  /**
   * Rounds a non-negative BigInt to its top 24 significant bits,
   * round-half-even — the exact mirror of the shader's top24Bits
   * (n lands in [2^23, 2^24], or m exactly when shorter).
   */
  const splitTop24Bits = (magnitude) => {
    if (magnitude === 0n) return { n: 0n, shift: 0n };
    const bitLength = BigInt(magnitude.toString(2).length);
    const shift = bitLength > 24n ? bitLength - 24n : 0n;
    const truncated = magnitude >> shift;
    if (shift === 0n) return { n: truncated, shift: 0n };
    const roundBit = (magnitude >> (shift - 1n)) & 1n;
    const sticky = (magnitude & ((1n << (shift - 1n)) - 1n)) !== 0n;
    let n = truncated;
    if (roundBit === 1n && (sticky || (truncated & 1n) === 1n)) n = truncated + 1n;
    return { n, shift };
  };

  /**
   * Returns n·2^(shift−148) as an f32 with the shader's explicit
   * subnormal→0 semantics — exact for every normal result (n ≤ 2^24
   * on a power-of-two scale), so GPU and oracle agree bit-for-bit
   * on FTZ and non-FTZ stacks alike.
   */
  const scaledF32 = (n, shift) => {
    if (n === 0n) return 0;
    const exponent = Number(shift) - REFERENCE_FRACTION_BITS;
    if (exponent < -126) return 0;
    return Math.fround(Number(n) * Math.pow(2, exponent));
  };

  /**
   * Returns one limb-format value's df32 pair as bit patterns —
   * the exact mirror of the shader's limbToF32Pair: hi = the value
   * rounded to its top 24 bits, lo = the exact limb residual
   * rounded to ITS top 24 bits.
   */
  const f32PairWords = (value) => {
    const negative = value < 0n;
    const magnitude = negative ? -value : value;
    const hiSplit = splitTop24Bits(magnitude);
    const hiScaled = hiSplit.n << hiSplit.shift;
    let residual = negative ? hiScaled - magnitude : magnitude - hiScaled;
    const residualNegative = residual < 0n; // set only when rounding went up past the value
    if (residualNegative) residual = -residual;
    const loSplit = splitTop24Bits(residual);
    const hi = scaledF32(hiSplit.n, hiSplit.shift);
    const lo = scaledF32(loSplit.n, loSplit.shift);
    return [f32BitsOf(negative ? -hi : hi), f32BitsOf(residualNegative ? -lo : lo)];
  };

  /** Returns the fixed test cases: label, center decimal strings, budget. */
  const referenceTestCases = () => [
    { label: 'seahorse deep (bounded boundary orbit)', centerXText: '-0.743643887037158704752191506114774', centerYText: '0.131825904205311970493132056385139', iterationBudget: 1000 },
    { label: 'interior c = 0', centerXText: '0', centerYText: '0', iterationBudget: 512 },
    { label: 'slow escape c = 0.26', centerXText: '0.26', centerYText: '0', iterationBudget: 1000 },
    { label: 'fast cap-crosser c = -2.1', centerXText: '-2.1', centerYText: '0', iterationBudget: 1000 },
    { label: 'dendrite c = i (small budget)', centerXText: '0', centerYText: '1', iterationBudget: 128 },
  ];

  /** Returns the first row whose GPU words differ from the oracle, or −1 when all compared rows match. */
  const findFirstRowMismatch = (oracleRows, tableWords, rowCount) => {
    for (let rowIndex = 0; rowIndex < rowCount; rowIndex++) {
      const realPair = f32PairWords(oracleRows[2 * rowIndex]);
      const imagPair = f32PairWords(oracleRows[2 * rowIndex + 1]);
      const rowBase = REFERENCE_TABLE_HEADER_WORDS + rowIndex * REFERENCE_TABLE_STRIDE;
      if (tableWords[rowBase] !== realPair[0]
        || tableWords[rowBase + 1] !== realPair[1]
        || tableWords[rowBase + 2] !== imagPair[0]
        || tableWords[rowBase + 3] !== imagPair[1]) {
        return rowIndex;
      }
    }
    return -1;
  };

  /** Compares the GPU table against the oracle and records the verdict. */
  const judgeReferenceCase = (testCase, tableWords) => {
    const centerReal = decimalStringToFixedPoint(testCase.centerXText, REFERENCE_FRACTION_BITS);
    const centerImag = decimalStringToFixedPoint(testCase.centerYText, REFERENCE_FRACTION_BITS);
    const oracleRows = buildOracleTable(centerReal, centerImag, testCase.iterationBudget);
    const oracleRowCount = oracleRows.length / 2;

    const gpuRowCount = tableWords[0];
    const firstMismatch = findFirstRowMismatch(oracleRows, tableWords, Math.min(gpuRowCount, oracleRowCount));
    const isPass = gpuRowCount === oracleRowCount && firstMismatch === -1;

    if (isPass) {
      check(true, `reference ${testCase.label}: ${gpuRowCount} rows match (${gpuRowCount === testCase.iterationBudget + 1 ? 'budget-capped' : 'cap-crosser stored'})`);
    } else {
      check(false, `reference ${testCase.label}: rowCount GPU ${gpuRowCount} vs oracle ${oracleRowCount}${firstMismatch >= 0 ? `, first row mismatch at ${firstMismatch}` : ''}`);
    }
  };

  /** Runs one case: one dispatch, table readback, oracle compare. */
  const runReferenceCase = async (testCase) => {
    const centerReal = decimalStringToFixedPoint(testCase.centerXText, REFERENCE_FRACTION_BITS);
    const centerImag = decimalStringToFixedPoint(testCase.centerYText, REFERENCE_FRACTION_BITS);
    const budget = testCase.iterationBudget;

    const inputWords = new Uint32Array(INPUT_WORD_COUNT);
    inputWords.set(limbsFromBigInt(centerReal, REFERENCE_LIMB_COUNT), 0);
    inputWords.set(limbsFromBigInt(centerImag, REFERENCE_LIMB_COUNT), REFERENCE_LIMB_COUNT);
    inputWords.set(limbsFromBigInt(STOP_MAGNITUDE_SQUARED << BigInt(2 * REFERENCE_FRACTION_BITS), WIDE_LIMB_COUNT), 2 * REFERENCE_LIMB_COUNT);
    const uniformWords = packKernelUniforms(REFERENCE_FRACTION_BITS, budget, 0, 0);

    const tableWordCount = REFERENCE_TABLE_HEADER_WORDS + (budget + 1) * REFERENCE_TABLE_STRIDE;
    const inputBuffer = device.createBuffer({ size: inputWords.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    const uniformBuffer = device.createBuffer({ size: uniformWords.byteLength, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const tableBuffer = device.createBuffer({ size: tableWordCount * U32_BYTE_WIDTH, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    const tableReadBuffer = device.createBuffer({ size: tableBuffer.size, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const bindGroup = device.createBindGroup({
      layout: referencePipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: uniformBuffer } },
        { binding: 1, resource: { buffer: inputBuffer } },
        { binding: 2, resource: { buffer: tableBuffer } },
      ],
    });
    device.queue.writeBuffer(inputBuffer, 0, inputWords);
    device.queue.writeBuffer(uniformBuffer, 0, uniformWords);

    const commandEncoder = device.createCommandEncoder();
    const computePass = commandEncoder.beginComputePass();
    computePass.setPipeline(referencePipeline);
    computePass.setBindGroup(0, bindGroup);
    computePass.dispatchWorkgroups(1);
    computePass.end();
    commandEncoder.copyBufferToBuffer(tableBuffer, 0, tableReadBuffer, 0, tableBuffer.size);
    device.queue.submit([commandEncoder.finish()]);
    await tableReadBuffer.mapAsync(GPUMapMode.READ);
    const tableWords = new Uint32Array(tableReadBuffer.getMappedRange().slice(0));
    tableReadBuffer.unmap();
    inputBuffer.destroy();
    uniformBuffer.destroy();
    tableBuffer.destroy();
    tableReadBuffer.destroy();

    judgeReferenceCase(testCase, tableWords);
  };

  for (const testCase of referenceTestCases()) await runReferenceCase(testCase);
}

/* ================================================================
 * Section 2 — perturb frame tolerance: shipped engine frames vs
 * the retired exact per-pixel kernel (the oracle)
 * ================================================================
 * Both kernels render a 7×7 rgba8unorm offscreen texture; the
 * bytes are compared word-for-word. The oracle runs the exact
 * kernel at a fixed 173-bit frame precision on the SAME
 * frame-quantized pixel grid the f32 engine's δ constants derive
 * from. The f32 delta iteration rounds ε every step (2⁻²³·|ε|),
 * so the ALLOWANCE is boundary-pixel flips: escaped-vs-interior
 * or ±1-count knife-edge orbits land on different palette
 * positions — any channel delta on a bounded fraction of pixels.
 * A systematic shift (all pixels off) fails; smooth-color drift
 * alone (~2⁻²⁰) rarely changes a channel at all. The rebase
 * frame's wider fraction also absorbs the adaptive-SSAA
 * single-sample drift: the oracle always averages the full 2×2
 * lattice, the engine returns sample 0 wherever its DE-flat
 * verdict fires — most of that frame, each differing pixel
 * only sub-band (measured 38/49, Δ ≤ 11).
 * ================================================================ */

/**
 * Runs the perturb-frame parity section over FRAME_SWEEP.
 * @param {GPUDevice} device - Shared device.
 * @param {GPUComputePipeline} referencePipeline - Compiled reference pipeline.
 * @returns {Promise<void>}
 */
async function perturbFrameTests(device, referencePipeline) {
  const FRAME_WIDTH_PX = 7;
  const FRAME_HEIGHT_PX = 7;
  const ORBIT_TEXTURE_FORMAT = 'rgba8unorm'; // byte 0 is R — read-back bytes match the packed color words
  const ORBIT_ROW_BYTES = 256;               // copyTextureToBuffer bytesPerRow must be 256-aligned — 7 px × 4 B = 28 rounds up
  const EXACT_LIMB_INTEGER_HEADROOM_BITS = 19;  // the retired exact engine's headroom (its frame-tier sizing)
  const EXACT_ORACLE_FRACTION_BITS = 173;        // oracle arithmetic precision — tier 6, one step finer than the reference limb format's 148 bits
  const EXACT_INPUT_VECTOR_COUNT = 7;           // the retired exact engine's input layout: threshold (2 units) + 5 N-limb vectors
  const MAX_MISMATCH_FRACTION = 0.05;         // default allowance: <= 5% of pixels may differ (boundary flips; at 49 px one flip is already 2%) ...
  const MAX_CHANNEL_DELTA = 255;              // ... by any channel amount - a flipped boundary pixel lands elsewhere on the palette
  const REBASE_CHAOS_MISMATCH_FRACTION = 0.85; // rebase frame: knife-edge chaos near c = -2, PLUS the adaptive-SSAA single-sample drift — the DE-flat verdict fires on most of this frame (measured 38/49) with only sub-band color shifts (Δ ≤ 11); a systematic all-pixel shift still fails at 1.0

  // The retired exact kernel (oracle), from the shared palette/render
  // chunk constants in perturb_kernel_wgsl.js: the limb lib + a tier
  // spliced ahead of the orbit + render chunks.
  const EXACT_ENGINE_KERNEL_WGSL = LIMB_ARRAY_MATH_WGSL
    + ENGINE_KERNEL_WGSL_BINDINGS
    + PERTURB_KERNEL_WGSL_PALETTE
    + ENGINE_KERNEL_WGSL_ORBIT
    + PERTURB_KERNEL_WGSL_RENDER;

  const exactPipelinesByLimbCount = new Map(); // frame tier → retired exact render pipeline

  // the delta render pipeline — pure f32, no splice; compiled once for the section
  const perturbModule = await compileShaderModule(device, PERTURB_KERNEL_WGSL, 'perturbation-delta-kernel');
  const perturbFramePipeline = await device.createRenderPipelineAsync({
    layout: 'auto',
    vertex: { module: perturbModule, entryPoint: 'vs' },
    fragment: { module: perturbModule, entryPoint: 'fs', targets: [{ format: ORBIT_TEXTURE_FORMAT }] },
    primitive: { topology: 'triangle-list' },
  });

  /**
   * Returns the exact-oracle frame config for a view — the pixel
   * grid quantized at the view's frame precision (the same grid
   * the perturbation engine visits), represented exactly at the
   * oracle's fixed 173-bit precision so the orbit arithmetic
   * out-resolves the reference limb format.
   */
  const buildExactFrameConfig = (view) => {
    const gridBits = fractionBitsForScale(view.s); // pixel-grid quantization — shared with the perturbation engine
    const gridScale = Number(1n << BigInt(gridBits));
    const gridShift = BigInt(EXACT_ORACLE_FRACTION_BITS - gridBits); // exact left shift onto the oracle grid
    const toOracleGrid = (gridValue) => gridValue << gridShift;
    const centerShift = BigInt(gridBits - view.centerFractionBits);
    const centerXFixed = centerShift >= 0n ? view.centerXFixed << centerShift : view.centerXFixed >> -centerShift;
    const centerYFixed = centerShift >= 0n ? view.centerYFixed << centerShift : view.centerYFixed >> -centerShift;
    const pixelStepFixed = BigInt(Math.max(1, Math.round(view.s * gridScale)));
    return {
      fractionBits: EXACT_ORACLE_FRACTION_BITS,
      limbCount: Math.ceil((EXACT_ORACLE_FRACTION_BITS + EXACT_LIMB_INTEGER_HEADROOM_BITS) / 32),
      escapeThreshold: BigInt(ESCAPE_RADIUS_SQUARED) << BigInt(2 * EXACT_ORACLE_FRACTION_BITS),
      planeLeft: toOracleGrid(centerXFixed - BigInt(Math.round((FRAME_WIDTH_PX / 2 - 0.5) * view.s * gridScale))),
      planeTop: toOracleGrid(centerYFixed + BigInt(Math.round((FRAME_HEIGHT_PX / 2 - 0.5) * view.s * gridScale))),
      pixelStep: toOracleGrid(pixelStepFixed),
      subStepX: toOracleGrid(pixelStepFixed >> BigInt(Math.log2(SAMPLE_COLUMNS_PER_PIXEL) + 1)),
      subStepY: toOracleGrid(pixelStepFixed >> BigInt(Math.log2(SAMPLE_ROWS_PER_PIXEL) + 1)),
    };
  };

  /** Packs the exact kernel's input + uniform words for a frame (mirrors the retired engine.js packing at frame precision). */
  const packExactFrameWords = (frameConfig, maxIter) => {
    const limbCount = frameConfig.limbCount;
    const inputWords = new Uint32Array(EXACT_INPUT_VECTOR_COUNT * limbCount);
    inputWords.set(limbsFromBigInt(frameConfig.escapeThreshold, 2 * limbCount), 0);
    inputWords.set(limbsFromBigInt(frameConfig.planeLeft, limbCount), 2 * limbCount);
    inputWords.set(limbsFromBigInt(frameConfig.planeTop, limbCount), 3 * limbCount);
    inputWords.set(limbsFromBigInt(frameConfig.pixelStep, limbCount), 4 * limbCount);
    inputWords.set(limbsFromBigInt(frameConfig.subStepX, limbCount), 5 * limbCount);
    inputWords.set(limbsFromBigInt(frameConfig.subStepY, limbCount), 6 * limbCount);
    return { inputWords, uniformWords: packKernelUniforms(frameConfig.fractionBits, maxIter, FRAME_WIDTH_PX, FRAME_HEIGHT_PX) };
  };

  /** Returns the settled readback bytes as packed rgba words, one per pixel, row-major. */
  const packedWordsFromBytes = (bytes) => {
    const words = [];
    for (let pixelIndex = 0; pixelIndex < FRAME_WIDTH_PX * FRAME_HEIGHT_PX; pixelIndex++) {
      const byteIndex = Math.floor(pixelIndex / FRAME_WIDTH_PX) * ORBIT_ROW_BYTES + (pixelIndex % FRAME_WIDTH_PX) * 4;
      words.push(bytes[byteIndex] | (bytes[byteIndex + 1] << 8) | (bytes[byteIndex + 2] << 16) | (bytes[byteIndex + 3] << 24));
    }
    return words;
  };

  /** Returns the compiled exact render pipeline for a frame tier (cached per tier, compilation-checked). */
  const exactPipelineForTier = async (limbCount) => {
    let pipeline = exactPipelinesByLimbCount.get(limbCount);
    if (!pipeline) {
      const shaderModule = await compileShaderModule(
        device,
        EXACT_ENGINE_KERNEL_WGSL.replaceAll('__LIMB_COUNT__', String(limbCount)),
        `exact-oracle-tier-${limbCount}`,
      );
      pipeline = await device.createRenderPipelineAsync({
        layout: 'auto',
        vertex: { module: shaderModule, entryPoint: 'vs' },
        fragment: { module: shaderModule, entryPoint: 'fs', targets: [{ format: ORBIT_TEXTURE_FORMAT }] },
        primitive: { topology: 'triangle-list' },
      });
      exactPipelinesByLimbCount.set(limbCount, pipeline);
    }
    return pipeline;
  };

  /**
   * Renders one frame through one render pipeline into a fresh
   * offscreen texture and returns the packed pixel words.
   */
  const renderFrameToWords = async (pipeline, inputWords, uniformWords) => {
    const inputBuffer = device.createBuffer({ size: inputWords.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    const uniformBuffer = device.createBuffer({ size: uniformWords.byteLength, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(inputBuffer, 0, inputWords);
    device.queue.writeBuffer(uniformBuffer, 0, uniformWords);
    const texture = device.createTexture({
      size: [FRAME_WIDTH_PX, FRAME_HEIGHT_PX],
      format: ORBIT_TEXTURE_FORMAT,
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
    });
    const readBuffer = device.createBuffer({
      size: ORBIT_ROW_BYTES * FRAME_HEIGHT_PX,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
    const bindGroup = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: uniformBuffer } },
        { binding: 1, resource: { buffer: inputBuffer } },
      ],
    });

    const commandEncoder = device.createCommandEncoder();
    const renderPass = commandEncoder.beginRenderPass({
      colorAttachments: [{
        view: texture.createView(),
        loadOp: 'clear',
        storeOp: 'store',
        clearValue: { r: 0, g: 0, b: 0, a: 1 },
      }],
    });
    renderPass.setPipeline(pipeline);
    renderPass.setBindGroup(0, bindGroup);
    renderPass.draw(3);
    renderPass.end();
    commandEncoder.copyTextureToBuffer({ texture }, { buffer: readBuffer, bytesPerRow: ORBIT_ROW_BYTES, rowsPerImage: FRAME_HEIGHT_PX }, [FRAME_WIDTH_PX, FRAME_HEIGHT_PX]);
    device.queue.submit([commandEncoder.finish()]);
    await readBuffer.mapAsync(GPUMapMode.READ);
    const bytes = new Uint8Array(readBuffer.getMappedRange().slice(0));
    readBuffer.unmap();
    inputBuffer.destroy();
    uniformBuffer.destroy();
    texture.destroy();
    readBuffer.destroy();
    return packedWordsFromBytes(bytes);
  };

  /**
   * Renders one frame through the perturbation engine's full path —
   * reference-orbit compute pass building the Z table, then the
   * delta render pass reading it — and returns the packed pixel
   * words.
   */
  const renderPerturbationFrameToWords = async (frameConfig, maxIter) => {
    // the f32 delta kernel's five frame constants (buildFixedPointConfig output), as bit patterns
    const frameInputWords = new Uint32Array(new Float32Array([
      frameConfig.planeLeftF32,
      frameConfig.planeTopF32,
      frameConfig.pixelStepF32,
      frameConfig.subStepXF32,
      frameConfig.subStepYF32,
    ]).buffer);
    const referenceInputWords = new Uint32Array(4 * REFERENCE_LIMB_COUNT); // centerReal + centerImag + stop threshold (2 units)
    referenceInputWords.set(limbsFromBigInt(frameConfig.centerReal, REFERENCE_LIMB_COUNT), 0);
    referenceInputWords.set(limbsFromBigInt(frameConfig.centerImag, REFERENCE_LIMB_COUNT), REFERENCE_LIMB_COUNT);
    referenceInputWords.set(limbsFromBigInt(frameConfig.referenceStopThreshold, 2 * REFERENCE_LIMB_COUNT), 2 * REFERENCE_LIMB_COUNT);
    const uniformWords = packKernelUniforms(REFERENCE_FRACTION_BITS, maxIter, FRAME_WIDTH_PX, FRAME_HEIGHT_PX);

    const frameInputBuffer = device.createBuffer({ size: frameInputWords.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    const referenceInputBuffer = device.createBuffer({ size: referenceInputWords.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    const uniformBuffer = device.createBuffer({ size: uniformWords.byteLength, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const tableBuffer = device.createBuffer({
      size: (1 + (maxIter + 1) * 4) * U32_BYTE_WIDTH, // header + (budget+1) df32 rows (reHi|reLo|imHi|imLo)
      usage: GPUBufferUsage.STORAGE,
    });
    device.queue.writeBuffer(frameInputBuffer, 0, frameInputWords);
    device.queue.writeBuffer(referenceInputBuffer, 0, referenceInputWords);
    device.queue.writeBuffer(uniformBuffer, 0, uniformWords);

    const referenceBindGroup = device.createBindGroup({
      layout: referencePipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: uniformBuffer } },
        { binding: 1, resource: { buffer: referenceInputBuffer } },
        { binding: 2, resource: { buffer: tableBuffer } },
      ],
    });
    const texture = device.createTexture({
      size: [FRAME_WIDTH_PX, FRAME_HEIGHT_PX],
      format: ORBIT_TEXTURE_FORMAT,
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
    });
    const readBuffer = device.createBuffer({ size: ORBIT_ROW_BYTES * FRAME_HEIGHT_PX, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });

    const frameBindGroup = device.createBindGroup({
      layout: perturbFramePipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: uniformBuffer } },
        { binding: 1, resource: { buffer: frameInputBuffer } },
        { binding: 2, resource: { buffer: tableBuffer } },
      ],
    });

    // the engine's frame shape: compute pass first, render pass second, one submit —
    // the implicit pass barrier orders the table writes before every fragment reads them
    const commandEncoder = device.createCommandEncoder();
    const computePass = commandEncoder.beginComputePass();
    computePass.setPipeline(referencePipeline);
    computePass.setBindGroup(0, referenceBindGroup);
    computePass.dispatchWorkgroups(1);
    computePass.end();
    const renderPass = commandEncoder.beginRenderPass({
      colorAttachments: [{
        view: texture.createView(),
        loadOp: 'clear',
        storeOp: 'store',
        clearValue: { r: 0, g: 0, b: 0, a: 1 },
      }],
    });
    renderPass.setPipeline(perturbFramePipeline);
    renderPass.setBindGroup(0, frameBindGroup);
    renderPass.draw(3);
    renderPass.end();
    commandEncoder.copyTextureToBuffer({ texture }, { buffer: readBuffer, bytesPerRow: ORBIT_ROW_BYTES, rowsPerImage: FRAME_HEIGHT_PX }, [FRAME_WIDTH_PX, FRAME_HEIGHT_PX]);
    device.queue.submit([commandEncoder.finish()]);
    await readBuffer.mapAsync(GPUMapMode.READ);
    const bytes = new Uint8Array(readBuffer.getMappedRange().slice(0));
    readBuffer.unmap();
    frameInputBuffer.destroy();
    referenceInputBuffer.destroy();
    uniformBuffer.destroy();
    tableBuffer.destroy();
    texture.destroy();
    readBuffer.destroy();
    return packedWordsFromBytes(bytes);
  };

  /** Compares two frames' packed words under the documented allowance and records the verdict. */
  const judgeFrame = (label, exactWords, perturbWords, allowance) => {
    let mismatchedPixels = 0;
    let maxChannelDelta = 0;
    let worstExactWord = 0;
    let worstPerturbWord = 0;
    for (let pixelIndex = 0; pixelIndex < exactWords.length; pixelIndex++) {
      if (exactWords[pixelIndex] === perturbWords[pixelIndex]) continue;
      mismatchedPixels++;
      for (const channelShift of [0, 8, 16]) {
        const channelDelta = Math.abs(((exactWords[pixelIndex] >> channelShift) & 0xFF) - ((perturbWords[pixelIndex] >> channelShift) & 0xFF));
        if (channelDelta > maxChannelDelta) {
          maxChannelDelta = channelDelta;
          worstExactWord = exactWords[pixelIndex];
          worstPerturbWord = perturbWords[pixelIndex];
        }
      }
    }
    const mismatchFraction = mismatchedPixels / exactWords.length;
    const isPass = mismatchFraction <= allowance.mismatchFraction && maxChannelDelta <= allowance.channelDelta;
    if (isPass) {
      passCount++;
      logLine(mismatchedPixels === 0 ? 'pass' : 'info',
        `parity ${label}: ${mismatchedPixels === 0 ? 'all 49 pixels identical' : `${mismatchedPixels}/${exactWords.length} pixels differ, max channel delta ${maxChannelDelta} (within allowance)`}`);
    } else {
      failCount++;
      logLine('fail', `parity ${label}: ${mismatchedPixels}/${exactWords.length} pixels differ, max channel delta ${maxChannelDelta} — exact 0x${(worstExactWord >>> 0).toString(16)} vs perturb 0x${(worstPerturbWord >>> 0).toString(16)}`);
    }
  };

  for (const frameCase of FRAME_SWEEP) {
    const view = makeView(frameCase);
    const exactConfig = buildExactFrameConfig(view);
    const exactPipeline = await exactPipelineForTier(exactConfig.limbCount);
    const exactPack = packExactFrameWords(exactConfig, FRAME_ITERATION_BUDGET);
    const exactWords = await renderFrameToWords(exactPipeline, exactPack.inputWords, exactPack.uniformWords);
    const perturbConfig = buildFixedPointConfig(view, FRAME_WIDTH_PX, FRAME_HEIGHT_PX);
    const perturbWords = await renderPerturbationFrameToWords(perturbConfig, FRAME_ITERATION_BUDGET);
    const allowance = frameCase.label === 'rebase path'
      ? { mismatchFraction: REBASE_CHAOS_MISMATCH_FRACTION, channelDelta: MAX_CHANNEL_DELTA }
      : { mismatchFraction: MAX_MISMATCH_FRACTION, channelDelta: MAX_CHANNEL_DELTA };
    judgeFrame(frameCase.label, exactWords, perturbWords, allowance);
  }
}

/* ================================================================
 * Section 3 — banded render: the GPU duty-cycle control's core
 * guarantee, through the REAL engine
 * ================================================================
 * A multi-band frame is BIT-IDENTICAL to the same frame rendered
 * in one pass. Real device, real pipelines, the engine's own base
 * texture — read back via copyTextureToBuffer through the base's
 * COPY_SRC usage. Band count is forced by shadowing
 * bandCountForFrame() per render; the pacing sleeps never fire on
 * these tiny frames (measured band ms ≈ 0).
 *
 * 3a — band partition unit check (pure integer math): every pixel
 *      row covered exactly once, remainder band last.
 * 3b — per FRAME_SWEEP view, one single-band frame vs one forced-
 *      band frame, words compared exactly (same format, same
 *      engine — channel order cancels); the degenerate-color
 *      guard runs only where a small window provably catches the
 *      boundary (deep views may sit wholly interior — their
 *      identity still catches clear/load bugs: band 0 clears
 *      black, the interior color is not black).
 * ================================================================ */

/**
 * Runs the banded-render section (boots the real engine on its own
 * device — init() compiles the app's own pipelines).
 * @returns {Promise<void>}
 */
async function bandedRenderTests() {
  const CANVAS_WIDTH_PX = 4;   // engine canvas in device px — the base texture runs at 8×8
  const CANVAS_HEIGHT_PX = 4;
  const FORCED_BAND_COUNT = 3; // on 8 base rows: bands 3+3+2 — exercises the remainder band
  const MIN_DISTINCT_COLORS = 2; // degenerate-frame guard — views flagged expectStructure must render actual structure
  const PARTITION_CASES = [ // [baseHeight, bandCount] — exact divisions, remainders, ceilings, degenerates
    [1, 1], [1, 64], [2, 4], [7, 4], [8, 3], [8, 8], [65, 64], [129, 5], [1080, 4], [1080, 64],
  ];

  /** Returns the engine's settled base texture as packed rgba words, one per pixel, row-major. */
  const readBaseTextureWords = async (engine) => {
    const texture = engine.baseTexture;
    const bytesPerRow = Math.ceil((texture.width * 4) / 256) * 256; // copyTextureToBuffer rows are 256-aligned
    const readBuffer = engine.gpuDevice.createBuffer({ size: bytesPerRow * texture.height, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const commandEncoder = engine.gpuDevice.createCommandEncoder();
    commandEncoder.copyTextureToBuffer({ texture }, { buffer: readBuffer, bytesPerRow, rowsPerImage: texture.height }, [texture.width, texture.height]);
    engine.gpuDevice.queue.submit([commandEncoder.finish()]);
    await readBuffer.mapAsync(GPUMapMode.READ);
    const bytes = new Uint8Array(readBuffer.getMappedRange().slice(0));
    readBuffer.unmap();
    readBuffer.destroy();
    const words = [];
    for (let pixelY = 0; pixelY < texture.height; pixelY++) {
      for (let pixelX = 0; pixelX < texture.width; pixelX++) {
        const byteIndex = pixelY * bytesPerRow + pixelX * 4;
        words.push(bytes[byteIndex] | (bytes[byteIndex + 1] << 8) | (bytes[byteIndex + 2] << 16) | (bytes[byteIndex + 3] << 24));
      }
    }
    return words;
  };

  /**
   * Checks bandLayoutForFrame against the plan's exact partition:
   * y0 = i·bandH, height = min(bandH, H − y0) — contiguous strips,
   * full coverage, no empty band, at most the requested count.
   */
  const bandPartitionChecks = (engine) => {
    for (const [baseHeight, bandCount] of PARTITION_CASES) {
      const bands = engine.bandLayoutForFrame(baseHeight, bandCount);
      const nominalBandHeight = Math.ceil(baseHeight / bandCount);
      const matchesPlanFormula = bands.every((band, bandIndex) =>
        band.y0 === bandIndex * nominalBandHeight
        && band.height === Math.min(nominalBandHeight, baseHeight - band.y0)
        && band.height > 0);
      const coversExactlyOnce = bands.length > 0
        && bands[0].y0 === 0
        && bands[bands.length - 1].y0 + bands[bands.length - 1].height === baseHeight
        && bands.every((band, bandIndex) => bandIndex === 0 || bands[bandIndex - 1].y0 + bands[bandIndex - 1].height === band.y0);
      check(matchesPlanFormula && coversExactlyOnce && bands.length <= bandCount,
        `partition H=${baseHeight} bands=${bandCount} → [${bands.map((band) => `${band.y0}:${band.y0 + band.height}`).join(' ')}]`);
    }
  };

  /**
   * Renders one view at a forced band count and returns its packed
   * base-texture words (fresh reference table per render — the
   * compute pass must ride band 0 in both shapes).
   */
  const renderForcedBandFrame = async (engine, view, bandCount) => {
    engine.resetRef();
    engine.bandCountForFrame = () => bandCount;
    await engine.render(view, FRAME_ITERATION_BUDGET);
    return await readBaseTextureWords(engine);
  };

  /** Boots the engine, then runs the partition check and the identity sweep. */
  const engine = new WebGPUPerturbationEngine();
  await engine.init(document.createElement('canvas'));
  engine.resize(CANVAS_WIDTH_PX, CANVAS_HEIGHT_PX);

  bandPartitionChecks(engine);
  for (const frameCase of FRAME_SWEEP) {
    const view = makeView(frameCase);
    const singleBandWords = await renderForcedBandFrame(engine, view, 1);
    const bandedWords = await renderForcedBandFrame(engine, view, FORCED_BAND_COUNT);
    const distinctColors = new Set(singleBandWords).size;
    const mismatchedPixels = singleBandWords.reduce((count, word, pixelIndex) => count + (word === bandedWords[pixelIndex] ? 0 : 1), 0);
    if (frameCase.expectStructure) {
      check(distinctColors >= MIN_DISTINCT_COLORS, `banded ${frameCase.label}: frame is non-degenerate (${distinctColors} distinct colors)`);
    }
    check(mismatchedPixels === 0,
      `banded ${frameCase.label}: single-band vs ${FORCED_BAND_COUNT}-band — ${mismatchedPixels === 0 ? `all ${singleBandWords.length} pixels bit-identical` : `${mismatchedPixels}/${singleBandWords.length} pixels differ`}`);
  }
}

/* ================================================================
 * Section 4 — present blit: the commit blit's shader, exact at
 * crop 1 (the app's only presentation crop)
 * ================================================================
 * The app blits the settled base at crop fraction 1 (the CSS
 * transform owns the zoom between settles), i.e. a 2:1 bilinear
 * downsample of the 2× base onto the canvas. The base pattern
 * r = x·16, g = y·16, b = 255 makes every texel addressable from
 * its color; b = 255 separates "sampled zero" from "drew nothing".
 * Each target pixel must land the exact bilinear average of its
 * 2×2 base block — (4i+1)·8 on the pattern, an integer the unorm
 * store reproduces exactly. The y-dependent g channel also pins
 * the v orientation: the vertex stage flips v against NDC y, and
 * a missing flip mirrors the frame vertically.
 * ================================================================ */

/**
 * Runs the present-blit section.
 * @param {GPUDevice} device - Shared device.
 * @returns {Promise<void>}
 */
async function presentBlitTests(device) {
  const BASE_SIZE_PX = 8;    // base "frame" texture edge — pattern span 0..112 per channel
  const TARGET_SIZE_PX = 4;  // blit target edge — the full base at 2:1
  const READ_ROW_BYTES = 256; // copyTextureToBuffer bytesPerRow must be 256-aligned — 4 px × 4 B = 16 rounds up
  const CHANNEL_MAX = 255;

  /**
   * Returns the expected pixel at crop 1: target pixel (x, y)
   * samples the exact center of base texel block [2x..2x+1] ×
   * [2y..2y+1], so linear filtering averages the four texels —
   * (4x+1)·8 on r and g, exactly representable in 8 bits.
   */
  const expectedWindowStartPixel = (x, y) => {
    const blockAverage = (4 * x + 1) * 8; // == ((2x)·16 + (2x+1)·16) / 2 — r and g share the pattern
    return [blockAverage, (4 * y + 1) * 8, CHANNEL_MAX, CHANNEL_MAX];
  };

  /** Builds the patterned base texture on the shared device. */
  const createPatternBaseTexture = () => {
    const texture = device.createTexture({
      size: [BASE_SIZE_PX, BASE_SIZE_PX],
      format: 'rgba8unorm',
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    });
    const texelBytes = new Uint8Array(BASE_SIZE_PX * BASE_SIZE_PX * 4);
    for (let y = 0; y < BASE_SIZE_PX; y++) {
      for (let x = 0; x < BASE_SIZE_PX; x++) {
        const offset = (y * BASE_SIZE_PX + x) * 4;
        texelBytes[offset] = x * 16;
        texelBytes[offset + 1] = y * 16;
        texelBytes[offset + 2] = CHANNEL_MAX;
        texelBytes[offset + 3] = CHANNEL_MAX;
      }
    }
    device.queue.writeTexture({ texture }, texelBytes, { bytesPerRow: BASE_SIZE_PX * 4 }, [BASE_SIZE_PX, BASE_SIZE_PX]);
    return texture;
  };

  /**
   * Blits the base into a fresh target at crop 1 and returns the
   * read-back pixels.
   */
  const blitAndReadPixels = async (pipeline, baseTexture, sampler, uniformBuffer) => {
    const target = device.createTexture({
      size: [TARGET_SIZE_PX, TARGET_SIZE_PX],
      format: 'rgba8unorm',
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
    });
    const readBuffer = device.createBuffer({
      size: READ_ROW_BYTES * TARGET_SIZE_PX,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(uniformBuffer, 0, new Float32Array([1])); // the app's presentation crop — always 1
    const bindGroup = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: uniformBuffer } },
        { binding: 1, resource: sampler },
        { binding: 2, resource: baseTexture.createView() },
      ],
    });
    const commandEncoder = device.createCommandEncoder();
    const renderPass = commandEncoder.beginRenderPass({
      colorAttachments: [{
        view: target.createView(),
        loadOp: 'clear',
        storeOp: 'store',
        clearValue: { r: 0, g: 0, b: 0, a: 1 },
      }],
    });
    renderPass.setPipeline(pipeline);
    renderPass.setBindGroup(0, bindGroup);
    renderPass.draw(3);
    renderPass.end();
    commandEncoder.copyTextureToBuffer({ texture: target }, { buffer: readBuffer, bytesPerRow: READ_ROW_BYTES }, [TARGET_SIZE_PX, TARGET_SIZE_PX]);
    device.queue.submit([commandEncoder.finish()]);
    await readBuffer.mapAsync(GPUMapMode.READ);
    const bytes = new Uint8Array(readBuffer.getMappedRange().slice(0));
    readBuffer.unmap();
    target.destroy();
    readBuffer.destroy();
    const pixels = [];
    for (let y = 0; y < TARGET_SIZE_PX; y++) {
      for (let x = 0; x < TARGET_SIZE_PX; x++) {
        const offset = y * READ_ROW_BYTES + x * 4;
        pixels.push([bytes[offset], bytes[offset + 1], bytes[offset + 2], bytes[offset + 3]]);
      }
    }
    return pixels;
  };

  /** Judges the blit against the CPU model of the crop and records the verdict. */
  const judgeBlit = (label, pixels, expectedPixel) => {
    let mismatches = 0;
    for (let y = 0; y < TARGET_SIZE_PX; y++) {
      for (let x = 0; x < TARGET_SIZE_PX; x++) {
        const actual = pixels[y * TARGET_SIZE_PX + x];
        const expected = expectedPixel(x, y);
        if (actual[0] !== expected[0] || actual[1] !== expected[1] || actual[2] !== expected[2] || actual[3] !== expected[3]) {
          if (mismatches < 4) logLine('fail', `${label}: pixel (${x},${y}) got rgba(${actual}) expected rgba(${expected})`);
          mismatches++;
        }
      }
    }
    check(mismatches === 0, `${label}: ${mismatches === 0 ? `all ${TARGET_SIZE_PX * TARGET_SIZE_PX} pixels exact` : `${mismatches}/${TARGET_SIZE_PX * TARGET_SIZE_PX} pixels off`}`);
  };

  const baseTexture = createPatternBaseTexture();
  const shaderModule = await compileShaderModule(device, PRESENT_BLIT_WGSL, 'present-crop-blit');
  const pipeline = await device.createRenderPipelineAsync({
    layout: 'auto',
    vertex: { module: shaderModule, entryPoint: 'vs' },
    fragment: { module: shaderModule, entryPoint: 'fs', targets: [{ format: 'rgba8unorm' }] },
    primitive: { topology: 'triangle-list' },
  });
  const sampler = device.createSampler({
    magFilter: 'linear', minFilter: 'linear',
    addressModeU: 'clamp-to-edge', addressModeV: 'clamp-to-edge',
  });
  const uniformBuffer = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });

  judgeBlit('crop 1 (the commit blit, 2:1 downsample)',
    await blitAndReadPixels(pipeline, baseTexture, sampler, uniformBuffer),
    expectedWindowStartPixel);
  baseTexture.destroy();
}

/* ================================================================
 * Section 5 — render/presentation split on mocks (no GPU work)
 * ================================================================
 * A mock GPUDevice drives the engine's render→settle→commit
 * lifecycle, and a mock engine + fake canvas drive dive.js's
 * window/generation state machine and the presenter's transform.
 *
 * Engine invariants:
 * - resize doubles the canvas backing (base resolution)
 * - a render submits compute+render passes into the OFFSCREEN
 *   base — the canvas is never a render target mid-flight
 * - commit (settle-time, queue empty) submits exactly one blit
 *   into the canvas view, crop uniform 1.0, sampling the rendered
 *   base; the bind group is cached per texture lifetime
 * - a cached-reference frame skips the compute pass
 * - commit after a resize (base destroyed) early-returns
 * - band sizing: unmeasured → DEFAULT_BAND_COUNT, measured →
 *   ceil(gpuMs/TARGET_BAND_GPU_MS) clamped to 1..MAX_BANDS
 * - pacing: sleeps gpu·(1/DUTY−1); a zero gap sleeps nothing
 *
 * Dive/presenter invariants:
 * - settle latches the window and commits exactly once; setZoom's
 *   magnification is windowStartScale / viewportScale (pure state)
 * - single flight: no second render submit while one is in flight
 * - the next base is requested exactly at the window end (f = ½)
 * - the transform keeps zooming past 1:1 through the render gap
 *   (the animated soft upscale) and only freezes when the window
 *   closes
 * - an invalidation mid-flight (click-zoom) orphans the settle:
 *   no commit, window closed, transform frozen until the fresh
 *   base lands
 * - a click maps through the presenter's frozen scale(m) rect to
 *   the UNtransformed canvas: an edge click lands on the edge
 *   device px at any m
 * - the zoom law is exact: s ← s·e^(−r·Δt); jumps clamp at MIN_S
 *
 * Adaptive-rate invariants (5c):
 * - a flight overrunning its window (flight × headroom >
 *   ln(BOF)/rate) slows the dive to the fitted rate, clamped
 *   SPEED_MIN..user rate; an enormous flight pins at SPEED_MIN
 * - comfortable flights climb ×ADAPTIVE_RECOVERY_STEP per settle,
 *   clamping exactly at the user rate; at the user rate they
 *   change nothing
 * - paused-time flights (the reference build) and flights spanning
 *   a hidden tab (clamped inter-band sleeps) are never samples
 * ================================================================ */

/**
 * Runs the mock section (clock/rAF shadows installed here stay for
 * the page — this section runs last).
 * @returns {Promise<void>}
 */
async function renderPresentSplitTests() {
  // shims: this section mocks the whole GPU — run even where WebGPU globals are absent
  if (typeof GPUTextureUsage === 'undefined') window.GPUTextureUsage = { RENDER_ATTACHMENT: 16, TEXTURE_BINDING: 4 };
  if (typeof GPUBufferUsage === 'undefined') window.GPUBufferUsage = { UNIFORM: 64, COPY_DST: 8, STORAGE: 512 };

  // deterministic clock + rAF (dive.js reads both; the shadows make every tick reproducible)
  let nowMs = 0;
  performance.now = () => nowMs; // own-property shadow — the real clock keeps running elsewhere
  let rafCallback = null;
  window.requestAnimationFrame = (callback) => { rafCallback = callback; };

  /* --- 5a: engine lifecycle on a mock GPUDevice (mocks injected; init()'s WebGPU plumbing is not under test) --- */
  const engineLifecycleTest = async () => {
    const CANVAS_VIEW = { __canvasView: true };
    const log = [];
    const mkPass = (kind) => ({
      kind,
      setPipeline(pipeline) { this.pipeline = pipeline; },
      setBindGroup(_i, bindGroup) { this.bindGroup = bindGroup; },
      setScissorRect() {}, // banded frames scissor each band's strip — layout asserted by section 3, not here
      dispatchWorkgroups(n) { this.dispatch = n; },
      draw(n) { this.drawCount = n; },
      end() {},
    });
    let bindGroupCount = 0;
    const device = {
      queue: {
        writeBuffer(buffer, _offset, data) { log.push(['writeBuffer', buffer.tag, Array.from(data).slice(0, 8)]); },
        submit(buffers) {
          log.push(['submit', buffers.length]);
          for (const buffer of buffers) {
            for (const pass of buffer.passes) log.push(['pass', pass.kind, pass.desc ? pass.desc.colorAttachments[0].view : null]);
          }
        },
        onSubmittedWorkDone: async () => { log.push(['settled']); },
      },
      createSampler: (spec) => ({ __sampler: spec }),
      createBuffer: (spec) => ({ ...spec, tag: spec.usage }),
      createTexture: (spec) => ({
        ...spec, width: spec.size[0], height: spec.size[1], destroyed: false,
        createView() { return { texture: this, id: `view-${spec.size[0]}x${spec.size[1]}` }; },
        destroy() { this.destroyed = true; },
      }),
      createBindGroup: (spec) => { bindGroupCount++; return { __bindGroup: spec }; },
      createCommandEncoder: () => ({
        passes: [],
        beginComputePass() { const pass = mkPass('compute'); this.passes.push(pass); return pass; },
        beginRenderPass(desc) { const pass = mkPass('render'); pass.desc = desc; this.passes.push(pass); return pass; },
        finish() { return { passes: this.passes }; },
      }),
    };
    const canvasContext = { configure() {}, getCurrentTexture: () => ({ createView: () => CANVAS_VIEW }) };
    const canvas = { width: 300, height: 150 };

    const engine = new WebGPUPerturbationEngine();
    engine.gpuDevice = device;               // injected — init()'s adapter/pipeline plumbing is not under test
    engine.canvas = canvas;
    engine.canvasContext = canvasContext;
    engine.canvasFormat = 'bgra8unorm';
    engine.blitPipeline = { getBindGroupLayout: () => 'layout' };
    engine.referencePipeline = { getBindGroupLayout: () => 'layout' }; // pipelines too — init() is not under test
    engine.framePipeline = { getBindGroupLayout: () => 'layout' };
    engine.blitSampler = device.createSampler({ magFilter: 'linear', minFilter: 'linear' });
    engine.blitCropUniformBuffer = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    engine.available = true;
    check(engine.available && engine.gpuDevice === device, 'A1 engine accepts the injected mock device');

    engine.resize(960, 540);
    check(canvas.width === 1920 && canvas.height === 1080, `A2 canvas backing at 2× (got ${canvas.width}×${canvas.height})`);

    const view = () => makeView({ centerXText: '-0.743643887', centerYText: '0.131825904', viewportScale: 0.0025 });
    log.length = 0;
    await engine.render(view(), 400);
    check(engine.baseTexture.width === 1920 && engine.baseTexture.height === 1080, 'A3 base texture at 2× canvas size');
    const passes = log.filter((entry) => entry[0] === 'pass').map((entry) => entry[1]);
    check(passes[0] === 'compute' && passes.length >= 2 && passes.slice(1).every((kind) => kind === 'render'),
      `A4 fresh-reference frame runs compute then banded render passes (${passes.join(',')})`);
    check(log.every((entry) => entry[0] !== 'pass' || entry[2] !== CANVAS_VIEW), 'A5 the render never targets the canvas');
    check(log.some((entry) => entry[0] === 'settled'), 'A6 render awaits the settle');

    log.length = 0;
    engine.commit();
    check(log.filter((entry) => entry[0] === 'submit').length === 1, 'A7 commit submits exactly once');
    check(log.some((entry) => entry[0] === 'pass' && entry[1] === 'render' && entry[2] === CANVAS_VIEW), 'A8 commit blits into the canvas view');
    const cropWrite = log.find((entry) => entry[0] === 'writeBuffer' && entry[1] === 72); // UNIFORM|COPY_DST
    check(!!cropWrite && cropWrite[2][0] === 1, 'A9 commit writes crop fraction 1.0 (full base)');
    const renderedView = engine.baseTextureView;
    const bindGroupsBefore = bindGroupCount;
    engine.commit();
    check(bindGroupCount === bindGroupsBefore, 'A10 the commit bind group is cached per texture lifetime');
    check(engine.baseTextureView === renderedView, 'A11 the base view is cached');

    log.length = 0;
    await engine.render(view(), 400); // same center + budget → cached reference
    check(!log.some((entry) => entry[0] === 'pass' && entry[1] === 'compute'), 'A12 cached-reference frame skips the compute pass');

    engine.resize(1280, 720);
    log.length = 0;
    engine.commit();
    check(log.filter((entry) => entry[0] === 'submit').length === 0, 'A13 commit after a resize (base destroyed) early-returns');

    await engine.render(view(), 400);
    engine.commit();
    check(engine.baseTexture.width === 2560 && log.some((entry) => entry[0] === 'pass' && entry[2] === CANVAS_VIEW), 'A14 re-render + commit at the new size');

    // band sizing + duty-cycle pacing — pure engine math on the same instance, no GPU calls
    engine.lastFrameTimingsMs = null;
    check(engine.bandCountForFrame() === DEFAULT_BAND_COUNT, `A15 unmeasured frame sizes at DEFAULT_BAND_COUNT (${DEFAULT_BAND_COUNT})`);
    for (const [gpuMs, expectedBands] of [[0, 1], [TARGET_BAND_GPU_MS / 2, 1], [TARGET_BAND_GPU_MS, 1], [TARGET_BAND_GPU_MS + 1, 2], [1e9, MAX_BANDS]]) {
      engine.lastFrameTimingsMs = { cpuMs: 0, gpuMs };
      check(engine.bandCountForFrame() === expectedBands, `A16 sizing gpuMs=${gpuMs} → ${expectedBands} band(s) (ceil, clamp 1..${MAX_BANDS})`);
    }
    const pacedDelaysMs = [];
    const realSetTimeout = window.setTimeout; // capture the sleep the pacing path actually requests
    window.setTimeout = (callback, delayMs) => { pacedDelaysMs.push(delayMs); return realSetTimeout(callback, delayMs); };
    await engine.paceAfterBand(0);   // zero gap — must not sleep at all
    await engine.paceAfterBand(90);  // 90 ms band at DUTY 0.9 → 10 ms idle
    window.setTimeout = realSetTimeout;
    const expectedIdleMs = 90 * (1 / GPU_DUTY_CYCLE - 1);
    check(pacedDelaysMs.length === 1 && Math.abs(pacedDelaysMs[0] - expectedIdleMs) < 1e-9,
      `A17 pacing sleeps gpu·(1/DUTY−1) = ${expectedIdleMs.toFixed(2)} ms; a zero gap skips the sleep`);
  };

  /* --- 5b: dive.js window state machine + presenter on a mock engine --- */
  const diveStateMachineTest = async () => {
    const calls = { commit: 0, resetRef: 0, renders: [] };
    let renderGate = null;
    let canvasScale = 1; // the presenter transform's m, frozen on the mock canvas's rect
    const mockCanvas = {
      clientWidth: 100, clientHeight: 50, // layout size — never touched by a transform
      getBoundingClientRect: () => ({ // POST-transform box, like the real DOM's under scale(canvasScale)
        left: (100 - 100 * canvasScale) / 2,
        top: (50 - 50 * canvasScale) / 2,
        width: 100 * canvasScale,
        height: 50 * canvasScale,
      }),
    };
    const mockEngine = {
      available: true, w: 100, h: 50,
      canvas: mockCanvas,
      resetRef() { calls.resetRef++; },
      commit() { calls.commit++; },
      render(view, maxIter) { calls.renders.push({ s: view.s, budget: maxIter }); return new Promise((resolve) => { renderGate = resolve; }); },
      lastFrameTimingsMs: { cpuMs: 1, gpuMs: 1234.5 },
    };
    const fakeCanvas = { style: {} };

    window.updatePauseButton = () => {}; // ui.js is not loaded on this page — no-op stand-ins for its contract
    window.updateHud = () => {};
    window.showFailureNotice = () => {};
    diveState.engine = mockEngine; // the mock replaces the real engine before the first control call
    initPresenter(fakeCanvas);
    rafCallback = onAnimationFrame; // app.js never schedules its own first frame (initApp does)

    const settleRender = async () => { renderGate(); await new Promise((resolve) => setTimeout(resolve, 0)); };
    const tick = () => { const callback = rafCallback; rafCallback = null; callback(); }; // one rAF step — the loop reschedules itself

    startDive({ x: '-0.743643887', y: '0.131825904' });
    check(calls.resetRef === 1 && diveState.baseWindowStartScale === 0, 'B1 startDive resets the reference and closes the window');
    const initialScale = diveState.viewportScale; // BASE_SCALE / h

    tick(); // paused, needsRepaint → first render
    check(calls.renders.length === 1 && diveState.isRenderInFlight, 'B2 first base render submits on the next tick');
    check(calls.renders[0].s === initialScale / BASE_OVERRENDER_FACTOR, `B3 the render pitch is the base pitch (window start ÷ ${BASE_OVERRENDER_FACTOR})`);
    tick();
    check(calls.renders.length === 1, 'B4 single flight — no second render while one is in flight');
    check(fakeCanvas.style.transform === undefined, 'B5 no transform write while the window is closed (hold)');

    await settleRender();
    check(calls.commit === 1 && diveState.baseWindowStartScale === initialScale, 'B6 settle latches the window and commits exactly once');
    check(diveState.lastFrameGpuMs === 1234.5, 'B7 the GPU timing reaches the HUD state');

    tick();
    check(fakeCanvas.style.transform === 'scale(1)', `B8 window open at factor 1 → transform scale(1) (got ${fakeCanvas.style.transform})`);

    togglePause(); // resume at DEFAULT_SPEED = 1/s
    nowMs += 100; // short step — stays inside the window (window runs ln2/DEFAULT_SPEED ≈ 693 ms)
    tick();
    const expectedScale = initialScale * Math.exp(-DEFAULT_SPEED * (100 / MS_PER_SECOND)); // Δt = 100 ms
    check(Math.abs(diveState.viewportScale - expectedScale) / expectedScale < 1e-12, 'B9 zoom law exact: s·e^(−r·Δt)');
    const expectedTransform = `scale(${initialScale / diveState.viewportScale})`;
    check(fakeCanvas.style.transform === expectedTransform, `B10 transform = windowStart/viewportScale (got ${fakeCanvas.style.transform})`);
    check(calls.renders.length === 1, 'B11 no resubmit inside the window');

    nowMs += 600; // cumulative 700 ms > ln2 s → the crop reaches ½ (window end)
    tick();
    check(calls.renders.length === 2 && diveState.isRenderInFlight, 'B12 the next base is requested at the window end (f = ½)');
    check(fakeCanvas.style.transform === `scale(${initialScale / diveState.viewportScale})`, `B13 the transform keeps zooming past 1:1 through the gap (got ${fakeCanvas.style.transform})`);

    clickToZoom({ clientX: 50, clientY: 25 }); // mid-flight invalidation: generation bump, window closed
    check(diveState.baseWindowStartScale === 0, 'B14 click-zoom closes the window');
    await settleRender();
    check(calls.commit === 1, 'B15 the orphaned settle never commits (generation mismatch)');
    tick(); // needsRepaint from the click → fresh base at the new view
    check(calls.renders.length === 3, 'B16 the fresh base submits after the invalidation');
    await settleRender();
    check(calls.commit === 2 && diveState.baseWindowStartScale > 0, 'B17 the fresh base commits and reopens the window');

    // floor clamp: a jump below MIN_S lands at the floor, paused
    goToView(DIVE_TARGETS[1].x, DIVE_TARGETS[1].y, MIN_S * 40); // height clamps: s = MIN_S
    check(diveState.viewportScale === MIN_S && !diveState.isRunning, 'B18 view jumps clamp at the MIN_S floor (paused)');

    // click mapping under the presenter transform: the mock rect is post-scale(2)
    // (center-origin, exactly what setZoom leaves mid-window) — an edge click must
    // still land on the edge device px, not bent (1 − 1/m) toward the center
    const sBeforeClick = diveState.viewportScale;
    const xBeforeClick = diveState.centerXFixed;
    const yBeforeClick = diveState.centerYFixed;
    canvasScale = 2;
    clickToZoom({ clientX: 100, clientY: 0 }); // top-right corner: fx = 1, fy = 0
    const centerScale = Number(1n << BigInt(CENTER_FIXED_POINT_BITS));
    const expectedDx = BigInt(Math.round((100 - (100 / 2 - 0.5)) * sBeforeClick * centerScale));
    const expectedDy = BigInt(Math.round((50 / 2 - 0.5) * sBeforeClick * centerScale));
    check(diveState.centerXFixed - xBeforeClick === expectedDx, 'B19 edge click maps through the presenter transform (x)');
    check(diveState.centerYFixed - yBeforeClick === expectedDy, 'B20 edge click maps through the presenter transform (y)');
  };

  /* --- 5c: adaptive zoom rate — flight-driven slowdown/recovery, driven through submitRender's settle path --- */
  const adaptiveRateTest = async () => {
    let renderGate = null;
    const mockEngine = {
      available: true, w: 100, h: 50,
      canvas: { clientWidth: 100, clientHeight: 50, getBoundingClientRect: () => ({ left: 0, top: 0, width: 100, height: 50 }) },
      resetRef() {}, commit() {}, // the commit/window bookkeeping is 5b's subject — only the rate transitions matter here
      render(_view, _maxIter) { return new Promise((resolve) => { renderGate = resolve; }); },
      lastFrameTimingsMs: { cpuMs: 1, gpuMs: 1 },
    };
    diveState.engine = mockEngine;
    diveState.isRunning = true; // flights requested while running are the adaptation samples
    const setUserRate = (rate) => { diveState.userZoomRatePerSecond = rate; diveState.zoomRatePerSecond = rate; };
    const settleFlight = async (flightMs) => {
      submitRender();
      nowMs += flightMs; // the shadowed clock: the flight takes exactly this long
      renderGate();
      await new Promise((resolve) => setTimeout(resolve, 0));
    };
    const fittedRate = (flightMs) => Math.log(BASE_OVERRENDER_FACTOR) / ((flightMs / 1000) * ADAPTIVE_FLIGHT_HEADROOM);
    const USER_RATE = 0.5; // a made-up slider setting — above every fitted rate below, under SPEED_MAX

    // slow flight: window at USER_RATE is ln2/0.5 ≈ 1386 ms ≪ 3000×1.25 → the fitted rate
    setUserRate(USER_RATE);
    await settleFlight(3000);
    check(Math.abs(diveState.zoomRatePerSecond - fittedRate(3000)) < 1e-12,
          `C1 a flight overrunning its window slows the dive to the fitted rate (got ${diveState.zoomRatePerSecond.toFixed(6)}, want ${fittedRate(3000).toFixed(6)})`);

    // enormous flight: fitted rate ln2/75 ≈ 0.0092 < SPEED_MIN → the floor
    await settleFlight(60000);
    check(diveState.zoomRatePerSecond === SPEED_MIN, `C2 an enormous flight pins at SPEED_MIN (got ${diveState.zoomRatePerSecond})`);

    // recovery: one comfortable settle climbs exactly ×1.15; enough of them clamp exactly at the user rate
    await settleFlight(100); // 100×1.25 ≪ the ≈13.9 s window at SPEED_MIN
    check(Math.abs(diveState.zoomRatePerSecond - SPEED_MIN * ADAPTIVE_RECOVERY_STEP) < 1e-12,
          `C3 a comfortable flight recovers ×${ADAPTIVE_RECOVERY_STEP} (got ${diveState.zoomRatePerSecond.toFixed(6)})`);
    for (let i = 0; i < 40 && diveState.zoomRatePerSecond < diveState.userZoomRatePerSecond; i++) await settleFlight(100);
    check(diveState.zoomRatePerSecond === diveState.userZoomRatePerSecond,
          `C4 recovery clamps exactly at the user rate (got ${diveState.zoomRatePerSecond})`);

    // comfortable flight while already at the user rate: no drift
    await settleFlight(100);
    check(diveState.zoomRatePerSecond === USER_RATE, `C5 comfortable flights at the user rate change nothing (got ${diveState.zoomRatePerSecond})`);

    // paused-time flight (the post-goToView reference build): never a sample, however long
    diveState.isRunning = false;
    setUserRate(USER_RATE);
    await settleFlight(60000);
    check(diveState.zoomRatePerSecond === USER_RATE, `C6 paused flights never adapt the rate (got ${diveState.zoomRatePerSecond})`);

    // hidden-tab flight: hidden tabs clamp the inter-band sleeps to ≥1 s each — the
    // wall time is minutes past the GPU cost, so it must not sample
    diveState.isRunning = true;
    submitRender(); // requested visible — the stamp latches before the hide
    Object.defineProperty(document, 'hidden', { configurable: true, get: () => true });
    document.dispatchEvent(new Event('visibilitychange')); // dive.js's listener stamps the hide
    check(!diveState.isRunning, `C7a the hide auto-pauses the running dive (isRunning=${diveState.isRunning})`);
    nowMs += 60000; // what the clamped sleeps would inflate the flight to
    renderGate();
    await new Promise((resolve) => setTimeout(resolve, 0));
    check(diveState.zoomRatePerSecond === USER_RATE,
          `C7 a flight spanning a hidden stretch is not a sample — the rate holds (got ${diveState.zoomRatePerSecond})`);

    // tab back: the stamp matches again — visible flights sample once more
    delete document.hidden; // restore the real accessor (the listener bumps only on hides)
    document.dispatchEvent(new Event('visibilitychange')); // the show transition: the auto-pause from the hide resumes
    check(diveState.isRunning, `C8a the dive auto-resumes when the tab returns (isRunning=${diveState.isRunning})`);
    await settleFlight(3000);
    check(Math.abs(diveState.zoomRatePerSecond - fittedRate(3000)) < 1e-12,
          `C8 visible flights sample again once the tab is back (got ${diveState.zoomRatePerSecond.toFixed(6)}, want ${fittedRate(3000).toFixed(6)})`);

    // Auto off (the footer checkbox): overrunning flights hold the user's rate exactly;
    // back on, flights retune it again. setUserRate mirrors what the checkbox's
    // restore does (ui.js is not loaded on this page).
    diveState.isAutoSpeedEnabled = false;
    setUserRate(USER_RATE);
    await settleFlight(60000);
    check(diveState.zoomRatePerSecond === USER_RATE,
          `C9 Auto off — overrunning flights never slow the dive (got ${diveState.zoomRatePerSecond})`);
    diveState.isAutoSpeedEnabled = true;
    await settleFlight(3000);
    check(Math.abs(diveState.zoomRatePerSecond - fittedRate(3000)) < 1e-12,
          `C10 Auto on again — flights retune the dive's rate (got ${diveState.zoomRatePerSecond.toFixed(6)}, want ${fittedRate(3000).toFixed(6)})`);
  };

  await engineLifecycleTest();
  await diveStateMachineTest();
  await adaptiveRateTest();
}

/* --- wiring / lifecycle --- */

/**
 * Boots WebGPU, runs all five sections in order, writes the
 * summary. No fallback — a WebGPU-less browser gets the honest
 * failure notice (section 5's mocks never make up for a missing
 * adapter).
 * @returns {Promise<void>}
 */
async function runTests() {
  if (!('gpu' in navigator)) {
    finishTests('FAIL — navigator.gpu missing (WebGPU off in this browser)');
    return;
  }
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) {
    finishTests('FAIL — no adapter');
    return;
  }
  const device = await adapter.requestDevice();
  const referencePipeline = await buildReferencePipeline(device);

  await referenceOrbitTests(device, referencePipeline);
  await perturbFrameTests(device, referencePipeline);
  await bandedRenderTests();
  await presentBlitTests(device);
  await renderPresentSplitTests();

  const total = passCount + failCount;
  finishTests(failCount === 0
    ? `PASS — reference orbit, frame parity, banding, blit, split all green (${total} checks)`
    : `FAIL — ${failCount}/${total} checks failed`);
}

runTests().catch((error) => {
  logLine('fail', `test crashed: ${error}`);
  finishTests('FAIL — crashed (see log)');
});
