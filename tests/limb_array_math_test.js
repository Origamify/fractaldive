'use strict';
/* ================================================================
 * Limb-array math tests — runs the WGSL lib (limb_array_math.js,
 * LIMB_ARRAY_MATH_WGSL) on a real WebGPU device and compares every
 * function against JS BigInt two's-complement twins (the oracle).
 *
 * How: for each precision tier (limb count 1..8) the lib source is
 * spliced into a one-thread test kernel; each case is one dispatch
 * with operands in a read-only storage buffer (uniforms reject
 * stride-4 limb arrays), the lib call's result read back from
 * storage, and compared limb-by-limb with the oracle.
 *
 * Owns nothing but this page; never touches the app's engine.
 * ================================================================ */

/* --- constants --- */
const TESTED_LIMB_COUNTS = [1, 2, 3, 4, 5, 6, 7, 8]; // every tier — validates the __LIMB_COUNT__ splice itself
const RANDOM_CASES_PER_GROUP = 4;                    // random vectors per edge group, per tier
const MAX_DISPLAYED_FAILURES = 20;                   // log cap — beyond that only counts
const INPUT_SCALAR_WORDS = 4;                        // opcode, scalar, wordA, wordB before the limb arrays

// Opcodes match the WGSL switch below — kind drives the comparison:
// 'limbs' compares result[0..2N), 'flag' compares the single bool slot.
const OP_MUL_WORD = 0;
const OP_IS_NEGATIVE = 1;
const OP_NEGATE = 2;
const OP_ADD = 3;
const OP_SUB = 4;
const OP_ADD_WIDE = 5;
const OP_SUB_WIDE = 6;
const OP_UNSIGNED_GREATER_WIDE = 7;
const OP_MULTIPLY_SIGNED = 8;
const OP_MULTIPLY_SMALL = 9;
const OP_SHIFT_RIGHT_ARITHMETIC_WIDE = 10;
const OP_TRUNCATE_LOW = 11;
const OP_SQUARE = 12;
const OP_INFO = {
  [OP_MUL_WORD]: { name: 'mulWord32', kind: 'limbs' },
  [OP_IS_NEGATIVE]: { name: 'limbIsNegative', kind: 'flag' },
  [OP_NEGATE]: { name: 'limbNegate', kind: 'limbs' },
  [OP_ADD]: { name: 'limbAdd', kind: 'limbs' },
  [OP_SUB]: { name: 'limbSub', kind: 'limbs' },
  [OP_ADD_WIDE]: { name: 'limbAddWide', kind: 'limbs' },
  [OP_SUB_WIDE]: { name: 'limbSubWide', kind: 'limbs' },
  [OP_UNSIGNED_GREATER_WIDE]: { name: 'limbUnsignedGreaterWide', kind: 'flag' },
  [OP_MULTIPLY_SIGNED]: { name: 'limbMultiplySigned', kind: 'limbs' },
  [OP_MULTIPLY_SMALL]: { name: 'limbMultiplySmall', kind: 'limbs' },
  [OP_SHIFT_RIGHT_ARITHMETIC_WIDE]: { name: 'limbShiftRightArithmeticWide', kind: 'limbs' },
  [OP_TRUNCATE_LOW]: { name: 'limbTruncateLow', kind: 'limbs' },
  [OP_SQUARE]: { name: 'limbSquare', kind: 'limbs' },
};

/* --- state --- */
const totalCounts = { pass: 0, fail: 0 };

/* --- BigInt oracle — two's-complement twins of the WGSL fns --- */
/* (limb/PRNG helpers come from test_kit.js) */

/**
 * Returns the expected result words of a case per the oracle —
 * limbs at product width (2N, zero-padded for narrow results).
 * @param {object} testCase - { opcode, scalar, wordA, wordB, a, b, wa, wb }.
 * @param {number} limbCount - Tier n.
 * @returns {number[]} 2n expected words.
 */
function expectedLimbsForCase(testCase, limbCount) {
  const wideCount = 2 * limbCount;
  switch (testCase.opcode) {
    case OP_MUL_WORD: {
      const product = BigInt(testCase.wordA) * BigInt(testCase.wordB);
      return [Number(product & 0xFFFFFFFFn), Number(product >> 32n)].concat(Array(wideCount - 2).fill(0));
    }
    case OP_NEGATE:
      return limbsFromBigInt(-unsignedValueOfLimbs(testCase.a), limbCount);
    case OP_ADD:
      return limbsFromBigInt(unsignedValueOfLimbs(testCase.a) + unsignedValueOfLimbs(testCase.b), limbCount);
    case OP_SUB:
      return limbsFromBigInt(unsignedValueOfLimbs(testCase.a) - unsignedValueOfLimbs(testCase.b), limbCount);
    case OP_ADD_WIDE:
      return limbsFromBigInt(unsignedValueOfLimbs(testCase.wa) + unsignedValueOfLimbs(testCase.wb), wideCount);
    case OP_SUB_WIDE:
      return limbsFromBigInt(unsignedValueOfLimbs(testCase.wa) - unsignedValueOfLimbs(testCase.wb), wideCount);
    case OP_MULTIPLY_SIGNED:
      return limbsFromBigInt(signedValueOfLimbs(testCase.a) * signedValueOfLimbs(testCase.b), wideCount);
    case OP_SQUARE:
      return limbsFromBigInt(signedValueOfLimbs(testCase.a) * signedValueOfLimbs(testCase.a), wideCount);
    case OP_MULTIPLY_SMALL:
      return limbsFromBigInt(unsignedValueOfLimbs(testCase.a) * BigInt(testCase.scalar), limbCount);
    case OP_SHIFT_RIGHT_ARITHMETIC_WIDE:
      return limbsFromBigInt(signedValueOfLimbs(testCase.wa) >> BigInt(testCase.scalar), wideCount);
    case OP_TRUNCATE_LOW:
      return testCase.wa.slice(0, limbCount).concat(Array(wideCount - limbCount).fill(0));
    default:
      return Array(wideCount).fill(0); // flag-kind cases carry no limb expectation
  }
}

/**
 * Returns the expected bool-slot word for flag-kind cases, else null.
 * @param {object} testCase - Same shape as expectedLimbsForCase.
 * @param {number} limbCount - Tier n.
 * @returns {?number} 0/1, or null when the opcode is not a predicate.
 */
function expectedFlagForCase(testCase, limbCount) {
  if (testCase.opcode === OP_IS_NEGATIVE) {
    return (testCase.a[limbCount - 1] & 0x80000000) !== 0 ? 1 : 0;
  }
  if (testCase.opcode === OP_UNSIGNED_GREATER_WIDE) {
    return unsignedValueOfLimbs(testCase.wa) > unsignedValueOfLimbs(testCase.wb) ? 1 : 0;
  }
  return null;
}

/* --- case builders (inputs per opcode; oracle derives expectations) --- */

/**
 * Returns the canonical edge vectors for a limb width: 0, 1, −1,
 * most-negative, most-positive, plus RANDOM_CASES_PER_GROUP randoms.
 * @param {number} limbCount - Width n.
 * @returns {number[][]} Limb vectors.
 */
function edgeValuesForLimbCount(limbCount) {
  const zero = Array(limbCount).fill(0);
  const one = [1].concat(Array(limbCount - 1).fill(0));
  const minusOne = Array(limbCount).fill(0xFFFFFFFF);
  const mostNegative = Array(limbCount).fill(0);
  mostNegative[limbCount - 1] = 0x80000000;
  const mostPositive = Array(limbCount).fill(0xFFFFFFFF);
  mostPositive[limbCount - 1] = 0x7FFFFFFF;
  const edges = [zero, one, minusOne, mostNegative, mostPositive];
  for (let randomIndex = 0; randomIndex < RANDOM_CASES_PER_GROUP; randomIndex++) {
    edges.push(Array.from({ length: limbCount }, nextRandomU32));
  }
  return edges;
}

/**
 * Returns the full case list for one tier — every lib function gets
 * edge vectors plus randoms; pair ops get carry-chain-prone combos.
 * @param {number} limbCount - Tier n.
 * @returns {object[]} Cases shaped for the uniform packer and oracle.
 */
function buildCaseList(limbCount) {
  prngState = PRNG_SEED; // same vectors per tier width — failures replay
  const wideCount = 2 * limbCount;
  const edges = edgeValuesForLimbCount(limbCount);
  const wideEdges = edgeValuesForLimbCount(wideCount);
  const randomPairs = () => {
    const pairs = [];
    for (let pairIndex = 0; pairIndex < RANDOM_CASES_PER_GROUP; pairIndex++) {
      pairs.push([Array.from({ length: limbCount }, nextRandomU32), Array.from({ length: limbCount }, nextRandomU32)]);
    }
    return pairs;
  };
  const narrowPairs = [[edges[0], edges[1]], [edges[1], edges[2]], [edges[2], edges[1]],
    [edges[3], edges[3]], [edges[4], edges[4]], [edges[2], edges[2]]].concat(randomPairs());
  const widePairs = [[wideEdges[0], wideEdges[1]], [wideEdges[1], wideEdges[2]], [wideEdges[2], wideEdges[1]],
    [wideEdges[3], wideEdges[3]], [wideEdges[4], wideEdges[4]]].concat(
      Array.from({ length: RANDOM_CASES_PER_GROUP }, () => [Array.from({ length: wideCount }, nextRandomU32), Array.from({ length: wideCount }, nextRandomU32)]));

  const cases = [];
  const caseWords = [0, 1, 0xFFFF, 0x7FFFFFFF, 0x80000000, 0xFFFFFFFF];
  for (let wordIndex = 0; wordIndex < caseWords.length; wordIndex++) {
    cases.push({ opcode: OP_MUL_WORD, wordA: caseWords[wordIndex], wordB: caseWords[wordIndex], label: `${caseWords[wordIndex]}>>>0 × itself` });
    cases.push({ opcode: OP_MUL_WORD, wordA: caseWords[wordIndex], wordB: nextRandomU32(), label: 'random word pair' });
  }
  for (const value of edges) {
    cases.push({ opcode: OP_IS_NEGATIVE, a: value, label: 'edge/random value' });
    cases.push({ opcode: OP_NEGATE, a: value, label: 'edge/random value' });
  }
  for (const [leftValue, rightValue] of narrowPairs) {
    cases.push({ opcode: OP_ADD, a: leftValue, b: rightValue, label: 'narrow pair' });
    cases.push({ opcode: OP_SUB, a: leftValue, b: rightValue, label: 'narrow pair' });
  }
  for (const [leftWide, rightWide] of widePairs) {
    cases.push({ opcode: OP_ADD_WIDE, wa: leftWide, wb: rightWide, label: 'wide pair' });
    cases.push({ opcode: OP_SUB_WIDE, wa: leftWide, wb: rightWide, label: 'wide pair' });
    cases.push({ opcode: OP_UNSIGNED_GREATER_WIDE, wa: leftWide, wb: rightWide, label: 'wide pair' });
  }
  cases.push({ opcode: OP_UNSIGNED_GREATER_WIDE, wa: wideEdges[2], wb: wideEdges[2], label: 'equal operands' });
  for (const [leftValue, rightValue] of [[edges[3], edges[3]], [edges[3], edges[4]], [edges[4], edges[4]],
    [edges[2], edges[3]], [edges[0], edges[4]]].concat(randomPairs())) {
    cases.push({ opcode: OP_MULTIPLY_SIGNED, a: leftValue, b: rightValue, label: 'signed pair' });
  }
  for (const value of edges.concat(randomPairs().map(([leftValue]) => leftValue))) {
    cases.push({ opcode: OP_SQUARE, a: value, label: 'square edge/random' });
  }
  for (const value of edges) {
    for (const scalar of [0, 1, 2, 0xFFFF, 0x7FFFFFFF, 0xFFFFFFFF]) {
      cases.push({ opcode: OP_MULTIPLY_SMALL, a: value, scalar, label: `value × scalar ${scalar}` });
    }
  }
  const shiftWideValues = [wideEdges[0], wideEdges[2], wideEdges[3], Array.from({ length: wideCount }, nextRandomU32)];
  for (const wideValue of shiftWideValues) {
    for (const bitCount of [0, 1, 31, 32, 33, 63, 64 * limbCount - 1, 64 * limbCount]) {
      cases.push({ opcode: OP_SHIFT_RIGHT_ARITHMETIC_WIDE, wa: wideValue, scalar: bitCount, label: `wide >> ${bitCount}` });
    }
  }
  for (const wideValue of wideEdges) {
    cases.push({ opcode: OP_TRUNCATE_LOW, wa: wideValue, label: 'wide value' });
  }
  return cases;
}

/* --- GPU plumbing --- */

/**
 * Returns the compiled one-thread test pipeline for a tier — the lib
 * spliced ahead of the dispatch switch, __LIMB_COUNT__ substituted.
 * Compilation problems are logged (shader messages + device error
 * scope) — an invalid pipeline would otherwise silently zero results.
 * @param {GPUDevice} gpuDevice - Active device.
 * @param {number} limbCount - Tier n.
 * @returns {Promise<GPUComputePipeline>} Compiled pipeline.
 * @throws {Error} When compilation info reports errors or the device errors.
 */
async function compileTestPipeline(gpuDevice, limbCount) {
  const kernelSource = (LIMB_ARRAY_MATH_WGSL + `
const RESULT_FLAG_OFFSET: u32 = PRODUCT_LIMB_COUNT; // bool slot after the 2N result words

// Case inputs live in a read-only storage buffer (the uniform
// address space rejects stride-4 limb arrays) — same word layout
// as inputWordsForCase on the JS side.
const OPCODE_OFFSET: u32 = 0u;
const SCALAR_OFFSET: u32 = 1u;   // limbMultiplySmall operand / shift bitCount
const WORD_A_OFFSET: u32 = 2u;   // mulWord32 left word
const WORD_B_OFFSET: u32 = 3u;   // mulWord32 right word
const OPERAND_A_OFFSET: u32 = 4u;
const OPERAND_B_OFFSET: u32 = OPERAND_A_OFFSET + LIMB_COUNT;
const OPERAND_WIDE_A_OFFSET: u32 = OPERAND_B_OFFSET + LIMB_COUNT;
const OPERAND_WIDE_B_OFFSET: u32 = OPERAND_WIDE_A_OFFSET + PRODUCT_LIMB_COUNT;

@group(0) @binding(0) var<storage, read> caseInputs: array<u32>;
@group(0) @binding(1) var<storage, read_write> result: array<u32>;

fn loadNarrowValue(baseOffset: u32) -> limb_array_int {
  var loadedLimbs: limb_array_int;
  for (var limbIndex = 0u; limbIndex < LIMB_COUNT; limbIndex++) { loadedLimbs[limbIndex] = caseInputs[baseOffset + limbIndex]; }
  return loadedLimbs;
}
fn loadWideValue(baseOffset: u32) -> limb_array_product {
  var loadedLimbs: limb_array_product;
  for (var limbIndex = 0u; limbIndex < PRODUCT_LIMB_COUNT; limbIndex++) { loadedLimbs[limbIndex] = caseInputs[baseOffset + limbIndex]; }
  return loadedLimbs;
}

fn storeNarrowValue(valueLimbs: limb_array_int) {
  for (var limbIndex = 0u; limbIndex < LIMB_COUNT; limbIndex++) { result[limbIndex] = valueLimbs[limbIndex]; }
  for (var limbIndex = LIMB_COUNT; limbIndex < PRODUCT_LIMB_COUNT; limbIndex++) { result[limbIndex] = 0u; }
  result[RESULT_FLAG_OFFSET] = 0u;
}
fn storeWideValue(valueLimbs: limb_array_product) {
  for (var limbIndex = 0u; limbIndex < PRODUCT_LIMB_COUNT; limbIndex++) { result[limbIndex] = valueLimbs[limbIndex]; }
  result[RESULT_FLAG_OFFSET] = 0u;
}

@compute @workgroup_size(1)
fn main() {
  switch (caseInputs[OPCODE_OFFSET]) {
    case ${OP_MUL_WORD}u: {
      var wideResult: limb_array_product = limb_array_product();
      let mulWord = mulWord32(caseInputs[WORD_A_OFFSET], caseInputs[WORD_B_OFFSET]);
      wideResult[0] = mulWord.lo; wideResult[1] = mulWord.hi;
      storeWideValue(wideResult);
    }
    case ${OP_IS_NEGATIVE}u: {
      storeNarrowValue(limb_array_int());
      result[RESULT_FLAG_OFFSET] = u32(limbIsNegative(loadNarrowValue(OPERAND_A_OFFSET)));
    }
    case ${OP_NEGATE}u: { storeNarrowValue(limbNegate(loadNarrowValue(OPERAND_A_OFFSET))); }
    case ${OP_ADD}u: { storeNarrowValue(limbAdd(loadNarrowValue(OPERAND_A_OFFSET), loadNarrowValue(OPERAND_B_OFFSET))); }
    case ${OP_SUB}u: { storeNarrowValue(limbSub(loadNarrowValue(OPERAND_A_OFFSET), loadNarrowValue(OPERAND_B_OFFSET))); }
    case ${OP_ADD_WIDE}u: { storeWideValue(limbAddWide(loadWideValue(OPERAND_WIDE_A_OFFSET), loadWideValue(OPERAND_WIDE_B_OFFSET))); }
    case ${OP_SUB_WIDE}u: { storeWideValue(limbSubWide(loadWideValue(OPERAND_WIDE_A_OFFSET), loadWideValue(OPERAND_WIDE_B_OFFSET))); }
    case ${OP_UNSIGNED_GREATER_WIDE}u: {
      storeWideValue(limb_array_product());
      result[RESULT_FLAG_OFFSET] = u32(limbUnsignedGreaterWide(loadWideValue(OPERAND_WIDE_A_OFFSET), loadWideValue(OPERAND_WIDE_B_OFFSET)));
    }
    case ${OP_MULTIPLY_SIGNED}u: { storeWideValue(limbMultiplySigned(loadNarrowValue(OPERAND_A_OFFSET), loadNarrowValue(OPERAND_B_OFFSET))); }
    case ${OP_MULTIPLY_SMALL}u: { storeNarrowValue(limbMultiplySmall(loadNarrowValue(OPERAND_A_OFFSET), caseInputs[SCALAR_OFFSET])); }
    case ${OP_SHIFT_RIGHT_ARITHMETIC_WIDE}u: { storeWideValue(limbShiftRightArithmeticWide(loadWideValue(OPERAND_WIDE_A_OFFSET), caseInputs[SCALAR_OFFSET])); }
    case ${OP_TRUNCATE_LOW}u: { storeNarrowValue(limbTruncateLow(loadWideValue(OPERAND_WIDE_A_OFFSET))); }
    case ${OP_SQUARE}u: { storeWideValue(limbSquare(loadNarrowValue(OPERAND_A_OFFSET))); }
    default: {}
  }
}
`).replaceAll('__LIMB_COUNT__', String(limbCount));
  const shaderModule = gpuDevice.createShaderModule({ code: kernelSource });
  const compilationInfo = await shaderModule.getCompilationInfo();
  const shaderErrors = compilationInfo.messages.filter((message) => message.type === 'error');
  if (shaderErrors.length > 0) {
    throw new Error(`tier ${limbCount} shader errors:\n` + shaderErrors.map((message) => `  line ${message.lineNum}: ${message.message}`).join('\n'));
  }
  gpuDevice.pushErrorScope('validation');
  const pipeline = gpuDevice.createComputePipeline({ layout: 'auto', compute: { module: shaderModule, entryPoint: 'main' } });
  const pipelineError = await gpuDevice.popErrorScope();
  if (pipelineError) throw new Error(`tier ${limbCount} pipeline: ${pipelineError.message}`);
  return pipeline;
}

/**
 * Returns the input words for a case — the JS-side mirror of the
 * kernel's caseInputs layout, in offset order.
 * @param {object} testCase - Case from buildCaseList.
 * @param {number} limbCount - Tier n.
 * @returns {number[]} Input words (opcode, scalar, wordA, wordB, limbs…).
 */
function inputWordsForCase(testCase, limbCount) {
  const wideCount = 2 * limbCount;
  const words = [testCase.opcode, testCase.scalar || 0, testCase.wordA || 0, testCase.wordB || 0];
  const pad = (limbs, width) => (limbs || []).concat(Array(width - (limbs || []).length).fill(0));
  return words.concat(pad(testCase.a, limbCount), pad(testCase.b, limbCount), pad(testCase.wa, wideCount), pad(testCase.wb, wideCount));
}

/**
 * Runs every case of one tier on the device and records results.
 * @param {GPUDevice} gpuDevice - Active device.
 * @param {number} limbCount - Tier n.
 * @param {function} reportCase - Called (testCase, passed, gotWords, expectedWords) per case.
 * @returns {Promise<void>} Resolves when the tier finished.
 */
async function runTier(gpuDevice, limbCount, reportCase) {
  const wideCount = 2 * limbCount;
  const pipeline = await compileTestPipeline(gpuDevice, limbCount);
  const bindGroupLayout = pipeline.getBindGroupLayout(0);
  const inputWordCount = INPUT_SCALAR_WORDS + 6 * limbCount;
  const inputBuffer = gpuDevice.createBuffer({ size: inputWordCount * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  const resultWordCount = wideCount + 1;
  const resultBuffer = gpuDevice.createBuffer({ size: resultWordCount * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
  const stagingBuffer = gpuDevice.createBuffer({ size: resultWordCount * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  const bindGroup = gpuDevice.createBindGroup({
    layout: bindGroupLayout,
    entries: [
      { binding: 0, resource: { buffer: inputBuffer } },
      { binding: 1, resource: { buffer: resultBuffer } },
    ],
  });
  const words = new Uint32Array(inputWordCount);
  for (const testCase of buildCaseList(limbCount)) {
    words.fill(0);
    words.set(inputWordsForCase(testCase, limbCount));
    gpuDevice.queue.writeBuffer(inputBuffer, 0, words);
    const encoder = gpuDevice.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(1);
    pass.end();
    encoder.copyBufferToBuffer(resultBuffer, 0, stagingBuffer, 0, resultWordCount * 4);
    gpuDevice.queue.submit([encoder.finish()]);
    await stagingBuffer.mapAsync(GPUMapMode.READ);
    const gotWords = Array.from(new Uint32Array(stagingBuffer.getMappedRange()).slice());
    stagingBuffer.unmap();
    reportCase(testCase, gotWords);
  }
  inputBuffer.destroy(); resultBuffer.destroy(); stagingBuffer.destroy();
}

/* --- reporting (logLine from test_kit.js) --- */

/**
 * Compares one case's readback against the oracle and logs failures.
 * @param {number} limbCount - Tier n.
 * @param {object} testCase - Case from buildCaseList.
 * @param {number[]} gotWords - Readback result words.
 */
function reportCaseResult(limbCount, testCase, gotWords) {
  const opInfo = OP_INFO[testCase.opcode];
  const wideCount = 2 * limbCount;
  const expectedWords = expectedLimbsForCase(testCase, limbCount);
  const expectedFlag = expectedFlagForCase(testCase, limbCount);
  const limbsMatch = opInfo.kind !== 'flag' && expectedWords.every((word, wordIndex) => word === gotWords[wordIndex]);
  const flagMatches = opInfo.kind === 'flag' ? expectedFlag === gotWords[wideCount] : true;
  const passed = (opInfo.kind === 'flag' ? flagMatches : limbsMatch);
  totalCounts[passed ? 'pass' : 'fail']++;
  if (!passed && totalCounts.fail <= MAX_DISPLAYED_FAILURES) {
    logLine('fail', `FAIL tier=${limbCount} ${opInfo.name} (${testCase.label})\n  expected ${wordsToHex(opInfo.kind === 'flag' ? [expectedFlag] : expectedWords)}\n  got      ${wordsToHex(opInfo.kind === 'flag' ? [gotWords[wideCount]] : gotWords.slice(0, wideCount))}`);
  }
}

/* --- wiring / lifecycle --- */

/**
 * Boots WebGPU, runs all tiers, writes the summary. No fallback —
 * a WebGPU-less browser gets the honest failure notice.
 */
async function runTests() {
  const summaryElement = document.getElementById('test-summary');
  try {
    if (!('gpu' in navigator)) throw new Error('navigator.gpu missing');
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) throw new Error('no adapter');
    const gpuDevice = await adapter.requestDevice();
    logLine('info', `device ok — running tiers ${TESTED_LIMB_COUNTS.join(', ')}`);
    for (const limbCount of TESTED_LIMB_COUNTS) {
      await runTier(gpuDevice, limbCount, (testCase, gotWords) => reportCaseResult(limbCount, testCase, gotWords));
      logLine('info', `tier ${limbCount} done — running total ${totalCounts.pass} pass / ${totalCounts.fail} fail`);
    }
    gpuDevice.destroy();
    const allPassed = totalCounts.fail === 0;
    summaryElement.textContent = `${allPassed ? 'PASS' : 'FAIL'} — ${totalCounts.pass} passed, ${totalCounts.fail} failed`;
    summaryElement.className = allPassed ? 'pass' : 'fail';
  } catch (error) {
    summaryElement.textContent = `cannot run: ${error.message}`;
    summaryElement.className = 'fail';
    logLine('fail', String(error));
  }
}

runTests();
