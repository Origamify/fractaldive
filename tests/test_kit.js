'use strict';
/* ================================================================
 * Test kit — helpers shared by the WGSL test pages (limb math lib,
 * engine kernel): deterministic PRNG, BigInt limb plumbing, f32 bit
 * casting, DOM log lines. Classic script, loaded first.
 * ================================================================ */

/* --- constants --- */
const PRNG_SEED = 0x600DF00D; // fixed → failures are reproducible

/* --- state --- */
let prngState = PRNG_SEED; // mulberry32 cursor — reset by callers for stable vectors

/* --- deterministic PRNG (mulberry32) --- */

/**
 * Returns the next pseudo-random u32 — deterministic across runs so
 * a failure can be replayed; reset prngState to re-run a sequence.
 * @returns {number} Uniform u32.
 */
function nextRandomU32() {
  prngState = (prngState + 0x6D2B79F5) | 0;
  let mixed = Math.imul(prngState ^ (prngState >>> 15), 1 | prngState);
  mixed = (mixed + Math.imul(mixed ^ (mixed >>> 7), 61 | mixed)) ^ mixed;
  return (mixed ^ (mixed >>> 14)) >>> 0;
}

/**
 * Returns value as n little-endian u32 limbs, reduced mod 2^(32·n)
 * (two's-complement wrap — the same drops the WGSL carries make).
 * @param {bigint} value - Any integer; negatives wrap.
 * @param {number} limbCount - Limb width n.
 * @returns {number[]} n limbs, each 0..0xFFFFFFFF.
 */
function limbsFromBigInt(value, limbCount) {
  const modulus = 1n << BigInt(32 * limbCount);
  let wrapped = ((value % modulus) + modulus) % modulus;
  const limbs = [];
  for (let limbIndex = 0; limbIndex < limbCount; limbIndex++) {
    limbs.push(Number(wrapped & 0xFFFFFFFFn));
    wrapped >>= 32n;
  }
  return limbs;
}

/**
 * Returns the unsigned value of little-endian limbs.
 * @param {number[]} limbs - Limb words.
 * @returns {bigint} Unsigned interpretation, 0..2^(32·n)−1.
 */
function unsignedValueOfLimbs(limbs) {
  let value = 0n;
  for (let limbIndex = limbs.length - 1; limbIndex >= 0; limbIndex--) {
    value = (value << 32n) | BigInt(limbs[limbIndex]);
  }
  return value;
}

/**
 * Returns the signed (two's-complement) value of n limbs.
 * @param {number[]} limbs - Limb words.
 * @returns {bigint} Signed interpretation, −2^(32·n−1)..2^(32·n−1)−1.
 */
function signedValueOfLimbs(limbs) {
  const modulus = 1n << BigInt(32 * limbs.length);
  const value = unsignedValueOfLimbs(limbs);
  return value >= modulus / 2n ? value - modulus : value;
}

/**
 * Returns value reduced mod 2^bits (always non-negative).
 * @param {bigint} value - Any integer.
 * @param {bigint} bits - Modulus exponent.
 * @returns {bigint} value mod 2^bits.
 */
function modPow2(value, bits) {
  const modulus = 1n << bits;
  return ((value % modulus) + modulus) % modulus;
}

/**
 * Returns the signed interpretation of value's low `bits` bits.
 * @param {bigint} value - Any integer (already wrapped or not).
 * @param {bigint} bits - Word width in bits.
 * @returns {bigint} Signed value in −2^(bits−1)..2^(bits−1)−1.
 */
function signedAtWidth(value, bits) {
  const modulus = 1n << bits;
  const wrapped = modPow2(value, bits);
  return wrapped >= modulus / 2n ? wrapped - modulus : wrapped;
}

/* --- f32 bit casting (WGSL talks u32 words) --- */

const f32Scratch = new Float32Array(1);
const f32BitsScratch = new Uint32Array(f32Scratch.buffer);

/**
 * Returns the f32 bit pattern of value (f64 → f32 rounding included).
 * @param {number} value - Any JS number.
 * @returns {number} u32 bit pattern.
 */
function f32BitsOf(value) {
  f32Scratch[0] = value;
  return f32BitsScratch[0];
}

/* --- reporting --- */

/**
 * Appends one line to #test-log.
 * @param {string} cssClass - 'pass' | 'fail' | 'info'.
 * @param {string} text - Line content.
 */
function logLine(cssClass, text) {
  const line = document.createElement('span');
  line.className = cssClass;
  line.textContent = text + '\n';
  document.getElementById('test-log').appendChild(line);
}

/**
 * Returns words as 0x-prefixed hex for failure lines.
 * @param {number[]} words - u32 words.
 * @returns {string} Space-joined hex words.
 */
function wordsToHex(words) {
  return words.map((word) => '0x' + word.toString(16).padStart(8, '0')).join(' ');
}
