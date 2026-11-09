'use strict';
/* ================================================================
 * Limb array math — the WGSL u32-limb primitives as a template
 * string, spliced ahead of the reference compute source in
 * reference_orbit_wgsl.js (WGSL requires declaration before use;
 * order below is simplest → most complex with every use after
 * its declaration).
 *
 * Pure u32 wrap arithmetic (WGSL has no u64 ALU):
 * - wide products split into 16-bit halves
 * - carries wrap-detected (sum < operand)
 * - mulWord32's high word stays below 2^32 by construction
 *   (bound proof in its comment)
 * __LIMB_COUNT__ is the tier splice point — engine.js substitutes
 * the frame's tier before compiling (WGSL array lengths are
 * compile-time constants). limb_array_int / limb_array_product
 * alias the two limb widths: callers spell the value type, not
 * the raw array type.
 * ================================================================ */

/* --- limb math source (WGSL, not JavaScript) --- */
const LIMB_ARRAY_MATH_WGSL = `
// --- limb-array math library (u32 wrap arithmetic only) ---

// --- constants: limb widths (tier template) ---
const LIMB_COUNT: u32 = __LIMB_COUNT__u;           // splice point — one variant per precision tier
const PRODUCT_LIMB_COUNT: u32 = 2u * LIMB_COUNT;   // fixed-point products double the limb width
const LOW_HALF_MASK: u32 = 0xFFFFu; // low 16 bits — the half-word split keeps every partial product under 2^32

// --- value types (aliases, not struct wrappers — structs would force .limbs indexing at every site) ---
alias limb_array_int = array<u32, LIMB_COUNT>;               // one fixed-point value at frame width
alias limb_array_product = array<u32, PRODUCT_LIMB_COUNT>;   // products/sums double the limb width

// --- helpers: simplest → most complex, every use after its declaration ---

// a < 0
// Returns true when the top limb's sign bit is set (two's complement).
fn limbIsNegative(limbs: limb_array_int) -> bool {
  return (limbs[LIMB_COUNT - 1u] & 0x80000000u) != 0u;
}

// -a
// Returns the two's-complement negation (bitwise NOT + carried 1).
// - carry propagates only past a 0xFFFFFFFF limb (wraps from +1)
// - a zero limb with carry 0 must NOT restart the chain
fn limbNegate(limbs: limb_array_int) -> limb_array_int {
  var resultLimbs: limb_array_int;
  var carry = 1u; // the +1 of two's complement
  for (var limbIndex = 0u; limbIndex < LIMB_COUNT; limbIndex++) {
    let sum = ~limbs[limbIndex] + carry; // wraps only from 0xFFFFFFFF + 1
    resultLimbs[limbIndex] = sum;
    carry = select(0u, 1u, sum == 0u && carry == 1u);
  }
  return resultLimbs;
}

// a + b
// Returns left + right; carry = wrap-detect (sum < operand).
// - a limb can wrap twice → the carry itself reaches 2, still u32-exact
fn limbAdd(leftLimbs: limb_array_int, rightLimbs: limb_array_int) -> limb_array_int {
  var sumLimbs: limb_array_int;
  var carry = 0u;
  for (var limbIndex = 0u; limbIndex < LIMB_COUNT; limbIndex++) {
    let partialSum = leftLimbs[limbIndex] + rightLimbs[limbIndex];
    let carryInto = select(0u, 1u, partialSum < leftLimbs[limbIndex]);
    let totalSum = partialSum + carry;
    let carryFrom = select(0u, 1u, totalSum < partialSum);
    sumLimbs[limbIndex] = totalSum;
    carry = carryInto + carryFrom;
  }
  return sumLimbs; // final carry dropped — widths carry integer headroom (fixed_point.js)
}

// a - b
// Returns left − right; borrows mirror limbAdd's carries.
fn limbSub(leftLimbs: limb_array_int, rightLimbs: limb_array_int) -> limb_array_int {
  var differenceLimbs: limb_array_int;
  var borrow = 0u;
  for (var limbIndex = 0u; limbIndex < LIMB_COUNT; limbIndex++) {
    let partialDifference = leftLimbs[limbIndex] - rightLimbs[limbIndex];
    let borrowInto = select(0u, 1u, leftLimbs[limbIndex] < rightLimbs[limbIndex]);
    let totalDifference = partialDifference - borrow;
    let borrowFrom = select(0u, 1u, partialDifference < borrow);
    differenceLimbs[limbIndex] = totalDifference;
    borrow = borrowInto + borrowFrom;
  }
  return differenceLimbs;
}

// a + b (at product width)
// Same add as limbAdd at product width (2N limbs).
fn limbAddWide(leftLimbs: limb_array_product, rightLimbs: limb_array_product) -> limb_array_product {
  var sumLimbs: limb_array_product;
  var carry = 0u;
  for (var limbIndex = 0u; limbIndex < PRODUCT_LIMB_COUNT; limbIndex++) {
    let partialSum = leftLimbs[limbIndex] + rightLimbs[limbIndex];
    let carryInto = select(0u, 1u, partialSum < leftLimbs[limbIndex]);
    let totalSum = partialSum + carry;
    let carryFrom = select(0u, 1u, totalSum < partialSum);
    sumLimbs[limbIndex] = totalSum;
    carry = carryInto + carryFrom;
  }
  return sumLimbs;
}

// a - b (at product width)
// Same sub as limbSub at product width (2N limbs).
fn limbSubWide(leftLimbs: limb_array_product, rightLimbs: limb_array_product) -> limb_array_product {
  var differenceLimbs: limb_array_product;
  var borrow = 0u;
  for (var limbIndex = 0u; limbIndex < PRODUCT_LIMB_COUNT; limbIndex++) {
    let partialDifference = leftLimbs[limbIndex] - rightLimbs[limbIndex];
    let borrowInto = select(0u, 1u, leftLimbs[limbIndex] < rightLimbs[limbIndex]);
    let totalDifference = partialDifference - borrow;
    let borrowFrom = select(0u, 1u, partialDifference < borrow);
    differenceLimbs[limbIndex] = totalDifference;
    borrow = borrowInto + borrowFrom;
  }
  return differenceLimbs;
}

// a > b (unsigned, at product width)
// Returns left > right as unsigned.
// - the escape check compares non-negative magnitudes only — no signed compare
fn limbUnsignedGreaterWide(leftLimbs: limb_array_product, rightLimbs: limb_array_product) -> bool {
  for (var limbIndex = PRODUCT_LIMB_COUNT; limbIndex > 0u; limbIndex--) {
    if (leftLimbs[limbIndex - 1u] != rightLimbs[limbIndex - 1u]) {
      return leftLimbs[limbIndex - 1u] > rightLimbs[limbIndex - 1u];
    }
  }
  return false;
}

// low_N(a)
// Returns the low N limbs of a product — valid truncation because
// values fit N limbs by integer-headroom sizing (fixed_point.js).
fn limbTruncateLow(limbs: limb_array_product) -> limb_array_int {
  var truncatedLimbs: limb_array_int;
  for (var limbIndex = 0u; limbIndex < LIMB_COUNT; limbIndex++) {
    truncatedLimbs[limbIndex] = limbs[limbIndex];
  }
  return truncatedLimbs;
}

// a >> n (arithmetic, at product width)
// Returns limbs >> bitCount with two's-complement sign fill, at
// product width (2·fractionBits → fractionBits scale).
// - bitShift 0 guarded: u << 32 shifts by 0 in WGSL (count wraps mod 32)
fn limbShiftRightArithmeticWide(limbs: limb_array_product, bitCount: u32) -> limb_array_product {
  let signFill = select(0u, 0xFFFFFFFFu, (limbs[PRODUCT_LIMB_COUNT - 1u] & 0x80000000u) != 0u);
  let limbShift = bitCount / 32u;
  let bitShift = bitCount % 32u;
  var shiftedLimbs: limb_array_product;
  for (var limbIndex = 0u; limbIndex < PRODUCT_LIMB_COUNT; limbIndex++) {
    let sourceIndex = limbIndex + limbShift;
    let sourceWord = select(limbs[sourceIndex], signFill, sourceIndex >= PRODUCT_LIMB_COUNT);
    let nextWord = select(limbs[sourceIndex + 1u], signFill, sourceIndex + 1u >= PRODUCT_LIMB_COUNT);
    var shiftedWord = sourceWord >> bitShift;
    if (bitShift != 0u) {
      shiftedWord = shiftedWord | (nextWord << (32u - bitShift));
    }
    shiftedLimbs[limbIndex] = shiftedWord;
  }
  return shiftedLimbs;
}

struct MulWordResult {
  lo: u32,  // low word of the product
  hi: u32,  // high word — < 2^32 by construction (bound proven in mulWord32)
};

// a × b (u32 × u32 -> u64)
// Returns both words of a u32×u32 product.
// - hi < 2^32 by construction: crossMid wraps at 2^32 → crossMid>>16 < 2^16
// - and (2^16−1)² + (2^16−1) + 2^16 + 1 = 2^32 − 1 — hi fits u32 exactly
fn mulWord32(leftWord: u32, rightWord: u32) -> MulWordResult {
  let leftLow = leftWord & LOW_HALF_MASK;
  let leftHigh = leftWord >> 16u;
  let rightLow = rightWord & LOW_HALF_MASK;
  let rightHigh = rightWord >> 16u;
  let crossLow = leftLow * rightHigh;    // < 2^32
  let crossHigh = leftHigh * rightLow;   // < 2^32
  let crossMid = crossLow + crossHigh;   // < 2^33 — wraps once; the lost bit is carryMid
  let carryMid = select(0u, 1u, crossMid < crossLow);
  let lowLow = leftLow * rightLow;       // < 2^32
  let lo = lowLow + (crossMid << 16u);   // wraps once at most
  let carryLo = select(0u, 1u, lo < lowLow);
  let hi = leftHigh * rightHigh + (crossMid >> 16u) + (carryMid << 16u) + carryLo;
  return MulWordResult(lo, hi);
}

// a[i] += b + carry
// Adds word (+ carry) into productLimbs[index]; returns the outgoing
// carry (0..2) — the wrap-detected accumulation chain of every
// schoolbook multiply/square row step.
fn limbAccumulateWord(productLimbs: ptr<function, limb_array_product>, index: u32, word: u32, carry: u32) -> u32 {
  let partialSum = (*productLimbs)[index] + word;
  let carryInto = select(0u, 1u, partialSum < word);
  let totalSum = partialSum + carry;
  let carryFrom = select(0u, 1u, totalSum < partialSum);
  (*productLimbs)[index] = totalSum;
  return carryInto + carryFrom;
}

// a × k (u32 scalar)
// Returns limbs × scalar — the per-pixel c = planeLeft + x·step assembly.
// - fits N limbs: x < canvas width < 2^32
fn limbMultiplySmall(limbs: limb_array_int, scalar: u32) -> limb_array_int {
  var resultLimbs: limb_array_int;
  var carry = 0u;
  for (var limbIndex = 0u; limbIndex < LIMB_COUNT; limbIndex++) {
    let mulWord = mulWord32(limbs[limbIndex], scalar);
    let partialSum = mulWord.lo + carry;
    let carryInto = select(0u, 1u, partialSum < mulWord.lo);
    resultLimbs[limbIndex] = partialSum;
    carry = mulWord.hi + carryInto; // < 2^32, consumed by the next limb
  }
  return resultLimbs;
}

// a × b
// Returns left × right as a 2N-limb product (sign-magnitude schoolbook).
// - negate negatives, multiply magnitudes, negate back on sign mismatch
// - carry: each step's mulWord hi consumed immediately → running
//   carry stays < 2^32 (hi_max + 2 wraps at most once; select catches it)
fn limbMultiplySigned(leftLimbs: limb_array_int, rightLimbs: limb_array_int) -> limb_array_product {
  let leftNegative = limbIsNegative(leftLimbs);
  let rightNegative = limbIsNegative(rightLimbs);
  var leftMagnitude = leftLimbs; // select() rejects arrays — branch instead
  if (leftNegative) { leftMagnitude = limbNegate(leftLimbs); }
  var rightMagnitude = rightLimbs;
  if (rightNegative) { rightMagnitude = limbNegate(rightLimbs); }
  var productLimbs: limb_array_product = limb_array_product();
  for (var leftIndex = 0u; leftIndex < LIMB_COUNT; leftIndex++) {
    var carry = 0u;
    for (var rightIndex = 0u; rightIndex < LIMB_COUNT; rightIndex++) {
      let mulWord = mulWord32(leftMagnitude[leftIndex], rightMagnitude[rightIndex]);
      carry = mulWord.hi + limbAccumulateWord(&productLimbs, leftIndex + rightIndex, mulWord.lo, carry); // < 2^32 by the mulWord32 bound
    }
    // row tail: fold the final carry in, propagating while limbs wrap
    var tailIndex = leftIndex + LIMB_COUNT;
    var tailCarry = carry;
    loop {
      if (tailCarry == 0u) { break; }
      let tailWord = productLimbs[tailIndex];
      let tailSum = tailWord + tailCarry;
      productLimbs[tailIndex] = tailSum;
      tailCarry = select(0u, 1u, tailSum < tailWord); // wrap-detect: valid for any carry < 2^32, not just 0/1
      tailIndex++;
    }
  }
  if (leftNegative != rightNegative) { // two's-complement negate at 2N width
    var negationCarry = 1u;
    for (var limbIndex = 0u; limbIndex < PRODUCT_LIMB_COUNT; limbIndex++) {
      let negationSum = ~productLimbs[limbIndex] + negationCarry;
      productLimbs[limbIndex] = negationSum;
      negationCarry = select(0u, 1u, negationSum == 0u && negationCarry == 1u);
    }
  }
  return productLimbs;
}

// a²
// Returns limbs × limbs as a 2N-limb product — the squaring
// specialization of limbMultiplySigned (~half the mulWord32
// calls, no final negation: a square is non-negative).
// - identity: a² = 2·Σ(a[i]·a[j], i<j) + Σ(a[i]²)
// - each cross/diagonal term folded ONCE (carry-safe single fold:
//   out = hi + ≤2 wraps < 2³²)
// - doubling happens on the summed cross total as one <<1 pass,
//   never per-term — 2·hi could wrap u32 and silently drop a
//   carry into the next-next limb
fn limbSquare(limbs: limb_array_int) -> limb_array_product {
  let negative = limbIsNegative(limbs);
  var magnitude = limbs; // select() rejects arrays — branch instead
  if (negative) { magnitude = limbNegate(limbs); }
  var productLimbs: limb_array_product = limb_array_product();
  // pass 1: cross pairs, each folded once (row/tail exactly like the signed multiply)
  for (var leftIndex = 0u; leftIndex < LIMB_COUNT; leftIndex++) {
    var carry = 0u;
    for (var rightIndex = leftIndex + 1u; rightIndex < LIMB_COUNT; rightIndex++) {
      let mulWord = mulWord32(magnitude[leftIndex], magnitude[rightIndex]);
      carry = mulWord.hi + limbAccumulateWord(&productLimbs, leftIndex + rightIndex, mulWord.lo, carry);
    }
    var tailIndex = leftIndex + LIMB_COUNT;
    var tailCarry = carry;
    loop {
      if (tailCarry == 0u) { break; }
      let tailWord = productLimbs[tailIndex];
      let tailSum = tailWord + tailCarry;
      productLimbs[tailIndex] = tailSum;
      tailCarry = select(0u, 1u, tailSum < tailWord); // wrap-detect: valid for any carry < 2^32, not just 0/1
      tailIndex++;
    }
  }
  // pass 2: double the cross sum in place — 2·Σcross
  var doubleCarry = 0u;
  for (var limbIndex = 0u; limbIndex < PRODUCT_LIMB_COUNT; limbIndex++) {
    let doubledWord = (productLimbs[limbIndex] << 1u) | doubleCarry;
    doubleCarry = productLimbs[limbIndex] >> 31u;
    productLimbs[limbIndex] = doubledWord;
  }
  // pass 3: diagonals, each folded once at position 2i, carry propagating up
  for (var leftIndex = 0u; leftIndex < LIMB_COUNT; leftIndex++) {
    let diagonalWord = mulWord32(magnitude[leftIndex], magnitude[leftIndex]);
    let diagonalCarry = diagonalWord.hi + limbAccumulateWord(&productLimbs, 2u * leftIndex, diagonalWord.lo, 0u);
    var tailIndex = 2u * leftIndex + 1u;
    var tailCarry = diagonalCarry;
    loop {
      if (tailCarry == 0u) { break; }
      let tailWord = productLimbs[tailIndex];
      let tailSum = tailWord + tailCarry;
      productLimbs[tailIndex] = tailSum;
      tailCarry = select(0u, 1u, tailSum < tailWord); // wrap-detect: valid for any carry < 2^32, not just 0/1
      tailIndex++;
    }
  }
  return productLimbs; // never negated — a square is non-negative
}
`;
