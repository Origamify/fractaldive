'use strict';
/* ================================================================
 * Fractal Dive — shared constants and helpers (1st script loaded).
 *
 * - standalone sibling of benchmark/js/common.js (suite lives on
 *   the `benchmark` branch); keeps only what the dive app needs
 * - classic script, one global lexical scope (no modules/exports):
 *   dive.js, app.js, ui.js and the engine files read these
 *   declarations directly
 * - perturb_kernel_wgsl.js splices SAMPLE_COLUMNS/ROWS_PER_PIXEL
 *   into its WGSL at script load, fixed_point.js derives the
 *   sub-step shifts from the same pair — don't rename those two
 *   without updating both; every kernel file loads after this one
 * ================================================================ */

/* --- constants: zoom / iteration config --- */
const MAX_ITER_CAP = 16384;  // iteration ceiling; also sizes the per-frame reference Z table (engine_frame.js)
const BASE_SCALE = 3.0;      // viewport height in the complex plane at dive start — whole set visible
const MIN_S = 1e-35; // dive floor: smallest viewport scale — ~5 decades past the f32-delta regime's external proof (2^-100, ALGORITHM.md §2); reference format still exact here (see fixed_point.js)
const CENTER_FIXED_POINT_BITS = 256; // canonical fractional bits of the BigInt dive center (~77 decimal digits — far past MIN_S, under the engine's 1000-bit frame cap)
const SPEED_MIN = 0.05;      // speed slider 0 — zoom factor per second
const SPEED_MAX = 4;         // speed slider 100 — zoom factor per second
const DEFAULT_SPEED = 1;     // initial zoom rate; the slider parks at the nearest step
const MS_PER_SECOND = 1000;  // ms→s divisor for the per-second zoom rate (dive.js's adaptation, app.js's stepping)
const VIEW_JSON_FRACTION_DIGITS = 78; // decimal fractional digits in the clipboard view JSON — ≥ log10(2^CENTER_FIXED_POINT_BITS), so the center round-trips exactly

/* --- constants: iteration schedule --- */
const SCHEDULE_BASE_ITERATIONS = 128;             // iteration budget at dive start (depth ×1)
const SCHEDULE_ITERATIONS_PER_DEPTH_DECADE = 300; // extra iterations per power of ten of zoom depth

/* --- constants: overrender zoom cache ---
 * The dive zooms into pre-rendered "base frames" instead of
 * re-rendering every scale:
 * - each base covers the window-start view at
 *   BASE_OVERRENDER_FACTOR × canvas resolution — its pixel pitch
 *   is the window-END pitch
 * - zooming in = crop+blit of the base texture until the 2×
 *   factor is consumed, then the next base renders
 * - fixed at 2: one octave per base frame; dive.js owns the
 *   window, engine_frame.js renders the 2× texture and engine.js crop-blits it */
const BASE_OVERRENDER_FACTOR = 2;

/* --- constants: banded rendering (GPU duty-cycle control) ---
 * A deep frame's single fullscreen pass pins the GPU for whole
 * seconds — the compositor starves and Windows' ~2 s TDR resets
 * the driver. The engine instead renders each base frame in
 * paced horizontal bands: short bursts with sleep gaps between,
 * bit-identical pixels (design: js/engine_frame.js's header). */
const GPU_DUTY_CYCLE = 0.9;     // fraction of render time the GPU may stay busy — 0.9 ⇒ a frame takes ~11% longer
const TARGET_BAND_GPU_MS = 100; // target GPU ms per band — drives the adaptive band count (engine_frame.js)

/* --- constants: supersampling lattice ---
 * Anti-aliasing by supersampling: each pixel runs the orbit at
 * SAMPLE_COLUMNS × SAMPLE_ROWS sub-pixel positions and averages
 * the packed channels (cost grows linearly with sample count).
 * - offsets at odd multiples of step/(2·columns) in x, step/(2·rows)
 *   in y (±step/4 per axis at the 2×2 default)
 * - counts must be powers of two: fixed_point.js turns them into
 *   sub-step shifts, perturb_kernel_wgsl.js splices them into the
 *   WGSL at load — keep both loading after this file */
const SAMPLE_COLUMNS_PER_PIXEL = 2; // x sub-positions per pixel
const SAMPLE_ROWS_PER_PIXEL = 2;    // y sub-positions per pixel — 2 × 2 = 4 exact orbits per pixel

/* --- constants: palette ---
 * Escaped-point color scheme:
 * - each RGB channel = cosine wave, same frequency, own phase `ph`
 *   (radians); interior points get the flat `interior` RGB (0..255)
 * - `ph` / `interior` are the engine contract (engine_frame.js reads
 *   palette.ph / palette.interior); kernel in perturb_kernel_wgsl.js
 * - phases reuse the benchmark offsets — colors match the parent
 *   project byte-for-byte */
const PALETTE = { ph: [0.0, 0.6283185307179586, 1.2566370614359172], interior: [4, 7, 14] };

/* --- constants: dive destinations ----------------
 * Fixed targets for the UI dropdown (dive.js). Sources:
 * - Seahorse deep — Zom-B's Wikimedia deep zoom (File:Mandelbrot
 *   sequence new.gif, center verified to 3.18e31); the float64
 *   target was its truncation
 * - Elephant / Triple spiral / Antenna / Scepter — benchmark
 *   branch's PRESETS / BENCH_VIEWS digits, kept as-is
 * - Dendrite c=i — Misiurewicz point at c=i; self-similar all the
 *   way down to MIN_S
 * Plain-decimal strings parsed once into BigInt fixed-point
 * (decimalStringToFixedPoint); float64 never touches the center. */
const DIVE_TARGETS = [
  { name: 'Snail',           x: '-0.743643798682766355449481514223719580885434673763992011569591030896912276471540',
                             y: '0.131825855119617217414972223456819427104345299352863223263572277591819676510188'},
  { name: 'Seahorse',        x: '-0.743643887037157538660883850651430026102966449340658436038986968068285773223319',
                             y: '0.131825904205311228052672782568086462645360384230426064100186040672041146201285'},
  { name: 'Elephant valley', x: '0.286931866425330254898210741789066615248100445947300845115890260353709212450758',
                             y: '0.014286694777853531059016128898981862396288537568968614441476776018056988215821'},
  { name: 'Triple spiral',   x: '-0.088238164088608622528664627214567064830159245703029946213421315863707268734297',
                             y: '0.654418808581611869185702100378007964059777573664164269908366472273145231045121'},
  { name: 'Antenna',         x: '-1.985527943458530891036738081208683502913474728899372139841951266139191169182478',
                             y: '0.000000000000000000422748152628983707839043367286824562930856210816871276737926'},
  { name: 'Scepter crown',   x: '-1.250659962596132298300965834890780792939414593377061793577006718534409081394205',
                             y: '0.020120398201075903960681139738619439004435566151216671089887349727105685850165'},
  { name: 'Dendrite c=i',    x: '0',                                    y: '1' },
];

/* --- helpers --- */

/**
 * Converts a plain decimal string ("-0.088", "0", "1.5") to BigInt
 * fixed-point, rounded to nearest.
 * Why: the exact center representation diveState keeps (float64
 * never touches the dive center); fixed_point.js rescales it
 * to per-frame precision.
 * @param {string} decimalText - Signed plain decimal, no exponent; any digit count.
 * @param {number} fractionBits - Fractional bits of the result (e.g. CENTER_FIXED_POINT_BITS).
 * @returns {bigint} The value scaled by 2^fractionBits.
 */
function decimalStringToFixedPoint(decimalText, fractionBits) {
    const isNegative = decimalText.startsWith('-');
    const unsignedText = isNegative ? decimalText.slice(1) : decimalText;
    const [integerDigits, fractionalDigits = ''] = unsignedText.split('.');
    const numerator = BigInt(integerDigits + fractionalDigits || '0');
    const denominator = 10n ** BigInt(fractionalDigits.length);
    const quotient = (numerator << BigInt(fractionBits + 1)) / denominator;
    const rounded = (quotient + 1n) >> 1n; // extra-bit trick: floor(v·2+1)/2 = round-half-up
    return isNegative ? -rounded : rounded;
}

/**
 * Converts BigInt fixed-point back to plain decimal — the inverse
 * of decimalStringToFixedPoint (serializes the center for the
 * clipboard view JSON).
 * With fractionalDigits ≥ log10(2^fractionBits) the round trip
 * back is exact (VIEW_JSON_FRACTION_DIGITS for the center).
 * @param {bigint} fixedValue - Value scaled by 2^fractionBits (e.g. a dive center coordinate).
 * @param {number} fractionBits - Fractional bits of the input (e.g. CENTER_FIXED_POINT_BITS).
 * @param {number} fractionalDigits - Decimal digits after the point in the result.
 * @returns {string} Signed plain decimal, no exponent part.
 */
function fixedPointToDecimalString(fixedValue, fractionBits, fractionalDigits) {
    const isNegative = fixedValue < 0n;
    const scaled = (isNegative ? -fixedValue : fixedValue) * 10n ** BigInt(fractionalDigits);
    const quotient = (scaled << 1n) / (1n << BigInt(fractionBits));
    const rounded = (quotient + 1n) >> 1n; // round-half-up, mirrors decimalStringToFixedPoint
    const digitsText = rounded.toString().padStart(fractionalDigits + 1, '0');
    const body = fractionalDigits
        ? `${digitsText.slice(0, -fractionalDigits)}.${digitsText.slice(-fractionalDigits)}`
        : digitsText;
    return isNegative ? `-${body}` : body;
}

/**
 * Returns the DOM element with the given id — the lookup alias used
 * by ui.js and dive.js wiring code.
 * @param {string} elementId - Value of the element's id attribute.
 * @returns {HTMLElement|null} The matching element, or null if absent (per the DOM API).
 */
const getElementById = (elementId) => document.getElementById(elementId);

/**
 * Clamps a value into [lowerBound, upperBound] after truncating
 * toward zero — guards iteration counts before the GPU buffers.
 * @param {number} value - Candidate number; coerced to int via |0.
 * @param {number} lowerBound - Minimum allowed value.
 * @param {number} upperBound - Maximum allowed value.
 * @returns {number} Clamped integer.
 */
function clampI(value, lowerBound, upperBound) {
  return Math.max(lowerBound, Math.min(upperBound, value | 0));
}

/**
 * Returns the iteration budget for a viewport scale.
 * Why: detail must grow with zoom depth — budget tracks
 * depth = log10(BASE_SCALE / scale), heuristic inherited from the
 * benchmark suite so behavior matches the parent.
 * @param {number} viewportScale - Complex units per pixel of the current view.
 * @returns {number} base + growth·log10(depth), clamped to [SCHEDULE_BASE_ITERATIONS, MAX_ITER_CAP].
 */
function iterationCountForScale(viewportScale) {
  const depthDecades = Math.max(0, Math.log10(BASE_SCALE / viewportScale));
  const budget = SCHEDULE_BASE_ITERATIONS + SCHEDULE_ITERATIONS_PER_DEPTH_DECADE * depthDecades;
  return clampI(Math.round(budget), SCHEDULE_BASE_ITERATIONS, MAX_ITER_CAP);
}

/**
 * Formats a number in forced-exponent form ("3.0e-7") for the HUD
 * depth readout.
 * @param {number|null} value - Number to format; null/NaN/±Infinity render as "—".
 * @returns {string} One-decimal scientific notation, or "—" for non-finite input.
 */
function formatEngineeringNotation(value) {
  if (value == null || !isFinite(value)) return '—';
  const exponent = Math.floor(Math.log10(Math.abs(value)));
  return (value / 10 ** exponent).toFixed(1) + 'e' + exponent;
}
