# FractalDive — the algorithm

FractalDive dives into the Mandelbrot set down to a viewport
height of 1e-35. It runs on WebGPU, on Firefox, without float64
in shaders and without optional GPU features.

The app renders through **perturbation with plain f32 deltas**:

- **One reference orbit per frame**, computed exactly in
  arbitrary-precision integer arithmetic on the GPU (u32 limbs).
- **One cheap f32 delta orbit per pixel**, replaying the
  reference as a table.
- Frames are correct **within a stated tolerance**, not
  bit-exact. The tolerance is checked against the exact engine
  in `tests/`, which serves as the oracle.

This one doc covers everything: the problem, the design
rationale, the math, the number formats, the kernels, the cost
model, and the honesty ledger.

Contents: [problem](#1-the-problem-floats-run-out-of-bits) ·
[why f32](#2-why-f32-perturbation) ·
[pipeline](#3-pipeline-overview) · [zoom law](#4-the-zoom-law) ·
[precision](#5-precision-how-many-bits-where) ·
[frame math](#6-fixed-point-frame-math-cpu) ·
[limb numbers](#7-limb-numbers) · [limb math](#8-limb-arithmetic-in-wgsl) ·
[reference orbit](#9-the-reference-orbit-compute-pass) ·
[delta math](#10-the-delta-iteration) ·
[rebase](#11-rebase-when-a-pixel-stops-tracking) ·
[delta kernel](#12-the-delta-kernel-render-pass) ·
[coloring](#13-coloring) · [delivery](#14-frame-delivery) ·
[schedule](#15-the-iteration-schedule) ·
[exact engine](#16-the-exact-engine-the-test-oracle) ·
[performance](#17-performance-how-the-engines-compare) ·
[not built](#18-not-built-series-approximation-and-other-formats) ·
[ledger](#19-what-is-exact-and-what-isnt)

---

## 1. The problem: floats run out of bits

The Mandelbrot iteration is z ← z² + c, starting at z = 0. It is
chaotic: a tiny error in `c` grows exponentially as you iterate.

Float64 gives `c` about 16 significant decimal digits (52
fraction bits). Past a zoom of roughly 1e-13, **adjacent pixels
differ in `c` by less than one float64 ulp** — the smallest step
a double can represent. Plain doubles then compute the same `c`
for many pixels. The image pixelates and the zoom "freezes".
Every deep zoomer hits this wall.

Two roads past it:

- **Perturbation** (Kalles Fraktaler, Fractal Zoomer, most deep
  zoomers): iterate *one* orbit in high precision — the
  **reference** — and per pixel iterate only the tiny difference
  from it — the **delta** — in low precision. Fast, but needs
  rebasing and glitch handling.
- **Exact per-pixel**: run every pixel's full orbit in
  arbitrary-precision integer arithmetic. Slower per pixel, but
  nothing can go subtly wrong.

The shipped engine is perturbation with f32 deltas (§2, §12). The
exact engine (§16) is kept in `tests/oracle/` as the tolerance
oracle. Both are documented here.

The dive floor is `MIN_S = 1e-35` (viewport height in complex
units, `js/common.js`). That is ~5 decades past where the f32
regime is externally *proven* (2⁻¹⁰⁰) — the stretch rides on
the tolerance contract (§2), not on an external proof. The math
stops far later: the limb format would keep going to ~1e-300
(`FIXED_POINT_MAX_BITS = 1000`).

## 2. Why f32 perturbation

One constraint shapes everything: **it must run on Firefox, and
it must not use the `shader-f64` WebGPU feature.** f64 in WGSL
is gated behind `shader-f64`, which wgpu (Firefox's WebGPU
foundation) marks *native-only* — no shipping browser exposes it
to web content. So every shader-side number must be built from
core WGSL types (f32, i32, u32), and the engine requests no
optional features at all — not even `shader-f16`.

Three designs fit that constraint, and the f32 engine wins on cost:

1. **The exact engine** — per-pixel limb orbits, 2–4 limbs
   depending on depth (§16). Correct by construction, but its
   per-iteration cost grows quadratically with depth — too slow
   for real-time diving.
2. **Exact-delta perturbation** — 6-limb fixed-point δ/ε (LSB
   2⁻¹⁴⁸), bit-exact frames. Per iteration it pays 150
   word-multiplies at a *constant* 6-limb width — 5 full-width
   products (2 escape-check squares the update cannot reuse, plus
   3 signed multiplies) — and reads 12 table words per iteration,
   with a single-invocation reference pass serialized ahead of
   every frame. Net: **5–17× the exact engine's per-iteration
   cost at every depth in the dive range**. The depth-invariant
   cost only breaks even near 1e-41 — below the floor, i.e. never
   in range.
3. **df32 (double-float) deltas** — the classic ~48-bit fix,
   structurally blocked on this stack: `fma` is fused and
   bit-exact, but every two-sum error-extraction formulation is
   compiler-rewritten to zero on Firefox/wgpu/NVIDIA — a
   spec-nonconforming fast-math transform that plain,
   select-guarded, storage-write→read and atomic formulations
   all produce identically. df32 needs exact two-sums; WGSL offers no
   precision controls. Road closed.

So the shipped engine **takes the f32 penalty** — only 24 useful
bits per stored ε, budget math in §10 — and contains it with
three levers:

- **A df32 reference table** — the exact limb orbit split into
  hi/lo f32 pairs. Table error enters scaled by ε: ~2⁻⁴⁵·|ε| per
  step. (The table is df32; only the per-pixel ε is plain f32.)
- **Zhuoran rebase as a precision reset** (§11) — re-anchor ε at
  fresh f32 precision whenever it grows.
- **A floor at 1e-35** — ~5 decades past the depth at which a
  shipping site (zoomingfractal.com, WebGL2) runs exactly this
  stack — f32 deltas + df32 table + rebase — before widening its
  delta format to floatexp. The external existence proof stops
  at 2⁻¹⁰⁰, and it was a proof of shippability, not of this
  stack; both gaps are what the tolerance contract covers.

**The tolerance contract.** Frames match the exact oracle within
a stated tolerance instead of bit-identically: iteration counts
equal modulo a bounded fraction of boundary pixels — a flipped
boundary pixel lands elsewhere on the palette, any channel
delta. Probe-verified on Firefox: the df32 table is bit-exact
vs the BigInt oracle (5/5); frames match the exact
oracle kernel pixel-identically on mid/deep/floor views, with one
boundary flip on the shallow view and sub-band single-sample
drift on most of the rebase view (the DE-flat adaptive SSAA,
§12) — 5/5 within tolerance.

Also decided, and therefore out of scope: bit-exactness,
float64 shortcuts, series approximation/BLA, multi-reference
management, floatexp deltas (the floor caps depth), and CPU
fallback — no WebGPU means the failure notice.

## 3. Pipeline overview

One frame = a paced loop of band submits (GPU duty-cycle
control), two pass kinds per first band
(`js/engine.js`, class `WebGPUPerturbationEngine`):

```
rAF tick (app.js onAnimationFrame)
  └─ stepZoom: s ← s·e^(−rate·Δt)                  zoom law, §4
  └─ setZoom(windowStart/s)                        presenter.js: the CSS zoom crop — no GPU work
  └─ view = { centerXFixed, centerYFixed,          exact BigInt center (256 frac bits),
              centerFractionBits: 256, s }         + float64 scale s
  └─ engine.render(view, maxIter)                  js/engine.js
       ├─ buildFixedPointConfig(view)              §6: exact reference center + f32 delta constants
       │                                           (js/fixed_point.js)
       ├─ pack + upload the input buffers
       ├─ COMPUTE pass (cached — skipped while the     §9: exact Z table for the frame center
       │  reference cache holds; see §9)
       │    └─ js/kernels/reference_orbit_wgsl.js over js/kernels/limb_array_math.js (§7–§8),
       │       rows stored as df32 pairs, GPU-side only
       └─ RENDER pass, one horizontal band           §12: per-pixel f32 delta orbits
          at a time (scissored fullscreen draw,
          submit + settle + duty-cycle sleep per band)
            └─ js/kernels/perturb_kernel_wgsl.js (no limb lib) + adaptive 2×2 supersampling
               (a distance estimator on the first lattice sample picks 1 or 4 orbits per pixel, §12),
               into the 2× offscreen base texture — bands combine
               bit-identically to one full pass
```

The Z table never returns to the CPU. The render pass reads the
storage buffer the compute pass just wrote — WebGPU inserts the
barrier between passes of one command buffer (the compute pass
rides the first band's buffer).

The CPU never touches a per-pixel orbit. It sizes the precision,
packs the reference center (limbs) and five f32 constants, and
submits once per band.

## 4. The zoom law

State (`dive.js`): the center as BigInt fixed-point (fixed for
the whole dive) and `viewportScale` s — **complex units per
pixel**.

Each tick advances the zoom by elapsed wall time:

```
s ← s · e^(−r·Δt)          r = zoom rate per second (slider ×0.05–×4)
```

Because the steps *multiply*, the zoom after any time T is
exactly e^(−rT) no matter how many frames the time was chopped
into — frame-rate independent, no accumulation drift. Pause
simply stops stepping (and zeroes Δt on resume so no jump
accrues). Depth for the HUD is `BASE_SCALE / (s·h)`; s clamps at
`MIN_S = 1e-35` where the dive parks.

The iteration budget grows with depth — §15.

## 5. Precision: how many bits, where

Two fixed-point widths coexist (`js/fixed_point.js`):

**The frame precision** — bits that grow with zoom depth. It
quantizes the pixel grid and feeds the exact reference center:

```
fractionBits = clamp( ceil(log2(1/s)) + 8, 16, 1000 )
```

One bit per power of two of zoom, plus 8 headroom bits
(`FIXED_POINT_HEADROOM_BITS`) so pixel centers stay distinct.

| scale s | bits(1/s) | fractionBits |
|---|---|---|
| 3e-3 (start) | ~8.4 | 16 (floor) |
| 1e-10 | ~33.2 | 42 |
| 2⁻¹⁰⁰ (external proof) | 100 | 108 |
| 1e-35 (floor) | ~117 | 125 |
| 1e-280 | ~931 | 939 |

**The reference limb format** — constant at every depth. 6 u32
limbs at LSB 2⁻¹⁴⁸ (`REFERENCE_FRACTION_BITS = 148`,
`REFERENCE_LIMB_COUNT = 6`). Sizing, bottom up:

- LSB 2⁻¹⁴⁸: at the dive floor the frame grid needs ~125
  fractional bits, so the limb format keeps 23 bits *below* the
  grid — the reference orbit is exact with wide margin at any
  depth the app dives.
- Integer range ±2⁴³ (`REFERENCE_INTEGER_HEADROOM_BITS = 44`):
  every stored row keeps |Z| ≤ 6.3 (the stop rule of §9), so
  this is safety margin far beyond anything the views produce.
- 148 + 44 = 192 bits = 6 u32 limbs exactly.

The frame precision never exceeds the limb format —
`buildFixedPointConfig` throws past it (unreachable above
MIN_S) — so the shift from frame width to limb width is always
an exact left shift.

WGSL array lengths must be compile-time constants, so the limb
count is spliced in as a constant (`__LIMB_COUNT__ = 6`) and the
reference pipeline compiles once per session. (The exact oracle's
kernel instead compiles one shader variant per depth tier, §16.)

## 6. Fixed-point frame math (CPU)

`buildFixedPointConfig` (`js/fixed_point.js`) computes, in
BigInt where it matters, the constants one frame needs. A
fixed-point value with f fraction bits means the stored integer
is value·2^f. The output has two halves.

**Reference half (exact):**

1. **Rescale the center.** The dive center is stored once at 256
   fraction bits (parsed from the target's decimal string,
   rounded once at parse — float64 never touches the center).
   Each frame shifts it to the frame precision: left shift when
   deepening (exact growth — bits are appended), right shift
   when shallower (drops only bits this frame cannot represent).
2. **Shift to limb width.** Always an exact left shift (§5).
3. **Stop threshold.** `REFERENCE_STOP_MAGNITUDE_SQUARED = 4`,
   stored at product scale: `4 << 2·148`.

**Delta half (f32, tolerance-rounded):**

4. **Pixel grid.** One pixel step is
   `pixelStep = max(1, round(s·2^f))` fixed-point units (≥ 1 so
   adjacent pixels can never collapse into the same value). The
   plane edges' offsets from the center are
   `round((w/2 − 0.5)·s·2^f)` and `round((h/2 − 0.5)·s·2^f)`.
   The supersampling sub-steps are exact shifts of the step:
   `subStepX = step >> (log2(columns)+1)`, same for y — for the
   2×2 default both are step >> 2, and sample offsets are odd
   multiples of them (±step/4 on both axes).
5. **Round to f32 at δ's own scale.** The reference sits at the
   frame center, so every δ term is step-scaled. Assembly error
   is ~2⁻²⁴·step — far below the step/4 supersampling lattice at
   every depth the app dives.

Packing: the reference inputs — centerReal (N limbs),
centerImag (N), stop threshold (2N) — go to one u32 buffer; the
five delta constants go as five f32 words; a small uniform
carries fractionBits, the iteration budget, the base texture
canvas dimensions, three
palette phases and the packed interior color (8 words, one
layout shared by both kernels).

The BigInt pixel-grid machinery of the exact engine survives
only where it is still load-bearing: the reference center C.

## 7. Limb numbers

A value is an `array<u32, N>` — **little-endian two's-complement
limbs**: limb 0 holds bits 0..31, limb 1 bits 32..63, …; the
sign lives in the top bit of limb N−1. This is exactly how
BigInt is packed on the CPU side (`bigIntToLimbs` wraps mod
2^(32·N) — two's-complement truncation).

Products use `array<u32, 2N>` (`limb_array_product`) — full
schoolbook width, so nothing is lost between multiply and
renormalize.

(The integer-headroom question — how many bits above the point —
has a different answer per user of the limb format: §5 for the
reference pass, §16 for the oracle's exact kernel.)

## 8. Limb arithmetic in WGSL

All primitives live in `js/kernels/limb_array_math.js` — a WGSL
template string spliced ahead of the kernel (WGSL needs
declaration before use). WGSL has no u64 ALU and no overflow
flags, so every carry is **detected by wraparound**: in u32, if
`a + b < a`, the addition wrapped and the carry is 1.

- **`mulWord32(a, b) → (lo, hi)`** — a full 32×32→64 product
  from 16-bit halves (all intermediates fit u32):

  ```
  crossMid = aLo·bHi + aHi·bLo          // < 2³³, wraps once; lost bit = carryMid
  lo       = aLo·bLo + (crossMid << 16) // wraps once; lost bit = carryLo
  hi       = aHi·bHi + (crossMid >> 16) + (carryMid << 16) + carryLo
  ```

  `hi < 2³²` holds by construction. No edge case.

- **Add / subtract** (`limbAdd`, `limbSub`, `…Wide` 2N variants)
  — schoolbook carry chains. A limb can wrap twice in one step,
  so the outgoing carry can be 2 — still u32-exact. Final carry
  out the top is dropped: the width carries headroom by design.

- **`limbMultiplySigned(a, b) → 2N-limb product`** —
  sign-magnitude schoolbook: negate negative operands, multiply
  magnitudes row by row (each row's `mulWord32` hi is consumed
  immediately, keeping the running carry < 2³²), propagate the
  row tail carry, negate back if the signs differed.

- **`limbSquare(a) → 2N-limb product`** — the squaring
  specialization via a² = 2·Σ(a[i]·a[j], i<j) + Σ(a[i]²): cross
  pairs fold once, the summed cross total doubles in one
  shift-left-1 pass, then the diagonals fold at position 2i —
  ~half the `mulWord32` calls of a full multiply, and never a
  final negation (a square is non-negative). Doubling the total
  rather than each term keeps every fold single-width: 2·hi of
  one product can wrap u32 and would silently drop a carry.

- **`limbMultiplySmall(a, scalar)`** — limb vector times one u32.

- **`limbShiftRightArithmeticWide(p, k)`** — arithmetic right
  shift with sign fill at product width; the fixed-point
  renormalizer, dropping a 2f-scale product back to f scale. The
  bitShift==0 case is guarded because WGSL, like JS, shifts by
  count mod 32 (`u << 32` would shift by 0).

- **`limbTruncateLow(p)`** — low N limbs of a product. Loss-free
  because headroom sizing guarantees values fit N limbs.

- **`limbUnsignedGreaterWide(a, b)`** — unsigned compare from
  the top limb down. Comparing the two's-complement encodings as
  unsigned is correct for non-negative magnitudes.

Every primitive is pinned by parity tests against a JS BigInt
oracle (`tests/limb_array_math_test`, tiers 1..8).

## 9. The reference orbit (compute pass)

`js/kernels/reference_orbit_wgsl.js` (over `js/kernels/limb_array_math.js`) —
one invocation per table build; `engine_frame.js` dispatches 1 workgroup of
1. It iterates, in exact u32-limb arithmetic:

```
Z₀ = 0        Zₙ₊₁ = Zₙ² + C          (exact, once per frame)
```

C is the frame center. The step matches the oracle kernel's z ←
z² + c exactly (escape check before update,
three-squares cross identity, arithmetic shift by the fraction
bits) — one orbit instead of ~10⁶ (§17: amortized to nothing).
The expensive precision is paid once per frame, never per pixel.

**Table format.** The table is an `array<vec4u>`: element 0 is
a header carrying the row count in `.x` (padded to the 16-byte
element stride the binding requires), then one vec4u per row —
a **df32 pair** `(re_hi, re_lo, im_hi, im_lo)`, f32 bit patterns,
~48 significant bits — made by splitting the exact limb value
into top bits + residual inside the compute pass. One row = one
16-byte vector load in the delta loop.

**What gets stored.** Rows 0 and 1 (Z₀ = 0 and Z₁ = C) always —
the render kernel's rebasing needs both (§11) — and the first
row with |Z|² > 4 (the cap-crosser). The table ends there or at
the budget, whichever comes first. The table lives entirely on
the GPU (binding 2); nothing returns to the CPU.

**The table cache.** The orbit is a pure function of (C, build
budget, stop threshold) — the viewport scale never enters it —
so the engine caches the on-GPU table across frames and rebuilds
only when the center moves (view jump, click-to-zoom) or the
schedule's budget outgrows the build's headroom
(`REFERENCE_TABLE_BUDGET_HEADROOM` in `js/engine_frame.js`). A zooming
dive (fixed center, shrinking s) therefore skips the serial
single-invocation compute pass on almost every frame. Frames
stay bit-identical to a fresh build at the frame budget:
`referenceIndex ≤ iterationsUsed` always (rebase only resets
the index), so the render loop's budget break fires before any
row past the frame budget is read — a cached table's headroom
rows are never touched, and an escape-capped table is identical
at any build budget (every build stops at the same cap-crosser
row). Builds carry their budget in the reference pass's own
copy of the shared uniform, so the render pass's budget stays
the schedule's.

## 10. The delta iteration

The idea: split the work by *role*, not by pixel.

- **One reference orbit** Zₙ in full precision, once per frame,
  for a single point C (§9).
- **Every pixel** iterates only its tiny offset from the
  reference: δ = c − C, and the orbit difference εₙ = zₙ − Zₙ.
  While the pixel tracks the reference, εₙ stays near δ's
  magnitude (pixel scale) — so it fits a small float with ~w
  *useful* bits **regardless of zoom depth**. The big
  coordinates live only in the reference. The expensive width is
  paid once, not per pixel.

**Derivation.** Write the pixel's orbit as zₙ = Zₙ + εₙ with
ε₀ = 0 and c = C + δ. Substitute into z ← z² + c:

```
Zₙ₊₁ + εₙ₊₁ = (Zₙ + εₙ)² + C + δ
            = Zₙ² + C  +  2·Zₙ·εₙ + εₙ² + δ
            = Zₙ₊₁     +  2·Zₙ·εₙ + εₙ² + δ
```

hence, exactly:

```
εₙ₊₁ = 2·Zₙ·εₙ + εₙ² + δ          ε₀ = 0     (all values complex)
```

This is **an algebraic identity, not an approximation** — for
now. Why it works numerically: |ε| ~ 10⁻³⁰ while |Z| ~ 1.
Squaring Z + ε directly in low precision is hopeless — ε² and
2Zε would be absorbed by rounding. The delta form keeps every
term (2Zε, ε², δ) at ε's own scale; nothing is added to a number
10³⁰ times larger until the very end (z = Z + ε, once, for the
escape check and coloring). δ itself is computed once per pixel
from its own scale down — only its significant low bits matter;
the high bits are the reference's business.

**The error budget — how many useful bits are enough.** Write
every stored ε with w significant bits. Per step:

1. ε is stored rounded — relative error 2⁻ʷ of |ε|.
2. Zₙ enters only multiplied by ε: rounding the stored Z table to
   u bits contributes absolute error ~2·2⁻ᵘ·|Z|·|ε| — *scaled by
   ε itself*, so the table never needs more than the delta
   format's own width. With the df32 table (u ≈ 47, |Z| ≤ 2 in
   the tracking regime): ≤ 2⁻⁴⁵·|ε|.

Per step the error is amplified by |2·Zₙ + εₙ| — the map's local
derivative along the orbit. On boundary orbits this expands on
average (Lyapunov exponent ln 2), so error grows roughly
geometrically. Two facts make that tolerable:

- Escaping pixels only need their *smooth iteration count* to a
  few digits; once the orbit is clearly outbound, rounding noise
  no longer changes the classification.
- ε stays ~δ-scaled while the pixel tracks the reference, so all
  w bits sit *above the signal's scale* — depth never enters.
  That is the whole reason cost is O(1) in zoom depth.

This is also the budget line that kills naive f32 deltas: plain
f32 gives w = 24 — per-step noise ~2⁻²³·|ε|, amplifying along
boundary orbits into visible speckle and wrong smooth counts.

> **How the shipped engine makes plain f32 work.** Accept
> exactly this f32 penalty, and pull the budget's other levers
> separately:
> - the **table** is df32 (u ≈ 47 ⇒ per-step table error
>   2⁻⁴⁵·|ε|, step 2 above) — the piece naive f32 perturbation
>   lacks;
> - **rebase re-anchors ε at full fresh f32 precision** whenever
>   |Z+ε| < |ε| (§11's identity, now load-bearing as a precision
>   reset, not free bookkeeping);
> - what remains is the stored-ε rounding 2⁻²³·|ε| per step —
>   boundary speckle and occasional ±1 smooth counts. That is
>   the tolerance contract (§2).
>
> The floor sits at 1e-35 — 5 decades past the external
> existence proof (§2), still inside the tolerance contract.

The failure mode this budget does not cover: the pixel *stops
tracking* the reference — ε grows toward |Z|-scale, z = Z + ε
must cancel two O(1) numbers down to O(ε), and the delta format
has nothing left to cancel with. Those pixels come out wrong
without rebasing — §11.

Escape for the delta loop is checked on the reconstructed full
value |Zₙ + εₙ|² > 65536 — one addition pair per iteration. At
escape |z| is O(1)–O(bailout), small compared to any format's
ceiling. Coloring then proceeds as in §13.

## 11. Rebase: when a pixel stops tracking

If the reference orbit escapes while a pixel's doesn't (or a
pixel's orbit passes near a critical point the reference's
doesn't), ε grows to O(1) and the delta arithmetic loses the
signal. The image shows flat blobs or noise — the infamous
perturbation glitches. Two mechanisms, in order of strength:

- **Pauldelbrot's criterion** — a pixel is suspect when
  `|zₙ| < 10⁻³·|Zₙ|`. Suspect pixels are re-rendered against a
  second reference orbit seeded inside the blob. (This is where
  reference *storage and management* enters a classical
  renderer.)
- **Zhuoran's rebasing** (2019, fractalforums) — mostly removes
  the need for extra references. Track the pixel as (m, ε)
  against reference index m; when `|Zₘ + ε| < |ε|`, *rebase*:
  set the new delta to ε′ = Zₘ + ε (the full pixel value, small
  by assumption) and restart the reference index at m = 0. This
  is again an exact identity — the pixel orbit is re-expressed
  against the same reference table from its start (Z₀ = 0 keeps
  the algebra clean). It costs nothing per iteration beyond the
  comparison already being made: after a rebase the loop consumes
  the Z table again from index 0, lookups it would do anyway.
  With rebasing, one reference serves almost every location;
  glitch classes that remain are handled by re-rendering with a
  second reference.

None of this depends on the number format — rebasing works the
same over f64 (CPU) or f32 (this engine).

FractalDive ships rebasing only: the delta kernel rebases when
|z|² < |ε|² **or when the table runs out** (the reference
escaped — the stop rule of §9; rows 0 and 1 are always stored so
the replay from row 0 is well defined). No second reference and
no glitch detection: at the accepted tolerance the residual
classes don't pay for the machinery (§2, out of scope). The exact
engine has no analogue of any of this — nothing to detect,
because nothing can go subtly wrong (§16).

## 12. The delta kernel (render pass)

`js/kernels/perturb_kernel_wgsl.js` holds the f32 delta
render kernel as five source chunks (bindings, palette, delta,
supersample, render) assembled in that order; the palette and
render chunk constants are shared with the fractaldive_test
oracle. The kernel is **pure f32** — no
limb library, no `__LIMB_COUNT__` splice. One fragment
invocation per pixel (fullscreen triangle; the render target is
exactly the frame, so every invocation is a real pixel — no
bounds check).

Per sub-pixel sample (`sampleColorPacked`, delta chunk):

- **δ assembly** (supersample chunk): base δ = planeLeft +
  x·step (real) and planeTop − y·step (imag) from integer pixel
  coords and the f32 constants, then odd multiples of the
  sub-steps for the lattice.
- **Loop.** Reconstruct z = (Z_hi + Z_lo) + ε from the df32
  table row; check escape on |z|² > 65536 — *before* the budget
  check (the exact engine's order, pinned by tests: a point
  crossing bailout on its final allowed iteration still
  escapes). Budget exhausted → interior color. Rebase when
  |z|² < |ε|² or the table's last row: ε ← z, replay from row 0
  (§11). Then update:

  ```
  ε ← 2·(Z_hi·ε + Z_lo·ε) + ε² + δ
  ```

  Z rides as its df32 pair: the hi products carry the term, the
  lo products patch it to ~2⁻⁴⁵ relative (product-of-parts, no
  error-free transforms — nothing for a fast-math compiler to
  rewrite, cf. §2/§18).
- **Derivative** (first lattice sample only — adaptive
  supersampling): D = dz/dc rides the same reconstructed z,
  D ← 2·z·D + 1 per iteration. Reference-free — rebase
  re-expresses ε, not z — so D carries across rebases
  unchanged. |D| grows ~2ⁿ and overflows f32 past n ≈ 128:
  exact 2⁻³² rescales under an i32 exponent, with the
  recurrence's +1 riding its own 2⁻³²ᵏ scale (an unscaled +1
  next to a rescaled D would swamp it).

Every value in the loop fits f32 trivially: |Z| ≤ 6.3, |ε| ≤
~7e4 between checks, |z|² ≤ 2¹⁸ at escape. At the floor 1e-35, δ
spans ~8e-39…4e-36 — the innermost sub-samples sit at f32's
minimum-normal line (~1.2e-38); a flushed-to-zero subnormal δ
simply renders that sub-sample as the reference exactly, an
error below its δ (sub-pixel, inside the tolerance contract).
Per sample this is **~20–30 full-rate f32 ops per iteration,
at any depth**.

**Supersampling.** A fragment runs the delta orbit at a 2×2
lattice of sub-pixel positions — **up to** 4 orbits per pixel
(`SAMPLE_COLUMNS_PER_PIXEL × SAMPLE_ROWS_PER_PIXEL`,
`js/common.js`; counts must be powers of two — the sub-steps are
exact shifts). Offsets are odd multiples of the two sub-step
constants: ±step/4 on both axes. The pixel color is the
per-channel average `(Σ + sampleCount/2) / sampleCount` — round
half up, integer-exact; alpha is 255 in every sample so it stays
255. Averaging happens strictly after the orbits, on colors
only — with one deliberate skip, keyed off the FIRST lattice
sample:

- **No interior early-out** — it is worth 3.9× on solid-interior
  screens, but unsound: one interior sub-sample proves nothing
  about its siblings. At the cardioid ∩ period-2-bulb tangency
  (the parity suite's shallow view) the first sub-sample
  classifies interior while its siblings escape at iteration
  ~50, and the skipped pixel rendered solid interior — a hard
  interior edge, on an 8×8 frame the whole image (the suite's
  degenerate-frame guard caught it). Interior first samples pay
  the derivative tax and run the full lattice like any other.
- **The distance-estimator early-out** (adaptive SSAA). At the
  first sample's escape, the exterior distance estimate

  ```
  DE ≈ |z|·ln|z| / |D|
  ```

  bounds both the distance to the set and the iteration-band
  density (|∇smooth| ≈ 1/(ln2·DE)). When
  `DE > SSAA_FLAT_DISTANCE_FACTOR·step` (factor 2.0, a
  delta-chunk constant) the pixel's color cannot change within
  the pixel: sample 0's color is returned directly and the
  remaining orbits never run — far-exterior pixels drop from 4
  orbits to ~1.4. The derivative costs ~4 multiplies per
  iteration on the one tracked orbit; interior pixels pay it
  for nothing (they never escape) and all-filament screens pay
  it everywhere — the honest cost of the scheme. It fails
  safe: any D inflation underestimates DE and falls back to
  the full lattice. Flat-classified pixels that the fixed
  lattice would have averaged carry at most sub-band color
  drift — the same divergence class as a boundary flip, inside
  the tolerance contract. Supersampled pixels are untouched:
  same offsets, same orbits (untracked), same average —
  bit-identical to the fixed-lattice engine. Verified in a
  node-side JS port of the exact formulas (f64 ground truth vs
  an f32-faithful port): renormalization bookkeeping exact to
  0 bits, flat verdicts agreeing across far-exterior /
  near-boundary / interior regimes — the WGSL compile and GPU
  behavior are checked by the browser test suite, not this
  probe.

## 13. Coloring

Same shape in both engines: interior points get the palette's
flat color; escaped points are colored **in-shader** by smooth
iteration count. All color math is f32 and never touches
positions.

1. **Smooth iterations.** At escape the orbit reports
   n − log₂log₂|z|². The exact kernel gets log₂|z|² from the top
   nonzero limb (exact integer limb index, f32 log2 of the one
   top word — the only float in the chain, ~2⁻²⁴ relative drift,
   invisible next to a palette cosine); the delta kernel computes
   it directly from the reconstructed z in native f32 log2. The
   inner log is floored at 1e-4 so near-threshold escapes can't
   push the count past +∞, and orbits escaping with
   |z|² ≤ 1.0001 skip the subtraction entirely.
2. **Sqrt warp.** The palette position is
   `t = frac( sqrt(n_smooth) / sqrt(16) )` — one full palette
   cycle per 16 smooth iterations, square-root warped so the
   band structure stays readable as the budget grows with depth.
3. **Cosine palette.** Each channel is its own cosine wave at
   one full cycle, phase-shifted per channel:
   ```
   channel = round( 127.5 + 127.5·cos(2π·t + phase_channel) )
   ```
   Phases and interior color come from `PALETTE` in
   `js/common.js` — same phase offsets ⇒ byte-identical colors
   to the benchmark suite.
4. **Packing.** `r | g<<8 | b<<16 | 255<<24` — little-endian
   rgba8. The fragment entry unpacks the word into its vec4
   output; the 8-bit unorm store maps k/255 straight back to k,
   so the word-to-pixel mapping is exact.
5. **Averaging.** The pixel's final color is the per-channel
   average over the 4 supersampled colors — round half up,
   integer-exact, after all orbits. Interior samples contribute
   the flat interior color, escapes their palette color, so
   boundary pixels blend instead of aliasing (classification
   lives per sample; mixed pixels are neither exactly interior
   nor exactly escape-colored) — except flat-classified
   pixels: the distance-estimator early-out (§12) returns
   sample 0's color directly, provably constant within the
   pixel up to sub-band drift.

## 14. Frame delivery

Back in `js/engine_frame.js`: each band's render pass draws the same
fullscreen triangle (three hardcoded clip-space vertices),
scissored to the band's strip, so exactly one fragment
invocation covers each pixel of the **offscreen 2× base
texture** (the overrender cache — the canvas is never a render
target while a frame grinds). A frame is a paced loop of bands:
per band, encoding to `submit` runs in one synchronous task,
`await onSubmittedWorkDone()` settles that band, then the
duty-cycle sleep (`GPU_DUTY_CYCLE`) paces the next — the GPU
works in short bursts, and the compositor gets slices between
them. Band count is adaptive: `ceil(lastGpuMs /
TARGET_BAND_GPU_MS)`, clamped to `1..MAX_BANDS` (`DEFAULT_BAND_COUNT`
for the first, unmeasured frame; 1 band = the plain single-pass
path). The HUD's GPU ms is the per-band settle waits summed,
sleeps excluded — the next frame's sizing reads the same number.
A resize mid-bands abandons the orphaned frame quietly (the
generation check in dive.js never commits it).

Presentation is the render/presentation split
([ARCHITECTURE.md](ARCHITECTURE.md)): at settle, dive.js calls
`commit()` — one blit of the full base into the canvas (the
queue is empty then, so presentation never queues behind a
render). Between settles the zoom is presenter.js's CSS
transform on the canvas element, re-sampling the base bitmap in
the compositor — no per-tick GPU work at all.

Rendering is **single-flight** (`dive.js`): the next render pass
is issued on the first rAF *after* the previous frame has fully
settled. The zoom between settles runs on the wall-clock law
(§4) via the CSS transform, never tied to render throughput. A
kernel failure marks the engine dead — no fallback by design;
the failure notice appears.

## 15. The iteration schedule

Detail must grow with depth: the deeper the zoom, the longer
orbits take to reveal structure. The budget
(`iterationCountForScale`, `js/common.js`) is a heuristic
inherited unchanged from the benchmark suite:

```
budget = clamp( 128 + 300·log10(depth), 128, 16384 )
          depth = BASE_SCALE / s     (zoom factor vs dive start)
```

Roughly 300 extra iterations per decimal decade of zoom, capped
at `MAX_ITER_CAP = 16384`. The cap also sizes the reference Z
table.

## 16. The exact engine: the test oracle

The second engine: **no perturbation at all** — every pixel's
full orbit in exact arbitrary-precision fixed point, the same
math as a CPU BigInt renderer, executed on the GPU (WGSL offers
only 32-bit integers, so wide values are u32 limb arrays — §7).
Not shipped in the app; its kernel lives in `tests/oracle/`
(`engine_kernel_wgsl_bindings.js`, `engine_kernel_wgsl_orbit.js`,
driven by `fractaldive_test`) as the oracle the tolerance tests
compare against. Its virtues:

- exact by construction — no glitch classes, no rebasing, no
  reference orbit;
- embarrassingly parallel — one GPU thread per pixel;
- one number format, one loop — simple to reason about.

Its per-iteration cost grows with depth:
`N = ceil((fractionBits + 19)/32)` limbs — the 19 integer
headroom bits cover the between-checks overshoot (from just
under bailout, the update can reach |z²| + |c| ≈ 2¹⁷ before the
next check sees it: 17 magnitude bits + sign + margin). One
shader variant per tier is compiled and cached — a dive touches
a new tier every ~9 decades of zoom.

Per supersample, the loop (kept conceptually verbatim in the
oracle):

```
loop:
    zr² = limbSquare(zr)           // 2N limbs, scale 2f
    zi² = limbSquare(zi)
    |z|² = zr² + zi²               // limbAddWide
    if |z|² >u threshold: escape → color, return
    if iterationsUsed ≥ budget: interior, return
    zi ← trunc(((zr+zi)² − zr² − zi²) ≫ f) + cᵢ
    zr ← trunc((zr² − zi²) ≫ f) + cᵣ
    iterationsUsed++
```

Three details worth noting:

- **Escape check before budget check** — same order the
  reference pass and the delta kernel inherit; the parity tests
  pin it.
- **The three-squares identity.** The zi update needs 2·zr·zi,
  and (zr+zi)² = zr² + 2·zr·zi + zi², so
  2·zr·zi = (zr+zi)² − zr² − zi²: one extra `limbSquare`
  (N(N+1)/2 word-mults) replaces `limbMultiplySigned` (N²
  word-mults plus negations) — ~17% fewer word-mults per
  iteration at the 4-limb tier. Exact over ℤ; no rounding path
  changes. The (zr+zi) overflow hazard is bounded by the same
  19-bit headroom: squaring only sees |z| ≤ 2¹⁸ between checks.
- **Everything renormalizes** through
  `limbShiftRightArithmeticWide` + `limbTruncateLow`: multiplies
  widen to 2N limbs at 2f scale, the shift/truncate pair brings
  the value back to N limbs at f scale before adding c. The
  orbit never accumulates width.

This loop is bit-for-bit the iteration a CPU BigInt renderer
performs; the whole difference is the ALU width. Supersampling
was the same 2×2 lattice and the same color averaging as
§12–§13 (4 exact orbits per pixel).

## 17. Performance: how the engines compare

**Per-iteration cost.** One complex delta step ε′ = 2Zε + ε² + δ:

- In df32 (the rejected ~48-bit delta format, §18): ~6 df
  multiplies + ~5 df
  adds ≈ 100–150 f32 flops; + escape ~30, rebase compare ~10
  (shared magnitudes), table fetch and loop ~20 → **~150–200
  full-rate f32 ops per pixel-iteration**.
- In plain f32 (shipped): **~20–30 full-rate f32 ops per
  iteration, flat at any depth** (§12).
- Exact engine: 3 limb squares = 3·N(N+1)/2 word-multiplies plus
  wide adds/shifts/compares; N grows with depth.

| | f32 perturbation (per sample) | exact engine (per supersample) |
|---|---|---|
| format | ε: f32 (df32 table + rebase) | N-limb u32 fixed point, N grows with depth |
| core work | ~20–30 f32 ops incl. escape + rebase | 3 squares = 3·N(N+1)/2 word-mults + wide adds/shifts/compare |
| at s = 2⁻¹⁰⁰ (N=4) | ~25 f32 ops | ≈ 400 u32 ops |
| samples per pixel | 4 (2×2 lattice) | 4 (2×2 lattice) |
| depth scaling | O(1) — flat | O(N²), N ≈ (log₂(1/s)+27)/32 |

The word-mult count is exact (limb-square algebra); the f32/u32
op counts are order-of-magnitude estimates from the primitive
expansions.

**How the gap grows with depth.** Word-mults per iteration for
the exact engine (3 squares) vs the flat f32 kernel:

| scale s | N | word-mults / iteration | f32 perturbation |
|---|---|---|---|
| 1e-10 | 2 | 9 | ~25 f32, flat |
| 1e-30 | 4 | 30 | ~25 f32, flat |
| 1e-100 | 12 | 234 | ~25 f32, flat |
| 1e-300 | 32 | 1584 | would need floatexp (~350–800) — out of scope |

The exact engine's cost per iteration grows quadratically with
the decade count; perturbation's never grows. At 1e-100 the gap
is ~30–70×; at 1e-300 it is orders of magnitude — on top of a
similar or larger iteration budget. That is why CPU perturbation
renderers produce 1e-1000-class images in minutes while
full-precision renderers measure the same locations in hours.

**Where the floor sits, and why.** Near the dive start the gap
between the two engines is small — at 1e-30 (N = 4, both engines
4 samples) only ~9× per pixel, absorbable in real time with every
number exact (§16's virtues). But each further ~9.6 decades of
zoom adds a limb, per-iteration cost grows quadratically while
the budget grows linearly, and exact math exits real-time long
before the format's ~1e-300 ceiling. The f32 engine's cost is
constant, so the floor is set by precision instead: 1e-35 sits
past the external proof (§2), still inside the format (grid 125
bits < LSB 2⁻¹⁴⁸) and under the iteration cap (floor budget
~10.8k < 16384).

**The overheads perturbation adds:**

- **Reference orbit** — one high-precision orbit per table
  build, cached across frames (§9): the same square cost as
  *one* pixel-iteration of the exact engine, amortized over
  ~10⁶ pixels *and* over every frame the cache serves.
  Negligible.
- **Reference storage/replay** — budget × 16 bytes as df32
  pairs; at 16384 iterations that is 256 KiB of Zₙ read per pass
  by every pixel. Bandwidth, not FLOPs, can become the limit;
  series approximation (§18) would cut both the table and the
  traffic.
- **Rebasing** — a comparison per iteration and occasional rebase
  resets; residual blobs would re-render with a second reference
  (a fractional frame cost) — not needed at the shipped
  tolerance.
- **Single-reference locality** — near minibrots the reference
  may need careful choice; exact math has no such failure mode.

None of this cancels the per-iteration gap; the overheads are
engineering friction, not asymptotics.

**Hardware notes:**

- On CPUs, f64 is full-rate — classic perturbation renderers are
  CPU-native for a reason. The discrete-GPU f64 penalty (1/16 on
  AMD, 1/32–1/64 on NVIDIA consumer parts) only matters to
  native renderers: on the web the f64-vs-f32 rate question does
  not exist, the feature simply isn't exposed (§2). Every op
  here is full-rate f32/i32/u32 on every GPU.
- Firefox specifics this project lives with: WebGPU ships on
  Windows since Firefox 141; Linux builds need
  `dom.webgpu.enabled` + WebRender prefs; and Firefox has shown
  shader-miscompile quirks before — any new kernel gets the same
  probe-based parity testing.

## 18. Not built: series approximation and other formats

**Series approximation** — perturbation's second half of the
classic speed story: don't even *run* the early iterations per
pixel. Expand the delta orbit as a polynomial in δ:

```
εₙ = Aₙ·δ + Bₙ·δ² + Cₙ·δ³ + …          A₀ = B₀ = … = 0
Aₙ₊₁ = 2Zₙ·Aₙ + 1
Bₙ₊₁ = 2Zₙ·Bₙ + Aₙ²
Cₙ₊₁ = 2Zₙ·Cₙ + 2AₙBₙ        (each next coefficient convolves)
```

The coefficients depend only on the reference orbit, so they are
computed once per frame; evaluating the polynomial at a pixel's
δ jumps ε (and the iteration counter) straight to iteration n
for the whole frame at once. An error bound — the dropped term's
size vs the rounding scale of §10 — picks the largest safe skip;
in deep locations this routinely skips most of the iteration
count. The modern refinement is Zhuoran/knighty's **bivariate
linear approximation (BLA)** — piecewise-linear z↦az+b segments
valid over ranges of the reference orbit, replacing both series
terms and big chunks of iteration with table lookups. Not needed
at the shipped floor and out of scope; noted because it dominates
current deep-zoom speed records.

**Number formats not used, and why:**

- **df32 (double-float)** — a value as hi + lo, two f32s, with
  |lo| ≤ ulp(hi)/2: a ~48-bit mantissa floating value with
  f32's exponent range. Sums via quick-two-sum/two-sum, products
  via the `fma` builtin — ~10–17 full-rate f32 flops per df
  multiply, ~3–9 per df add. Unusable as a *delta* format on
  this stack: WGSL's compiler rewrites every two-sum
  error-extraction to zero on Firefox/wgpu/NVIDIA (§2), and
  df32 needs exact two-sums. (WGSL permits flushing f32
  subnormals; that never bites this app — lo sits within 2⁻²⁴
  of hi and every magnitude stays far above f32's minimum
  normal at the depths in range.) Survives only as the *table*
  format, where plain hi/lo products need no error-free
  transforms.
- **Floatexp** — df32 mantissa + i32 exponent, renormalized
  after each operation: arbitrary exponent range at ~2–4× df32
  cost. The known ladder past df32's ~1e-36 frame-height
  comfort; out of scope — the floor caps depth; at 1e-35 the
  innermost sub-sample δs sit at the minimum-normal line and
  the tolerance contract absorbs the flush.
- For comparison, the **CPU ladder**: f64 (1× vs f64) → long
  double (~2–3×) → f64-mantissa floatexp (~5–10×) — cheaper per
  value, because desktop CPUs have f64 in hardware. The web pays
  a constant emulation factor instead; in exchange every op is
  full-rate f32/fma/i32 on every GPU.

## 19. What is exact and what isn't

Honest ledger of the shipped engine, stage by stage:

| Stage | Exactness |
|---|---|
| Dive center (target literal → 256-bit fixed-point) | exact (rounded once, at parse, half-up) |
| Zoom scale s | float64 **ratio** — relative error 2⁻⁵³ is invisible next to the 8 headroom bits that separate pixel centers |
| Center rescale to frame precision | exact shift when deepening; drops only sub-frame bits when shallowing |
| Reference orbit + Z table (all limb math) | arithmetic exact; df32 rows round to ~2⁻⁴⁵·\|ε\| per step (§10) |
| Frame delta constants (plane edges, steps) | f32-rounded at δ's scale — ~2⁻²⁴·step, far below the step/4 lattice |
| Delta orbits (per pixel) | identity exact, evaluation rounded every step: ~2⁻²³·\|ε\| stored-ε rounding — the tolerance contract |
| Escape classification, rebase decisions | tolerance-evaluated on the reconstructed z |
| Supersampling (sample positions, channel average) | positions an exact lattice of f32 constants; average integer-exact round-half-up, on colors only; DE-flat pixels render sample 0's color unaveraged (sub-band drift on flat screens — inside the boundary-flip allowance) |
| Escape coloring (log2, smooth iterations, palette cosines) | f32 — approximate *color* only (~2⁻²⁴ relative drift in the palette position) |
| Canvas output | bit-exact — the 8-bit unorm store maps k/255 back to k exactly |

And the mirror ledger — what the exact engine gets for free, and
what the f32 engine trades for constant cost:

| property | exact limb engine | f32 perturbation (shipped) |
|---|---|---|
| arithmetic exactness | by construction (integer wrap = exact) | identity exact, evaluation rounded every step (f32: w = 24, df32 table: u ≈ 47) |
| correctness proof | parity vs BigInt oracle, bit-for-bit | tolerance checks vs the exact engine (bounded boundary fraction, channel deltas ≤ 1) |
| failure modes | none below the format ceiling | glitch classes contained by rebase; residual risk accepted |
| machinery | one loop, one number format | reference table, rebasing, tolerance tests |
| GPU number format | u32 limbs — core WGSL | f32 + df32 pairs — core WGSL |
| browser reach | any WebGPU (Firefox incl.), no optional features | identical requirements — the no-f64 constraint is the web's, not a differentiator |
| cost curve | O(N²) per iteration, N ∝ depth | O(1) per iteration at any depth |
| verification story | "compare against BigInt" — trivially sound | "matches the oracle within tolerance, empirically, at every depth the app can reach" |

The ledger above is the accepted price of constant-cost deep
zoom on WebGPU — an informed choice, not an accident (§2).

Correctness is pinned by the tests in `tests/`:
`limb_array_math_test` runs every limb primitive against a JS
BigInt oracle at tiers 1..8; `fractaldive_test` checks the df32
reference table bit-exact against the BigInt oracle and shipped
frames against the exact oracle (`tests/oracle/`) within tolerance
(plus the banded-frame bit-identity, commit-blit, and
render/presentation-split sections).

---

*Cross-references: architecture and file map —
[project.md](../project.md) + [ARCHITECTURE.md](ARCHITECTURE.md);
the external f32 existence proof —
zoomingfractal.com (WebGL2: f32 deltas + df32 table + rebase
below 2¹⁰⁰, floatexp beyond).*
