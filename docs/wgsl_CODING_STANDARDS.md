# FractalDive — WGSL Coding Standards

These rules exist so a brand-new developer can read any WGSL source
in the project and understand it without prior context. Shader code
is debugged by reading, not stepping — optimize for the reader.
Every rule below is already the style of the existing WGSL sources;
match it, don't invent a new one.

## 0. Project shape (why the rules are what they are)

- **WGSL lives as template strings inside JS wrapper files, not
  standalone `.wgsl` files.** `file://` pages never expose external
  `<script type="text/wgsl">` text and `fetch` is CORS-blocked — a
  JS string is the only `file://`-safe single source of truth. The
  wrapper file's header comment owns the why; the string owns the
  shader.
- **Library first, then kernel — one concatenation.** The render
  kernel splices the shared limb-math library ahead of itself
  because WGSL requires declaration before use. Inside each source,
  order helpers simplest → most complex while keeping every use
  after its declaration.
- **Two splice mechanisms, don't confuse them.** A placeholder token
  is replaced by the engine at compile time (one variant per
  precision tier — WGSL array lengths must be compile-time
  constants); `${…}`-style interpolations are JS constants spliced
  at script load. A splice point is a named constant with a comment
  saying which side substitutes it.
- **The GPU mirrors the CPU.** Kernel semantics (escape check
  before budget check, `zi` before `zr`, palette math) must match
  the CPU implementation exactly — parity is validated against
  BigInt/f64 oracle test pages, one tier set to the next.
- **No u64 ALU exists in WGSL.** All wide math is u32 wrap
  arithmetic: products split into 16-bit halves, carries are
  wrap-detected (`sum < operand`), and bounds are proven in
  comments, not hoped for.

## 1. Naming — clarity over brevity

**Same law as the JS standards** — no short identifiers outside
their established domain (`zr`/`zi` in the Mandelbrot orbit,
`lo`/`hi` word halves). Every identifier says what it holds, units
in the trailing comment.

| Bad | Good |
|---|---|
| `fn add(a, b)` | `fn limbSubtract(left: limb_fixed_int, right: limb_fixed_int) -> limb_fixed_int` |
| `const OFF = 0u;` | `const THRESHOLD_INPUT_OFFSET: u32 = 0u; // limbInputs index where threshold starts` |
| `var t = 0u;` | `var stepsSpent = 0u; // orbit steps consumed so far` |

Rules:
- **Module-scope constants are `SCREAMING_SNAKE_CASE` with an
  explicit type and a why/units comment**:
  `const PALETTE_CYCLE_LENGTH: f32 = 16.0; // one palette cycle per 16 smooth iterations`.
  Buffer layout is a chain of computed offsets, never magic
  numbers: each offset derives from the previous one plus a field
  length (`NEXT_INPUT_OFFSET = PREV_INPUT_OFFSET + FIELD_LENGTH_LIMBS`).
- **Functions are verbs**: `loadLimbInputs()`, `escapeColorUnpacked()`.
  Library primitives may name the operation: `limbSubtract`,
  `limbIsPositive`, `limbShiftRightArithmeticWide`.
- **Library prefix and width suffix are the contract.**
  `limb*` = limb-array operation on the value alias; `*Wide` =
  the same operation at product width (2N limbs) on the
  product alias; word-domain primitives (`wordMul32`) spell
  their domain. A new width variant follows the suffix, not a new
  name.
- **Booleans read as questions**: `limbIsPositive()`.
- **Types: aliases snake_case, structs CamelCase.**
  `alias limb_fixed_int` / `alias limb_product_int` spell the
  *value type*, not the raw array — callers say what the value is,
  never `array<u32, N>` at call sites. Structs
  (`ShadeParams`, `MulWordsResult`) are CamelCase with every field
  commented.
- **Frozen names**: the limb-count placeholder, the width
  constants, the two value aliases, the uniform struct's fields,
  the binding numbers (which number is uniforms, which is storage),
  the packed color layout `r | g<<8 | b<<16 | 255<<24`, and the
  entry-point names (terse on purpose — the pipeline setup
  references them). Changing one means updating every consumer in
  the same commit.

## 2. Modularity — one job per source string

- **The shared limb-math library is kernel-agnostic.** It takes
  u32 limbs in, returns limbs out — no `params`, no bindings, no
  palette, no pixel concepts. If a limb helper wants to read
  `params`, it belongs in the kernel source.
- **The kernel owns the frame.** Bindings, uniforms, palette,
  supersampling, entry points live only in the render-kernel
  source. The uniform struct and buffer offsets are declared once
  at the top and documented in the header box (what each binding
  holds, in limb order).
- **Mirror helpers keep their CPU twin's semantics and shape**
  (same check order, same zi-before-zr) so a diff across the CPU
  and GPU sides reads as a translation, not a rewrite. Comment the
  mirror in the header.
- **Aliases, not struct wrappers, for value arrays.** A struct
  would force `.limbs` indexing at every call site; the alias
  keeps expressions reading as math.
- **Nothing is read back to the CPU.** The fragment colors the
  canvas texture directly; if you find yourself wanting a
  readback, that's a design change, not an edit.

## 3. Size limits — nothing gets blocked up

| Unit | Cap | Action when exceeded |
|---|---|---|
| Function body | ~40 lines | Extract helpers (each gets its own doc comment); multi-pass algorithms may stretch to ~60 with `// pass N:` banners |
| Nesting | 2 levels | `break`/early `return` guards; compute offsets before loops |
| Loop bodies | one concern | Accumulators + index math only; anything clever gets a named `let` |

The test: **can you state the function's contract as a math
identity plus a returns-line?** If not, it's doing two things.

## 4. Doc comments — short, topic-style, identity first

Every function that isn't a one-liner gets a comment block
directly above it. Line 1 is the math identity — **and the whole
block stays short. It's an index, not an essay.**

```wgsl
// a × k (u32 scalar)
// - returns limbs × scalar, the per-pixel c = planeLeft + x·step
// - fits N limbs: x < canvas width < 2^32
fn limbScaleSmall(limbs: limb_fixed_int, scalar: u32) -> limb_fixed_int {
```

Format:
- Line 1: the identity or equation (`// a + b`, `// a >> n (arithmetic, at
  product width)`, `// ch(t) = round(1/2 + 1/2·cos(2π·t + φ))`,
  `// rgba8(t) = r | g<<8 | b<<16 | 255<<24`).
- Next: `Returns …` — one short line of plain-English meaning.
- Extra facts (bounds, edge cases, parity) are **one-line bullets,
  not wrapped sentences**.
- **Wrap arithmetic gets its bound proof as a bullet** — `// - hi word ≤
  0xFFFF…: a,b < 2^16`. If a reader could ask "can this overflow?",
  the answer is already written — in one line.
- Entry-point conventions get a one-line note: why no y-flip, why
  `+0.5` pixel centers.

Hard rules:
- **≤ 6 lines of prose** (identity + bullets) per doc comment —
  headers carry no `@param` luggage, so this is the total cap.
  Beyond it, split the function or move context to the header box.
- **Topics, not prose.** Facts are scannable bullets
  (`// - x·step fits N limbs: x < 2^32`), never paragraph
  sentences.
- **No narrative** — the reader wants labels to scan while
  reading code, not a tutorial mid-shader.

## 5. Source structure — same order everywhere

Inside each template string:

```wgsl
// ================================================================
// Header box: what this source owns, the bindings contract,
// the splice points, the CPU-parity notes.
// ================================================================

// --- constants: buffer layout ---
// --- constants: <domain> (palette, lattice, limb widths) ---
// --- value types (aliases) / structs ---
// --- bindings (@group/@binding + uniform/storage vars) ---
// --- helpers: simplest → most complex, every use after its declaration ---
// --- render entries: one fullscreen triangle, one fragment per pixel ---
```

`// --- section ---` banners whenever there's more than one
section's worth of content; long helpers split with sub-banners
(`// pass 1: cross pairs …`). Struct fields get per-field trailing
comments. The JS wrapper around the string keeps its own boxed
header owning the file:// rationale and the splice mechanics.

## 6. General style — WGSL-specific discipline

- **`let` by default; `var` only when the value is actually
  reassigned** — one exception: a lookup table indexed at runtime
  (the vertex entry point's position table) stays `var`, because
  by-value dynamic indexing of `let` arrays is spec-legal but
  implementation-spotty (naga gained it late — gpuweb #2086,
  wgpu #4337). Declare the type on zero-init:
  `var zr: limb_fixed_int = limb_fixed_int();`.
- **Literal suffixes always**: `0u` / `0xFFu` for u32 (never a
  bare `0` mixing abstract-int), `2.0` not `2` for f32. Masks are
  hex (`0xFFFFu`, `0x80000000u`).
- **`select()` for value choice, `if` for control flow** — WGSL
  has no ternary. `select(0u, 1u, sum < operand)` is the wrap-detect
  idiom. `select()` rejects arrays — branch and assign instead,
  with the `// select() rejects arrays` comment so nobody "fixes"
  it back.
- **Respect the wrap semantics the spec forces on you.** Shift
  counts wrap mod 32 — guard `if (bitShift != 0u)` before
  `nextWord << (32u - bitShift)`; signed shifts need explicit
  sign-fill. These traps get a comment at every site.
- **u32 loop counters** from `0u`; general loops use
  `loop { if (cond) { break; } … }` when the exit isn't a plain
  counter.
- **Comments explain why, not what — as topics, not paragraphs.**
  `// wraps only from 0xFFFFFFFF + 1`, `// complex y grows up
  (screen y down)` — one fact per line. Wrap proofs are bullets,
  not essays; anything reading like a paragraph gets rewritten
  as bullets or cut.
  Delete dead code and commented-out experiments; the shader
  compiles per tier and unused anything is noise.
- **Precision deviations from the CPU paint are called out.**
  The f32 palette pipeline is the one allowed deviation and every
  piece of it says so in a comment; exact limb math never degrades
  silently.
- **No premature abstraction.** No helper until the second call
  site; no uniform field until the shader reads it; no new
  binding until the data actually crosses the boundary.

## 7. Definition of done (check before committing)

- [ ] Compiles clean through naga's front end for every precision
      tier (the per-tier splice means you compile N variants, not
      one).
- [ ] The limb-math test page (new lib helpers) and/or the
      frame-level kernel/palette test page pass against the
      BigInt/f64 oracles on WebGPU; exact-arithmetic paths match
      exactly, f32 palette paths within stated tolerance.
- [ ] Every identifier self-explanatory; constants SCREAMING_SNAKE
      with explicit types, units, and why-comments.
- [ ] Doc comment on every non-trivial function: identity line,
      `Returns` line, overflow bound where wrap math is involved —
      **≤ 6 lines, bullet topics, no paragraphs.**
- [ ] Buffer offsets derive from the offset-constant chain — no
      fresh magic numbers in load/store code.
- [ ] Frozen contract names (the library prefix/width-suffix
      scheme, the value aliases, the uniform struct, binding
      numbers, packed color layout, the limb-count placeholder)
      untouched, or the engine and both wrapper headers updated in
      the same change.
- [ ] Any semantic change to the orbit (check order, truncation,
      escape rule) mirrors the CPU implementation — or the
      deviation is documented in both headers and covered by a
      new oracle test.
- [ ] `unverified: <what> - <why>` in review if you couldn't run
      WebGPU — do not claim parity you didn't execute.
