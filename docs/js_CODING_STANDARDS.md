# FractalDive — JavaScript Coding Standards

These rules exist so a brand-new developer can read any JS file in the
project and understand it without prior context. Code is read far more often
than it is written — optimize for the reader. Every rule below is
already the style of the existing files; match it, don't invent a new
one.

## 0. Project shape (why the rules are what they are)

- **Classic scripts, no modules.** The entry HTML loads the files in
  dependency order — shared helpers first, then pure math, then kernel
  sources, then the engine, then UI, then the entry point; all files
  share one global lexical scope. No `import`/`export`, no bundler, no
  deps, no server — the app must keep working from `file://`.
- **No build step, ever.** What you write is what runs.
- Ownership: a shared-constants file (constants + helpers) · pure
  fixed-point math files · WGSL kernel wrapper files (a u32-limb
  library, a Z-table compute shader, an f32 render kernel) · a render
  engine · a UI module (footer controls / HUD / hotkeys) · an
  entry-point state machine + render loop.

## 1. Naming — clarity over brevity

**No short variables.** Never use single letters or abbreviations
outside their natural domain (`i`/`j` in a loop, `x`/`y` for complex
plane or pixel coordinates, `zr`/`zi` in the Mandelbrot kernel where
they are the established math names). Every identifier must say what
it holds.

| Bad | Good |
|---|---|
| `let s = 0.25;` | `let unitsPerPixel = 0; // complex units per pixel` |
| `function fit(cv)` | `function resizeCanvasToViewport()` |
| `const t = 300;` | `const DEPTH_DECADE_ITERATION_BUDGET = 300;` |

Rules:
- **Variables use nouns or noun phrases**, units in the trailing
  comment: `zoomRatePerSecond`, `canvasWidthPx`, `elapsedMs`.
- **Functions and methods use verbs**: `resizeCanvasToViewport()`,
  `iterationsForScale()`, `shouldRenderFrame()`.
- **Booleans read as questions**: `isFramePending`, `hasHitDepthCap`,
  `canSkipRender`. Prefix with `is` / `has` / `can` / `should`.
- **Constants are `SCREAMING_SNAKE_CASE`.** Shared constants live in
  the shared constants file; module-local ones live at the top of the
  owning file. No magic numbers buried in logic.
- **Contract field names are frozen**: the view struct's fields, the
  palette fields, the engine's public API. Changing one of these names
  means updating every consumer in the same commit.

If you have to hover over an identifier to understand it, rename it.

## 2. Modularity — one job per file, one owner per concern

- **One concern per file.** The UI module owns the footer and never
  steps the zoom; the dive module owns the dive model and never
  touches DOM elements; engine files compute and never read `window`.
  If you're editing a file for an unrelated reason, you've found a new
  module.
- **One mutable model per concern.** The dive state object in the dive
  module is the single dive model — the UI reads it at runtime but
  never at load time, and only its owner mutates it. No shadow copies
  of dive state anywhere else.
- **Engine ↔ app talk only through the contract.** The app calls the
  engine's public API and friends; it never reaches into engine
  internals. A new engine (e.g. a BigInt CPU engine) is a drop-in
  class with the same API.
- **DOM access only in the UI / wiring layer.** Engine and math
  helpers take values in, return values out, so they stay testable
  in the browser test pages.

## 3. Size limits — nothing gets blocked up

| Unit | Cap | Action when exceeded |
|---|---|---|
| Function body | ~40 lines | Extract helpers; each extracted helper gets its own docstring |
| File | ~350 lines | Split along ownership lines (e.g. engine kernel vs. plumbing) |
| Function parameter list | 4 | Group them into a single options object — which is exactly what the view struct is |

The test: **can you write one complete sentence that describes the
whole function?** If not, it's doing two things — extract until you
can.

## 4. Docstrings — short, topic-style, every function

Every function that isn't a one-liner gets a JSDoc block directly
above it (existing files do this; keep doing it) — **and it stays
short. A docstring is an index, not an essay.**

```js
/**
 * Returns the iteration budget for a viewport scale.
 * Why: detail must grow with zoom depth — budget tracks depth decades.
 * @param {number} unitsPerPixel - Complex units per pixel.
 * @returns {number} Iteration count for the current depth.
 */
function iterationsForScale(unitsPerPixel) { ... }
```

Format:
- Line 1: imperative summary — **one line, one sentence.**
  (`Returns…`, `Maps…`, `Clamps…`), not "This function…".
- Optional why: **one short line or bullet, not a paragraph.**
  If the why needs more than a sentence, the function is too
  complex or the context belongs in the file header box.
- `@param` / `@returns`: type + plain-English meaning in a few
  words. No ranges/stories unless a reader would actually be
  misled without them.
- Skip docstrings only for one-liners whose name is the whole
  documentation (`getElementById` alias).

Hard rules:
- **≤ 6 lines of prose** (summary + bullets) before the
  `@param` block; each `@param` / `@returns` stays one short
  line. Beyond the cap, split the function or move context to
  the file header box.
- **Topics, not prose.** Multiple facts → short bullets
  (`// - x·step fits N limbs: x < 2^32`), never wrapped
  paragraph sentences.
- **No narrative** ("First we… then we… note that…"). The reader
  wants labels to scan, not a tutorial to read end-to-end.

## 5. File structure — same order everywhere

Every file opens with `'use strict';` then a boxed header comment,
then banner sections in a fixed order:

```js
'use strict';
/* ================================================================
 * <File name> — <one line: what the file owns, what it must never
 * touch, load-order notes>.
 * ================================================================ */

/* --- constants --- */
const MAX_CANVAS_PIXEL_RATIO = 2; // DPR cap — keeps hi-dpi phones from rendering 4× the pixels

/* --- state --- */
const appState = { ... }; // the one mutable model, field-by-field commented

/* --- helpers (internal, then public) --- */
/** Docstring… */
function currentIterationCount() { ... }

/* --- wiring / lifecycle (UI / entry-point files only) --- */
```

Sections: header box → constants → state → internal helpers → public
functions → wiring. Use `/* --- … --- */` banner comments whenever a
file has more than one section's worth of content; split long
sections with sub-banners (`/* --- speed --- */`).

Inline comments carry the *why* plus units:
`this.unitsPerPixel = WORLD_SPAN / this.canvasHeightPx; // whole set visible: span 3.0`.

## 6. General style

- **Comments explain why, not what — as topics, not paragraphs.**
  `// never reuse the previous target's reference orbit` — yes.
  `// multiply by two` — no. One fact per line, scannable labels
  over wrapped prose; a comment block that reads like a paragraph
  gets rewritten as bullets or cut.
- **No dead code, no commented-out blocks, no TODO scaffolding.**
  If it's not used, delete it.
- **Early returns over nested `if`s**; guard clauses at the top:
  `if (width === this.width && height === this.height) return;`.
- **No premature abstraction.** No helper until the second call site
  exists; no config knob until something reads it. YAGNI is law —
  the project deliberately has no manual zoom, no Julia sets, no
  build step.
- **Match the existing files** — 4-space indent, semicolons,
  trailing commas in multiline literals. Consistency beats personal
  preference; when in doubt, copy the style of the function above
  the one you're writing.
- **Failure is honest.** A feature that can't run (no WebGPU) shows
  a notice; it never silently no-ops. Say `unverified: <what>` in
  review when you couldn't actually run it.

## 7. Definition of done (check before committing)

- [ ] Opens from `file://` in Chromium; dive runs, HUD updates, no console errors.
- [ ] Every identifier is self-explanatory; no single-letter vars
      outside loop/coordinate/kernel-math names.
- [ ] Functions are verbs, booleans are questions, each file has one job.
- [ ] Docstring on every non-trivial function with `@param` /
      `@returns` — **≤ 6 lines, topic-style, no paragraphs.**
- [ ] New constants sit in the shared constants file (shared) or the
      owning file's constants section — no fresh magic numbers in
      logic.
- [ ] Contract names (view fields, palette fields, engine API)
      untouched, or all consumers updated in the same change.
- [ ] Pure-math changes (kernels, iteration schedules) got a smoke
      test in a browser test page — sanity-check outputs (interior
      vs. exterior points, edge values).
