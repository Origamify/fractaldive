# FractalDive — project.md (for agents)

Operational map: what's where, how to run and change things.
The math and design decisions live in
[docs/ALGORITHM.md](docs/ALGORITHM.md); the module split in
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Status

Active. Current engine: **f32 tolerance perturbation on WebGPU**
(`js/engine.js`) — one exact BigInt-limb reference orbit per dive
center, f32 delta orbits per pixel, rendered in paced bands.
Frames are correct within a stated tolerance, not bit-exact
(ALGORITHM.md §19). The exact per-pixel engine (not shipped)
survives
as the tests' oracle (ALGORITHM.md §16); a multi-engine
benchmark lives on the `benchmark` branch.

## Scope

Does: continuous auto-zoom dives into the Mandelbrot set down to
viewport height 1e-35 (`MIN_S`), exact reference centers (plain
decimal strings — float64 never touches the center), target
picker, speed control, pause/reset, click-to-zoom, copy/paste
views as JSON.

Doesn't: manual pan, image export, recording, CPU rendering,
depths past `MIN_S`, user-supplied target coordinates. Not-built
formats (series approximation, BLA, floatexp): ALGORITHM.md §18.

## Run & setup

Static site — no build, no server, no deps, no env vars:

    xdg-open index.html        # needs a WebGPU browser; init() throws otherwise

Tests run as webpages (same WebGPU requirement, no CLI runner):

- `tests/limb_array_math_test.html` — limb lib primitives vs the
  BigInt oracle, tiers 1..8
- `tests/fractaldive_test.html` — everything else, five sections
  in run order: df32 reference table vs BigInt oracle; frames vs
  frames vs the exact oracle engine; banded vs single-pass frames;
  the commit blit; the render/presentation split on mocks
  (shadows the clock and rAF — runs last)
- `tests/run_selenium.py` — opens both pages in Firefox, reports
  each verdict

## File map

| Path | Role |
|---|---|
| `index.html` | Page shell + script load order (classic scripts for `file://`; order matters — see the comment block) |
| `js/common.js` | Shared constants & helpers: dive targets, palette, iteration schedule, zoom/speed constants, DOM/math helpers. "To change what the app does → start here" |
| `js/fixed_point.js` | BigInt fixed-point frame math for the engine (pure, no DOM/GPU) |
| `js/engine.js` | `WebGPUPerturbationEngine`: the frozen API (`init`/`resize`/`render`/`resetRef`/`commit`) + pipeline compilation |
| `js/engine_frame.js` | The engine's frame pipeline (prototype extension): buffers, reference-orbit cache, one frame = a paced loop of band submits into the offscreen 2× base texture. Render correctness → here |
| `js/presenter.js` | Presentation layer: the canvas's CSS zoom transform (`initPresenter` / `setZoom`) — the only zoom process |
| `js/dive.js` | Dive state machine + view edits (go-to / click-zoom) + base-frame submission; adaptive zoom rate, overrender window bookkeeping, generation/invalidation. Dive behavior → here |
| `js/app.js` | Entry point: render loop (`onAnimationFrame`, zoom-law stepping), canvas sizing, `initApp()` boot |
| `js/ui.js` | Footer controls, HUD, toast, failure notice, hotkeys. Controls/keys → here |
| `js/kernels/limb_array_math.js` | u32-limb WGSL math lib (spliced ahead of the reference kernel; the test oracle's exact kernel too) |
| `js/kernels/reference_orbit_wgsl.js` | Assembles the reference-orbit compute kernel |
| `js/kernels/perturb_kernel_wgsl.js` | Delta/render kernel source in five chunks + their assembly |
| `js/kernels/present_blit_wgsl.js` | The commit blit's shader (crop 1 — the CSS transform crops) |
| `style.css` | Canvas + footer styling |
| `docs/ALGORITHM.md` | Algorithm, formats, and the design rationale |
| `docs/ARCHITECTURE.md` | The render/presentation separation of responsibilities |
| `docs/js_CODING_STANDARDS.md`, `docs/wgsl_CODING_STANDARDS.md` | Language rules for JS / WGSL here |
| `tests/` | The test pages above; `tests/oracle/` holds the exact oracle engine's kernel sources |

## Conventions

- Classic scripts, no modules (must run from `file://`); all of
  `js/` shares one global lexical scope. Load order in
  `index.html` matters: common → limb lib → kernels → engine →
  app (`perturb_kernel_wgsl.js` splices common.js's
  `SAMPLE_COLUMNS_PER_PIXEL` / `SAMPLE_ROWS_PER_PIXEL` into its
  WGSL at script load, and `fixed_point.js` derives the sub-step
  shifts from them — don't rename those without updating both).
- Engine↔app contract and the presentation invariants: see
  [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md). In short —
  `render(view, maxIter)` takes `view.s` at BASE pixel pitch,
  dive.js owns the window semantics and calls `commit()` once per
  settled render; the zoom between settles is presenter.js's CSS
  transform.
- Test pages that load dive.js and app.js set `FRACTALDIVE_NO_AUTOBOOT`
  first — without it app.js auto-boots the app on load.

## Settings

All runtime settings are constants in code (grep for the name);
none come from env vars or config files.

| Setting | Where | Meaning | Default |
|---|---|---|---|
| Speed slider (0–100) | `js/common.js` `SPEED_MIN`/`SPEED_MAX`/`DEFAULT_SPEED` | Zoom factor per second | 1/s (range 0.05–4) |
| Auto speed checkbox | `js/dive.js` `diveState.isAutoSpeedEnabled` (wired in `js/ui.js`) | Master switch for the adaptive zoom-rate slowdown | on |
| `SPEED_HOTKEY_STEP` | `js/ui.js` | `+`/`-` speed step | 0.05 |
| `MIN_S` | `js/common.js` | Dive floor (smallest viewport scale) | 1e-35 |
| `BASE_SCALE` / `INITIAL_VIEW_HEIGHT` | `js/common.js` | Reset / page-load viewport height | 3.0 / 8.57e-28 over target 0 |
| `DIVE_TARGETS` | `js/common.js` | Dropdown dive destinations (plain-decimal strings) | 7 targets |
| `CENTER_FIXED_POINT_BITS` | `js/common.js` | Fractional bits of the BigInt dive center | 256 |
| `MAX_ITER_CAP` | `js/common.js` | Iteration ceiling; sizes the reference Z table | 16384 |
| `SCHEDULE_BASE_ITERATIONS` / `SCHEDULE_ITERATIONS_PER_DEPTH_DECADE` | `js/common.js` | Frame iteration budget | 128, +300 per depth decade |
| `SAMPLE_COLUMNS_PER_PIXEL` / `SAMPLE_ROWS_PER_PIXEL` | `js/common.js` | Supersampling lattice | 2 × 2 |
| `SSAA_FLAT_DISTANCE_FACTOR` | `js/kernels/perturb_kernel_wgsl.js` | Adaptive SSAA clearance (pixel steps) required to render single-sample | 2.0 |
| `PALETTE` | `js/common.js` | Cosine palette phases + interior color | fixed (interior rgb 4,7,14) |
| `MAX_DEVICE_PIXEL_RATIO` / `MIN_CANVAS_SIZE_PX` | `js/app.js` | DPR cap / min canvas | 2 / 64 px |
| `BASE_OVERRENDER_FACTOR` | `js/common.js` | Base frame resolution factor per axis (zoom window width) | 2 |
| `BASE_WINDOW_END_CROP_FRACTION` | `js/app.js` | Crop fraction that requests the next base frame (the 1:1 window end) | 1/2 |
| `CLICK_ZOOM_FACTOR` | `js/dive.js` | Scale divisor per canvas click | 2 |
| `REFERENCE_TABLE_BUDGET_HEADROOM` | `js/engine_frame.js` | Extra table rows past the frame budget (~3 decades of cache per build) | 1024 |
| `GPU_DUTY_CYCLE` | `js/common.js` | Fraction of render time the GPU may stay busy (inter-band sleep) | 0.9 |
| `ADAPTIVE_FLIGHT_HEADROOM` / `ADAPTIVE_RECOVERY_STEP` | `js/dive.js` | Auto-slowdown threshold / rate climb back to user speed | 1.25 / 1.15 |
| `TARGET_BAND_GPU_MS` | `js/common.js` | Target GPU ms per band (drives the adaptive band count) | 100 |
| `MAX_BANDS` | `js/engine_frame.js` | Band count ceiling | 64 |
| `DEFAULT_BAND_COUNT` | `js/engine_frame.js` | Bands for the first, unmeasured frame | 12 |

Hotkeys: Space = pause/resume, `+`/`-` = speed.
