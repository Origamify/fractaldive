# ARCHITECTURE — the separation of responsibilities

Why the app is split the way it is. The math lives in
[ALGORITHM.md](ALGORITHM.md); the file map in
[../project.md](../project.md).

## Why a split at all

- WebGPU gives each device exactly **one queue**, and submissions
  execute strictly in order (FIFO, no priority).
- The app has two GPU-adjacent jobs of very different shape:
  rendering a base frame (rare, expensive — up to seconds at
  depth) and presenting the zoom (cheap, every rAF tick).
- Driving both through the one queue stalled the presentation
  behind every render — the zoom visibly froze for the length of
  each base render, then jumped. The split removes the coupling:
  **the GPU renders; the DOM presents.**

## The modules

| Module | Responsibility | Never touches |
|---|---|---|
| `js/engine.js` | `WebGPUPerturbationEngine` class: lifecycle (`init`/`resize`), the frozen API (`render`/`resetRef`/`commit`), pipeline compilation | dive state, the DOM, crop fractions |
| `js/engine_frame.js` | The engine's frame pipeline (prototype extension of `WebGPUPerturbationEngine`): buffers, reference-orbit cache, one frame = a paced loop of band submits into the offscreen 2× base texture | dive state, the DOM, presentation |
| `js/presenter.js` | Presentation: the canvas element's CSS zoom transform (the compositor re-samples the base bitmap) | WebGPU, dive state |
| `js/dive.js` | The dive model: zoom state machine, overrender-window bookkeeping, generation/invalidation, single-flight submission, hidden-tab pause/resume | GPU details, the render loop |
| `js/app.js` | The entry point: render loop (zoom law stepping, fps sampling), canvas sizing, boot (`initApp`) | dive-model mutation beyond stepping it |
| `js/ui.js` | Controls, HUD, notices | rendering |
| `js/common.js`, `js/fixed_point.js`, `js/kernels/*` | Shared constants, exact frame math, WGSL sources | — |

## Data flow

Per rAF tick — no GPU work at all:

```
stepZoom: s ← s·e^(−rate·Δt)
setZoom(baseWindowStartScale / s)     presenter: CSS transform = the visible crop
submitRender() if due (window end / invalidation / repaint)
```

At settle — the only presentation submit, on an empty queue:

```
render settled → generation matches?
  yes → latch window start, engine.commit()   one full-base blit, crop 1
  no  → nothing; the canvas holds the last frame at its frozen transform
```

The canvas backing store runs at the base resolution (2× device
px); the transform shows the center sub-rect — the same geometry
the old crop-blit shader had (2:1 downsample at window start,
1:1 texels at window end, soft upscale past 1:1).

## Contracts

- Engine (frozen): `init / resize / render / resetRef / commit`
  (+ `.available / .w / .h`). `commit()` is called only at settle
  — the queue is empty then, so presentation can never queue
  behind a render.
- Presenter: `initPresenter(canvas)` once; `setZoom(m)` per tick.
  Stateless besides a change gate — when app.js stops calling
  (window closed), the last transform simply holds.
- dive.js owns all window semantics; nothing else decides when a
  base renders or commits.

## Invariants

- One zoom process: the CSS transform, always. Nothing blits the
  canvas per tick.
- The transform is a pure function of dive state
  (`windowStartScale / viewportScale`) — no swap choreography; a
  new base lands already correctly cropped (only soft→sharp).
- An orphaned settle (invalidation mid-flight) never commits.
- Renders target the offscreen base only; the canvas receives
  `commit()`'s single blit at settle.
