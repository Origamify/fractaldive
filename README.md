# FractalDive

Standalone auto-zoom dive into the Mandelbrot set, glitch-free down
to ~1e-35. Renders on WebGPU via perturbation: one exact
BigInt-limb reference orbit per dive center, plus cheap f32 delta
orbits per pixel — no float64 in shaders, no libraries, no build
step, nothing to install.

**Live demo:** https://origamify.github.io/fractaldive/

## Why this project exists

I built FractalDive to **learn WebGPU and WGSL from scratch** — and
it turns out a deep Mandelbrot dive is an excellent project for that:
compute shaders, render passes, buffers, bind groups, textures,
frame pacing, GPU/CPU synchronization and more.

I made a JS version 10+ years ago by hand, with no GPU usage. This
time I redid it with AI and technologies I hadn't worked with before
— mostly written with GLM 5.3. It was awesome to learn while doing
and grow the project at a good pace.

If you want to learn WebGPU/WGSL too, this codebase might be good
inspiration: zero dependencies, no build step, plain classic scripts.

- [docs/ALGORITHM.md](docs/ALGORITHM.md) — the perturbation math, tolerance model.
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — the render/presentation split and module responsibilities.
- [project.md](project.md) — file map, settings, and how to run/change things.

## Build your own — the steps this project took

In the order this project actually built them, each step motivating
the next:

1. **A small BigInt implementation on CPU.** The first version: one exact
   reference orbit for the dive center, computed in JS `BigInt` fixed
   point — slow and compute-intensive, but you can already click
   the canvas and zoom arbitrarily deep.
2. **Supersampling.** Each pixel runs the orbit at a 2×2 sub-pixel
   lattice and averages the four colors — the cheap trick that turns
   mushy gradients into beautiful clean images (§13 of ALGORITHM.md).
3. **Move the BigInt math onto the GPU with limb arithmetic.**
   Implement u32 limb add/carry/multiply in WGSL as a compute
   shader, so the exact reference orbit is built on-device
   instead of shipped from JS (§7–§9 of ALGORITHM.md).
4. **Perturbation on the GPU.** Upload the reference orbit as a
   storage buffer; per pixel iterate only the *delta* in f32
   (`zₙ = Zₙ + ε`, the reference value plus a small correction).
5. **Push the depth.** Rebase deltas onto the reference when a
   pixel stops tracking, keeping the dive glitch-free all the way
   down to the 1e-35 floor (§11 of ALGORITHM.md).
6. **Diving and zooming logic.** The zoom state machine (`js/dive.js`):
   named dive targets, the wall-clock zoom law, click-to-zoom (one
   octave per click), an adaptive rate that auto-slows when the GPU
   can't keep up, and hidden-tab pausing.
7. **Adaptive supersampling + presentation.** A distance estimator for
   adaptive sampling, banded frame submits for GPU duty-cycle,
   and a CSS-transform presenter between frames (§12–§14 of
   ALGORITHM.md, ARCHITECTURE.md).

## Usage

Open `index.html` in a WebGPU-capable browser (Firefox/Chrome; no
server, no deps — works straight from `file://`) — the dive starts
on its own, just watch it zoom.

When you want to steer: pick a dive target from the dropdown, adjust
Speed with `+`/`-`, click the canvas to jump one zoom octave, Space
to pause/resume. The **Auto** checkbox lets the speed follow the
GPU instead of your slider. ⧉ Copy / ⤳ Go to share or jump to a view.
More detail: [about.md](about.md).

## License

MIT — see [LICENSE](LICENSE).
