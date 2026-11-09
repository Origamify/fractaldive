# FractalDive

Standalone auto-zoom dive into the Mandelbrot set, glitch-free down
to ~1e-35. Renders on WebGPU via perturbation: one exact
BigInt-limb reference orbit per dive center, plus cheap f32 delta
orbits per pixel — no float64 in shaders, no libraries, no build
step, nothing to install.

## Usage

Open it in a WebGPU-capable browser (Firefox/Chrome; no server, no
deps — works straight from `file://`):

    xdg-open index.html

Typical workflow:

1. Pick a target from the dropdown (e.g. Seahorse deep) and watch it dive.
2. Adjust Speed (slider or `+`/`-`), click the canvas to zoom one
   octave, Space to pause/resume.
3. ⧉ Copy / ⤳ Go to share or jump to a view; the dive stops at max
   depth (~1e-35) — Reset to go again.

Internals, file map, settings: see `project.md`.
