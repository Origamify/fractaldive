'use strict';
/* ================================================================
 * Present blit WGSL — the overrender zoom cache's presentation
 * pass: crops a center sub-rect of the 2× base frame texture and
 * linear-samples it across the canvas (engine.js commit()).
 *
 * - fullscreen triangle = the perturb kernel's own vertex shape
 * - fragment maps each canvas pixel's uv into the cropped window
 *   [½ − f/2, ½ + f/2] per axis, f = cropFraction = dive's
 *   viewportScale / windowStartScale (1 at window start → 2:1
 *   downsample; 1/BASE_OVERRENDER_FACTOR at window end → 1:1
 *   texel alignment)
 * - NDC y = +1 is the canvas TOP while texture v = 0 is the base's
 *   top row → the vertex stage flips v (uv.y = ½ − y/2); without
 *   it the crop lands mirrored
 * - bilinear filtering smooths in-between scales; at 1:1 the
 *   texel centers align exactly — each canvas pixel reproduces
 *   its base texel
 * ================================================================ */

/* --- kernel source (WGSL, not JavaScript) --- */
const PRESENT_BLIT_WGSL = `
// ================================================================
// Present blit — one textured triangle, one fragment per canvas
// pixel. Bindings: (0) the crop fraction uniform, (1) a linear
// clamp-to-edge sampler, (2) the base frame texture view (2×
// canvas resolution, canvas format).
// ================================================================

struct PresentParams {
  cropFraction: f32, // visible fraction of the base frame per axis, (0, 1] — dive's viewportScale / windowStartScale
};
@group(0) @binding(0) var<uniform> presentParams: PresentParams;
@group(0) @binding(1) var baseSampler: sampler;
@group(0) @binding(2) var baseTexture: texture_2d<f32>;

struct BlitVertexOutput {
  @builtin(position) position: vec4f,
  @location(0) uv: vec2f, // u,v in the base texture — u = 0 left, v = 0 top row (v flipped against NDC y, which points up)
};

@vertex
fn vs(@builtin(vertex_index) vertexIndex: u32) -> BlitVertexOutput {
  var clipPositions = array<vec2f, 3>(vec2f(-1.0, -1.0), vec2f(3.0, -1.0), vec2f(-1.0, 3.0));
  let screenPosition = clipPositions[vertexIndex];
  var output: BlitVertexOutput;
  output.position = vec4f(screenPosition, 0.0, 1.0);
  // x: NDC and u both grow rightward. y: NDC +1 is the canvas top, v = 0 is the base's top row — flip.
  output.uv = vec2f(screenPosition.x * 0.5 + 0.5, 0.5 - screenPosition.y * 0.5);
  return output;
}

// --- constants: crop clamp ---
const MIN_CROP_FRACTION: f32 = 1e-6; // degenerate-crop floor — JS already floors at one texel

@fragment
fn fs(fragmentInput: BlitVertexOutput) -> @location(0) vec4f {
  let cropFraction = clamp(presentParams.cropFraction, MIN_CROP_FRACTION, 1.0);
  let cropOrigin = vec2f(0.5) - vec2f(0.5 * cropFraction);        // center sub-rect
  return textureSample(baseTexture, baseSampler, cropOrigin + fragmentInput.uv * cropFraction);
}
`;
