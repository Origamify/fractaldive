'use strict';
/* ================================================================
 * Engine — the WebGPU perturbation engine (class
 * WebGPUPerturbationEngine).
 *
 * Owns rendering and nothing else: never reads dive state, never
 * touches the footer — dive.js drives it only through the frozen
 * contract init / resize / render / resetRef / commit
 * (+ .available / .w / .h). No CPU fallback:
 * no WebGPU → init() throws → dive.js shows the failure notice.
 *
 * This file: the class, its lifecycle (init / resize), the
 * contract API (render / commit) and pipeline compilation. The
 * frame pipeline behind render() — bands, buffers, packing, the
 * two passes — lives in engine_frame.js (prototype extension);
 * load it right after this file.
 *
 * Presentation (render/presentation split — docs/ARCHITECTURE.md):
 * - render() draws each base frame 2×-resolution offscreen; the
 *   canvas backing store runs at that same 2× resolution
 * - commit() blits the FULL base into the canvas (crop fraction 1,
 *   present_blit_wgsl.js) — dive.js calls it exactly once per
 *   settled render, when the queue is empty (single-flight), so
 *   the blit executes immediately and can never queue behind a
 *   render (the single-queue FIFO stall, docs/ARCHITECTURE.md)
 * - the zoom between settles is presenter.js's CSS transform on
 *   the canvas element (the compositor re-samples the full base
 *   every frame) — the engine never submits presentation work
 *   while a render runs
 * - dive.js owns the window; the engine renders what it is handed
 *   and commits what settled
 * ================================================================ */

/* --- constants: GPU buffer packing (module-local) --- */
const BLIT_UNIFORM_BYTES = 16; // the commit blit's crop-fraction uniform buffer — one f32 padded to the comfortable 16-byte size (always written 1.0: the full base; the presenter's CSS transform does the cropping)

/* --- engine: the public API contract (dive.js constructs & drives it) --- */

class WebGPUPerturbationEngine {
    constructor() {
        this.available = false;                  // init() succeeded — dive.js gates rendering on this
        this.w = 0;                              // canvas width in device px
        this.h = 0;                              // canvas height in device px
        this.canvas = null;                      // render target handed in by init()
        this.canvasContext = null;               // its WebGPU context — the render pass draws through it
        this.canvasFormat = null;                // preferred 8-bit canvas format (fixed per session)
        this.gpuDevice = null;                   // WebGPU device — null until init() succeeds
        this.referencePipeline = null;           // compute pipeline: the exact Z-table builder (constant limb tier)
        this.framePipeline = null;               // render pipeline: the f32 delta kernel
        this.blitPipeline = null;                // render pipeline: the present crop-blit (present_blit_wgsl.js)
        this.frameBuffers = null;                // reused uniform/input/table GPU buffers
        this.referenceCache = null;              // { centerReal, centerImag, budget } the on-GPU Z table was built for — the table survives frames while the dive center holds and the frame budget stays under the build budget (see engine_frame.js)
        this.frameBindGroup = null;               // cached render-pass bind group — valid until ensureFrameBuffers reallocates the buffers
        this.referenceBindGroup = null;           // cached reference-pass bind group — same lifetime as frameBindGroup
        this.baseTexture = null;                    // the BASE frame texture (offscreen, 2× canvas device size): every render targets it, commit() copies it to the canvas at settle — nothing samples it mid-render, so one texture suffices
        this.baseTextureView = null;                // cached view of the base texture — valid until the texture is (re)created (render pass + commit blit share it)
        this.commitBindGroup = null;                // cached commit-blit bind group (crop uniform + sampler + base view) — same lifetime as the view
        this.blitSampler = null;                    // linear clamp-to-edge sampler for the commit blit (fixed per session)
        this.blitCropUniformBuffer = null;          // the blit's crop-fraction uniform (one f32, written 1.0 per commit)
        this.lastFrameTimingsMs = null;          // CPU/GPU split of the last settled frame — dive.js reads gpuMs for the HUD, the next frame's band sizing reads it too
        this.lastFrameBandCount = 0;             // bands the last settled frame rendered in — dive.js surfaces it in the HUD next to the GPU time
    }

    /* --- contract API (dive.js calls only these) --- */

    /**
      * Initializes the engine on a canvas: WebGPU up, both kernels
      * compiled (a broken shader dies here, not at first render).
      * No fallback — WebGPU or a thrown error; dive.js shows the
      * failure notice on throw.
      * @param {HTMLCanvasElement} canvas - The render target (#cv).
      * @returns {Promise<string>} Human-readable engine details.
      * @throws {Error} WebGPU unavailable, device request fails, or a kernel miscompiles.
      */
    async init(canvas) {
        this.canvas = canvas;
        this.w = 0; // canvas device px, sized by the first resize() — initApp resizes before the first tick (never render before it)
        this.h = 0;
        if (!('gpu' in navigator)) throw new Error('navigator.gpu missing');
        const adapter = await navigator.gpu.requestAdapter();
        if (!adapter) throw new Error('no adapter');
        this.gpuDevice = await adapter.requestDevice({
            // the base texture and the canvas backing run at 2× the canvas
            // device size — ask for the adapter's texture-dimension ceiling
            // (the default device limit, 8192, would crop a DPR-2 4K+ viewport)
            requiredLimits: { maxTextureDimension2D: adapter.limits.maxTextureDimension2D },
        });
        this.canvasFormat = navigator.gpu.getPreferredCanvasFormat();
        this.canvasContext = canvas.getContext('webgpu');
        this.canvasContext.configure({ device: this.gpuDevice, format: this.canvasFormat, alphaMode: 'opaque', usage: GPUTextureUsage.RENDER_ATTACHMENT }); // RENDER_ATTACHMENT: commit() blits the canvas through a render pass
        const tierSplice = (source) => source.replaceAll('__LIMB_COUNT__', String(REFERENCE_LIMB_COUNT));
        this.referencePipeline = await this.compileReferencePipeline(tierSplice(REFERENCE_KERNEL_WGSL));
        this.framePipeline = await this.compileFramePipeline(PERTURB_KERNEL_WGSL); // pure f32 — no tier splice
        this.blitPipeline = await this.compileFramePipeline(PRESENT_BLIT_WGSL, 'present-crop-blit');   // same vs/fs entries + canvas target — the frame-compile path serves it as-is
        this.blitSampler = this.gpuDevice.createSampler({
            magFilter: 'linear', minFilter: 'linear',
            addressModeU: 'clamp-to-edge', addressModeV: 'clamp-to-edge',
        });
        this.blitCropUniformBuffer = this.gpuDevice.createBuffer({
            size: BLIT_UNIFORM_BYTES,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        this.available = true;
        return `WebGPU perturbation engine (${BASE_OVERRENDER_FACTOR}× overrender base frames, f32 delta over ${REFERENCE_LIMB_COUNT}-limb reference)`;
    }

    /**
      * Resizes the render target; a no-op when the size already
      * matches.
      * - the canvas backing store runs at BASE_OVERRENDER_FACTOR ×
      *   the handed size (base frame resolution — the presenter's
      *   CSS transform scales the bitmap to the viewport)
      * - no buffer depends on the pixel count — the reused input
      *   buffers stay valid across resizes
      * @param {number} canvasWidthPx - New width in device px.
      * @param {number} canvasHeightPx - New height in device px.
      * @returns {void}
      */
    resize(canvasWidthPx, canvasHeightPx) {
        if (canvasWidthPx === this.w && canvasHeightPx === this.h) return;
        this.w = canvasWidthPx;
        this.h = canvasHeightPx;
        this.baseTexture?.destroy(); // the single base texture is wrong-sized for the new canvas
        this.baseTexture = null;
        this.baseTextureView = null;  // the cached view died with the texture
        this.commitBindGroup = null;  // so did the commit bind group (it binds that view)
        if (this.canvas) {
            this.canvas.width = canvasWidthPx * BASE_OVERRENDER_FACTOR; // backing at base resolution — the presenter samples it down
            this.canvas.height = canvasHeightPx * BASE_OVERRENDER_FACTOR;
        }
        // configuration survives size changes per spec — the reconfigure is belt-and-braces,
        // guarded because resize can fire before async init() finishes (canvasContext not yet set)
        if (this.canvasContext) this.canvasContext.configure({ device: this.gpuDevice, format: this.canvasFormat, alphaMode: 'opaque', usage: GPUTextureUsage.RENDER_ATTACHMENT });
    }

    /**
      * Drops the cached reference Z table. The cache key would
      * drop it anyway on any center change — this is the explicit
      * contract call for a view jump (goToView); clickToZoom
      * self-invalidates via the key.
      * @returns {void}
      */
    resetRef() { this.referenceCache = null; }

    /**
      * Renders one base frame through the two kernels into the 2×
      * base texture (the pipeline steps: engine_frame.js's header).
      * The caller (dive.js) owns the window semantics: view.s at
      * the base pixel PITCH (window-start / OVERRENDER_FACTOR) and
      * a budget for that same end-of-window scale. A kernel failure
      * marks the engine dead rather than rendering something wrong.
      * @param {object} view - Exact dive view at the base pixel pitch (see buildFixedPointConfig).
      * @param {number} maxIter - Iteration budget (dive schedule, at the base pitch scale).
      * @returns {Promise<void>} Resolves when the frame is fully drawn.
      * @throws {Error} A pass failed (engine goes unavailable).
      */
    async render(view, maxIter) {
        if (!this.available) throw new Error('engine not initialized');
        try {
            await this.renderPerturbationFrame(view, maxIter);
        } catch (error) {
            this.available = false; // dive.js notices next tick and stops the loop — honest death, no wrong pixels
            throw error;
        }
    }

    /**
      * Blits the latest settled base frame into the canvas — the
      * FULL 2× base at crop fraction 1: the zoom's crop is the
      * presenter's CSS transform (presenter.js), not this blit.
      * - dive.js calls this exactly once per settled render, with
      *   the queue empty (single-flight) — the blit executes
      *   immediately, never queues behind a render (the FIFO
      *   stall this split exists to kill, docs/ARCHITECTURE.md)
      * - a stale settle after a resize mid-flight holds the last
      *   committed frame (size check below)
      * @returns {void}
      */
    commit() {
        if (!this.available || !this.canvasContext || !this.w) return;
        const baseTexture = this.baseTexture;
        const baseWidth = this.w * BASE_OVERRENDER_FACTOR;
        const baseHeight = this.h * BASE_OVERRENDER_FACTOR;
        if (!baseTexture || baseTexture.width !== baseWidth || baseTexture.height !== baseHeight) return; // stale settle after a resize mid-flight — hold the last committed frame
        if (!this.baseTextureView) this.baseTextureView = baseTexture.createView(); // recreated with the texture
        if (!this.commitBindGroup) { // same lifetime as the view
            this.commitBindGroup = this.gpuDevice.createBindGroup({
                layout: this.blitPipeline.getBindGroupLayout(0),
                entries: [
                    { binding: 0, resource: { buffer: this.blitCropUniformBuffer } },
                    { binding: 1, resource: this.blitSampler },
                    { binding: 2, resource: this.baseTextureView },
                ],
            });
        }
        this.gpuDevice.queue.writeBuffer(this.blitCropUniformBuffer, 0, new Float32Array([1])); // full base — the presenter's transform crops
        const commandEncoder = this.gpuDevice.createCommandEncoder();
        const blitPass = commandEncoder.beginRenderPass({
            colorAttachments: [{
                view: this.canvasContext.getCurrentTexture().createView(),
                loadOp: 'clear',
                storeOp: 'store',
                clearValue: { r: 0, g: 0, b: 0, a: 1 },
            }],
        });
        blitPass.setPipeline(this.blitPipeline);
        blitPass.setBindGroup(0, this.commitBindGroup);
        blitPass.draw(3);
        blitPass.end();
        this.gpuDevice.queue.submit([commandEncoder.finish()]);
    }

    /* --- kernel compilation (init's internals) --- */

    /**
      * Returns the compiled compute pipeline for the reference
      * Z-table builder. Async on purpose: an invalid pipeline does
      * not throw synchronously, but createComputePipelineAsync
      * rejects — a broken shader dies at init instead of at the
      * first render.
      * @param {string} kernelSource - Tier-spliced reference WGSL.
      * @returns {Promise<object>} GPUComputePipeline.
      */
    async compileReferencePipeline(kernelSource) {
        const shaderModule = this.gpuDevice.createShaderModule({ code: kernelSource, label: 'perturbation-reference-orbit' });
        return await this.gpuDevice.createComputePipelineAsync({
            layout: 'auto',
            compute: { module: shaderModule, entryPoint: 'main' },
        });
    }

    /**
      * Returns the compiled render pipeline for a fullscreen-
      * triangle fragment kernel (the delta kernel and the present
      * blit share the shape).
      * @param {string} kernelSource - Render WGSL (no splice points).
      * @param {string} [shaderLabel='perturbation-delta-kernel'] - Shader module label for debug logs.
      * @returns {Promise<object>} GPURenderPipeline.
      */
    async compileFramePipeline(kernelSource, shaderLabel = 'perturbation-delta-kernel') {
        const shaderModule = this.gpuDevice.createShaderModule({ code: kernelSource, label: shaderLabel });
        return await this.gpuDevice.createRenderPipelineAsync({
            layout: 'auto',
            vertex: { module: shaderModule, entryPoint: 'vs' },
            fragment: { module: shaderModule, entryPoint: 'fs', targets: [{ format: this.canvasFormat }] },
            primitive: { topology: 'triangle-list' },
        });
    }
}
