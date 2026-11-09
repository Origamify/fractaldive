'use strict';
/* ================================================================
 * Engine frame — WebGPUPerturbationEngine's frame pipeline
 * (prototype extension: render()'s internals).
 *
 * One frame = a paced loop of band submits over the base
 * texture's rows (GPU duty-cycle control), top to bottom:
 *   1. buildFixedPointConfig — frame constants (fixed_point.js)
 *   2. pack + upload the input buffers
 *   3. appendReferenceOrbitPass — COMPUTE pass, SKIPPED while the
 *      reference cache holds: the Z table depends only on the
 *      center + build budget, never on zoom scale — a dive reuses
 *      the on-GPU table frame after frame (rebuild on center move
 *      or budget outgrowing headroom). Cached frames are
 *      bit-identical to a fresh build: referenceIndex ≤
 *      iterationsUsed always (rebase only resets it — the budget
 *      break fires before any row past the frame budget is read,
 *      headroom rows stay untouched), and an escape-capped table
 *      is identical at any build budget (both builds stop at the
 *      same cap-crosser row); rides the first band's
 *      command buffer
 *   4. appendFrameRenderPass — RENDER pass per horizontal band:
 *      the same fullscreen triangle, scissored to the band's strip
 *      of the 2× offscreen base texture — one fragment per base
 *      pixel replays the table through the plain-f32 delta
 *      iteration; bands combine bit-identically to one full pass
 *      (position-based mapping, no blending)
 *   5. per band: queue.submit + await onSubmittedWorkDone, then
 *      the duty-cycle sleep — the GPU works in short bursts and
 *      the compositor gets slices between them (GPU_DUTY_CYCLE /
 *      TARGET_BAND_GPU_MS, common.js; band count adaptive from
 *      the last frame's measured gpuMs)
 * - the Z table never returns to the CPU: the render pass reads
 *   the storage buffer the compute pass wrote (implicit barrier
 *   between passes of one command buffer)
 * Load after engine.js (the class), before dive.js.
 * ================================================================ */

/* --- constants: GPU buffer & uniform packing (module-local) --- */
// reference-limb-format constants (REFERENCE_FRACTION_BITS, REFERENCE_LIMB_COUNT) and the frame-config
// builder live in fixed_point.js — this file consumes their output
const FRAME_INPUT_WORD_COUNT = 5;         // f32 words in the frame input buffer: planeLeft + planeTop + pixelStep + subStepX + subStepY (bit patterns, mirrors the kernel's input offsets)
const REFERENCE_INPUT_VECTOR_COUNT = 2;    // u32 limb vectors in the reference input buffer: centerReal + centerImag (stop threshold rides 2N wide after them)
const REFERENCE_TABLE_STRIDE = 4;          // u32 words per table row: reHi | reLo | imHi | imLo (df32 pairs)
const U32_BYTE_WIDTH = 4;                  // bytes per u32 word — every GPU buffer below is sized in these
const FRAME_UNIFORM_WORD_COUNT = 8;        // mirrors struct KernelParams (both kernels — one layout): 2 frame scalars + 2 canvas scalars + 3 palette phases + packed interior
const UNIFORM_PALETTE_PHASE_WORD_INDEX = 4; // KernelParams words 4..6 — the three f32 cosine phases
const UNIFORM_INTERIOR_WORD_INDEX = 7;      // KernelParams word 7 — interior rgb packed r | g<<8 | b<<16
const REFERENCE_TABLE_HEADER_WORDS = 4;   // one vec4u header element: rowCount in .x — padded to 4 words so the table buffer size stays a multiple of the array<vec4u> element stride (16 bytes) WebGPU requires
const REFERENCE_TABLE_BUDGET_HEADROOM = 1024; // extra row cap a table BUILD carries past the frame budget — the schedule adds ~300 iterations per depth decade, so one build covers ~3 decades of zoom before the cache goes stale (rebuild ~every 2 s at max speed, ~every 16 s at default)
const MAX_BANDS = 64;           // band count ceiling — the finest pacing granularity
const DEFAULT_BAND_COUNT = 12;  // bands for the first, unmeasured frame (no lastFrameTimingsMs yet)

/* --- frame pipeline (WebGPUPerturbationEngine.prototype) --- */
Object.assign(WebGPUPerturbationEngine.prototype, {

    /**
      * Returns the band count for the next frame, sized from the
      * last frame's measured GPU ms so a band lands near
      * TARGET_BAND_GPU_MS (common.js). The unmeasured first frame
      * takes DEFAULT_BAND_COUNT; a fast GPU collapses to 1 band —
      * today's single-pass path, no pacing sleep.
      * @returns {number} Band count, clamped to 1..MAX_BANDS.
      */
    bandCountForFrame() {
        const lastGpuMs = this.lastFrameTimingsMs?.gpuMs;
        if (lastGpuMs == null) return DEFAULT_BAND_COUNT;
        return clampI(Math.ceil(lastGpuMs / TARGET_BAND_GPU_MS), 1, MAX_BANDS);
    },

    /**
      * Returns the frame's horizontal partition of the base
      * texture: equal-height strips of ceil(H/bandCount), the last
      * band taking the remainder — every pixel row covered exactly
      * once, no seams. Pure integer math (the partition unit test
      * checks it directly).
      * @param {number} baseHeight - Base texture height in px.
      * @param {number} bandCount - Requested band count (already clamped).
      * @returns {Array<object>} { y0, height } per band, top to bottom.
      */
    bandLayoutForFrame(baseHeight, bandCount) {
        const nominalBandHeight = Math.ceil(baseHeight / bandCount);
        const bands = [];
        for (let y0 = 0; y0 < baseHeight; y0 += nominalBandHeight) {
            bands.push({ y0, height: Math.min(nominalBandHeight, baseHeight - y0) });
        }
        return bands;
    },

    /**
      * Idles out the duty-cycle gap after a band: sleeps
      * bandGpuMs·(1/GPU_DUTY_CYCLE − 1) so other GPU clients (the
      * compositor, other apps) get slices between the frame's
      * bursts. A zero/negative gap (fast band) skips the sleep.
      * @param {number} bandGpuMs - The previous band's measured GPU ms.
      * @returns {Promise<void>} Resolves after the gap.
      */
    async paceAfterBand(bandGpuMs) {
        const idleMs = bandGpuMs * (1 / GPU_DUTY_CYCLE - 1);
        if (idleMs <= 0) return;
        await new Promise((resolve) => setTimeout(resolve, idleMs));
    },

    /**
      * Returns the reused GPU buffers for a frame, reallocating
      * when the table budget outgrows them.
      * - a realloc drops the cached bind groups — they reference
      *   the destroyed buffers; re-created lazily on the next pass
      * - no buffer depends on the pixel count — the render target
      *   is the offscreen base texture
      * @param {number} frameInputWordCount - u32 words of f32 frame inputs.
      * @param {number} referenceInputWordCount - u32 words of reference inputs.
      * @param {number} tableWordCount - u32 words of the df32 Z table (grows with the budget).
      * @returns {object} { uniformBuffer, frameInputBuffer, referenceInputBuffer, referenceTableBuffer }.
      */
    ensureFrameBuffers(frameInputWordCount, referenceInputWordCount, tableWordCount) {
        const areExistingBuffersReusable = this.frameBuffers
            && this.frameBuffers.frameInputWordCount >= frameInputWordCount
            && this.frameBuffers.referenceInputWordCount >= referenceInputWordCount
            && this.frameBuffers.tableWordCount >= tableWordCount;
        if (!areExistingBuffersReusable) {
            if (this.frameBuffers) Object.values(this.frameBuffers.buffers).forEach((buffer) => buffer.destroy());
            this.frameBindGroup = null;   // the cached bind groups reference the destroyed buffers — the next pass re-creates them lazily
            this.referenceBindGroup = null;
            const device = this.gpuDevice;
            const buffers = {
                uniformBuffer: device.createBuffer({ size: FRAME_UNIFORM_WORD_COUNT * U32_BYTE_WIDTH, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST }),
                referenceUniformBuffer: device.createBuffer({ size: FRAME_UNIFORM_WORD_COUNT * U32_BYTE_WIDTH, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST }), // the reference pass's own copy of the shared layout — carries the table BUILD budget (with cache headroom), never the render pass's frame budget
                frameInputBuffer: device.createBuffer({ size: frameInputWordCount * U32_BYTE_WIDTH, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST }),
                referenceInputBuffer: device.createBuffer({ size: referenceInputWordCount * U32_BYTE_WIDTH, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST }),
                // written by the compute pass, read by the render pass — GPU-side only, never copied to the CPU
                referenceTableBuffer: device.createBuffer({ size: tableWordCount * U32_BYTE_WIDTH, usage: GPUBufferUsage.STORAGE }),
            };
            this.frameBuffers = { buffers, frameInputWordCount, referenceInputWordCount, tableWordCount };
        }
        return this.frameBuffers.buffers;
    },

    /**
      * Renders one frame: pack → upload → compute pass (unless
      * the cache holds) → banded render passes → per band:
      * submit, settle, measure, pace — one synchronous task per
      * band, so the writeBuffers queue ahead of the passes.
      * Band-0 clears / cache proofs / resize and device-loss
      * exits: file header + inline at the loop.
      * @param {object} view - Exact dive view (see buildFixedPointConfig).
      * @param {number} maxIter - Iteration budget (from the dive schedule).
      * @returns {Promise<void>} Resolves when the frame is fully drawn.
      */
    async renderPerturbationFrame(view, maxIter) {
        const cpuStartMs = performance.now();
        const baseWidth = this.w * BASE_OVERRENDER_FACTOR;   // the base texture (and the canvas backing) run at 2× the canvas device size
        const baseHeight = this.h * BASE_OVERRENDER_FACTOR;  // dive.js hands in view.s at the base pixel pitch (window-end pitch)
        const frameConfig = buildFixedPointConfig(view, baseWidth, baseHeight);
        const cachedReference = this.referenceCache;
        const isCachedTableValid = cachedReference !== null
            && cachedReference.centerReal === frameConfig.centerReal
            && cachedReference.centerImag === frameConfig.centerImag
            && maxIter <= cachedReference.budget;
        const buildBudget = isCachedTableValid
            ? cachedReference.budget
            : Math.min(MAX_ITER_CAP, maxIter + REFERENCE_TABLE_BUDGET_HEADROOM);
        const buffers = this.ensureFrameBuffers(
            FRAME_INPUT_WORD_COUNT,
            (REFERENCE_INPUT_VECTOR_COUNT + 2) * REFERENCE_LIMB_COUNT,
            REFERENCE_TABLE_HEADER_WORDS + (buildBudget + 1) * REFERENCE_TABLE_STRIDE); // padded vec4u header + (build budget+1) df32 row elements — the buffer always holds the cached table's full width
        if (!this.baseTexture || this.baseTexture.width !== baseWidth || this.baseTexture.height !== baseHeight) {
            this.baseTexture?.destroy();
            this.baseTexture = this.gpuDevice.createTexture({
                size: [baseWidth, baseHeight],
                format: this.canvasFormat,
                usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC, // render target + the commit blit's sample source + the test pages' readback
            });
            this.baseTextureView = null; // the cached view died with the old texture — recreated lazily by the next pass that needs it
            this.commitBindGroup = null; // and so did the commit bind group (it binds that view)
        }
        const frameBaseTexture = this.baseTexture; // mid-band resize sentinel — resize() nulls/replaces the field
        this.gpuDevice.queue.writeBuffer(buffers.uniformBuffer, 0, this.packFrameUniformWords(maxIter));
        this.gpuDevice.queue.writeBuffer(buffers.frameInputBuffer, 0, this.packFrameInputWords(frameConfig));
        const bands = this.bandLayoutForFrame(baseHeight, this.bandCountForFrame());
        let firstBandSubmitMs = 0;
        let previousBandGpuMs = 0;
        let bandGpuMsSum = 0;
        for (let bandIndex = 0; bandIndex < bands.length; bandIndex++) {
            if (this.baseTexture !== frameBaseTexture) return; // a resize orphaned the target mid-bands — abandon quietly, never submit into a destroyed texture
            if (bandIndex > 0) {
                await this.paceAfterBand(previousBandGpuMs); // the duty-cycle gap — other GPU clients get their slice
                if (this.baseTexture !== frameBaseTexture) return; // the sleep is an event-loop turn — a resize may land in it; abandon before the null texture throws
            }
            const commandEncoder = this.gpuDevice.createCommandEncoder();
            if (bandIndex === 0 && !isCachedTableValid) {
                this.gpuDevice.queue.writeBuffer(buffers.referenceInputBuffer, 0, this.packReferenceInputWords(frameConfig));
                this.gpuDevice.queue.writeBuffer(buffers.referenceUniformBuffer, 0, this.packReferenceUniformWords(buildBudget));
                this.appendReferenceOrbitPass(commandEncoder); // rides band 0's buffer — the FIFO queue lands the table before any fragment reads it
                this.referenceCache = { centerReal: frameConfig.centerReal, centerImag: frameConfig.centerImag, budget: buildBudget };
            }
            this.appendFrameRenderPass(commandEncoder, bands[bandIndex], bandIndex === 0);
            const bandSubmitMs = performance.now();
            this.gpuDevice.queue.submit([commandEncoder.finish()]);
            if (bandIndex === 0) firstBandSubmitMs = bandSubmitMs;
            await this.gpuDevice.queue.onSubmittedWorkDone(); // settles the band — also rejects on device loss, keeping honest death inside render's catch
            previousBandGpuMs = performance.now() - bandSubmitMs;
            bandGpuMsSum += previousBandGpuMs;
        }
        // frame-time split (1 decimal, ms): CPU = prologue through band 0's submit; GPU = the bands' settle waits summed
        this.lastFrameTimingsMs = {
            cpuMs: firstBandSubmitMs - cpuStartMs,
            gpuMs: bandGpuMsSum,
        };
        this.lastFrameBandCount = bands.length;
        // promotion is dive.js's call: its settle handler runs commit() now (the queue is empty) — the engine holds no presentation state
    },

    /**
      * Packs the delta kernel's five f32 frame constants
      * (buildFixedPointConfig output) as bit patterns in the
      * kernel's input layout: planeLeft, planeTop, pixelStep,
      * then the two supersampling sub-steps.
      * @param {object} frameConfig - buildFixedPointConfig output for this frame.
      * @returns {Uint32Array} FRAME_INPUT_WORD_COUNT words, ready to upload.
      */
    packFrameInputWords(frameConfig) {
        const frameWords = new Float32Array([
            frameConfig.planeLeftF32,
            frameConfig.planeTopF32,
            frameConfig.pixelStepF32,
            frameConfig.subStepXF32,
            frameConfig.subStepYF32,
        ]);
        return new Uint32Array(frameWords.buffer);
    },

    /**
      * Packs the reference pass's inputs: the frame center at
      * reference limb width (Cr, Ci — N limbs each), then the stop threshold (2N
      * limbs, |Z|² cap that ends the table).
      * @param {object} frameConfig - buildFixedPointConfig output for this frame.
      * @returns {Uint32Array} (REFERENCE_INPUT_VECTOR_COUNT + 2) × N words, ready to upload.
      */
    packReferenceInputWords(frameConfig) {
        const inputWords = new Uint32Array((REFERENCE_INPUT_VECTOR_COUNT + 2) * REFERENCE_LIMB_COUNT);
        inputWords.set(bigIntToLimbs(frameConfig.centerReal, REFERENCE_LIMB_COUNT), 0);
        inputWords.set(bigIntToLimbs(frameConfig.centerImag, REFERENCE_LIMB_COUNT), REFERENCE_LIMB_COUNT);
        inputWords.set(bigIntToLimbs(frameConfig.referenceStopThreshold, 2 * REFERENCE_LIMB_COUNT), 2 * REFERENCE_LIMB_COUNT);
        return inputWords;
    },

    /**
      * Packs the shared uniform (struct KernelParams in both
      * kernels): frame scalars (the base texture's dimensions — the
      * render target the pass covers), then palette phases and the
      * packed interior rgb.
      * @param {number} maxIter - Iteration budget (from the dive schedule).
      * @returns {Uint32Array} FRAME_UNIFORM_WORD_COUNT words.
      */
    packFrameUniformWords(maxIter) {
        const uniformWords = new Uint32Array(FRAME_UNIFORM_WORD_COUNT);
        uniformWords.set([REFERENCE_FRACTION_BITS, maxIter, this.w * BASE_OVERRENDER_FACTOR, this.h * BASE_OVERRENDER_FACTOR]);
        new Float32Array(uniformWords.buffer).set(PALETTE.ph, UNIFORM_PALETTE_PHASE_WORD_INDEX);
        uniformWords[UNIFORM_INTERIOR_WORD_INDEX] =
            PALETTE.interior[0] | (PALETTE.interior[1] << 8) | (PALETTE.interior[2] << 16); // little-endian byte order
        return uniformWords;
    },

    /**
     * Packs the reference pass's own copy of the shared uniform
     * (struct KernelParams layout): only referenceFractionBits and the
     * table BUILD budget matter to the compute shader — the row
     * cap the table is built to, carrying headroom past the render
     * budget so the cached table serves later frames. The render
     * pass keeps its own uniform at the frame budget.
     * @param {number} buildBudget - Row cap for this table build.
     * @returns {Uint32Array} FRAME_UNIFORM_WORD_COUNT words.
     */
    packReferenceUniformWords(buildBudget) {
        const uniformWords = new Uint32Array(FRAME_UNIFORM_WORD_COUNT);
        uniformWords.set([REFERENCE_FRACTION_BITS, buildBudget, 0, 0, 0, 0, 0, 0]);
        return uniformWords;
    },

    /**
      * Appends the reference-orbit compute pass: one invocation
      * builds the Z table into the shared storage buffer.
      * - precedes the render pass in the same command buffer → the
      *   implicit barrier guarantees completion before any read
      * - skipped entirely on cache-valid frames (the only pass a
      *   cached frame omits)
      * @param {object} commandEncoder - The frame's encoder.
      * @returns {void}
      */
    appendReferenceOrbitPass(commandEncoder) {
        if (!this.referenceBindGroup) { // created once per buffer lifetime — ensureFrameBuffers nulls it on realloc
            const buffers = this.frameBuffers.buffers;
            this.referenceBindGroup = this.gpuDevice.createBindGroup({
                layout: this.referencePipeline.getBindGroupLayout(0),
                entries: [
                    { binding: 0, resource: { buffer: buffers.referenceUniformBuffer } },
                    { binding: 1, resource: { buffer: buffers.referenceInputBuffer } },
                    { binding: 2, resource: { buffer: buffers.referenceTableBuffer } },
                ],
            });
        }
        const computePass = commandEncoder.beginComputePass();
        computePass.setPipeline(this.referencePipeline);
        computePass.setBindGroup(0, this.referenceBindGroup);
        computePass.dispatchWorkgroups(1); // one orbit per frame — the whole expensive half of perturbation
        computePass.end();
    },

    /**
      * Appends one band's render pass: the fullscreen triangle,
      * scissored to the band's strip — out-of-band fragments are
      * discarded before the shader runs.
      * - band 0 clears the target, later bands load (a pass's
      *   clear ignores the scissor)
      * - nothing samples the base texture mid-render
      * @param {object} commandEncoder - The band's encoder.
      * @param {object} band - The band's strip ({ y0, height } from bandLayoutForFrame).
      * @param {boolean} isFirstBand - True for band 0 (its pass clears the target).
      * @returns {void}
      */
    appendFrameRenderPass(commandEncoder, band, isFirstBand) {
        if (!this.frameBindGroup) { // created once per buffer lifetime — ensureFrameBuffers nulls it on realloc
            const buffers = this.frameBuffers.buffers;
            this.frameBindGroup = this.gpuDevice.createBindGroup({
                layout: this.framePipeline.getBindGroupLayout(0),
                entries: [
                    { binding: 0, resource: { buffer: buffers.uniformBuffer } },
                    { binding: 1, resource: { buffer: buffers.frameInputBuffer } },
                    { binding: 2, resource: { buffer: buffers.referenceTableBuffer } },
                ],
            });
        }
        if (!this.baseTextureView) { // recreated with the texture — resize or a render size change nulls it
            this.baseTextureView = this.baseTexture.createView();
        }
        const renderPass = commandEncoder.beginRenderPass({
            colorAttachments: [{
                view: this.baseTextureView,
                loadOp: isFirstBand ? 'clear' : 'load', // the whole attachment loads/clears — later bands must keep band 0's rows
                storeOp: 'store',
                clearValue: { r: 0, g: 0, b: 0, a: 1 }, // read only by the first band ('load' passes ignore it)
            }],
        });
        renderPass.setPipeline(this.framePipeline);
        renderPass.setBindGroup(0, this.frameBindGroup);
        renderPass.setScissorRect(0, band.y0, this.baseTexture.width, band.height);
        renderPass.draw(3);
        renderPass.end();
    },
});
