/**
 * AudioWorklet processor for resampling audio to 16kHz PCM16LE
 *
 * Converts browser's native sample rate (typically 44.1/48kHz) to 16kHz
 * using linear interpolation, then converts float32 to int16 PCM.
 *
 * Accumulates 100ms chunks (1600 samples at 16kHz) before posting to main thread.
 */

class ResamplerProcessor extends AudioWorkletProcessor {
  constructor() {
    super();

    // Output sample rate (16kHz for ASR)
    this.outputSampleRate = 16000;

    // Input sample rate from AudioContext
    this.inputSampleRate = sampleRate; // Global from AudioWorkletGlobalScope

    // Resampling ratio: number of input samples per output sample
    this.ratio = this.inputSampleRate / this.outputSampleRate;

    // Output buffer accumulator (100ms = 1600 samples at 16kHz)
    this.targetSamples = Math.floor(this.outputSampleRate * 0.1); // 100ms
    this.outputBuffer = new Float32Array(this.targetSamples);
    this.outputIndex = 0;

    // Fractional position in input stream (carries across process() calls)
    // Advances by `ratio` for each output sample produced
    this.srcPos = 0;

    // Stop flag
    this.shouldStop = false;

    // Debug counter
    this.processCount = 0;
    this.chunksSent = 0;

    console.log(
      '[Worklet] Initialized - Input:', this.inputSampleRate,
      'Hz, Output:', this.outputSampleRate,
      'Hz, Ratio:', this.ratio.toFixed(2),
      'Target samples per chunk:', this.targetSamples
    );

    // Listen for messages from main thread
    this.port.onmessage = (event) => {
      if (event.data === 'stop') {
        console.log('[Worklet] Stop signal received');
        this.shouldStop = true;
      }
    };
  }

  /**
   * Process audio samples (called automatically by AudioWorklet)
   * @param {Float32Array[][]} inputs - Input audio channels
   * @param {Float32Array[][]} outputs - Output audio channels
   * @param {Object} parameters - Audio parameters (unused)
   * @returns {boolean} - Return true to keep processor alive
   */
  process(inputs, outputs, parameters) {
    if (this.shouldStop) {
      return false; // Stop processing
    }

    this.processCount++;
    if (this.processCount === 1) {
      console.log('[Worklet] First process() call - audio pipeline is active!');
    }

    const input = inputs[0];
    if (!input || !input.length) {
      return true; // Keep alive even with no input
    }

    // Get first channel (mono)
    const ch = input[0];
    if (!ch || ch.length === 0) {
      return true;
    }

    const N = ch.length;

    // Write silence to output to keep graph active
    if (outputs[0] && outputs[0][0]) {
      outputs[0][0].fill(0);
    }

    // Downsample: step through input at `ratio` spacing per output sample.
    // srcPos is a fractional index into the current input block.
    // For each output sample, we read the input at position srcPos using
    // linear interpolation, then advance srcPos by ratio.
    while (this.srcPos < N) {
      const idx = Math.floor(this.srcPos);
      const frac = this.srcPos - idx;

      // Linear interpolation between ch[idx] and ch[idx+1]
      const s0 = ch[idx];
      const s1 = idx + 1 < N ? ch[idx + 1] : ch[idx];
      const sample = s0 + frac * (s1 - s0);

      // Clip to [-1, 1] and add to output buffer
      this.outputBuffer[this.outputIndex++] = Math.max(-1.0, Math.min(1.0, sample));

      // Send chunk when we've accumulated enough samples
      if (this.outputIndex >= this.targetSamples) {
        this.sendChunk();
      }

      this.srcPos += this.ratio;
    }

    // Carry forward fractional remainder for next process() call
    this.srcPos -= N;

    return true; // Keep processor alive
  }

  /**
   * Convert accumulated float32 samples to int16 PCM and send to main thread
   */
  sendChunk() {
    // Convert float32 [-1, 1] to int16 [-32768, 32767]
    const pcm16 = new Int16Array(this.outputIndex);
    for (let i = 0; i < this.outputIndex; i++) {
      const sample = this.outputBuffer[i];
      pcm16[i] = Math.round(sample * 32767);
    }

    // Send to main thread with zero-copy transfer
    const buffer = pcm16.buffer;
    this.port.postMessage(buffer, [buffer]);

    this.chunksSent++;
    if (this.chunksSent === 1 || this.chunksSent % 50 === 0) {
      console.log(`[Worklet] Sent chunk #${this.chunksSent}, size: ${buffer.byteLength} bytes`);
    }

    // Reset accumulator
    this.outputIndex = 0;
  }
}

// Register the processor
registerProcessor('resampler-worklet', ResamplerProcessor);
