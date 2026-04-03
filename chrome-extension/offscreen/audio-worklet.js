class ChunkAccumulator extends AudioWorkletProcessor {
  constructor(options) {
    super();
    const chunkSeconds = (options.processorOptions && options.processorOptions.chunkSeconds) || 8;
    this._chunkSamples = Math.ceil(chunkSeconds * sampleRate);
    this._buffer = new Float32Array(this._chunkSamples);
    this._writeIndex = 0;
    this._silenceCheckSamples = Math.ceil(3 * sampleRate); // first 3 seconds
    this._silenceRmsSum = 0;
    this._silenceSampleCount = 0;
    this._silenceChecked = false;
    this._captureActive = true;

    // Listen for capture toggle from offscreen.js
    this.port.onmessage = (e) => {
      if (e.data.type === "SET_CAPTURE_ACTIVE") {
        this._captureActive = e.data.active;
        // Reset buffer on any toggle to avoid stale/mixed samples
        this._writeIndex = 0;
        this._buffer.fill(0);
      }
    };
  }

  process(inputs) {
    const input = inputs[0];
    if (!input || !input[0]) return true;

    const channelData = input[0];

    // Silence detection on first 3 seconds
    if (!this._silenceChecked) {
      for (let i = 0; i < channelData.length && this._silenceSampleCount < this._silenceCheckSamples; i++) {
        this._silenceRmsSum += channelData[i] * channelData[i];
        this._silenceSampleCount++;
      }
      if (this._silenceSampleCount >= this._silenceCheckSamples) {
        this._silenceChecked = true;
        const rms = Math.sqrt(this._silenceRmsSum / this._silenceSampleCount);
        if (rms < 0.001) {
          this.port.postMessage({ type: "silence" });
        }
      }
    }

    // Only accumulate when capture is active
    if (!this._captureActive) return true;

    // Accumulate samples
    for (let i = 0; i < channelData.length; i++) {
      this._buffer[this._writeIndex++] = channelData[i];
      if (this._writeIndex >= this._chunkSamples) {
        // Send the full chunk
        this.port.postMessage({
          type: "chunk",
          samples: this._buffer.slice(),
          sampleRate: sampleRate,
        });
        this._writeIndex = 0;
      }
    }

    return true;
  }
}

registerProcessor("chunk-accumulator", ChunkAccumulator);
