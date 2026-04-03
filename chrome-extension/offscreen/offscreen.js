const BASE_URL =
  "https://mollysandler--sports-translation-api-translatorservice-f-6a7378.modal.run";

let audioContext = null;
let mediaStream = null;
let workletNode = null;
let playbackCtx = null;
let sessionId = null;
let sourceLang = "en";
let targetLang = "hi";
let chunkCount = 0;

// Global AbortController — aborts ALL in-flight chunk requests on stop
let globalAbort = new AbortController();

// -- Buffered playback state --------------------------------------------------
let playbackState = "idle";           // "idle" | "buffering" | "playing"
let firstChunkSentAt = null;
let firstSegmentReceivedAt = null;
let initialLatencyMs = null;
let captureStartedAt = null;          // Date.now() when capture began
let playbackStartedAt = null;         // Date.now() when audio playback started
let nextPlayTime = 0;                 // AudioContext timeline cursor for gapless scheduling
let playbackStartedCtxTime = null;    // AudioContext time when playback started
let decodedQueue = [];                // pre-decoded AudioBuffers ready to schedule
let bufferedDuration = 0;             // total seconds of decoded audio in buffer
let captionQueue = [];                // captions held during buffering, released on play
let segmentCount = 0;                 // total segments scheduled
let bufferFallbackTimer = null;       // fallback to start playing with partial buffer
let totalAudioSentSec = 0;           // cumulative seconds of audio sent to backend
const TARGET_BUFFER_SEC = 16;         // buffer enough audio to survive the replay→capture transition gap

// Single reusable context for decoding audio during the buffering phase.
let decodeCtx = null;

// -- WAV encoding (ported from AudioInput.jsx) --------------------------------

function encodeWav(samples, sampleRate) {
  const buffer = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(buffer);
  const write = (off, str) => {
    for (let i = 0; i < str.length; i++) view.setUint8(off + i, str.charCodeAt(i));
  };
  write(0, "RIFF");
  view.setUint32(4, 36 + samples.length * 2, true);
  write(8, "WAVE");
  write(12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  write(36, "data");
  view.setUint32(40, samples.length * 2, true);
  let offset = 44;
  for (let i = 0; i < samples.length; i++, offset += 2) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7fff, true);
  }
  return buffer;
}

// -- SSE parser (ported from AudioInput.jsx) ----------------------------------

async function* readSseEvents(response) {
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop();
    for (const line of lines) {
      if (line.startsWith("data: ")) {
        try {
          yield JSON.parse(line.slice(6));
        } catch (_) {}
      }
    }
  }
}

// -- Resample to 16kHz and send chunk -----------------------------------------

async function sendPcmChunk(samples, nativeSR) {
  if (!sessionId || samples.length < 100 || globalAbort.signal.aborted) return;

  chunkCount++;
  const chunkNum = chunkCount;
  console.log(`[offscreen] Sending chunk #${chunkNum}, ${samples.length} samples at ${nativeSR}Hz`);

  // First chunk: transition to buffering state
  if (!captureStartedAt) {
    captureStartedAt = Date.now();
    playbackState = "buffering";
    // Fallback: start playing after 45s even if buffer target not reached
    bufferFallbackTimer = setTimeout(() => {
      if (playbackState === "buffering" && decodedQueue.length > 0) {
        console.log("[offscreen] Buffer fallback: starting with partial buffer");
        startPlayback();
      }
    }, 45000);
  }

  const chunkDurationSec = samples.length / nativeSR;
  totalAudioSentSec += chunkDurationSec;

  let wavBlob;
  try {
    const offlineCtx = new OfflineAudioContext(
      1,
      Math.ceil((samples.length * 16000) / nativeSR),
      16000
    );
    const buf = offlineCtx.createBuffer(1, samples.length, nativeSR);
    buf.getChannelData(0).set(samples);
    const srcNode = offlineCtx.createBufferSource();
    srcNode.buffer = buf;
    srcNode.connect(offlineCtx.destination);
    srcNode.start();
    const resampled = await offlineCtx.startRendering();
    const pcm16k = resampled.getChannelData(0);
    wavBlob = new Blob([encodeWav(pcm16k, 16000)], { type: "audio/wav" });
    console.log(`[offscreen] Chunk #${chunkNum} encoded to ${wavBlob.size} bytes WAV`);
  } catch (err) {
    console.error("[offscreen] Resample/encode failed:", err);
    return;
  }

  const formData = new FormData();
  formData.append("audio", wavBlob, "chunk.wav");
  formData.append("session_id", sessionId);
  formData.append("source_lang", sourceLang);
  formData.append("target_lang", targetLang);

  // Record when the first chunk is sent for latency measurement
  if (chunkCount === 1) {
    firstChunkSentAt = Date.now();
  }

  let response;
  // Abort on either: 60s per-chunk timeout OR global stop signal
  const chunkController = new AbortController();
  const chunkTimeout = setTimeout(() => chunkController.abort(), 60000);
  globalAbort.signal.addEventListener("abort", () => chunkController.abort(), { once: true });
  try {
    console.log(`[offscreen] POSTing chunk #${chunkNum} to /translate-live`);
    response = await fetch(`${BASE_URL}/translate-live`, {
      method: "POST",
      body: formData,
      signal: chunkController.signal,
    });
    console.log(`[offscreen] Chunk #${chunkNum} response status: ${response.status}`);
  } catch (err) {
    clearTimeout(chunkTimeout);
    const errorMsg = err.name === "AbortError"
      ? "Translation request timed out. The server may be overloaded."
      : "Could not reach the translation server.";
    console.error("[offscreen] translate-live fetch error:", err);
    chrome.runtime.sendMessage({
      type: "CHUNK_ERROR",
      error: errorMsg,
    }).catch(() => {});
    return;
  }
  clearTimeout(chunkTimeout);

  if (!response.ok) {
    const text = await response.text().catch(() => "");
    console.error(`[offscreen] Backend error ${response.status}: ${text}`);
    chrome.runtime.sendMessage({
      type: "CHUNK_ERROR",
      error: `Server error (${response.status}). Translation may be interrupted.`,
    }).catch(() => {});
    return;
  }

  // SSE stall detection: if no event received within 45s, report error
  let stallTimer = setTimeout(() => {
    console.error("[offscreen] SSE stall detected for chunk #" + chunkNum);
    chrome.runtime.sendMessage({
      type: "CHUNK_ERROR",
      error: "Translation server stopped responding.",
    }).catch(() => {});
  }, 45000);

  try {
    for await (const item of readSseEvents(response)) {
      // Bail out if stop was called while reading SSE events
      if (globalAbort.signal.aborted) {
        console.log(`[offscreen] Chunk #${chunkNum} SSE loop aborted by stop`);
        break;
      }
      // Reset stall timer on each event
      clearTimeout(stallTimer);
      stallTimer = setTimeout(() => {
        console.error("[offscreen] SSE stall detected for chunk #" + chunkNum);
        chrome.runtime.sendMessage({
          type: "CHUNK_ERROR",
          error: "Translation server stopped responding.",
        }).catch(() => {});
      }, 45000);

      console.log("[offscreen] SSE event:", item.type, item);
      if (item.type === "segment") {
        // Measure first-segment latency for video sync
        if (!firstSegmentReceivedAt && firstChunkSentAt) {
          firstSegmentReceivedAt = Date.now();
          initialLatencyMs = firstSegmentReceivedAt - firstChunkSentAt;
          console.log("[offscreen] First segment latency:", initialLatencyMs, "ms");
          chrome.runtime.sendMessage({
            type: "FIRST_SEGMENT_LATENCY",
            latencyMs: initialLatencyMs,
          }).catch(() => {});
        }
        // Pre-decode immediately on arrival
        decodeAndQueue(item.audio_b64);
        // Hold captions during buffering, send immediately when playing
        if (playbackState === "playing") {
          chrome.runtime.sendMessage({
            type: "CAPTION",
            caption: item.caption,
          }).catch(() => {});
        } else {
          captionQueue.push(item.caption);
        }
      } else if (item.type === "language_detected") {
        chrome.runtime.sendMessage({
          type: "STATUS",
          status: "streaming",
        }).catch(() => {});
      } else if (item.type === "error") {
        console.error("[offscreen] Backend error:", item.message);
      }
    }
  } catch (err) {
    // Expected when globalAbort fires mid-read — the reader.read() throws AbortError
    if (err.name !== "AbortError") {
      console.error(`[offscreen] SSE read error for chunk #${chunkNum}:`, err);
    } else {
      console.log(`[offscreen] Chunk #${chunkNum} SSE connection aborted`);
    }
  }
  clearTimeout(stallTimer);
}

// -- Gapless audio playback via scheduled buffers -----------------------------

function getDecodeContext() {
  if (playbackCtx) return playbackCtx;
  if (!decodeCtx) decodeCtx = new AudioContext();
  return decodeCtx;
}

function decodeAndQueue(b64) {
  const raw = atob(b64);
  const bytes = new Uint8Array(raw.length);
  for (let i = 0; i < raw.length; i++) bytes[i] = raw.charCodeAt(i);

  const ctx = getDecodeContext();

  ctx.decodeAudioData(bytes.buffer.slice(0)).then((audioBuffer) => {
    decodedQueue.push(audioBuffer);
    bufferedDuration += audioBuffer.duration;

    if (playbackState === "buffering") {
      console.log(`[offscreen] Buffering: ${bufferedDuration.toFixed(1)}s / ${TARGET_BUFFER_SEC}s`);
      if (bufferedDuration >= TARGET_BUFFER_SEC) {
        startPlayback();
      }
    } else if (playbackState === "playing") {
      scheduleBuffers();
    }
  }).catch((err) => {
    console.warn("[offscreen] Audio decode failed:", err);
  });
}

function startPlayback() {
  console.log(`[offscreen] Starting playback with ${bufferedDuration.toFixed(1)}s buffered`);
  playbackState = "playing";
  if (bufferFallbackTimer) { clearTimeout(bufferFallbackTimer); bufferFallbackTimer = null; }

  // Mute the worklet — stop capturing during the replay zone.
  // This cleanly avoids re-capture instead of the fragile skip-counting approach.
  if (workletNode) {
    workletNode.port.postMessage({ type: "SET_CAPTURE_ACTIVE", active: false });
  }

  // Measure throughput from first segment received (backend warm), not from
  // captureStartedAt (which includes chunk accumulation time and first-chunk
  // Whisper warmup, making the rate pessimistically slow).
  const measureFrom = firstSegmentReceivedAt || captureStartedAt;
  const elapsedSec = (Date.now() - measureFrom) / 1000;
  const measuredRate = elapsedSec > 0 ? Math.min(bufferedDuration / elapsedSec, 1.0) : 0.75;
  const safeRate = Math.max(measuredRate * 0.9, 0.5); // small safety margin, floor at 0.5
  console.log(`[offscreen] Measured throughput: ${measuredRate.toFixed(2)}, safe rate: ${safeRate.toFixed(2)}`);

  // Close the temporary decode context — playbackCtx takes over decoding
  if (decodeCtx) { decodeCtx.close().catch(() => {}); decodeCtx = null; }

  // Create the AudioContext NOW — not earlier — so currentTime starts near 0.
  playbackCtx = new AudioContext();
  if (playbackCtx.state === "suspended") playbackCtx.resume();
  playbackCtx.addEventListener("statechange", () => {
    if (playbackCtx && playbackCtx.state === "running" && decodedQueue.length > 0) {
      scheduleBuffers();
    }
  });

  // Record sync anchors
  playbackStartedCtxTime = playbackCtx.currentTime;
  playbackStartedAt = Date.now();

  // Schedule all buffered segments
  scheduleBuffers();

  // Release buffered captions
  for (const caption of captionQueue) {
    chrome.runtime.sendMessage({ type: "CAPTION", caption }).catch(() => {});
  }
  captionQueue = [];

  // Tell service worker to transition: seek video, set rate, start polling
  chrome.runtime.sendMessage({
    type: "TRANSITION_TO_PLAYBACK",
    totalAudioSentSec,
    measuredRate: safeRate,
    initialLatencyMs: initialLatencyMs || 0,
  }).catch(() => {});
}

function scheduleBuffers() {
  if (playbackState !== "playing" || !playbackCtx) return;
  if (playbackCtx.state === "suspended") playbackCtx.resume();

  while (decodedQueue.length > 0) {
    const audioBuffer = decodedQueue.shift();

    // Queue underrun recovery: if nextPlayTime has fallen behind the current
    // AudioContext time (backend was slow, queue ran dry), jump forward to now
    if (nextPlayTime < playbackCtx.currentTime) {
      console.log(
        `[offscreen] Queue underrun: nextPlayTime=${nextPlayTime.toFixed(3)}, ` +
        `currentTime=${playbackCtx.currentTime.toFixed(3)}, resetting`
      );
      nextPlayTime = playbackCtx.currentTime;
    }

    const source = playbackCtx.createBufferSource();
    source.buffer = audioBuffer;
    source.connect(playbackCtx.destination);
    source.start(nextPlayTime);
    nextPlayTime += audioBuffer.duration;

    segmentCount++;
    source.onended = () => { checkBuffer(); };
  }
}

// -- Buffer health monitoring -------------------------------------------------

function checkBuffer() {
  if (!playbackCtx || playbackState !== "playing") return;

  const bufferAheadSec = nextPlayTime - playbackCtx.currentTime;
  const bufferAheadMs = bufferAheadSec * 1000;

  console.log(`[offscreen] Buffer: ${bufferAheadSec.toFixed(1)}s ahead`);

  chrome.runtime.sendMessage({
    type: "BUFFER_STATUS",
    bufferAheadMs,
  }).catch(() => {});
}

// -- Start/stop capture -------------------------------------------------------

async function startCapture(streamId) {
  console.log("[offscreen] startCapture called with streamId:", streamId);

  try {
    mediaStream = await navigator.mediaDevices.getUserMedia({
      audio: {
        mandatory: {
          chromeMediaSource: "tab",
          chromeMediaSourceId: streamId,
        },
      },
    });
    console.log("[offscreen] Got media stream, tracks:", mediaStream.getTracks().length);
  } catch (err) {
    console.error("[offscreen] getUserMedia failed:", err);
    chrome.runtime.sendMessage({
      type: "CAPTURE_ERROR",
      error: `Audio capture failed: ${err.message}`,
    }).catch(() => {});
    return;
  }

  try {
    audioContext = new AudioContext();
    console.log("[offscreen] AudioContext created, sampleRate:", audioContext.sampleRate);
    const source = audioContext.createMediaStreamSource(mediaStream);

    await audioContext.audioWorklet.addModule("audio-worklet.js");
    console.log("[offscreen] AudioWorklet loaded");
    workletNode = new AudioWorkletNode(audioContext, "chunk-accumulator", {
      processorOptions: { chunkSeconds: 8 },
    });

    workletNode.port.onmessage = (e) => {
      if (e.data.type === "chunk") {
        console.log("[offscreen] Got chunk from worklet, samples:", e.data.samples.length);
        sendPcmChunk(e.data.samples, e.data.sampleRate);
      } else if (e.data.type === "silence") {
        console.log("[offscreen] Silence detected");
        chrome.runtime.sendMessage({ type: "SILENCE_WARNING" }).catch(() => {});
      }
    };

    source.connect(workletNode);
    // Do NOT connect workletNode to audioContext.destination — mute the
    // original tab audio so only translated audio plays (keeps video in sync).

    console.log("[offscreen] Audio pipeline connected, waiting for chunks...");
    chrome.runtime.sendMessage({
      type: "STATUS",
      status: "connecting",
    }).catch(() => {});
  } catch (err) {
    console.error("[offscreen] Audio pipeline setup failed:", err);
    chrome.runtime.sendMessage({
      type: "CAPTURE_ERROR",
      error: `Audio setup failed: ${err.message}`,
    }).catch(() => {});
  }
}

async function stopCapture() {
  console.log("[offscreen] stopCapture called");

  globalAbort.abort();
  globalAbort = new AbortController();

  if (workletNode) {
    workletNode.disconnect();
    workletNode = null;
  }
  if (audioContext) {
    await audioContext.close().catch(() => {});
    audioContext = null;
  }
  if (mediaStream) {
    mediaStream.getTracks().forEach((t) => t.stop());
    mediaStream = null;
  }
  if (playbackCtx) {
    await playbackCtx.close().catch(() => {});
    playbackCtx = null;
  }
  decodedQueue = [];
  sessionId = null;
  chunkCount = 0;

  // Reset sync timing
  playbackState = "idle";
  firstChunkSentAt = null;
  firstSegmentReceivedAt = null;
  initialLatencyMs = null;
  captureStartedAt = null;
  playbackStartedAt = null;
  nextPlayTime = 0;
  playbackStartedCtxTime = null;
  bufferedDuration = 0;
  captionQueue = [];
  segmentCount = 0;
  totalAudioSentSec = 0;
  if (bufferFallbackTimer) { clearTimeout(bufferFallbackTimer); bufferFallbackTimer = null; }
  if (decodeCtx) { decodeCtx.close().catch(() => {}); decodeCtx = null; }
}

// -- Message listener ---------------------------------------------------------

chrome.runtime.onMessage.addListener((message) => {
  if (message.type === "OFFSCREEN_START") {
    console.log("[offscreen] Received OFFSCREEN_START");
    sessionId = message.sessionId;
    sourceLang = message.sourceLang;
    targetLang = message.targetLang;
    startCapture(message.streamId).catch((err) => {
      console.error("[offscreen] startCapture error:", err);
    });
  }

  if (message.type === "OFFSCREEN_STOP") {
    stopCapture();
  }

  // Service worker tells us the replay zone is over — resume capturing
  if (message.type === "RESUME_CAPTURE") {
    console.log("[offscreen] Resuming capture (replay zone ended)");
    if (workletNode) {
      workletNode.port.postMessage({ type: "SET_CAPTURE_ACTIVE", active: true });
    }
  }

  // Service worker can query buffer status
  if (message.type === "BUFFER_QUERY") {
    if (playbackCtx && playbackState === "playing") {
      const bufferAheadMs = (nextPlayTime - playbackCtx.currentTime) * 1000;
      chrome.runtime.sendMessage({ type: "BUFFER_STATUS", bufferAheadMs }).catch(() => {});
    }
  }
});

console.log("[offscreen] Offscreen document loaded");
