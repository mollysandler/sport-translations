/**
 * Offscreen document: audio capture, WebSocket streaming, and playback.
 *
 * Audio pipeline (both sync modes):
 *   1. Capture tab audio continuously via AudioWorklet (200ms frames)
 *   2. Stream to backend via WebSocket for translation
 *   3. Buffer translated audio, then start playback
 *   4. Measure pipeline latency and send to content script
 *
 * Canvas mode: content script delays video frames; offscreen just plays audio.
 * Seekback mode: offscreen tells content script to seek back + adjusts rate.
 */

/* global chrome */

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

const WS_URL_BASE = "ws://localhost:8765";

const TARGET_BUFFER_SEC = 3;     // seconds of translated audio before playback (canvas mode needs less)
const FALLBACK_START_SEC = 15;   // start playback regardless after this
const HEARTBEAT_MS = 10000;      // WebSocket keepalive interval
const SW_KEEPALIVE_MS = 25000;   // service worker keepalive interval

const SILENCE_THRESHOLD = 0.005;
const SILENCE_WARN_FRAMES = 50;

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

let ws = null;
let captureStream = null;
let audioCtx = null;
let workletNode = null;
let playbackCtx = null;
let nextPlayTime = 0;
let decodedQueue = [];
let bufferedDurationSec = 0;
let isPlaying = false;
let totalAudioCapturedSec = 0;
let heartbeatInterval = null;
let swKeepaliveInterval = null;
let fallbackTimer = null;
let silenceFrames = 0;

let currentUtterance = null;
let lastOriginalEndSec = 0;
let translationOverrun = 0; // cumulative seconds that translated audio exceeds original duration

// Synchronous dedup (must fire before any await in finalizeUtterance)
let seenUtteranceKeys = new Set();
let highWaterEndSec = 0;

// Pipeline latency measurement
let firstFrameSentTime = 0;
let firstUtteranceReceivedTime = 0;
let measuredLatencySec = 0;
let latencySentToContent = false;

// Sync mode reported by content script
let syncMode = "canvas"; // "canvas" or "seekback"

// Caption dedup — prevent relaying the same translated text twice
let recentCaptions = [];

// Caption sync — hold captions until their corresponding audio starts playing.
// Keyed by seq number. Whichever arrives first (caption or audio schedule) stores
// its data; the second arrival triggers the timed delivery.
let pendingCaptions = new Map(); // seq -> caption object (caption arrived before audio scheduled)
let scheduledAudioTimes = new Map(); // seq -> nextPlayTime (audio scheduled before caption arrived)

// Seekback replay zone — suppress frames that are re-captured content.
// After seekback, the video replays from 0 and the tab produces the same audio
// again. We must NOT send this to the backend or it creates duplicate translations
// and a silence gap that permanently desyncs audio and video.
// Uses integer frame counting (not floating-point time) to avoid FP precision bugs.
let capturedFrameCount = 0;  // total frames captured (always increments)
let seekbackFrameMark = 0;   // capturedFrameCount at the moment of seekback
let inReplayZone = false;

// ---------------------------------------------------------------------------
// Message relay to service worker
// ---------------------------------------------------------------------------

function sendToSW(msg) {
  try { chrome.runtime.sendMessage(msg).catch(() => {}); } catch (e) {}
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

chrome.runtime.onMessage.addListener((msg, _sender, sendResponse) => {
  switch (msg.type) {
    case "START_CAPTURE":
      startCapture(msg.streamId, msg.sourceLang, msg.targetLang)
        .then(() => sendResponse({ ok: true }))
        .catch((err) => sendResponse({ error: err.message }));
      return true;

    case "STOP_CAPTURE":
      stopCapture();
      sendResponse({ ok: true });
      break;

    case "SYNC_MODE_REPORT":
      syncMode = msg.mode || "canvas";
      console.log(`[offscreen] Content script sync mode: ${syncMode}`);
      break;
  }
});

// ---------------------------------------------------------------------------
// Capture pipeline
// ---------------------------------------------------------------------------

async function startCapture(streamId, sourceLang, targetLang) {
  stopCapture();

  totalAudioCapturedSec = 0;
  bufferedDurationSec = 0;
  decodedQueue = [];
  isPlaying = false;
  silenceFrames = 0;
  currentUtterance = null;
  lastOriginalEndSec = 0;
  translationOverrun = 0;
  seenUtteranceKeys = new Set();
  highWaterEndSec = 0;
  firstFrameSentTime = 0;
  firstUtteranceReceivedTime = 0;
  measuredLatencySec = 0;
  latencySentToContent = false;
  recentCaptions = [];
  capturedFrameCount = 0;
  seekbackFrameMark = 0;
  inReplayZone = false;
  syncMode = "canvas";

  // 1. Get tab audio stream
  captureStream = await navigator.mediaDevices.getUserMedia({
    audio: {
      mandatory: {
        chromeMediaSource: "tab",
        chromeMediaSourceId: streamId,
      },
    },
  });

  // 2. AudioContext for capture + playback
  audioCtx = new AudioContext();
  playbackCtx = new AudioContext();
  const source = audioCtx.createMediaStreamSource(captureStream);

  await audioCtx.audioWorklet.addModule("stream-processor.js");
  workletNode = new AudioWorkletNode(audioCtx, "stream-processor");
  workletNode.port.onmessage = (e) => {
    if (e.data.type === "frame") handleAudioFrame(e.data);
  };
  source.connect(workletNode); // NOT to destination = mutes original

  // 3. WebSocket to backend
  const wsUrl = `${WS_URL_BASE}/ws/translate?source=${sourceLang}&target=${targetLang}`;
  ws = new WebSocket(wsUrl);
  ws.binaryType = "arraybuffer";

  ws.onopen = () => {
    sendToSW({ type: "STATUS", status: "streaming" });
    sendToSW({ type: "SHOW_OVERLAY", text: "Buffering translation..." });
    // Tell content script to detect DRM and choose sync mode
    sendToSW({ type: "START_SYNC" });
  };
  ws.onmessage = (event) => {
    if (typeof event.data === "string") {
      try {
        handleTextMessage(JSON.parse(event.data));
      } catch (e) {
        console.error("[offscreen] Failed to parse WS message:", e);
      }
    } else {
      handleBinaryMessage(event.data);
    }
  };
  ws.onerror = () => {
    sendToSW({ type: "CAPTURE_ERROR", error: "WebSocket connection failed" });
  };
  ws.onclose = () => stopCapture();

  // 4. Keepalives
  heartbeatInterval = setInterval(() => {
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: "heartbeat" }));
    }
  }, HEARTBEAT_MS);
  swKeepaliveInterval = setInterval(() => sendToSW({ type: "keepalive" }), SW_KEEPALIVE_MS);

  // 5. Fallback timer
  fallbackTimer = setTimeout(() => {
    if (!isPlaying && decodedQueue.length > 0) startPlayback();
  }, FALLBACK_START_SEC * 1000);
}

function stopCapture() {
  if (ws) {
    try {
      if (ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify({ type: "end_stream" }));
      ws.close();
    } catch (e) {}
    ws = null;
  }
  if (workletNode) { workletNode.disconnect(); workletNode = null; }
  if (audioCtx) { audioCtx.close().catch(() => {}); audioCtx = null; }
  if (captureStream) { captureStream.getTracks().forEach((t) => t.stop()); captureStream = null; }
  if (playbackCtx) { playbackCtx.close().catch(() => {}); playbackCtx = null; }

  clearInterval(heartbeatInterval);
  clearInterval(swKeepaliveInterval);
  clearTimeout(fallbackTimer);
  heartbeatInterval = null;
  swKeepaliveInterval = null;
  fallbackTimer = null;

  isPlaying = false;
  decodedQueue = [];
  currentUtterance = null;
  lastOriginalEndSec = 0;
  pendingCaptions.clear();
  scheduledAudioTimes.clear();

  sendToSW({ type: "HIDE_OVERLAY" });
  sendToSW({ type: "STATUS", status: "idle" });
}

// ---------------------------------------------------------------------------
// Audio frame handling (capture -> resample -> WebSocket)
// ---------------------------------------------------------------------------

async function handleAudioFrame(frame) {
  if (!ws || ws.readyState !== WebSocket.OPEN) return;

  const frameDuration = frame.samples.length / frame.sampleRate;
  capturedFrameCount++;

  // Seekback replay zone: the video is replaying already-translated content.
  // Suppress frames so the backend doesn't produce duplicate translations.
  // Uses integer frame count to avoid floating-point precision issues.
  // NOTE: totalAudioCapturedSec is NOT incremented during the replay zone
  // because these frames are duplicate content that the backend never sees.
  if (inReplayZone) {
    if (capturedFrameCount >= seekbackFrameMark * 2) {
      inReplayZone = false;
      console.log("[offscreen] Replay zone ended — resuming capture to backend");
    }
    return; // suppress ALL replay zone frames including the boundary
  }

  totalAudioCapturedSec += frameDuration;

  if (frame.rms < SILENCE_THRESHOLD) {
    silenceFrames++;
    if (silenceFrames === SILENCE_WARN_FRAMES) sendToSW({ type: "SILENCE_WARNING" });
  } else {
    silenceFrames = 0;
  }

  const pcm16 = await resampleTo16kPCM16(frame.samples, frame.sampleRate);
  ws.send(pcm16);

  if (firstFrameSentTime === 0) firstFrameSentTime = Date.now();

  if (!isPlaying) {
    const progress = Math.min(100, Math.round((bufferedDurationSec / TARGET_BUFFER_SEC) * 100));
    sendToSW({
      type: "OVERLAY_PROGRESS",
      text: `Buffering translation... ${Math.round(bufferedDurationSec)}s / ${TARGET_BUFFER_SEC}s`,
      progress,
    });
  }
}

function resampleTo16kPCM16(samples, sourceSampleRate) {
  const targetRate = 16000;
  if (sourceSampleRate === targetRate) return float32ToPCM16(samples);

  // Linear interpolation resampler — synchronous, no GC pressure from
  // creating a new OfflineAudioContext for every 200ms frame (5/sec).
  const ratio = sourceSampleRate / targetRate;
  const outputLength = Math.ceil(samples.length / ratio);
  const output = new Float32Array(outputLength);

  for (let i = 0; i < outputLength; i++) {
    const srcIdx = i * ratio;
    const srcFloor = Math.floor(srcIdx);
    const frac = srcIdx - srcFloor;
    const s0 = samples[srcFloor] || 0;
    const s1 = samples[Math.min(srcFloor + 1, samples.length - 1)] || 0;
    output[i] = s0 + frac * (s1 - s0);
  }

  return float32ToPCM16(output);
}

function float32ToPCM16(float32Array) {
  const pcm16 = new Int16Array(float32Array.length);
  for (let i = 0; i < float32Array.length; i++) {
    const s = Math.max(-1, Math.min(1, float32Array[i]));
    pcm16[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
  }
  return pcm16.buffer;
}

// ---------------------------------------------------------------------------
// WebSocket message handling (from backend)
// ---------------------------------------------------------------------------

function handleTextMessage(data) {
  switch (data.type) {
    case "session_ready":
      break;
    case "utterance_start":
      currentUtterance = { seq: data.seq, speakerId: data.speaker_id, chunks: [] };
      break;
    case "utterance_end":
      if (currentUtterance && currentUtterance.seq === data.seq) {
        finalizeUtterance(currentUtterance, data.original_start_sec || 0, data.original_end_sec || 0);
        currentUtterance = null;
      }
      break;
    case "caption": {
      const translated = data.translated || "";
      if (translated && recentCaptions.includes(translated)) break;
      recentCaptions.push(translated);
      if (recentCaptions.length > 20) recentCaptions.shift();

      const caption = {
        speaker: `Speaker ${data.speaker_id}`,
        original: data.original,
        translated,
      };

      // Sync caption with audio: delay delivery until the audio starts playing.
      if (!playbackCtx || data.seq === undefined) {
        // No audio context or no seq — send immediately (fallback)
        sendToSW({ type: "CAPTION", caption });
      } else {
        const timing = scheduledAudioTimes.get(data.seq);
        if (timing) {
          const delayMs = Math.max(0, (timing - playbackCtx.currentTime) * 1000);
          if (delayMs < 50) {
            sendToSW({ type: "CAPTION", caption });
          } else {
            setTimeout(() => sendToSW({ type: "CAPTION", caption }), delayMs);
          }
          scheduledAudioTimes.delete(data.seq);
        } else {
          pendingCaptions.set(data.seq, caption);
        }
      }
      break;
    }
    case "error":
      if (!data.recoverable) { sendToSW({ type: "CAPTURE_ERROR", error: data.message }); stopCapture(); }
      else sendToSW({ type: "CHUNK_ERROR", error: data.message });
      break;
    case "heartbeat_ack":
      break;
  }
}

function handleBinaryMessage(arrayBuffer) {
  if (currentUtterance) currentUtterance.chunks.push(arrayBuffer);
}

// ---------------------------------------------------------------------------
// MP3 silence trimming
// ---------------------------------------------------------------------------

function trimSilence(audioBuffer, threshold = 0.005) {
  const data = audioBuffer.getChannelData(0);
  let start = 0;
  while (start < data.length && Math.abs(data[start]) < threshold) start++;
  let end = data.length - 1;
  while (end > start && Math.abs(data[end]) < threshold) end--;
  if (start < 10 && end > data.length - 10) return audioBuffer;
  const trimmedLength = Math.max(1, end - start + 1);
  const trimmed = new AudioBuffer({
    length: trimmedLength,
    numberOfChannels: audioBuffer.numberOfChannels,
    sampleRate: audioBuffer.sampleRate,
  });
  for (let ch = 0; ch < audioBuffer.numberOfChannels; ch++) {
    trimmed.getChannelData(ch).set(audioBuffer.getChannelData(ch).subarray(start, end + 1));
  }
  return trimmed;
}

// ---------------------------------------------------------------------------
// Utterance finalization
// ---------------------------------------------------------------------------

async function finalizeUtterance(utterance, originalStartSec, originalEndSec) {
  if (utterance.chunks.length === 0) return;

  // Measure pipeline latency on first utterance
  if (firstUtteranceReceivedTime === 0 && firstFrameSentTime > 0) {
    firstUtteranceReceivedTime = Date.now();
    measuredLatencySec = (firstUtteranceReceivedTime - firstFrameSentTime) / 1000;
    console.log(`[offscreen] Measured pipeline latency: ${measuredLatencySec.toFixed(1)}s`);
    // Send to content script so canvas knows how much to delay
    sendToSW({ type: "SET_DELAY", delaySec: measuredLatencySec });
    latencySentToContent = true;
  }

  // ---- SYNCHRONOUS dedup (must complete before any await) ----

  // 1. Exact key match: same start+end = same Deepgram segment
  const dedupKey = `${originalStartSec.toFixed(3)}|${originalEndSec.toFixed(3)}`;
  if (seenUtteranceKeys.has(dedupKey)) {
    console.log(`[offscreen] Dedup: dropping duplicate seq=${utterance.seq} key=${dedupKey}`);
    return;
  }
  seenUtteranceKeys.add(dedupKey);
  if (seenUtteranceKeys.size > 200) {
    const iter = seenUtteranceKeys.values();
    for (let i = 0; i < 100; i++) seenUtteranceKeys.delete(iter.next().value);
  }

  // 2. Timestamp overlap: skip if this utterance's time range is already covered
  if (originalStartSec > 0 && originalStartSec < highWaterEndSec - 0.1) {
    console.log(
      `[offscreen] Dedup: skipping overlapping seq=${utterance.seq} ` +
      `(start=${originalStartSec.toFixed(1)}s < highWater=${highWaterEndSec.toFixed(1)}s)`
    );
    return;
  }
  // Update high-water SYNCHRONOUSLY before the await
  if (originalEndSec > highWaterEndSec) highWaterEndSec = originalEndSec;

  // ---- end synchronous dedup ----

  const totalSize = utterance.chunks.reduce((s, c) => s + c.byteLength, 0);
  const combined = new Uint8Array(totalSize);
  let offset = 0;
  for (const chunk of utterance.chunks) {
    combined.set(new Uint8Array(chunk), offset);
    offset += chunk.byteLength;
  }

  try {
    let audioBuffer = await playbackCtx.decodeAudioData(combined.buffer.slice(0));
    audioBuffer = trimSilence(audioBuffer);

    decodedQueue.push({
      audioBuffer,
      seq: utterance.seq,
      speakerId: utterance.speakerId,
      originalStartSec,
      originalEndSec,
    });
    bufferedDurationSec += audioBuffer.duration;

    if (isPlaying) {
      scheduleBufferedAudio();
    } else if (bufferedDurationSec >= TARGET_BUFFER_SEC) {
      startPlayback();
    }
  } catch (e) {
    console.error("[offscreen] Failed to decode audio:", e);
  }
}

// ---------------------------------------------------------------------------
// Playback
// ---------------------------------------------------------------------------

function startPlayback() {
  if (isPlaying) return;
  isPlaying = true;

  clearTimeout(fallbackTimer);
  fallbackTimer = null;

  if (playbackCtx.state === "suspended") playbackCtx.resume();
  nextPlayTime = playbackCtx.currentTime + 0.1;

  // Capture the first utterance's video position BEFORE draining the queue —
  // the content script needs this to align the canvas to the same position.
  const audioStartSec = decodedQueue.length > 0 ? decodedQueue[0].originalStartSec : 0;

  scheduleBufferedAudio();

  if (syncMode === "seekback") {
    seekbackFrameMark = capturedFrameCount;
    inReplayZone = true;
    sendToSW({ type: "VIDEO_SEEK_BACK", seekBackSec: totalAudioCapturedSec });
  } else {
    sendToSW({ type: "PLAYBACK_STARTED", audioStartSec });
  }

  sendToSW({ type: "HIDE_OVERLAY" });

  console.log(
    `[offscreen] Playback started (${syncMode} mode). ` +
    `Buffer: ${bufferedDurationSec.toFixed(1)}s, ` +
    `measured latency: ${measuredLatencySec.toFixed(1)}s`
  );
}

function scheduleBufferedAudio() {
  while (decodedQueue.length > 0) {
    scheduleAudioItem(decodedQueue.shift());
  }
}

function scheduleAudioItem(item) {
  if (!playbackCtx) return;

  const source = playbackCtx.createBufferSource();
  source.buffer = item.audioBuffer;
  source.connect(playbackCtx.destination);

  if (nextPlayTime < playbackCtx.currentTime) {
    nextPlayTime = playbackCtx.currentTime + 0.05;
  }

  // Track how much translated audio overruns the original speech duration.
  // If a 2s original utterance produces 2.6s of TTS, that's +0.6s overrun.
  // This accumulates across utterances as translations consistently expand.
  const originalDuration = (item.originalEndSec || 0) - (item.originalStartSec || 0);
  if (originalDuration > 0) {
    translationOverrun += item.audioBuffer.duration - originalDuration;
    if (translationOverrun < 0) translationOverrun = 0; // don't go negative
  }

  // Insert natural gap from original speech timing, but REDUCE it by the
  // accumulated overrun. This self-corrects translation expansion drift:
  // if translations are running long, gaps shrink to compensate.
  // The audio content itself is NEVER cut or sped up — only silence shrinks.
  if (item.originalStartSec > lastOriginalEndSec) {
    const gap = item.originalStartSec - lastOriginalEndSec;
    const adjustedGap = Math.max(0, gap - translationOverrun);
    nextPlayTime += Math.min(adjustedGap, 3.0);
    // Consume the overrun we just compensated for
    translationOverrun = Math.max(0, translationOverrun - gap);
  }

  source.start(nextPlayTime);

  // Sync caption delivery with audio playback.
  const caption = pendingCaptions.get(item.seq);
  if (caption) {
    const delayMs = Math.max(0, (nextPlayTime - playbackCtx.currentTime) * 1000);
    if (delayMs < 50) {
      sendToSW({ type: "CAPTION", caption });
    } else {
      setTimeout(() => sendToSW({ type: "CAPTION", caption }), delayMs);
    }
    pendingCaptions.delete(item.seq);
  } else if (item.seq !== undefined) {
    scheduledAudioTimes.set(item.seq, nextPlayTime);
  }

  nextPlayTime += item.audioBuffer.duration;

  if (item.originalEndSec > 0) lastOriginalEndSec = item.originalEndSec;

  source.onended = () => {
    if (decodedQueue.length > 0) scheduleBufferedAudio();
  };
}
