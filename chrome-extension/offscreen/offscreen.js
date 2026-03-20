const BASE_URL =
  "https://mollysandler--sports-translation-api-translatorservice-f-6a7378.modal.run";

let audioContext = null;
let mediaStream = null;
let workletNode = null;
let playbackCtx = null;
let isPlayingAudio = false;
let audioQueue = [];
let sessionId = null;
let sourceLang = "en";
let targetLang = "hi";
let chunkCount = 0;

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
  if (!sessionId || samples.length < 100) return;

  chunkCount++;
  const chunkNum = chunkCount;
  console.log(`[offscreen] Sending chunk #${chunkNum}, ${samples.length} samples at ${nativeSR}Hz`);

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

  let response;
  try {
    console.log(`[offscreen] POSTing chunk #${chunkNum} to /translate-live`);
    response = await fetch(`${BASE_URL}/translate-live`, {
      method: "POST",
      body: formData,
    });
    console.log(`[offscreen] Chunk #${chunkNum} response status: ${response.status}`);
  } catch (err) {
    console.error("[offscreen] translate-live fetch error:", err);
    return;
  }

  if (!response.ok) {
    const text = await response.text().catch(() => "");
    console.error(`[offscreen] Backend error ${response.status}: ${text}`);
    return;
  }

  for await (const item of readSseEvents(response)) {
    console.log("[offscreen] SSE event:", item.type, item);
    if (item.type === "segment") {
      audioQueue.push(item.audio_b64);
      drainAudioQueue();
      // Send caption to service worker which relays to side panel
      chrome.runtime.sendMessage({
        type: "CAPTION",
        caption: item.caption,
      }).catch(() => {});
    } else if (item.type === "language_detected") {
      chrome.runtime.sendMessage({
        type: "STATUS",
        status: "streaming",
      }).catch(() => {});
    } else if (item.type === "error") {
      console.error("[offscreen] Backend error:", item.message);
    }
  }
}

// -- Audio playback queue -----------------------------------------------------

function drainAudioQueue() {
  if (isPlayingAudio || audioQueue.length === 0) return;
  isPlayingAudio = true;

  const b64 = audioQueue.shift();
  const raw = atob(b64);
  const bytes = new Uint8Array(raw.length);
  for (let i = 0; i < raw.length; i++) bytes[i] = raw.charCodeAt(i);

  if (!playbackCtx) playbackCtx = new AudioContext();

  playbackCtx.decodeAudioData(bytes.buffer.slice(0)).then((audioBuffer) => {
    const source = playbackCtx.createBufferSource();
    source.buffer = audioBuffer;
    source.connect(playbackCtx.destination);
    source.onended = () => {
      isPlayingAudio = false;
      drainAudioQueue();
    };
    source.start();
  }).catch((err) => {
    console.warn("[offscreen] Audio decode failed:", err);
    isPlayingAudio = false;
    drainAudioQueue();
  });
}

// -- Start/stop capture -------------------------------------------------------

async function startCapture(streamId) {
  console.log("[offscreen] startCapture called with streamId:", streamId);

  try {
    // Get the tab audio stream
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

    // Load and connect AudioWorklet
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
    // Connect to destination so the original tab audio keeps playing
    workletNode.connect(audioContext.destination);

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
  audioQueue = [];
  isPlayingAudio = false;
  sessionId = null;
  chunkCount = 0;
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
});

console.log("[offscreen] Offscreen document loaded");
