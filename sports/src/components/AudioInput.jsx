"use client";

import { useState, useRef, useEffect } from "react";
import "./AudioInput.css";

const BASE_URL =
  "https://mollysandler--sports-translation-api-translatorservice-f-6a7378.modal.run";

const MIN_CHUNK_SECONDS = 5;
const MAX_CHUNK_SECONDS = 16;

/**
 * Scan backwards from maxSamples toward minSamples looking for consecutive
 * low-energy windows (~120ms of silence). Two-pass: first strict silence,
 * then fallback to quietest point (no hard splits).
 */
function findSilenceSplit(samples, sr, minSamples, maxSamples) {
  const windowSamples = Math.floor(sr * 0.03); // 30ms windows
  const consecutive = 4;
  const rmsThreshold = 0.02;
  const end = Math.min(samples.length, maxSamples);
  const scanStart = minSamples;

  if (end - windowSamples * consecutive < scanStart) return -1;

  // Pre-compute RMS for every window in scan range
  const firstPos = scanStart;
  const lastPos = end - windowSamples;
  const nWindows = Math.floor((lastPos - firstPos) / windowSamples) + 1;
  if (nWindows <= 0) return -1;

  const rmsValues = new Float32Array(nWindows);
  for (let i = 0; i < nWindows; i++) {
    const wStart = firstPos + i * windowSamples;
    const wEnd = wStart + windowSamples;
    let sum = 0;
    for (let k = wStart; k < wEnd; k++) sum += samples[k] * samples[k];
    rmsValues[i] = Math.sqrt(sum / windowSamples);
  }

  // Pass 1: scan backwards for `consecutive` windows all below threshold
  for (let i = nWindows - consecutive; i >= 0; i--) {
    let allSilent = true;
    for (let j = 0; j < consecutive; j++) {
      if (rmsValues[i + j] >= rmsThreshold) { allSilent = false; break; }
    }
    if (allSilent) return firstPos + i * windowSamples;
  }

  // Pass 2: find quietest region (sliding average over `consecutive` windows)
  let bestAvg = Infinity;
  let bestI = 0;
  if (nWindows >= consecutive) {
    for (let i = 0; i <= nWindows - consecutive; i++) {
      let sum = 0;
      for (let j = 0; j < consecutive; j++) sum += rmsValues[i + j];
      const avg = sum / consecutive;
      if (avg < bestAvg) { bestAvg = avg; bestI = i; }
    }
  } else {
    for (let i = 0; i < nWindows; i++) {
      if (rmsValues[i] < bestAvg) { bestAvg = rmsValues[i]; bestI = i; }
    }
  }
  return firstPos + bestI * windowSamples;
}

// -- WAV encoding helpers ----------------------------------------------------

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
  view.setUint16(20, 1, true);   // PCM
  view.setUint16(22, 1, true);   // mono
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

// -- SSE line parser ---------------------------------------------------------

async function* readSseEvents(response) {
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop(); // keep incomplete last line
    for (const line of lines) {
      if (line.startsWith("data: ")) {
        try {
          yield JSON.parse(line.slice(6));
        } catch {
          // skip malformed
        }
      }
    }
  }
}

// -- Shared waveform drawing -------------------------------------------------

function drawWaveform(canvas, analyser) {
  if (!canvas || !analyser) return null;

  const ctx = canvas.getContext("2d");
  const bufferLength = analyser.frequencyBinCount;
  const dataArray = new Uint8Array(bufferLength);

  const draw = () => {
    const frameId = requestAnimationFrame(draw);
    // Store frameId on the canvas so caller can cancel
    canvas._animFrameId = frameId;
    analyser.getByteFrequencyData(dataArray);

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    const barWidth = (canvas.width / bufferLength) * 2.5;
    let x = 0;
    for (let i = 0; i < bufferLength; i++) {
      const barHeight = (dataArray[i] / 255) * canvas.height;
      const gradient = ctx.createLinearGradient(0, canvas.height, 0, canvas.height - barHeight);
      gradient.addColorStop(0, "#3b82f6");
      gradient.addColorStop(1, "#1e40af");
      ctx.fillStyle = gradient;
      ctx.fillRect(x, canvas.height - barHeight, barWidth - 1, barHeight);
      x += barWidth;
      if (x > canvas.width) break;
    }
  };
  draw();
}

function stopWaveform(canvas, analyser) {
  if (canvas && canvas._animFrameId) {
    cancelAnimationFrame(canvas._animFrameId);
    canvas._animFrameId = null;
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);
  }
  if (analyser) {
    try { analyser.disconnect(); } catch { /* ignore */ }
  }
}

// -- Component ---------------------------------------------------------------

export default function AudioInput({
  sourceLanguage,
  targetLanguage,
  onAudioSelected,
  onLiveCaptionAdded,
  onConnectionStatusChange,
  onLanguageDetected,
  audioSegmentsRef,
  liveAudioControlRef,
  isMuted,
  showToast,
}) {
  const [inputMethod, setInputMethod] = useState("stream");
  const [streamUrl, setStreamUrl] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [liveStatus, setLiveStatus] = useState("idle"); // idle | connecting | streaming | stopping
  const [streamStatus, setStreamStatus] = useState("idle"); // idle | connecting | streaming | stopping

  const sessionIdRef = useRef(null);
  const micStreamRef = useRef(null);
  const audioCtxRef = useRef(null);       // playback AudioContext
  const captureCtxRef = useRef(null);     // capture AudioContext
  const processorRef = useRef(null);      // ScriptProcessorNode
  const sourceNodeRef = useRef(null);     // MediaStreamSourceNode
  const pcmBufferRef = useRef(new Float32Array(0));
  const nativeSRRef = useRef(44100);
  const audioQueueRef = useRef([]);
  const isPlayingAudioRef = useRef(false);
  const streamAbortRef = useRef(null);    // AbortController for stream URL
  const sessionAbortRef = useRef(null);   // AbortController for /session/start
  const firstSegmentReceivedRef = useRef(false);
  const canvasRef = useRef(null);
  const analyserRef = useRef(null);
  const streamCanvasRef = useRef(null);
  const streamAnalyserRef = useRef(null);
  const playbackGainRef = useRef(null);

  // -- Sync mute state with playback gain node --------------------------------
  useEffect(() => {
    if (playbackGainRef.current) {
      playbackGainRef.current.gain.value = isMuted ? 0 : 1;
    }
  }, [isMuted]);

  // -- Batch file upload (unchanged) ----------------------------------------

  const processAudioOnBackend = async (audioBlob, fileName) => {
    setIsLoading(true);
    const formData = new FormData();
    formData.append("audio", audioBlob, fileName);
    formData.append("source_lang", sourceLanguage);
    formData.append("target_lang", targetLanguage);

    try {
      const response = await fetch(`${BASE_URL}/translate-audio`, {
        method: "POST",
        body: formData,
      });
      if (!response.ok) throw new Error("Network response was not ok");

      const data = await response.json();

      if (data.detected_language && onLanguageDetected) {
        onLanguageDetected(data.detected_language);
      }

      const binary = window.atob(data.audio_base64);
      const bytes = new Uint8Array(binary.length);
      for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
      const audioBlobResponse = new Blob([bytes], { type: "audio/mp3" });
      const translatedAudioUrl = URL.createObjectURL(audioBlobResponse);

      onAudioSelected({
        type: "file",
        source: translatedAudioUrl,
        name: `Translated - ${fileName}`,
        timestamp: new Date(),
        captions: data.captions,
      });
    } catch (error) {
      console.error("Error translating audio:", error);
      if (showToast) showToast("Failed to translate audio. See console for details.", "error");
    } finally {
      setIsLoading(false);
    }
  }

  const handleFileUpload = (e) => {
    const file = e.target.files[0];
    if (file) processAudioOnBackend(file, file.name);
  };

  const handleStartStream = async (e) => {
    e.preventDefault();
    if (!streamUrl.trim()) return;

    setStreamStatus("connecting");
    if (onConnectionStatusChange) onConnectionStatusChange(true);
    firstSegmentReceivedRef.current = false;

    // Reset accumulated audio segments
    if (audioSegmentsRef) audioSegmentsRef.current = [];

    // Create session (with abort support)
    const sessionAbort = new AbortController();
    sessionAbortRef.current = sessionAbort;

    let session_id;
    try {
      const res = await fetch(`${BASE_URL}/session/start`, {
        method: "POST",
        signal: sessionAbort.signal,
      });
      const data = await res.json();
      session_id = data.session_id;
    } catch (err) {
      if (err.name === "AbortError") {
        setStreamStatus("idle");
        if (onConnectionStatusChange) onConnectionStatusChange(false);
        return;
      }
      console.error("Session start error:", err);
      setStreamStatus("idle");
      if (onConnectionStatusChange) onConnectionStatusChange(false);
      return;
    }
    sessionIdRef.current = session_id;

    // Playback AudioContext + Analyser for stream waveform
    const ctx = new AudioContext();
    audioCtxRef.current = ctx;
    if (liveAudioControlRef) {
      liveAudioControlRef.current = {
        pause: () => ctx.suspend(),
        resume: () => ctx.resume(),
      };
    }
    const analyser = ctx.createAnalyser();
    analyser.fftSize = 256;
    streamAnalyserRef.current = analyser;
    const gainNode = ctx.createGain();
    gainNode.gain.value = isMuted ? 0 : 1;
    playbackGainRef.current = gainNode;
    audioQueueRef.current = [];
    isPlayingAudioRef.current = false;

    // Signal live mode to parent
    onAudioSelected({
      type: "live",
      source: null,
      name: "Stream Translation",
      timestamp: new Date(),
      captions: [],
    });

    // Start SSE stream
    const abortController = new AbortController();
    streamAbortRef.current = abortController;

    const formData = new FormData();
    formData.append("url", streamUrl.trim());
    formData.append("session_id", session_id);
    formData.append("source_lang", sourceLanguage);
    formData.append("target_lang", targetLanguage);

    try {
      const response = await fetch(`${BASE_URL}/translate-stream`, {
        method: "POST",
        body: formData,
        signal: abortController.signal,
      });

      for await (const item of readSseEvents(response)) {
        if (item.type === "language_detected") {
          if (onLanguageDetected) onLanguageDetected(item.language);
        } else if (item.type === "segment") {
          // Transition from connecting -> streaming on first segment
          if (!firstSegmentReceivedRef.current) {
            firstSegmentReceivedRef.current = true;
            setStreamStatus("streaming");
            if (onConnectionStatusChange) onConnectionStatusChange(false);
            // Start waveform visualizer for stream
            if (streamCanvasRef.current && streamAnalyserRef.current) {
              drawWaveform(streamCanvasRef.current, streamAnalyserRef.current);
            }
          }
          audioQueueRef.current.push(item.audio_b64);
          if (audioSegmentsRef) audioSegmentsRef.current.push(item.audio_b64);
          drainAudioQueue();
          if (onLiveCaptionAdded) onLiveCaptionAdded(item.caption);
        } else if (item.type === "error") {
          console.error("Stream error:", item.message);
        } else if (item.type === "done") {
          break;
        }
      }
    } catch (err) {
      if (err.name !== "AbortError") {
        console.error("Stream fetch error:", err);
      }
    }

    // Stream ended naturally or was stopped
    stopWaveform(streamCanvasRef.current, streamAnalyserRef.current);
    streamAnalyserRef.current = null;
    if (onConnectionStatusChange) onConnectionStatusChange(false);
    setStreamStatus("idle");
  };

  const handleStopStream = async () => {
    setStreamStatus("stopping");
    if (onConnectionStatusChange) onConnectionStatusChange(false);

    // Stop waveform
    stopWaveform(streamCanvasRef.current, streamAnalyserRef.current);
    streamAnalyserRef.current = null;

    // Abort session start if still in progress
    if (sessionAbortRef.current) {
      sessionAbortRef.current.abort();
      sessionAbortRef.current = null;
    }

    // Abort the SSE fetch
    if (streamAbortRef.current) {
      streamAbortRef.current.abort();
      streamAbortRef.current = null;
    }

    // Clean up session
    if (sessionIdRef.current) {
      const fd = new FormData();
      fd.append("session_id", sessionIdRef.current);
      await fetch(`${BASE_URL}/session/end`, { method: "POST", body: fd }).catch(() => {});
      sessionIdRef.current = null;
    }

    if (audioCtxRef.current) {
      await audioCtxRef.current.close().catch(() => {});
      audioCtxRef.current = null;
    }

    setStreamStatus("idle");
  };

  // -- Audio playback queue (Web Audio API) ----------------------------------

  const drainAudioQueue = async () => {
    if (isPlayingAudioRef.current) return;
    isPlayingAudioRef.current = true;
    while (audioQueueRef.current.length > 0) {
      const b64 = audioQueueRef.current.shift();
      const ctx = audioCtxRef.current;
      if (!ctx) break;
      try {
        const binary = atob(b64);
        const bytes = new Uint8Array(binary.length);
        for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
        const audioBuffer = await ctx.decodeAudioData(bytes.buffer.slice(0));
        await new Promise((resolve) => {
          const src = ctx.createBufferSource();
          src.buffer = audioBuffer;
          // Route through gain node for mute control
          const gain = playbackGainRef.current;
          const analyser = streamAnalyserRef.current;
          if (analyser && gain) {
            src.connect(analyser);
            analyser.connect(gain);
            gain.connect(ctx.destination);
          } else if (gain) {
            src.connect(gain);
            gain.connect(ctx.destination);
          } else {
            src.connect(ctx.destination);
          }
          src.onended = resolve;
          src.start();
        });
      } catch (err) {
        console.warn("Audio decode/play error:", err);
      }
    }
    isPlayingAudioRef.current = false;
  };

  // -- Resample raw PCM -> 16 kHz WAV and POST to /translate-live -----------

  const sendPcmChunk = async (samples) => {
    if (!sessionIdRef.current || samples.length < 100) return;
    const nativeSR = nativeSRRef.current;

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
    } catch (err) {
      console.warn("PCM resample/encode failed:", err);
      return;
    }

    const formData = new FormData();
    formData.append("audio", wavBlob, "chunk.wav");
    formData.append("session_id", sessionIdRef.current);
    formData.append("source_lang", sourceLanguage);
    formData.append("target_lang", targetLanguage);

    let response;
    try {
      response = await fetch(`${BASE_URL}/translate-live`, {
        method: "POST",
        body: formData,
      });
    } catch (err) {
      console.error("translate-live fetch error:", err);
      return;
    }

    for await (const item of readSseEvents(response)) {
      if (item.type === "language_detected") {
        if (onLanguageDetected) onLanguageDetected(item.language);
      } else if (item.type === "segment") {
        // Transition from connecting -> streaming on first segment
        if (!firstSegmentReceivedRef.current) {
          firstSegmentReceivedRef.current = true;
          setLiveStatus("streaming");
          if (onConnectionStatusChange) onConnectionStatusChange(false);
        }
        audioQueueRef.current.push(item.audio_b64);
        if (audioSegmentsRef) audioSegmentsRef.current.push(item.audio_b64);
        drainAudioQueue();
        if (onLiveCaptionAdded) onLiveCaptionAdded(item.caption);
      } else if (item.type === "error") {
        console.error("Live chunk error:", item.message);
      }
    }
  };

  // -- Start / stop live ----------------------------------------------------

  const handleStartLive = async () => {
    setLiveStatus("connecting");
    if (onConnectionStatusChange) onConnectionStatusChange(true);
    firstSegmentReceivedRef.current = false;

    // Reset accumulated audio segments
    if (audioSegmentsRef) audioSegmentsRef.current = [];

    // Session (with abort support)
    const sessionAbort = new AbortController();
    sessionAbortRef.current = sessionAbort;

    let session_id;
    try {
      const res = await fetch(`${BASE_URL}/session/start`, {
        method: "POST",
        signal: sessionAbort.signal,
      });
      const data = await res.json();
      session_id = data.session_id;
    } catch (err) {
      if (err.name === "AbortError") {
        setLiveStatus("idle");
        if (onConnectionStatusChange) onConnectionStatusChange(false);
        return;
      }
      console.error("Session start error:", err);
      setLiveStatus("idle");
      if (onConnectionStatusChange) onConnectionStatusChange(false);
      return;
    }
    sessionIdRef.current = session_id;

    // Playback AudioContext
    const playCtx = new AudioContext();
    audioCtxRef.current = playCtx;
    if (liveAudioControlRef) {
      liveAudioControlRef.current = {
        pause: () => playCtx.suspend(),
        resume: () => playCtx.resume(),
      };
    }
    const gainNode = playCtx.createGain();
    gainNode.gain.value = isMuted ? 0 : 1;
    playbackGainRef.current = gainNode;
    audioQueueRef.current = [];
    isPlayingAudioRef.current = false;

    // Microphone
    let stream;
    try {
      stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    } catch (err) {
      console.error("Microphone access denied:", err);
      setLiveStatus("idle");
      if (onConnectionStatusChange) onConnectionStatusChange(false);
      return;
    }
    micStreamRef.current = stream;

    // Signal to parent: live mode started
    onAudioSelected({
      type: "live",
      source: null,
      name: "Live Translation",
      timestamp: new Date(),
      captions: [],
    });

    // -- ScriptProcessor for raw PCM capture --------------------------------
    const captureCtx = new AudioContext();
    captureCtxRef.current = captureCtx;
    const nativeSR = captureCtx.sampleRate;
    nativeSRRef.current = nativeSR;
    const MIN_CHUNK_SAMPLES = Math.ceil(MIN_CHUNK_SECONDS * nativeSR);
    const MAX_CHUNK_SAMPLES = Math.ceil(MAX_CHUNK_SECONDS * nativeSR);

    pcmBufferRef.current = new Float32Array(0);

    const sourceNode = captureCtx.createMediaStreamSource(stream);
    sourceNodeRef.current = sourceNode;

    // Analyser for waveform visualization
    const analyser = captureCtx.createAnalyser();
    analyser.fftSize = 256;
    analyserRef.current = analyser;
    sourceNode.connect(analyser);

    const processor = captureCtx.createScriptProcessor(4096, 1, 1);
    processorRef.current = processor;

    processor.onaudioprocess = (e) => {
      const input = e.inputBuffer.getChannelData(0);
      const prev = pcmBufferRef.current;
      const next = new Float32Array(prev.length + input.length);
      next.set(prev);
      next.set(input, prev.length);
      pcmBufferRef.current = next;

      if (pcmBufferRef.current.length >= MIN_CHUNK_SAMPLES) {
        const splitIdx = findSilenceSplit(
          pcmBufferRef.current, nativeSR, MIN_CHUNK_SAMPLES, MAX_CHUNK_SAMPLES
        );
        let chunkEnd;
        if (splitIdx > 0) {
          chunkEnd = splitIdx;
        } else if (pcmBufferRef.current.length >= MAX_CHUNK_SAMPLES) {
          chunkEnd = MAX_CHUNK_SAMPLES;
        } else {
          return; // wait for more data or a silence gap
        }
        const chunk = pcmBufferRef.current.slice(0, chunkEnd);
        pcmBufferRef.current = pcmBufferRef.current.slice(chunkEnd);
        sendPcmChunk(chunk);
      }
    };

    analyser.connect(processor);
    const muteGain = captureCtx.createGain();
    muteGain.gain.value = 0;
    processor.connect(muteGain);
    muteGain.connect(captureCtx.destination);

    // Start waveform visualizer
    drawWaveform(canvasRef.current, analyserRef.current);
  };

  const handleStopLive = async () => {
    setLiveStatus("stopping");
    if (onConnectionStatusChange) onConnectionStatusChange(false);
    stopWaveform(canvasRef.current, analyserRef.current);
    analyserRef.current = null;

    // Abort session start if still in progress
    if (sessionAbortRef.current) {
      sessionAbortRef.current.abort();
      sessionAbortRef.current = null;
    }

    // Flush any remaining buffered audio (> 1 second)
    const remaining = pcmBufferRef.current;
    if (remaining.length > nativeSRRef.current) {
      sendPcmChunk(remaining);
    }
    pcmBufferRef.current = new Float32Array(0);

    if (processorRef.current) {
      processorRef.current.disconnect();
      processorRef.current = null;
    }
    if (sourceNodeRef.current) {
      sourceNodeRef.current.disconnect();
      sourceNodeRef.current = null;
    }
    if (captureCtxRef.current) {
      await captureCtxRef.current.close().catch(() => {});
      captureCtxRef.current = null;
    }
    if (micStreamRef.current) {
      micStreamRef.current.getTracks().forEach((t) => t.stop());
      micStreamRef.current = null;
    }
    if (sessionIdRef.current) {
      const fd = new FormData();
      fd.append("session_id", sessionIdRef.current);
      await fetch(`${BASE_URL}/session/end`, { method: "POST", body: fd }).catch(() => {});
      sessionIdRef.current = null;
    }
    if (audioCtxRef.current) {
      await audioCtxRef.current.close().catch(() => {});
      audioCtxRef.current = null;
    }

    setLiveStatus("idle");
  };

  // -- Render ---------------------------------------------------------------

  return (
    <div className="audio-input-card">
      {isLoading && (
        <div className="loading-overlay">Translating... Please wait.</div>
      )}
      <div className="audio-input-card">
        <h2>Select Audio Source</h2>
        <div className="input-method-tabs">
          <button
            className={`tab-button ${inputMethod === "upload" ? "active" : ""}`}
            onClick={() => setInputMethod("upload")}
          >
            Upload File
          </button>
          <button
            className={`tab-button ${inputMethod === "stream" ? "active" : ""}`}
            onClick={() => setInputMethod("stream")}
          >
            Stream URL
          </button>
          <button
            className={`tab-button ${inputMethod === "record" ? "active" : ""}`}
            onClick={() => setInputMethod("record")}
          >
            Record Live
          </button>
        </div>

        <div className="input-method-content">
          {inputMethod === "upload" && (
            <div className="upload-section">
              <label htmlFor="audio-file" className="file-input-label">
                <span className="upload-icon">📤</span>
                <span>Click to upload a file</span>
                <span className="upload-hint">MP3, WAV, M4A (Max 500MB)</span>
              </label>
              <input
                id="audio-file"
                type="file"
                accept="audio/*"
                onChange={handleFileUpload}
                className="file-input"
              />
            </div>
          )}

          {inputMethod === "stream" && (
            <div className="stream-section">
              {streamStatus === "connecting" ? (
                <div className="connecting-indicator">
                  <div className="connecting-spinner"></div>
                  <p>Connecting to server...</p>
                  <p className="connecting-hint">
                    This may take 30-60s on first request while the GPU warms up
                  </p>
                  <button onClick={handleStopStream} className="record-button stop">
                    Cancel
                  </button>
                </div>
              ) : streamStatus === "streaming" ? (
                <div className="recording-active">
                  <div className="recording-indicator">
                    <span className="recording-dot"></span>
                    Streaming — translating live...
                  </div>
                  <canvas ref={streamCanvasRef} className="waveform-canvas" width={300} height={60} />
                  <p className="stream-url-display">{streamUrl}</p>
                  <button onClick={handleStopStream} className="record-button stop">
                    Stop Stream
                  </button>
                </div>
              ) : streamStatus === "stopping" ? (
                <div className="recording-active">
                  <p>Stopping...</p>
                </div>
              ) : (
                <form onSubmit={handleStartStream} className="stream-form">
                  <div className="stream-input-group">
                    <input
                      type="url"
                      placeholder="Paste a URL to an audio file or stream"
                      value={streamUrl}
                      onChange={(e) => setStreamUrl(e.target.value)}
                      className="stream-input"
                    />
                    <button type="submit" className="submit-button">
                      Start
                    </button>
                  </div>
                  <p className="stream-hint">
                    Paste any direct link to an audio stream. Translates continuously in real time.
                  </p>
                  <div className="sample-links">
                    <span className="sample-label">Try a sample:</span>
                    {[
                      { label: "BBC News (EN)", url: "https://stream.live.vc.bbcmedia.co.uk/bbc_world_service" },
                      { label: "France Info (FR)", url: "https://stream.radiofrance.fr/franceinfo/franceinfo_hifi.m3u8" },
                      { label: "Deutsche Welle (DE)", url: "https://rbmn-live.akamaized.net/hls/live/590198/dwstream5/index.m3u8" },
                      { label: "NHK World (JA)", url: "https://nhkworld.webcdn.stream.ne.jp/www11/nhkworld-tv/domestic/hlslive/radio/audio/nhkworld.m3u8" },
                      { label: "RAI Radio 1 (IT)", url: "https://icestreaming.rai.it/1.mp3" },
                    ].map((s) => (
                      <button
                        key={s.url}
                        type="button"
                        className="sample-link-button"
                        onClick={() => setStreamUrl(s.url)}
                      >
                        {s.label}
                      </button>
                    ))}
                  </div>
                </form>
              )}
            </div>
          )}

          {inputMethod === "record" && (
            <div className="record-section">
              {liveStatus === "connecting" ? (
                <div className="connecting-indicator">
                  <div className="connecting-spinner"></div>
                  <p>Connecting to server...</p>
                  <p className="connecting-hint">
                    This may take 30-60s on first request while the GPU warms up
                  </p>
                  <canvas ref={canvasRef} className="waveform-canvas" width={300} height={60} />
                  <button onClick={handleStopLive} className="record-button stop">
                    Cancel
                  </button>
                </div>
              ) : liveStatus === "streaming" ? (
                <div className="recording-active">
                  <div className="recording-indicator">
                    <span className="recording-dot"></span>
                    Live — translating continuously...
                  </div>
                  <canvas ref={canvasRef} className="waveform-canvas" width={300} height={60} />
                  <button
                    onClick={handleStopLive}
                    className="record-button stop"
                  >
                    Stop Live
                  </button>
                </div>
              ) : liveStatus === "stopping" ? (
                <div className="recording-active">
                  <p>Stopping...</p>
                </div>
              ) : (
                <button
                  onClick={handleStartLive}
                  className="record-button start"
                >
                  Start Live Translation
                </button>
              )}
              <p className="record-hint">
                Translates microphone audio in real time
              </p>
            </div>
          )}
        </div>
      </div>

    </div>
  );
}
