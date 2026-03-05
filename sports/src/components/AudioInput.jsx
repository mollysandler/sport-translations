"use client";

import { useState, useRef } from "react";
import "./AudioInput.css";

const BASE_URL =
  "https://mollysandler--sports-translation-api-translatorservice-f-6a7378.modal.run";

const CHUNK_SECONDS = 8;

// ── WAV encoding helpers ──────────────────────────────────────────────────────

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

// ── SSE line parser ───────────────────────────────────────────────────────────

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
        } catch (_) {
          // skip malformed
        }
      }
    }
  }
}

// ── Component ─────────────────────────────────────────────────────────────────

export default function AudioInput({
  sourceLanguage,
  targetLanguage,
  onAudioSelected,
  onLiveCaptionAdded,
}) {
  const [inputMethod, setInputMethod] = useState("upload");
  const [streamUrl, setStreamUrl] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [liveStatus, setLiveStatus] = useState("idle"); // idle | streaming | stopping
  const [streamStatus, setStreamStatus] = useState("idle"); // idle | streaming | stopping

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

  // ── Batch file upload (unchanged) ──────────────────────────────────────────

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
      alert("Failed to translate audio. See console for details.");
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

    setStreamStatus("streaming");

    // Create session
    const res = await fetch(`${BASE_URL}/session/start`, { method: "POST" });
    const { session_id } = await res.json();
    sessionIdRef.current = session_id;

    // Playback AudioContext
    audioCtxRef.current = new AudioContext();
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
        if (item.type === "segment") {
          audioQueueRef.current.push(item.audio_b64);
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
    setStreamStatus("idle");
  };

  const handleStopStream = async () => {
    setStreamStatus("stopping");

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

  // ── Audio playback queue (Web Audio API) ───────────────────────────────────

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
          src.connect(ctx.destination);
          src.onended = resolve;
          src.start();
        });
      } catch (err) {
        console.warn("Audio decode/play error:", err);
      }
    }
    isPlayingAudioRef.current = false;
  };

  // ── Resample raw PCM → 16 kHz WAV and POST to /translate-live ─────────────
  //
  // Accepts a Float32Array captured at nativeSR.  Resamples via OfflineAudioContext
  // (same approach as blobToWav16kMono, but without the WebM decode step that
  // silently fails on header-less MediaRecorder fragments after the first chunk).

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
      if (item.type === "segment") {
        audioQueueRef.current.push(item.audio_b64);
        drainAudioQueue();
        if (onLiveCaptionAdded) onLiveCaptionAdded(item.caption);
      } else if (item.type === "error") {
        console.error("Live chunk error:", item.message);
      }
    }
  };

  // ── Start / stop live ─────────────────────────────────────────────────────

  const handleStartLive = async () => {
    setLiveStatus("streaming");

    // Session
    const res = await fetch(`${BASE_URL}/session/start`, { method: "POST" });
    const { session_id } = await res.json();
    sessionIdRef.current = session_id;

    // Playback AudioContext
    audioCtxRef.current = new AudioContext();
    audioQueueRef.current = [];
    isPlayingAudioRef.current = false;

    // Microphone
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    micStreamRef.current = stream;

    // Signal to parent: live mode started
    onAudioSelected({
      type: "live",
      source: null,
      name: "Live Translation",
      timestamp: new Date(),
      captions: [],
    });

    // ── ScriptProcessor for raw PCM capture ──────────────────────────────────
    // Unlike MediaRecorder, ScriptProcessor yields raw Float32 PCM on every
    // callback.  Every chunk is independently encodable — no WebM header
    // fragmentation that silently breaks decodeAudioData after the first blob.

    const captureCtx = new AudioContext();
    captureCtxRef.current = captureCtx;
    const nativeSR = captureCtx.sampleRate;
    nativeSRRef.current = nativeSR;
    const CHUNK_SAMPLES = Math.ceil(CHUNK_SECONDS * nativeSR);

    pcmBufferRef.current = new Float32Array(0);

    const sourceNode = captureCtx.createMediaStreamSource(stream);
    sourceNodeRef.current = sourceNode;

    const processor = captureCtx.createScriptProcessor(4096, 1, 1);
    processorRef.current = processor;

    processor.onaudioprocess = (e) => {
      const input = e.inputBuffer.getChannelData(0);
      // Append incoming samples to rolling buffer
      const prev = pcmBufferRef.current;
      const next = new Float32Array(prev.length + input.length);
      next.set(prev);
      next.set(input, prev.length);
      pcmBufferRef.current = next;

      if (pcmBufferRef.current.length >= CHUNK_SAMPLES) {
        const chunk = pcmBufferRef.current.slice(0, CHUNK_SAMPLES);
        pcmBufferRef.current = pcmBufferRef.current.slice(CHUNK_SAMPLES);
        sendPcmChunk(chunk);
      }
    };

    sourceNode.connect(processor);
    // Muted gain node required: Chrome only fires onaudioprocess when the graph
    // is connected to a destination; gain=0 avoids mic-to-speaker feedback.
    const muteGain = captureCtx.createGain();
    muteGain.gain.value = 0;
    processor.connect(muteGain);
    muteGain.connect(captureCtx.destination);
  };

  const handleStopLive = async () => {
    setLiveStatus("stopping");

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

  // ── Render ────────────────────────────────────────────────────────────────

  return (
    <div className="audio-input-card">
      {isLoading && (
        <div className="loading-overlay">Translating... Please wait.</div>
      )}
      <div className="audio-input-card">
        <h2>Select Commentary Source</h2>
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
                <span>Click to upload or drag and drop</span>
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
              {streamStatus === "streaming" ? (
                <div className="recording-active">
                  <div className="recording-indicator">
                    <span className="recording-dot"></span>
                    Streaming — translating every 8s...
                  </div>
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
                    Paste any direct link to an audio stream. Translates continuously in 8-second chunks.
                  </p>
                  <div className="sample-links">
                    <span className="sample-label">Try a sample:</span>
                    {[
                      { label: "BBC News (EN)", url: "https://stream.live.vc.bbcmedia.co.uk/bbc_world_service" },
                      { label: "France Info (FR)", url: "https://stream.radiofrance.fr/franceinfo/franceinfo_hifi.m3u8" },
                      { label: "Deutsche Welle (DE)", url: "https://rbmn-live.akamaized.net/hls/live/590198/dwstream5/index.m3u8" },
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
              {liveStatus === "streaming" ? (
                <div className="recording-active">
                  <div className="recording-indicator">
                    <span className="recording-dot"></span>
                    Live — translating every 8s...
                  </div>
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
                  🎤 Start Live Translation
                </button>
              )}
              <p className="record-hint">
                Translates microphone audio in real-time, 8-second chunks
              </p>
            </div>
          )}
        </div>
      </div>

    </div>
  );
}
