// AudioInput.jsx
import React, { useEffect, useRef, useState } from "react";

export default function AudioInput(props) {
  const {
    sourceLanguage,
    targetLanguage,
    onAudioSelected,
    onError,
    apiBaseUrl,
  } = props;

  const API_BASE = (
    apiBaseUrl ||
    "https://mollysandler--sports-translation-api-translatorservice-f-6a7378.modal.run"
  ).replace(/\/$/, "");

  const [busy, setBusy] = useState(false);
  const [status, setStatus] = useState("");
  const [recording, setRecording] = useState(false);

  const mediaRecorderRef = useRef(null);
  const recordedChunksRef = useRef([]);

  // Local inputs fallback if parent doesn’t pass languages
  const [sourceLang, setSourceLang] = useState(sourceLanguage || "en");
  const [targetLang, setTargetLang] = useState(targetLanguage || "hi");

  useEffect(() => {
    if (sourceLanguage) setSourceLang(sourceLanguage);
  }, [sourceLanguage]);

  useEffect(() => {
    if (targetLanguage) setTargetLang(targetLanguage);
  }, [targetLanguage]);

  function emitError(err) {
    console.error(err);
    onError?.(err);
  }

  function blobToObjectUrlFromBase64(base64, mimeType = "audio/mpeg") {
    const byteChars = atob(base64);
    const bytes = new Uint8Array(byteChars.length);
    for (let i = 0; i < byteChars.length; i++)
      bytes[i] = byteChars.charCodeAt(i);
    const blob = new Blob([bytes], { type: mimeType });
    return URL.createObjectURL(blob);
  }

  async function streamTranslate(file, filename = "audio.wav") {
    setBusy(true);
    setStatus("Uploading & streaming…");

    // Show streaming card immediately
    onAudioSelected?.({
      type: "file",
      source: null,
      name: `Streaming - ${filename}`,
      timestamp: new Date(),
      captions: [],
      isStreaming: true,
    });

    try {
      const form = new FormData();
      form.append("audio", file, filename);
      form.append("source_lang", sourceLang || "en");
      form.append("target_lang", targetLang || "hi");
      form.append("buffer_sec", "120");

      const res = await fetch(`${API_BASE}/translate-audio-stream`, {
        method: "POST",
        body: form,
      });

      if (!res.ok || !res.body) {
        const text = await res.text().catch(() => "");
        throw new Error(
          `Stream failed (${res.status}): ${text || res.statusText}`,
        );
      }

      const reader = res.body.getReader();
      const decoder = new TextDecoder("utf-8");

      let buffer = "";
      let captionsAcc = [];

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() || "";

        for (const line of lines) {
          if (!line.trim()) continue;

          let evt;
          try {
            evt = JSON.parse(line);
          } catch {
            continue;
          }

          if (evt.type === "buffering") {
            setStatus(
              `Buffering… ${Math.floor(evt.buffered_until_sec || 0)}s / ${Math.floor(
                evt.buffer_target_sec || 120,
              )}s`,
            );
          }

          if (evt.type === "segment" && evt.caption) {
            captionsAcc = [...captionsAcc, evt.caption];

            // IMPORTANT: update parent on every segment
            onAudioSelected?.({
              type: "file",
              source: null,
              name: `Streaming - ${filename}`,
              timestamp: new Date(),
              captions: captionsAcc,
              isStreaming: true,
            });
          }

          if (evt.type === "final") {
            const audioBase64 = evt.audio_base64;
            const finalCaptions = evt.captions || captionsAcc;

            const audioUrl = audioBase64
              ? blobToObjectUrlFromBase64(audioBase64, "audio/mpeg")
              : null;

            setStatus("Done ✅");

            onAudioSelected?.({
              type: "file",
              source: audioUrl, // CommentaryPlayer expects audioInput.source
              name: filename,
              timestamp: new Date(),
              captions: finalCaptions,
              isStreaming: false, // switch UI out of streaming mode
            });
          }

          if (evt.type === "error") {
            throw new Error(evt.message || "Unknown stream error");
          }
        }
      }
    } finally {
      setBusy(false);
      setTimeout(() => setStatus(""), 2000);
    }
  }

  async function onFilePicked(e) {
    const file = e.target.files?.[0];
    if (!file) return;

    if (!file.type.startsWith("audio/") && !file.type.startsWith("video/")) {
      emitError(
        new Error("Please upload an audio or video file (mp3/wav/mp4/etc)."),
      );
      return;
    }

    try {
      await streamTranslate(file, file.name);
    } catch (err) {
      setStatus("Error ❌");
      emitError(err);
      setBusy(false);
    } finally {
      e.target.value = "";
    }
  }

  async function startRecording() {
    try {
      setStatus("Requesting microphone…");
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      recordedChunksRef.current = [];

      const preferredTypes = [
        "audio/webm;codecs=opus",
        "audio/webm",
        "audio/ogg;codecs=opus",
        "audio/ogg",
      ];

      let mimeType = "";
      for (const t of preferredTypes) {
        if (MediaRecorder.isTypeSupported(t)) {
          mimeType = t;
          break;
        }
      }

      const mr = new MediaRecorder(stream, mimeType ? { mimeType } : undefined);

      mr.ondataavailable = (evt) => {
        if (evt.data && evt.data.size > 0)
          recordedChunksRef.current.push(evt.data);
      };

      mr.onstart = () => {
        setRecording(true);
        setStatus("Recording…");
      };

      mr.onstop = async () => {
        setRecording(false);
        setStatus("Preparing upload…");
        stream.getTracks().forEach((t) => t.stop());

        const blob = new Blob(recordedChunksRef.current, {
          type: mimeType || "audio/webm",
        });

        try {
          const ext = (mimeType || "").includes("ogg") ? "ogg" : "webm";
          await streamTranslate(blob, `recording.${ext}`);
        } catch (err) {
          setStatus("Error ❌");
          emitError(err);
          setBusy(false);
        }
      };

      mediaRecorderRef.current = mr;
      mr.start(1000);
    } catch (err) {
      setStatus("Mic error ❌");
      emitError(err);
      setBusy(false);
    }
  }

  function stopRecording() {
    const mr = mediaRecorderRef.current;
    if (!mr) return;
    if (mr.state === "recording") mr.stop();
  }

  return (
    <div style={{ border: "1px solid #e5e7eb", borderRadius: 12, padding: 16 }}>
      <div
        style={{
          display: "flex",
          gap: 12,
          flexWrap: "wrap",
          alignItems: "center",
        }}
      >
        <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
          <label style={{ fontSize: 12, opacity: 0.8 }}>Source</label>
          <input
            value={sourceLang}
            onChange={(e) => setSourceLang(e.target.value)}
            placeholder="en"
            style={{
              padding: "8px 10px",
              borderRadius: 10,
              border: "1px solid #d1d5db",
            }}
            disabled={busy || recording}
          />
        </div>

        <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
          <label style={{ fontSize: 12, opacity: 0.8 }}>Target</label>
          <input
            value={targetLang}
            onChange={(e) => setTargetLang(e.target.value)}
            placeholder="hi"
            style={{
              padding: "8px 10px",
              borderRadius: 10,
              border: "1px solid #d1d5db",
            }}
            disabled={busy || recording}
          />
        </div>

        <div style={{ flex: 1 }} />

        <div
          style={{
            display: "flex",
            gap: 10,
            alignItems: "center",
            flexWrap: "wrap",
          }}
        >
          <label
            style={{
              display: "inline-flex",
              alignItems: "center",
              gap: 10,
              padding: "8px 12px",
              borderRadius: 10,
              border: "1px solid #d1d5db",
              cursor: busy || recording ? "not-allowed" : "pointer",
              opacity: busy || recording ? 0.6 : 1,
            }}
          >
            <input
              type="file"
              accept="audio/*,video/*"
              onChange={onFilePicked}
              disabled={busy || recording}
              style={{ display: "none" }}
            />
            <span>Upload audio/video</span>
          </label>

          {!recording ? (
            <button
              onClick={startRecording}
              disabled={busy}
              style={{
                padding: "8px 12px",
                borderRadius: 10,
                border: "1px solid #d1d5db",
                background: "white",
                cursor: busy ? "not-allowed" : "pointer",
                opacity: busy ? 0.6 : 1,
              }}
            >
              Record mic
            </button>
          ) : (
            <button
              onClick={stopRecording}
              style={{
                padding: "8px 12px",
                borderRadius: 10,
                border: "1px solid #d1d5db",
                background: "white",
                cursor: "pointer",
              }}
            >
              Stop
            </button>
          )}
        </div>
      </div>

      <div style={{ marginTop: 12, fontSize: 13, opacity: 0.85 }}>
        <div>
          <strong>API:</strong> {API_BASE}/translate-audio-stream
        </div>
        <div style={{ minHeight: 18, marginTop: 6 }}>
          {busy || recording ? (
            <span>{status}</span>
          ) : status ? (
            <span>{status}</span>
          ) : null}
        </div>
      </div>
    </div>
  );
}
