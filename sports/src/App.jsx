"use client";

import { useState, useRef, useCallback } from "react";
import LanguageSelector from "./components/LanguageSelector";
import AudioInput from "./components/AudioInput";
import CommentaryPlayer from "./components/CommentaryPlayer";
import "./App.css";

export default function App() {
  const [sourceLanguage, setSourceLanguage] = useState("en");
  const [targetLanguage, setTargetLanguage] = useState("hi");
  const [audioInput, setAudioInput] = useState(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [captions, setCaptions] = useState([]);
  const [isConnecting, setIsConnecting] = useState(false);

  const audioSegmentsRef = useRef([]);

  const handleAudioSelected = (data) => {
    setAudioInput(data);
    if (data.captions && data.captions.length > 0) {
      setCaptions(data.captions);
    } else {
      setCaptions([]);
    }
  };

  const handleLiveCaptionAdded = (caption) => {
    setCaptions((prev) => [...prev, caption]);
  };

  const handleConnectionStatusChange = useCallback((connecting) => {
    setIsConnecting(connecting);
  }, []);

  const downloadSrt = useCallback(() => {
    if (captions.length === 0) return;

    const pad = (n) => String(n).padStart(2, "0");
    const formatTs = (sec) => {
      const h = Math.floor(sec / 3600);
      const m = Math.floor((sec % 3600) / 60);
      const s = Math.floor(sec % 60);
      const ms = Math.round((sec % 1) * 1000);
      return `${pad(h)}:${pad(m)}:${pad(s)},${String(ms).padStart(3, "0")}`;
    };

    let srt = "";
    captions.forEach((cap, i) => {
      const start = cap.startTime ?? i * 8;
      const end = cap.endTime ?? start + 8;
      srt += `${i + 1}\n`;
      srt += `${formatTs(start)} --> ${formatTs(end)}\n`;
      srt += `[${cap.speaker}] ${cap.translated || cap.original}\n\n`;
    });

    const blob = new Blob([srt], { type: "text/srt" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "translation.srt";
    a.click();
    URL.revokeObjectURL(url);
  }, [captions]);

  const downloadAudio = useCallback(() => {
    // For batch mode, reuse the existing audio source
    if (audioInput?.type === "file" && audioInput?.source) {
      const a = document.createElement("a");
      a.href = audioInput.source;
      a.download = "translated-audio.mp3";
      a.click();
      return;
    }

    // For live/stream mode, concatenate accumulated MP3 segments
    const segments = audioSegmentsRef.current;
    if (segments.length === 0) return;

    const byteArrays = segments.map((b64) => {
      const binary = atob(b64);
      const bytes = new Uint8Array(binary.length);
      for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
      return bytes;
    });

    const blob = new Blob(byteArrays, { type: "audio/mp3" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "translated-audio.mp3";
    a.click();
    URL.revokeObjectURL(url);
  }, [audioInput]);

  return (
    <div className="app-container">
      <header className="app-header">
        <div className="header-content">
          <h1>Sports Commentary Translator</h1>
          <p>Translate live sports commentary into your language</p>
        </div>
      </header>

      <main className="app-main">
        <LanguageSelector
          sourceLanguage={sourceLanguage}
          targetLanguage={targetLanguage}
          onSourceChange={setSourceLanguage}
          onTargetChange={setTargetLanguage}
        />

        <AudioInput
          sourceLanguage={sourceLanguage}
          targetLanguage={targetLanguage}
          onAudioSelected={handleAudioSelected}
          onLiveCaptionAdded={handleLiveCaptionAdded}
          onConnectionStatusChange={handleConnectionStatusChange}
          audioSegmentsRef={audioSegmentsRef}
        />

        {audioInput && (
          <CommentaryPlayer
            audioInput={audioInput}
            captions={captions}
            sourceLanguage={sourceLanguage}
            targetLanguage={targetLanguage}
            isPlaying={isPlaying}
            onPlayingChange={setIsPlaying}
            isConnecting={isConnecting}
            onDownloadSrt={downloadSrt}
            onDownloadAudio={downloadAudio}
          />
        )}
      </main>
    </div>
  );
}
