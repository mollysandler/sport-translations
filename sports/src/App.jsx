"use client";

import { useState, useRef, useCallback, useEffect, useMemo } from "react";
import LanguageSelector from "./components/LanguageSelector";
import AudioInput from "./components/AudioInput";
import CommentaryPlayer from "./components/CommentaryPlayer";
import FeedbackPanel from "./components/FeedbackPanel";
import { ToastProvider, useToast } from "./components/Toast";
import "./App.css";

// -- localStorage helpers ----------------------------------------------------

const LS_CAPTIONS = "st_captions";
const LS_SOURCE_LANG = "st_source_lang";
const LS_TARGET_LANG = "st_target_lang";
const LS_DETECTED_LANG = "st_detected_lang";
const LS_SPEAKER_NAMES = "st_speaker_names";

function loadJson(key, fallback) {
  try {
    const raw = localStorage.getItem(key);
    return raw ? JSON.parse(raw) : fallback;
  } catch {
    return fallback;
  }
}

function saveJson(key, value) {
  try {
    localStorage.setItem(key, JSON.stringify(value));
  } catch {
    // quota exceeded — ignore
  }
}

// ---------------------------------------------------------------------------

function AppInner() {
  const [sourceLanguage, setSourceLanguage] = useState(
    () => loadJson(LS_SOURCE_LANG, null) || "auto"
  );
  const [targetLanguage, setTargetLanguage] = useState(
    () => loadJson(LS_TARGET_LANG, null) || "hi"
  );
  const [audioInput, setAudioInput] = useState(() => {
    // Restore a minimal audioInput if we have saved captions
    const saved = loadJson(LS_CAPTIONS, []);
    if (saved.length > 0) {
      return { type: "restored", source: null, name: "Previous Session", captions: saved };
    }
    return null;
  });
  const [isPlaying, setIsPlaying] = useState(false);
  const [captions, setCaptions] = useState(() => loadJson(LS_CAPTIONS, []));
  const [isConnecting, setIsConnecting] = useState(false);
  const [livePaused, setLivePaused] = useState(false);
  const [isMuted, setIsMuted] = useState(false);
  const [detectedLanguage, setDetectedLanguage] = useState(
    () => loadJson(LS_DETECTED_LANG, null)
  );
  const [speakerNames, setSpeakerNames] = useState(
    () => loadJson(LS_SPEAKER_NAMES, {})
  );

  const audioSegmentsRef = useRef([]);
  const liveAudioControlRef = useRef({});
  const captionBufferRef = useRef([]);
  const showToast = useToast();

  // Refs to allow keyboard handler to access current callbacks
  const downloadSrtRef = useRef(null);
  const downloadAudioRef = useRef(null);

  // -- Persist language prefs -----------------------------------------------

  useEffect(() => {
    saveJson(LS_SOURCE_LANG, sourceLanguage);
  }, [sourceLanguage]);

  useEffect(() => {
    saveJson(LS_TARGET_LANG, targetLanguage);
  }, [targetLanguage]);

  useEffect(() => {
    saveJson(LS_DETECTED_LANG, detectedLanguage);
  }, [detectedLanguage]);

  // -- Persist captions (skip storing audio base64) -------------------------

  useEffect(() => {
    saveJson(LS_CAPTIONS, captions);
  }, [captions]);

  // -- Persist speaker names ------------------------------------------------

  useEffect(() => {
    saveJson(LS_SPEAKER_NAMES, speakerNames);
  }, [speakerNames]);

  const handleSpeakerNameChange = useCallback((speakerId, newName) => {
    setSpeakerNames((prev) => ({ ...prev, [speakerId]: newName }));
  }, []);

  // -- Clear saved session --------------------------------------------------

  const clearSession = useCallback(() => {
    setAudioInput(null);
    setCaptions([]);
    setSpeakerNames({});
    setDetectedLanguage(null);
    localStorage.removeItem(LS_CAPTIONS);
    localStorage.removeItem(LS_SPEAKER_NAMES);
    localStorage.removeItem(LS_DETECTED_LANG);
  }, []);

  // -- Handlers -------------------------------------------------------------

  const handleAudioSelected = (data) => {
    setAudioInput(data);
    if (data.captions && data.captions.length > 0) {
      setCaptions(data.captions);
    } else if (data.type !== "live") {
      // New non-live audio clears old captions; live appends via handleLiveCaptionAdded
      setCaptions([]);
    }
  };

  const handleLiveCaptionAdded = useCallback((caption) => {
    if (livePaused) {
      captionBufferRef.current.push(caption);
    } else {
      setCaptions((prev) => [...prev, caption]);
    }
  }, [livePaused]);

  // Flush buffered captions when unpaused
  useEffect(() => {
    if (!livePaused && captionBufferRef.current.length > 0) {
      const buffered = captionBufferRef.current.splice(0);
      setCaptions((prev) => [...prev, ...buffered]);
    }
  }, [livePaused]);

  const handleLivePauseChange = useCallback((paused) => {
    setLivePaused(paused);
    const ctrl = liveAudioControlRef.current;
    if (ctrl) {
      paused ? ctrl.pause?.() : ctrl.resume?.();
    }
  }, []);

  const handleConnectionStatusChange = useCallback((connecting) => {
    setIsConnecting(connecting);
  }, []);

  const handleLanguageDetected = useCallback((langCode) => {
    setDetectedLanguage(langCode);
    const LANG_NAMES = {
      en: "English", es: "Spanish", fr: "French", de: "German",
      it: "Italian", pt: "Portuguese", hi: "Hindi", ja: "Japanese",
      zh: "Chinese", ar: "Arabic",
    };
    const name = LANG_NAMES[langCode] || langCode;
    showToast(`Detected language: ${name}`, "info");
  }, [showToast]);

  // -- Speaker helpers ------------------------------------------------------

  const getSpeakerColor = (speaker) => {
    const colors = ["#2563eb", "#dc2626", "#16a34a", "#d97706", "#9333ea"];
    const num = parseInt(String(speaker).replace(/\D/g, "") || "0", 10);
    return colors[num % colors.length];
  };

  const getDisplayName = (speaker) => {
    return (speakerNames && speakerNames[speaker]) || speaker;
  };

  const uniqueSpeakers = useMemo(() => {
    const seen = new Set();
    return captions.reduce((acc, cap) => {
      if (cap.speaker && !seen.has(cap.speaker)) {
        seen.add(cap.speaker);
        acc.push(cap.speaker);
      }
      return acc;
    }, []);
  }, [captions]);

  // -- Downloads ------------------------------------------------------------

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
      const name = (speakerNames[cap.speaker]) || cap.speaker;
      srt += `${i + 1}\n`;
      srt += `${formatTs(start)} --> ${formatTs(end)}\n`;
      srt += `[${name}] ${cap.translated || cap.original}\n\n`;
    });

    const blob = new Blob([srt], { type: "text/srt" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "translation.srt";
    a.click();
    URL.revokeObjectURL(url);
  }, [captions, speakerNames]);

  const downloadAudio = useCallback(() => {
    if (audioInput?.type === "file" && audioInput?.source) {
      const a = document.createElement("a");
      a.href = audioInput.source;
      a.download = "translated-audio.mp3";
      a.click();
      return;
    }

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

  // Keep refs in sync for keyboard handler
  downloadSrtRef.current = downloadSrt;
  downloadAudioRef.current = downloadAudio;

  // -- Keyboard shortcuts ---------------------------------------------------

  useEffect(() => {
    const handler = (e) => {
      const tag = document.activeElement?.tagName;
      if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") return;

      if (e.code === "Space") {
        e.preventDefault();
        if (audioInput?.type === "file") {
          const audio = document.querySelector("audio");
          if (audio) {
            audio.paused ? audio.play() : audio.pause();
          }
        }
      } else if (e.code === "KeyD" && !e.metaKey && !e.ctrlKey) {
        e.preventDefault();
        downloadAudioRef.current?.();
      } else if (e.code === "KeyM" && !e.metaKey && !e.ctrlKey) {
        e.preventDefault();
        setIsMuted((m) => !m);
      } else if (e.code === "KeyS" && !e.metaKey && !e.ctrlKey) {
        e.preventDefault();
        downloadSrtRef.current?.();
      }
    };
    document.addEventListener("keydown", handler);
    return () => document.removeEventListener("keydown", handler);
  }, [audioInput]);

  // -- Render ---------------------------------------------------------------

  const showPlayer = audioInput !== null;

  return (
    <div className="app-container">
      <header className="app-header">
        <div className="header-content">
          <h1>Live Audio Translator</h1>
          <p>Translate live audio into your language</p>
        </div>
      </header>

      <main className="app-main">
        <LanguageSelector
          sourceLanguage={sourceLanguage}
          targetLanguage={targetLanguage}
          onSourceChange={setSourceLanguage}
          onTargetChange={setTargetLanguage}
          detectedLanguage={detectedLanguage}
        />

        <AudioInput
          sourceLanguage={sourceLanguage}
          targetLanguage={targetLanguage}
          onAudioSelected={handleAudioSelected}
          onLiveCaptionAdded={handleLiveCaptionAdded}
          onConnectionStatusChange={handleConnectionStatusChange}
          onLanguageDetected={handleLanguageDetected}
          audioSegmentsRef={audioSegmentsRef}
          liveAudioControlRef={liveAudioControlRef}
          isMuted={isMuted}
          showToast={showToast}
        />

        {showPlayer && (
          <>
            {audioInput.type === "restored" && (
              <div className="restored-banner">
                <span>Showing captions from your previous session</span>
                <button onClick={clearSession}>Clear</button>
              </div>
            )}

            <div className="app-content-grid">
              <div className="app-content-main">
                <CommentaryPlayer
                  audioInput={audioInput}
                  captions={captions}
                  sourceLanguage={sourceLanguage}
                  targetLanguage={targetLanguage}
                  detectedLanguage={detectedLanguage}
                  isPlaying={isPlaying}
                  onPlayingChange={setIsPlaying}
                  isConnecting={isConnecting}
                  onDownloadSrt={downloadSrt}
                  onDownloadAudio={downloadAudio}
                  speakerNames={speakerNames}
                  livePaused={livePaused}
                  onLivePauseChange={handleLivePauseChange}
                  isMuted={isMuted}
                  onMuteToggle={() => setIsMuted((m) => !m)}
                />
              </div>

              <div className="app-sidebar">
                {uniqueSpeakers.length > 0 && (
                  <div className="speaker-panel">
                    <h4>Speakers <span className="speaker-panel-hint">(edit names below)</span></h4>
                    <div className="speaker-panel-list">
                      {uniqueSpeakers.map((spk) => (
                        <div key={spk} className="speaker-panel-item">
                          <span
                            className="speaker-panel-dot"
                            style={{ backgroundColor: getSpeakerColor(spk) }}
                          />
                          <input
                            type="text"
                            className="speaker-panel-input"
                            value={getDisplayName(spk)}
                            onChange={(e) => handleSpeakerNameChange(spk, e.target.value)}
                            onBlur={(e) => {
                              const trimmed = e.target.value.trim();
                              if (!trimmed) handleSpeakerNameChange(spk, spk);
                            }}
                            style={{ borderColor: getSpeakerColor(spk) }}
                          />
                          <span className="speaker-panel-id">{spk}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                <FeedbackPanel
                  sourceLanguage={sourceLanguage}
                  targetLanguage={targetLanguage}
                  audioInput={audioInput}
                  showToast={showToast}
                />
              </div>
            </div>
          </>
        )}
      </main>
    </div>
  );
}

export default function App() {
  return (
    <ToastProvider>
      <AppInner />
    </ToastProvider>
  );
}
