"use client";

import { useState } from "react";
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
        />

        {audioInput && (
          <CommentaryPlayer
            audioInput={audioInput}
            captions={captions}
            sourceLanguage={sourceLanguage}
            targetLanguage={targetLanguage}
            isPlaying={isPlaying}
            onPlayingChange={setIsPlaying}
          />
        )}
      </main>
    </div>
  );
}
