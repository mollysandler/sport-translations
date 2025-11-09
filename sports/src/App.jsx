"use client";

import { useState } from "react";
import LanguageSelector from "./components/LanguageSelector";
import AudioInput from "./components/AudioInput";
import CommentaryPlayer from "./components/CommentaryPlayer";
import FeedbackPanel from "./components/FeedbackPanel";
import "./App.css";

export default function App() {
  const [sourceLanguage, setSourceLanguage] = useState("en");
  const [targetLanguage, setTargetLanguage] = useState("hi");
  const [speakerGender, setSpeakerGender] = useState("male");
  const [audioInput, setAudioInput] = useState(null);
  const [isPlaying, setIsPlaying] = useState(false);

  return (
    <div className="app-container">
      <header className="app-header">
        <div className="header-content">
          <h1>Sports Commentary Translator</h1>
          <p>Translate live sports commentary into your language</p>
        </div>
      </header>

      <main className="app-main">
        <div className="content-grid">
          <div className="primary-section">
            <LanguageSelector
              sourceLanguage={sourceLanguage}
              targetLanguage={targetLanguage}
              onSourceChange={setSourceLanguage}
              onTargetChange={setTargetLanguage}
            />

            <AudioInput
              sourceLanguage={sourceLanguage}
              targetLanguage={targetLanguage}
              onAudioSelected={setAudioInput}
            />

            {audioInput && (
              <CommentaryPlayer
                audioInput={audioInput}
                sourceLanguage={sourceLanguage}
                targetLanguage={targetLanguage}
                speakerGender={speakerGender}
                onSpeakerGenderChange={setSpeakerGender}
                isPlaying={isPlaying}
                onPlayingChange={setIsPlaying}
              />
            )}
          </div>

          {audioInput && (
            <aside className="feedback-section">
              <FeedbackPanel
                sourceLanguage={sourceLanguage}
                targetLanguage={targetLanguage}
                audioInput={audioInput}
              />
            </aside>
          )}
        </div>
      </main>
    </div>
  );
}
