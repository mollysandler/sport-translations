"use client"

import "./LanguageSelector.css"

const LANGUAGES = [
  { code: "auto", name: "Auto-detect", flag: "🌐" },
  { code: "en", name: "English", flag: "🇬🇧" },
  { code: "es", name: "Spanish", flag: "🇪🇸" },
  { code: "fr", name: "French", flag: "🇫🇷" },
  { code: "de", name: "German", flag: "🇩🇪" },
  { code: "it", name: "Italian", flag: "🇮🇹" },
  { code: "pt", name: "Portuguese", flag: "🇵🇹" },
  { code: "hi", name: "Hindi", flag: "🇮🇳" },
  { code: "ja", name: "Japanese", flag: "🇯🇵" },
  { code: "zh", name: "Chinese", flag: "🇨🇳" },
  { code: "ar", name: "Arabic", flag: "🇸🇦" },
]

export default function LanguageSelector({ sourceLanguage, targetLanguage, onSourceChange, onTargetChange, detectedLanguage }) {
  const handleSwap = () => {
    const temp = sourceLanguage
    onSourceChange(targetLanguage)
    onTargetChange(temp)
  }

  return (
    <div className="language-selector-card">
      <h2>Select Languages</h2>
      <div className="language-controls">
        <div className="language-select-group">
          <label htmlFor="source-lang">
            From:
            {sourceLanguage === "auto" && detectedLanguage && (() => {
              const detected = LANGUAGES.find((l) => l.code === detectedLanguage);
              return (
                <span className="detected-language-indicator">
                  {detected ? `${detected.flag} ${detected.name}` : detectedLanguage}
                </span>
              );
            })()}
          </label>
          <select
            id="source-lang"
            value={sourceLanguage}
            onChange={(e) => onSourceChange(e.target.value)}
            className="language-dropdown"
          >
            {LANGUAGES.map((lang) => (
              <option key={lang.code} value={lang.code}>
                {lang.flag} {lang.name}
              </option>
            ))}
          </select>
        </div>

        {sourceLanguage !== "auto" && (
          <button onClick={handleSwap} className="swap-button" title="Swap languages">
            ⇄
          </button>
        )}

        <div className="language-select-group">
          <label htmlFor="target-lang">To:</label>
          <select
            id="target-lang"
            value={targetLanguage}
            onChange={(e) => onTargetChange(e.target.value)}
            className="language-dropdown"
          >
            {LANGUAGES.map((lang) => (
              <option key={lang.code} value={lang.code}>
                {lang.flag} {lang.name}
              </option>
            ))}
          </select>
        </div>
      </div>
    </div>
  )
}
