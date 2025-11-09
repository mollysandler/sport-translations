"use client"

import { useState, useRef, useEffect } from "react"
import "./CommentaryPlayer.css"

export default function CommentaryPlayer({
  audioInput,
  sourceLanguage,
  targetLanguage,
  speakerGender,
  onSpeakerGenderChange,
  isPlaying,
  onPlayingChange,
}) {
  const audioRef = useRef(null)
  const [currentTime, setCurrentTime] = useState(0)
  const [duration, setDuration] = useState(0)
  const [mockCommentary, setMockCommentary] = useState([
    { time: 0, original: "Welcome to today's match", translated: "Bienvenido al partido de hoy" },
    { time: 5, original: "The home team takes the field", translated: "El equipo local toma el campo" },
    { time: 10, original: "What an incredible pass!", translated: "¡Qué pase increíble!" },
  ])

  useEffect(() => {
    const audio = audioRef.current
    if (!audio) return

    const handlePlay = () => onPlayingChange(true)
    const handlePause = () => onPlayingChange(false)
    const handleTimeUpdate = () => setCurrentTime(audio.currentTime)
    const handleLoadedMetadata = () => setDuration(audio.duration)

    audio.addEventListener("play", handlePlay)
    audio.addEventListener("pause", handlePause)
    audio.addEventListener("timeupdate", handleTimeUpdate)
    audio.addEventListener("loadedmetadata", handleLoadedMetadata)

    return () => {
      audio.removeEventListener("play", handlePlay)
      audio.removeEventListener("pause", handlePause)
      audio.removeEventListener("timeupdate", handleTimeUpdate)
      audio.removeEventListener("loadedmetadata", handleLoadedMetadata)
    }
  }, [onPlayingChange])

  const togglePlay = () => {
    if (audioRef.current) {
      isPlaying ? audioRef.current.pause() : audioRef.current.play()
    }
  }

  const handleProgressChange = (e) => {
    if (audioRef.current) {
      audioRef.current.currentTime = Number.parseFloat(e.target.value)
    }
  }

  const formatTime = (seconds) => {
    if (!seconds || isNaN(seconds)) return "0:00"
    const mins = Math.floor(seconds / 60)
    const secs = Math.floor(seconds % 60)
    return `${mins}:${secs.toString().padStart(2, "0")}`
  }

  const currentCommentary = mockCommentary.find((c) => c.time <= currentTime && currentTime < c.time + 5)

  return (
    <div className="player-card">
      <div className="player-header">
        <div className="audio-info">
          <h3>{audioInput.name}</h3>
          <p>
            {audioInput.type === "file" && "📁 Uploaded File"}
            {audioInput.type === "stream" && "🌐 Live Stream"}
            {audioInput.type === "recording" && "🎙️ Live Recording"}
          </p>
        </div>
      </div>

      <audio ref={audioRef} src={audioInput.source} />

      <div className="player-controls">
        <button onClick={togglePlay} className="play-button">
          {isPlaying ? "⏸" : "▶"}
        </button>

        <div className="progress-container">
          <input
            type="range"
            min="0"
            max={duration || 0}
            value={currentTime}
            onChange={handleProgressChange}
            className="progress-slider"
          />
          <div className="time-display">
            <span>{formatTime(currentTime)}</span>
            <span>{formatTime(duration)}</span>
          </div>
        </div>
      </div>

      <div className="speaker-selector">
        <label>Speaker Gender:</label>
        <div className="gender-options">
          {["male", "female", "mixed"].map((gender) => (
            <button
              key={gender}
              className={`gender-button ${speakerGender === gender ? "active" : ""}`}
              onClick={() => onSpeakerGenderChange(gender)}
            >
              {gender === "male" && "♂ Male"}
              {gender === "female" && "♀ Female"}
              {gender === "mixed" && "♂♀ Mixed"}
            </button>
          ))}
        </div>
      </div>

      <div className="commentary-display">
        <div className="commentary-box">
          <div className="commentary-section">
            <h4>Original ({sourceLanguage.toUpperCase()})</h4>
            <p className="commentary-text">
              {currentCommentary ? currentCommentary.original : "Waiting for commentary..."}
            </p>
          </div>

          <div className="commentary-divider"></div>

          <div className="commentary-section">
            <h4>Translation ({targetLanguage.toUpperCase()})</h4>
            <p className="commentary-text translated">
              {currentCommentary ? currentCommentary.translated : "Waiting for translation..."}
            </p>
          </div>
        </div>
      </div>
    </div>
  )
}
