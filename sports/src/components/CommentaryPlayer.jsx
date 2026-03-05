"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import "./CommentaryPlayer.css";

export default function CommentaryPlayer({
  audioInput,
  captions = [],
  sourceLanguage,
  targetLanguage,
  isPlaying,
  onPlayingChange,
  isConnecting,
  onDownloadSrt,
  onDownloadAudio,
}) {
  const audioRef = useRef(null);
  const scrollRef = useRef(null);

  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);

  const isLive = audioInput.type === "live";

  useEffect(() => {
    if (!isLive) return;
    const el = scrollRef.current;
    if (!el) return;
    el.scrollTop = el.scrollHeight;
  }, [isLive, captions.length]);

  useEffect(() => {
    if (isLive) return;

    const audio = audioRef.current;
    if (!audio) return;

    const handlePlay = () => onPlayingChange(true);
    const handlePause = () => onPlayingChange(false);
    const handleTimeUpdate = () => setCurrentTime(audio.currentTime);
    const handleLoadedMetadata = () => setDuration(audio.duration);

    audio.addEventListener("play", handlePlay);
    audio.addEventListener("pause", handlePause);
    audio.addEventListener("timeupdate", handleTimeUpdate);
    audio.addEventListener("loadedmetadata", handleLoadedMetadata);

    return () => {
      audio.removeEventListener("play", handlePlay);
      audio.removeEventListener("pause", handlePause);
      audio.removeEventListener("timeupdate", handleTimeUpdate);
      audio.removeEventListener("loadedmetadata", handleLoadedMetadata);
    };
  }, [isLive, onPlayingChange]);

  const togglePlay = () => {
    if (isLive) return;
    const a = audioRef.current;
    if (!a) return;
    isPlaying ? a.pause() : a.play();
  };

  const handleProgressChange = (e) => {
    if (isLive) return;
    const a = audioRef.current;
    if (!a) return;
    a.currentTime = Number.parseFloat(e.target.value);
  };

  const formatTime = (seconds) => {
    if (!seconds || isNaN(seconds)) return "0:00";
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, "0")}`;
  };

  const getSpeakerColor = (speaker) => {
    const colors = ["#2563eb", "#dc2626", "#16a34a", "#d97706", "#9333ea"];
    const num = parseInt(String(speaker).replace(/\D/g, "") || "0", 10);
    return colors[num % colors.length];
  };

  const captionsToShow = useMemo(() => {
    if (isLive) return captions;
    return captions.filter(
      (c) => currentTime >= c.startTime && currentTime <= c.endTime,
    );
  }, [isLive, captions, currentTime]);

  return (
    <div className="player-card">
      <div className="player-header">
        <div className="audio-info">
          <h3>{audioInput?.name || "Translation"}</h3>
          <p>
            {audioInput.type === "file" && "Uploaded File"}
            {audioInput.type === "stream" && "Stream"}
            {audioInput.type === "live" && "Live Translation"}
          </p>
        </div>
        {captions.length > 0 && (
          <div className="download-buttons">
            <button className="download-button" onClick={onDownloadSrt}>
              Download SRT
            </button>
            <button className="download-button" onClick={onDownloadAudio}>
              Download Audio
            </button>
          </div>
        )}
      </div>

      {!isLive && <audio ref={audioRef} src={audioInput.source} />}

      {!isLive && (
        <div className="player-controls">
          <button onClick={togglePlay} className="play-button">
            {isPlaying ? "||" : ">"}
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
      )}

      <div className="commentary-display">
        <div
          className="commentary-box"
          ref={scrollRef}
          style={isLive ? { maxHeight: 420, overflowY: "auto" } : {}}
        >
          <div className="commentary-section">
            <h4>Original ({sourceLanguage.toUpperCase()})</h4>
            <div className="captions-list">
              {captionsToShow.length > 0 ? (
                captionsToShow.map((cap, index) => (
                  <div key={index} className="caption-item">
                    <span
                      className="speaker-label"
                      style={{ backgroundColor: getSpeakerColor(cap.speaker) }}
                    >
                      {cap.speaker}
                    </span>
                    <p>{cap.original}</p>
                  </div>
                ))
              ) : (
                <p className="placeholder-text">
                  {isConnecting
                    ? "Waiting for server..."
                    : isLive
                      ? "Streaming translation..."
                      : "Waiting for commentary..."}
                </p>
              )}
            </div>
          </div>

          <div className="commentary-divider"></div>

          <div className="commentary-section">
            <h4>Translation ({targetLanguage.toUpperCase()})</h4>
            <div className="captions-list">
              {captionsToShow.length > 0 ? (
                captionsToShow.map((cap, index) => (
                  <div key={index} className="caption-item">
                    <span
                      className="speaker-label"
                      style={{ backgroundColor: getSpeakerColor(cap.speaker) }}
                    >
                      {cap.speaker}
                    </span>
                    <p className="translated-text">{cap.translated}</p>
                  </div>
                ))
              ) : (
                <p className="placeholder-text">
                  {isConnecting
                    ? "Waiting for server..."
                    : isLive
                      ? "Streaming translation..."
                      : "Waiting for translation..."}
                </p>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
