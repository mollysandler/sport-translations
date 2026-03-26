"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import "./CommentaryPlayer.css";

export default function CommentaryPlayer({
  audioInput,
  captions = [],
  sourceLanguage,
  targetLanguage,
  detectedLanguage,
  isPlaying,
  onPlayingChange,
  isConnecting,
  onDownloadSrt,
  onDownloadAudio,
  speakerNames,
  livePaused,
  onLivePauseChange,
  isMuted,
  onMuteToggle,
}) {
  const audioRef = useRef(null);
  const scrollRef = useRef(null);

  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);

  const isLive = audioInput.type === "live";
  const isRestored = audioInput.type === "restored";
  const isScrollable = isLive || isRestored;

  useEffect(() => {
    if (!isScrollable) return;
    const el = scrollRef.current;
    if (!el) return;
    el.scrollTop = el.scrollHeight;
  }, [isScrollable, captions.length]);

  useEffect(() => {
    if (isScrollable) return;

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
  }, [isScrollable, onPlayingChange]);

  const togglePlay = () => {
    if (isRestored) return;
    if (isLive) {
      onLivePauseChange?.(!livePaused);
      return;
    }
    const a = audioRef.current;
    if (!a) return;
    isPlaying ? a.pause() : a.play();
  };

  const handleProgressChange = (e) => {
    if (isScrollable) return;
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

  const getDisplayName = (speaker) => {
    return (speakerNames && speakerNames[speaker]) || speaker;
  };

  const captionsToShow = useMemo(() => {
    if (isScrollable) return captions;
    return captions.filter(
      (c) => currentTime >= c.startTime && currentTime <= c.endTime,
    );
  }, [isScrollable, captions, currentTime]);

  const livePlayLabel = livePaused ? "Resume" : "Pause";
  const batchPlayLabel = isPlaying ? "Pause" : "Play";

  return (
    <div className="player-card">
      <div className="player-header">
        <div className="audio-info">
          <h3>{audioInput?.name || "Translation"}</h3>
          <p>
            {audioInput.type === "file" && "Uploaded File"}
            {audioInput.type === "stream" && "Stream"}
            {audioInput.type === "live" && "Live Translation"}
            {audioInput.type === "restored" && "Saved Session"}
          </p>
        </div>
        {captions.length > 0 && (
          <div className="download-buttons">
            <button className="download-button" onClick={onDownloadSrt} title="Keyboard: S">
              Download SRT
            </button>
            {!isRestored && (
              <button className="download-button" onClick={onDownloadAudio} title="Keyboard: D">
                Download Audio
              </button>
            )}
          </div>
        )}
      </div>

      {!isScrollable && audioInput.source && (
        <audio ref={audioRef} src={audioInput.source} muted={isMuted} />
      )}

      {/* Play/Pause controls — shown for batch and live, not restored */}
      {!isRestored && (
        <div className="player-controls">
          <button onClick={togglePlay} className="play-button" title="Keyboard: Space">
            {isLive
              ? (livePaused ? "\u25B6" : "\u23F8")
              : (isPlaying ? "\u23F8" : "\u25B6")}
          </button>
          <span className="play-label">
            {isLive ? livePlayLabel : batchPlayLabel}
          </span>

          <button onClick={onMuteToggle} className="mute-button" title="Keyboard: M">
            {isMuted ? "\uD83D\uDD07" : "\uD83D\uDD0A"}
          </button>

          {!isLive && (
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
          )}

          {isLive && livePaused && (
            <span className="paused-indicator">Audio paused</span>
          )}
        </div>
      )}

      <div className="commentary-display">
        <div
          className="commentary-box"
          ref={scrollRef}
          style={isScrollable ? { maxHeight: 420, overflowY: "auto" } : {}}
        >
          <div className="commentary-section">
            <h4>Original ({(detectedLanguage || sourceLanguage).toUpperCase()})</h4>
            <div className="captions-list">
              {captionsToShow.length > 0 ? (
                captionsToShow.map((cap, index) => (
                  <div key={index} className="caption-item">
                    <span
                      className="speaker-label"
                      style={{ backgroundColor: getSpeakerColor(cap.speaker) }}
                    >
                      {getDisplayName(cap.speaker)}
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
                      {getDisplayName(cap.speaker)}
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

      <div className="keyboard-hints">
        <span>Space: play/pause</span>
        <span>D: download audio</span>
        <span>M: mute/unmute</span>
        <span>S: download SRT</span>
      </div>
    </div>
  );
}
