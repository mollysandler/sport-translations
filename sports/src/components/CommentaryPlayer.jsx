// "use client";

// import { useState, useRef, useEffect } from "react";
// import "./CommentaryPlayer.css";

// export default function CommentaryPlayer({
//   audioInput,
//   captions = [],
//   sourceLanguage,
//   targetLanguage,
//   speakerGender,
//   onSpeakerGenderChange,
//   isPlaying,
//   onPlayingChange,
// }) {
//   const audioRef = useRef(null);
//   const [currentTime, setCurrentTime] = useState(0);
//   const [duration, setDuration] = useState(0);
//   const isStreaming = !!audioInput?.isStreaming;
//   const captionsToShow = isStreaming
//     ? captions
//     : captions.filter(
//         (c) => currentTime >= c.startTime && currentTime <= c.endTime,
//       );

//   useEffect(() => {
//     const audio = audioRef.current;
//     if (!audio) return;

//     const handlePlay = () => onPlayingChange(true);
//     const handlePause = () => onPlayingChange(false);
//     const handleTimeUpdate = () => setCurrentTime(audio.currentTime);
//     const handleLoadedMetadata = () => setDuration(audio.duration);

//     audio.addEventListener("play", handlePlay);
//     audio.addEventListener("pause", handlePause);
//     audio.addEventListener("timeupdate", handleTimeUpdate);
//     audio.addEventListener("loadedmetadata", handleLoadedMetadata);

//     return () => {
//       audio.removeEventListener("play", handlePlay);
//       audio.removeEventListener("pause", handlePause);
//       audio.removeEventListener("timeupdate", handleTimeUpdate);
//       audio.removeEventListener("loadedmetadata", handleLoadedMetadata);
//     };
//   }, [onPlayingChange]);

//   const togglePlay = () => {
//     if (audioRef.current) {
//       isPlaying ? audioRef.current.pause() : audioRef.current.play();
//     }
//   };

//   const handleProgressChange = (e) => {
//     if (audioRef.current) {
//       audioRef.current.currentTime = Number.parseFloat(e.target.value);
//     }
//   };

//   const formatTime = (seconds) => {
//     if (!seconds || isNaN(seconds)) return "0:00";
//     const mins = Math.floor(seconds / 60);
//     const secs = Math.floor(seconds % 60);
//     return `${mins}:${secs.toString().padStart(2, "0")}`;
//   };

//   const activeCaptions = captions.filter(
//     (c) => currentTime >= c.startTime && currentTime <= c.endTime,
//   );

//   const translatedCaptionsToShow = isStreaming
//     ? captionsToShow
//     : activeCaptions;

//   const getSpeakerColor = (speaker) => {
//     // Simple hash to pick a color based on speaker number
//     const colors = ["#2563eb", "#dc2626", "#16a34a", "#d97706", "#9333ea"];
//     const num = parseInt(speaker.replace(/\D/g, "") || "0");
//     return colors[num % colors.length];
//   };

//   return (
//     <div className="player-card">
//       <div className="player-header">
//         <div className="audio-info">
//           <h3>{audioInput.name}</h3>
//           <p>
//             {audioInput.type === "file" && "📁 Uploaded File"}
//             {audioInput.type === "stream" && "🌐 Live Stream"}
//             {audioInput.type === "recording" && "🎙️ Live Recording"}
//           </p>
//         </div>
//       </div>

//       {!isStreaming && <audio ref={audioRef} src={audioInput.source} />}
//       {!isStreaming && (
//         <div className="player-controls">
//           <button onClick={togglePlay} className="play-button">
//             {isPlaying ? "⏸" : "▶"}
//           </button>

//           <div className="progress-container">
//             <input
//               type="range"
//               min="0"
//               max={duration || 0}
//               value={currentTime}
//               onChange={handleProgressChange}
//               className="progress-slider"
//             />
//             <div className="time-display">
//               <span>{formatTime(currentTime)}</span>
//               <span>{formatTime(duration)}</span>
//             </div>
//           </div>
//         </div>
//       )}

//       <div className="speaker-selector">
//         <label>Speaker Gender:</label>
//         <div className="gender-options">
//           {["male", "female", "mixed"].map((gender) => (
//             <button
//               key={gender}
//               className={`gender-button ${
//                 speakerGender === gender ? "active" : ""
//               }`}
//               onClick={() => onSpeakerGenderChange(gender)}
//             >
//               {gender === "male" && "♂ Male"}
//               {gender === "female" && "♀ Female"}
//               {gender === "mixed" && "♂♀ Mixed"}
//             </button>
//           ))}
//         </div>
//       </div>

//       <div className="commentary-display">
//         <div className="commentary-box">
//           {/* ORIGINAL COLUMN */}
//           <div className="commentary-section">
//             <h4>Original ({sourceLanguage.toUpperCase()})</h4>
//             <div className="captions-list">
//               {translatedCaptionsToShow.length > 0 ? (
//                 translatedCaptionsToShow.map((cap, index) => (
//                   <div key={index} className="caption-item">
//                     <span
//                       className="speaker-label"
//                       style={{ backgroundColor: getSpeakerColor(cap.speaker) }}
//                     >
//                       {cap.speaker}
//                     </span>
//                     <p>{cap.original}</p>
//                   </div>
//                 ))
//               ) : (
//                 <p className="placeholder-text">
//                   {isStreaming
//                     ? "Streaming translation..."
//                     : "Waiting for commentary..."}
//                 </p>
//               )}
//             </div>
//           </div>

//           <div className="commentary-divider"></div>

//           {/* TRANSLATED COLUMN */}
//           <div className="commentary-section">
//             <h4>Translation ({targetLanguage.toUpperCase()})</h4>
//             <div className="captions-list">
//               {captionsToShow.length > 0 ? (
//                 captionsToShow.map((cap, index) => (
//                   <div key={index} className="caption-item">
//                     <span
//                       className="speaker-label"
//                       style={{ backgroundColor: getSpeakerColor(cap.speaker) }}
//                     >
//                       {cap.speaker}
//                     </span>
//                     <p className="translated-text">{cap.translated}</p>
//                   </div>
//                 ))
//               ) : (
//                 <p className="placeholder-text">Waiting for translation...</p>
//               )}
//             </div>
//           </div>
//         </div>
//       </div>
//     </div>
//   );
// }

"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import "./CommentaryPlayer.css";

export default function CommentaryPlayer({
  audioInput,
  captions = [],
  sourceLanguage,
  targetLanguage,
  speakerGender,
  onSpeakerGenderChange,
  isPlaying,
  onPlayingChange,
}) {
  const audioRef = useRef(null);
  const scrollRef = useRef(null);

  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);

  const isStreaming = !!audioInput?.isStreaming;

  // Autoscroll to bottom on new captions while streaming
  useEffect(() => {
    if (!isStreaming) return;
    const el = scrollRef.current;
    if (!el) return;
    el.scrollTop = el.scrollHeight;
  }, [isStreaming, captions.length]);

  useEffect(() => {
    if (isStreaming) return; // no audio events in streaming mode

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
  }, [isStreaming, onPlayingChange]);

  const togglePlay = () => {
    if (isStreaming) return;
    const a = audioRef.current;
    if (!a) return;
    isPlaying ? a.pause() : a.play();
  };

  const handleProgressChange = (e) => {
    if (isStreaming) return;
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

  // When streaming: show ALL captions as they arrive
  // When not streaming: show only those for current playback time
  const captionsToShow = useMemo(() => {
    if (isStreaming) return captions;
    return captions.filter(
      (c) => currentTime >= c.startTime && currentTime <= c.endTime,
    );
  }, [isStreaming, captions, currentTime]);

  return (
    <div className="player-card">
      <div className="player-header">
        <div className="audio-info">
          <h3>{audioInput?.name || "Translation"}</h3>
          <p>
            {audioInput?.type === "file" && "📁 Uploaded File"}
            {audioInput?.type === "stream" && "🌐 Live Stream"}
            {audioInput?.type === "recording" && "🎙️ Live Recording"}
            {isStreaming && " • ⏳ Streaming…"}
          </p>
        </div>
      </div>

      {!isStreaming && audioInput?.source && (
        <audio ref={audioRef} src={audioInput.source} />
      )}

      {!isStreaming && audioInput?.source && (
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
      )}

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
        <div
          className="commentary-box"
          ref={scrollRef}
          style={isStreaming ? { maxHeight: 420, overflowY: "auto" } : {}}
        >
          {/* ORIGINAL COLUMN */}
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
                  {isStreaming
                    ? "Streaming translation…"
                    : "Waiting for commentary…"}
                </p>
              )}
            </div>
          </div>

          <div className="commentary-divider"></div>

          {/* TRANSLATED COLUMN */}
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
                  {isStreaming
                    ? "Streaming translation…"
                    : "Waiting for translation…"}
                </p>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
