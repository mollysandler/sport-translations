"use client";

import { useState } from "react";
import "./AudioInput.css";

export default function AudioInput({
  sourceLanguage,
  targetLanguage,
  onAudioSelected,
}) {
  const [inputMethod, setInputMethod] = useState("upload");
  const [streamUrl, setStreamUrl] = useState("");
  const [isRecording, setIsRecording] = useState(false);
  const [mediaRecorder, setMediaRecorder] = useState(null);
  const [isLoading, setIsLoading] = useState(false);

  // NEW FUNCTION: To process audio via backend
  const processAudioOnBackend = async (audioBlob, fileName) => {
    setIsLoading(true);
    const formData = new FormData();
    formData.append("audio", audioBlob, fileName);

    try {
      const response = await fetch("http://127.0.0.1:5000/translate-audio", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error("Network response was not ok");
      }

      // The backend sends back the translated audio file
      const translatedAudioBlob = await response.blob();
      const translatedAudioUrl = URL.createObjectURL(translatedAudioBlob);

      // We now pass the NEW translated audio URL to the player
      onAudioSelected({
        type: "file", // Or whatever type is appropriate
        source: translatedAudioUrl, // This is the key change
        name: `Translated - ${fileName}`,
        timestamp: new Date(),
      });
    } catch (error) {
      console.error("Error translating audio:", error);
      alert("Failed to translate audio. See console for details.");
    } finally {
      setIsLoading(false);
    }
  };

  const handleFileUpload = (e) => {
    const file = e.target.files[0];
    if (file) {
      processAudioOnBackend(file, file.name);
    }
  };

  const handleStreamUrlSubmit = (e) => {
    e.preventDefault();
    if (streamUrl.trim()) {
      onAudioSelected({
        type: "stream",
        source: streamUrl,
        name: "Live Stream",
        timestamp: new Date(),
      });
      setStreamUrl("");
    }
  };

  const handleStartRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const recorder = new MediaRecorder(stream);
      const chunks = [];

      recorder.ondataavailable = (e) => chunks.push(e.data);
      recorder.onstop = () => {
        const blob = new Blob(chunks, { type: "audio/webm" });
        const url = URL.createObjectURL(blob);
        onAudioSelected({
          type: "recording",
          source: url,
          name: "Live Recording",
          timestamp: new Date(),
        });
        stream.getTracks().forEach((track) => track.stop());
      };

      recorder.start();
      setMediaRecorder(recorder);
      setIsRecording(true);
    } catch (err) {
      alert("Microphone access denied: " + err.message);
    }
  };

  const handleStopRecording = () => {
    if (mediaRecorder) {
      mediaRecorder.onstop = (e) => {
        const blob = new Blob(mediaRecorder.chunks, { type: "audio/webm" });
        processAudioOnBackend(blob, "live_recording.webm");
        mediaRecorder.chunks = [];
        stream.getTracks().forEach((track) => track.stop());
      };
      mediaRecorder.stop();
      setIsRecording(false);
    }
  };

  return (
    <div className="audio-input-card">
      {/* Add a simple loading overlay */}
      {isLoading && (
        <div className="loading-overlay">Translating... Please wait.</div>
      )}
      <div className="audio-input-card">
        <h2>Select Commentary Source</h2>
        <div className="input-method-tabs">
          <button
            className={`tab-button ${inputMethod === "upload" ? "active" : ""}`}
            onClick={() => setInputMethod("upload")}
          >
            Upload File
          </button>
          <button
            className={`tab-button ${inputMethod === "stream" ? "active" : ""}`}
            onClick={() => setInputMethod("stream")}
          >
            Stream URL
          </button>
          <button
            className={`tab-button ${inputMethod === "record" ? "active" : ""}`}
            onClick={() => setInputMethod("record")}
          >
            Record Live
          </button>
        </div>

        <div className="input-method-content">
          {inputMethod === "upload" && (
            <div className="upload-section">
              <label htmlFor="audio-file" className="file-input-label">
                <span className="upload-icon">📤</span>
                <span>Click to upload or drag and drop</span>
                <span className="upload-hint">MP3, WAV, M4A (Max 500MB)</span>
              </label>
              <input
                id="audio-file"
                type="file"
                accept="audio/*"
                onChange={handleFileUpload}
                className="file-input"
              />
            </div>
          )}

          {inputMethod === "stream" && (
            <form onSubmit={handleStreamUrlSubmit} className="stream-form">
              <div className="stream-input-group">
                <input
                  type="url"
                  placeholder="Enter stream URL (e.g., https://example.com/stream.m3u8)"
                  value={streamUrl}
                  onChange={(e) => setStreamUrl(e.target.value)}
                  className="stream-input"
                />
                <button type="submit" className="submit-button">
                  Connect Stream
                </button>
              </div>
              <p className="stream-hint">
                Supports HLS, DASH, and direct audio streams
              </p>
            </form>
          )}

          {inputMethod === "record" && (
            <div className="record-section">
              {isRecording ? (
                <div className="recording-active">
                  <div className="recording-indicator">
                    <span className="recording-dot"></span>
                    Recording...
                  </div>
                  <button
                    onClick={handleStopRecording}
                    className="record-button stop"
                  >
                    Stop Recording
                  </button>
                </div>
              ) : (
                <button
                  onClick={handleStartRecording}
                  className="record-button start"
                >
                  🎤 Start Recording
                </button>
              )}
              <p className="record-hint">
                Record live commentary directly from your microphone
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
