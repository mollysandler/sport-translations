const BASE_URL =
  "https://mollysandler--sports-translation-api-translatorservice-f-6a7378.modal.run";

let currentSessionId = null;
let captureTabId = null;
let videoFound = false;
let initialLatencyMs = null;
let captureStartVideoTime = 0;

// Adaptive rate state
let currentVideoRate = 1.0;
let replayZonePollTimer = null;
let replayZoneEnd = 0;            // video position where replay zone ends
let captureResumed = false;       // true after RESUME_CAPTURE — only then adapt rate
const RATE_ADJUST_COOLDOWN_MS = 2000;
let lastRateAdjustAt = 0;

// AbortController for cancelling an in-progress start
let startAbortController = null;

// Open side panel when extension icon is clicked
chrome.action.onClicked.addListener((tab) => {
  chrome.sidePanel.open({ tabId: tab.id });
});

// Listen for messages from side panel and offscreen document
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.type === "START_CAPTURE") {
    handleStartCapture(message).then(sendResponse).catch((err) => {
      console.error("[sw] START_CAPTURE error:", err);
      sendResponse({ error: err.message });
    });
    return true; // async response
  }

  if (message.type === "STOP_CAPTURE") {
    handleStopCapture().then(sendResponse).catch((err) => {
      console.error("[sw] STOP_CAPTURE error:", err);
      sendResponse({ error: err.message });
    });
    return true;
  }

  // Content script reports video element status
  if (message.type === "VIDEO_FOUND") {
    videoFound = true;
    captureStartVideoTime = message.currentTime || 0;
    console.log("[sw] Video element found at", captureStartVideoTime.toFixed(1), "s");
    chrome.runtime.sendMessage({
      type: "VIDEO_SYNC_STATUS",
      synced: false,
      status: "buffering",
    }).catch(() => {});
  }
  if (message.type === "VIDEO_NOT_FOUND") {
    videoFound = false;
    console.log("[sw] No video element found in tab");
    chrome.runtime.sendMessage({
      type: "VIDEO_SYNC_STATUS",
      synced: false,
      status: "no_video",
    }).catch(() => {});
  }

  // Offscreen reports the measured round-trip latency for the first segment
  if (message.type === "FIRST_SEGMENT_LATENCY") {
    initialLatencyMs = message.latencyMs;
    console.log("[sw] First segment latency:", initialLatencyMs, "ms");
  }

  // Buffer full — offscreen is now playing audio. Seek video back + set rate.
  if (message.type === "TRANSITION_TO_PLAYBACK") {
    handleTransitionToPlayback(message);
  }

  // Offscreen reports buffer health — adjust video rate adaptively
  if (message.type === "BUFFER_STATUS") {
    handleBufferStatus(message.bufferAheadMs);
  }

  // Messages from offscreen that need to reach the side panel
  if (message.type === "CAPTION") {
    console.log("[sw] Caption:", message.caption?.translated || message.caption?.text);
  }
  if (message.type === "STATUS") {
    console.log("[sw] Status update:", message.status);
  }
  if (message.type === "SILENCE_WARNING") {
    console.log("[sw] Silence warning from offscreen");
  }
  if (message.type === "CAPTURE_ERROR") {
    console.error("[sw] Capture error:", message.error);
    chrome.runtime.sendMessage({
      type: "CAPTURE_ERROR",
      error: message.error,
    }).catch(() => {});
  }
  if (message.type === "CHUNK_ERROR") {
    console.error("[sw] Chunk error:", message.error);
    chrome.runtime.sendMessage({
      type: "CHUNK_ERROR",
      error: message.error,
    }).catch(() => {});
  }
});

// -- Phase 2: Transition to playback ------------------------------------------

function handleTransitionToPlayback(msg) {
  const { totalAudioSentSec, measuredRate } = msg;

  currentVideoRate = measuredRate;
  captureResumed = false; // don't adapt rate until capture resumes
  if (captureTabId && videoFound) {
    // Query current video position, then compute seek target.
    // Seek to (currentPos - totalAudioSentSec) because that's where
    // the captured audio actually starts — NOT captureStartVideoTime,
    // which is where the video was when VIDEO_FOUND fired (potentially
    // 30+ seconds earlier due to backend cold start).
    chrome.tabs.sendMessage(captureTabId, { type: "VIDEO_REPORT_TIME" })
      .then((response) => {
        const currentPos = (response && response.currentTime) || 0;
        const seekTarget = Math.max(currentPos - totalAudioSentSec, 0);
        console.log(`[sw] Transition: video at ${currentPos.toFixed(1)}s, sent ${totalAudioSentSec.toFixed(1)}s, seeking to ${seekTarget.toFixed(1)}s, rate=${measuredRate.toFixed(2)}`);

        chrome.tabs.sendMessage(captureTabId, {
          type: "VIDEO_SEEK",
          time: seekTarget,
        }).catch(() => {});
        chrome.tabs.sendMessage(captureTabId, {
          type: "VIDEO_ADJUST_RATE",
          rate: currentVideoRate,
        }).catch(() => {});

        // Replay zone ends at seekTarget + totalAudioSentSec
        replayZoneEnd = seekTarget + totalAudioSentSec;
        startReplayZonePoll();
      })
      .catch(() => {});
  }

  chrome.runtime.sendMessage({
    type: "VIDEO_SYNC_STATUS",
    synced: true,
    status: "syncing",
    videoRate: currentVideoRate,
  }).catch(() => {});
}

// -- Replay zone polling: resume capture when video passes the zone -----------

function startReplayZonePoll() {
  stopReplayZonePoll();
  replayZonePollTimer = setInterval(() => {
    if (!captureTabId) { stopReplayZonePoll(); return; }

    chrome.tabs.sendMessage(captureTabId, { type: "VIDEO_REPORT_TIME" })
      .then((response) => {
        if (!response || !response.ok) return;
        const pos = response.currentTime;
        // Resume capture 3s before the zone ends (overlap handled by backend dedup)
        if (pos >= replayZoneEnd - 3) {
          console.log(`[sw] Replay zone ended at video pos ${pos.toFixed(1)}s (target: ${replayZoneEnd.toFixed(1)}s)`);
          stopReplayZonePoll();
          captureResumed = true;
          // Tell offscreen to resume capturing
          chrome.runtime.sendMessage({ type: "RESUME_CAPTURE" }).catch(() => {});
        }
      })
      .catch(() => {});
  }, 500);
}

function stopReplayZonePoll() {
  if (replayZonePollTimer) {
    clearInterval(replayZonePollTimer);
    replayZonePollTimer = null;
  }
}

// -- Phase 3: Adaptive video rate based on buffer health ----------------------

function handleBufferStatus(bufferAheadMs) {
  if (!videoFound || !captureTabId) return;

  const bufferSec = bufferAheadMs / 1000;
  const now = Date.now();

  // Adaptive rate: only after capture has resumed (not during replay zone,
  // where the buffer is finite and would drain if we speed up).
  if (captureResumed && now - lastRateAdjustAt >= RATE_ADJUST_COOLDOWN_MS) {
    let newRate = currentVideoRate;

    if (bufferSec > 8) {
      newRate = Math.min(currentVideoRate + 0.10, 1.0);   // buffer very healthy → speed up fast
    } else if (bufferSec > 6) {
      newRate = Math.min(currentVideoRate + 0.05, 1.0);   // buffer healthy → speed up
    } else if (bufferSec < 2) {
      newRate = Math.max(currentVideoRate - 0.05, 0.5);   // buffer low → slow down
    }
    // buffer 2-6s → hold rate

    if (newRate !== currentVideoRate) {
      console.log(`[sw] Buffer ${bufferSec.toFixed(1)}s → rate ${currentVideoRate.toFixed(2)} → ${newRate.toFixed(2)}`);
      currentVideoRate = newRate;
      lastRateAdjustAt = now;
      chrome.tabs.sendMessage(captureTabId, {
        type: "VIDEO_ADJUST_RATE",
        rate: currentVideoRate,
      }).catch(() => {});
    }
  }

  // Report to sidepanel
  chrome.runtime.sendMessage({
    type: "VIDEO_SYNC_STATUS",
    synced: bufferSec > 1,
    bufferAheadMs,
    videoRate: currentVideoRate,
  }).catch(() => {});
}

// -- Start/stop capture -------------------------------------------------------

async function handleStartCapture({ sourceLang, targetLang }) {
  if (startAbortController) {
    startAbortController.abort();
    startAbortController = null;
  }
  startAbortController = new AbortController();
  const signal = startAbortController.signal;

  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  if (!tab) throw new Error("No active tab found");
  captureTabId = tab.id;
  console.log("[sw] Starting capture on tab:", tab.id, tab.url);

  // Reset sync state
  videoFound = false;
  captureStartVideoTime = 0;
  initialLatencyMs = null;
  currentVideoRate = 1.0;
  lastRateAdjustAt = 0;
  captureResumed = false;
  stopReplayZonePoll();

  // Inject content script early (before backend call which may cold-start)
  try {
    await chrome.scripting.executeScript({
      target: { tabId: tab.id },
      files: ["content-script.js"],
    });
    console.log("[sw] Content script injected");
  } catch (err) {
    console.warn("[sw] Could not inject content script (sync unavailable):", err.message);
  }

  // Start a backend session
  console.log("[sw] Creating backend session...");
  let res;
  try {
    res = await fetch(`${BASE_URL}/session/start`, {
      method: "POST",
      signal,
    });
  } catch (err) {
    if (signal.aborted) {
      console.log("[sw] Start capture was cancelled");
      throw new Error("Capture was cancelled.");
    }
    const errorMsg = err.name === "AbortError"
      ? "Backend is taking too long to respond. Please try again."
      : "Could not reach the translation server.";
    chrome.runtime.sendMessage({
      type: "CAPTURE_ERROR",
      error: errorMsg,
    }).catch(() => {});
    throw new Error(errorMsg);
  }

  if (signal.aborted) {
    const data = await res.json().catch(() => null);
    if (data?.session_id) {
      const fd = new FormData();
      fd.append("session_id", data.session_id);
      fetch(`${BASE_URL}/session/end`, { method: "POST", body: fd }).catch(() => {});
    }
    throw new Error("Capture was cancelled.");
  }

  if (!res.ok) {
    const errorMsg = `Server error (${res.status}). Please try again.`;
    chrome.runtime.sendMessage({
      type: "CAPTURE_ERROR",
      error: errorMsg,
    }).catch(() => {});
    throw new Error(errorMsg);
  }
  const data = await res.json();
  currentSessionId = data.session_id;
  console.log("[sw] Session created:", currentSessionId);

  if (signal.aborted) {
    await cleanupSession();
    throw new Error("Capture was cancelled.");
  }

  console.log("[sw] Getting tab capture stream ID...");
  const streamId = await chrome.tabCapture.getMediaStreamId({
    targetTabId: tab.id,
  });
  console.log("[sw] Got streamId:", streamId.substring(0, 20) + "...");

  if (signal.aborted) {
    await cleanupSession();
    throw new Error("Capture was cancelled.");
  }

  await ensureOffscreenDocument();
  console.log("[sw] Offscreen document ready");

  if (signal.aborted) {
    await cleanupSession();
    throw new Error("Capture was cancelled.");
  }

  await chrome.runtime.sendMessage({
    type: "OFFSCREEN_START",
    streamId,
    sessionId: currentSessionId,
    sourceLang,
    targetLang,
  });
  console.log("[sw] Sent OFFSCREEN_START");

  startAbortController = null;
  return { sessionId: currentSessionId };
}

async function cleanupSession() {
  if (currentSessionId) {
    const fd = new FormData();
    fd.append("session_id", currentSessionId);
    await fetch(`${BASE_URL}/session/end`, { method: "POST", body: fd }).catch(() => {});
    currentSessionId = null;
  }
  captureTabId = null;
}

async function handleStopCapture() {
  if (startAbortController) {
    startAbortController.abort();
    startAbortController = null;
  }

  if (captureTabId) {
    // Reset video rate to normal before cleanup
    chrome.tabs.sendMessage(captureTabId, { type: "VIDEO_RESET_RATE" }).catch(() => {});
    chrome.tabs.sendMessage(captureTabId, { type: "VIDEO_CLEANUP" }).catch(() => {});
  }

  chrome.runtime.sendMessage({ type: "OFFSCREEN_STOP" }).catch(() => {});
  try {
    await chrome.offscreen.closeDocument();
    console.log("[sw] Offscreen document closed");
  } catch (_) {}

  if (currentSessionId) {
    const fd = new FormData();
    fd.append("session_id", currentSessionId);
    await fetch(`${BASE_URL}/session/end`, { method: "POST", body: fd }).catch(() => {});
    currentSessionId = null;
  }

  videoFound = false;
  initialLatencyMs = null;
  currentVideoRate = 1.0;
  captureTabId = null;
  stopReplayZonePoll();
  return { ok: true };
}

async function ensureOffscreenDocument() {
  const contexts = await chrome.runtime.getContexts({
    contextTypes: ["OFFSCREEN_DOCUMENT"],
  });
  if (contexts.length > 0) return;

  await chrome.offscreen.createDocument({
    url: "offscreen/offscreen.html",
    reasons: ["USER_MEDIA", "AUDIO_PLAYBACK"],
    justification: "Tab audio capture, processing, and translated audio playback",
  });
}

// Auto-stop when the captured tab is closed or navigated away
chrome.tabs.onRemoved.addListener((tabId) => {
  if (tabId === captureTabId) {
    handleStopCapture();
    chrome.runtime.sendMessage({ type: "STATUS", status: "idle" }).catch(() => {});
  }
});

chrome.tabs.onUpdated.addListener((tabId, changeInfo) => {
  if (tabId === captureTabId && changeInfo.url) {
    handleStopCapture();
    chrome.runtime.sendMessage({ type: "STATUS", status: "idle" }).catch(() => {});
  }
});
