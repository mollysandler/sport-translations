/**
 * Service worker: orchestrator for the streaming translation extension.
 *
 * Responsibilities:
 *   - Open side panel on action click
 *   - Create offscreen document and get tab stream ID
 *   - Relay messages between offscreen doc, content script, and side panel
 *   - Keepalive handling to prevent SW suspension
 */

/* global chrome */

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

let activeTabId = null;
let sessionActive = false;

// ---------------------------------------------------------------------------
// Action click -> open side panel
// ---------------------------------------------------------------------------

chrome.action.onClicked.addListener((tab) => {
  chrome.sidePanel.open({ tabId: tab.id });
});

// ---------------------------------------------------------------------------
// Message handling
// ---------------------------------------------------------------------------

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  switch (message.type) {
    case "START_CAPTURE":
      handleStartCapture(message.sourceLang, message.targetLang)
        .then((result) => sendResponse(result))
        .catch((err) => sendResponse({ error: err.message }));
      return true;

    case "STOP_CAPTURE":
      handleStopCapture()
        .then(() => sendResponse({ ok: true }))
        .catch(() => sendResponse({ ok: true }));
      return true;

    // --------------- Relay from offscreen -> side panel ---------------
    case "CAPTION":
    case "STATUS":
    case "SILENCE_WARNING":
    case "CAPTURE_ERROR":
    case "CHUNK_ERROR":
    case "VIDEO_SYNC_STATUS":
      broadcastToExtension(message);
      break;

    // --------------- Overlay commands (offscreen -> content script) ---------------
    case "SHOW_OVERLAY":
      sendToContentScript({ type: "SHOW_OVERLAY", text: message.text });
      break;

    case "HIDE_OVERLAY":
      sendToContentScript({ type: "HIDE_OVERLAY" });
      break;

    case "OVERLAY_PROGRESS":
      sendToContentScript({
        type: "UPDATE_OVERLAY",
        text: message.text,
        progress: message.progress,
      });
      break;

    // --------------- Sync initialization (offscreen -> content script) ---------------
    case "START_SYNC":
      sendToContentScript({ type: "START_SYNC" }, (response) => {
        if (response && response.mode) {
          // Report chosen sync mode back to offscreen
          sendToOffscreen({ type: "SYNC_MODE_REPORT", mode: response.mode });
        }
      });
      break;

    case "SET_DELAY":
      // Relay measured pipeline latency to content script for canvas delay
      sendToContentScript({ type: "SET_DELAY", delaySec: message.delaySec });
      break;

    case "PLAYBACK_STARTED":
      // Tell content script that audio playback began — start drawing canvas frames.
      // audioStartSec tells the canvas which video position the audio starts from.
      sendToContentScript({ type: "PLAYBACK_STARTED", audioStartSec: message.audioStartSec });
      break;

    // --------------- Sync mode report (content script -> offscreen) ---------------
    case "SYNC_MODE":
      sendToOffscreen({ type: "SYNC_MODE_REPORT", mode: message.mode });
      break;

    // --------------- Seekback fallback commands (offscreen -> content script) ------
    case "VIDEO_SEEK_BACK":
      sendToContentScript({
        type: "VIDEO_SEEK_BACK",
        seekBackSec: message.seekBackSec,
      });
      break;

    case "VIDEO_ADJUST_RATE":
      sendToContentScript({ type: "VIDEO_ADJUST_RATE", rate: message.rate });
      break;

    // --------------- Pause / Resume (side panel <-> offscreen + content script) ----
    case "PAUSE_ALL":
      sendToOffscreen({ type: "PAUSE_ALL" });
      sendToContentScript({ type: "PAUSE_ALL" });
      break;

    case "RESUME_ALL":
      sendToOffscreen({ type: "RESUME_ALL" });
      sendToContentScript({ type: "RESUME_ALL" });
      break;

    // From content script when user pauses/resumes the video directly
    case "USER_PAUSED_VIDEO":
    case "USER_RESUMED_VIDEO":
      broadcastToExtension(message);
      break;

    // --------------- Keepalive ---------------
    case "keepalive":
      break;
  }
});

// ---------------------------------------------------------------------------
// Start capture
// ---------------------------------------------------------------------------

async function handleStartCapture(sourceLang, targetLang) {
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  if (!tab) throw new Error("No active tab found");
  activeTabId = tab.id;

  // Inject content script
  try {
    await chrome.scripting.executeScript({
      target: { tabId: activeTabId },
      files: ["content-script.js"],
    });
  } catch (e) {
    console.warn("Content script injection failed (may already be injected):", e);
  }

  // Get tab media stream ID
  const streamId = await chrome.tabCapture.getMediaStreamId({
    targetTabId: activeTabId,
  });

  // Create offscreen document
  await ensureOffscreenDocument();

  // Mute the tab so the user only hears translated audio.
  // Tab capture still receives audio even when the tab is muted.
  await chrome.tabs.update(activeTabId, { muted: true });

  // Tell offscreen to start capture
  sessionActive = true;
  return await chrome.runtime.sendMessage({
    type: "START_CAPTURE",
    streamId,
    sourceLang,
    targetLang,
  });
}

async function handleStopCapture() {
  sessionActive = false;
  try {
    await chrome.runtime.sendMessage({ type: "STOP_CAPTURE" });
  } catch (e) {}
  if (activeTabId) {
    // Unmute the tab
    chrome.tabs.update(activeTabId, { muted: false }).catch(() => {});
    sendToContentScript({ type: "VIDEO_CLEANUP" });
    activeTabId = null;
  }
}

// ---------------------------------------------------------------------------
// Offscreen document management
// ---------------------------------------------------------------------------

async function ensureOffscreenDocument() {
  const contexts = await chrome.runtime.getContexts({
    contextTypes: ["OFFSCREEN_DOCUMENT"],
  });
  if (contexts.length > 0) return;

  await chrome.offscreen.createDocument({
    url: "offscreen/offscreen.html",
    reasons: ["USER_MEDIA", "AUDIO_PLAYBACK"],
    justification: "Capture tab audio and play translated audio",
  });
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function sendToContentScript(msg, callback) {
  if (!activeTabId) return;
  chrome.tabs.sendMessage(activeTabId, msg, callback || (() => {}));
}

function sendToOffscreen(msg) {
  chrome.runtime.sendMessage(msg).catch(() => {});
}

function broadcastToExtension(msg) {
  chrome.runtime.sendMessage(msg).catch(() => {});
}
