const BASE_URL =
  "https://mollysandler--sports-translation-api-translatorservice-f-6a7378.modal.run";

let currentSessionId = null;
let captureTabId = null;

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

  // Messages from offscreen that need to reach the side panel:
  // chrome.runtime.sendMessage broadcasts to ALL other contexts,
  // so the side panel already receives these directly from offscreen.
  // No relay needed — just log for debugging.
  if (message.type === "CAPTION") {
    console.log("[sw] Caption received from offscreen:", message.caption?.translated || message.caption?.text);
  }
  if (message.type === "STATUS") {
    console.log("[sw] Status update:", message.status);
  }
  if (message.type === "SILENCE_WARNING") {
    console.log("[sw] Silence warning from offscreen");
  }
  if (message.type === "CAPTURE_ERROR") {
    console.error("[sw] Capture error:", message.error);
  }
});

async function handleStartCapture({ sourceLang, targetLang }) {
  // Get the active tab
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  if (!tab) throw new Error("No active tab found");
  captureTabId = tab.id;
  console.log("[sw] Starting capture on tab:", tab.id, tab.url);

  // Start a backend session
  console.log("[sw] Creating backend session...");
  const res = await fetch(`${BASE_URL}/session/start`, { method: "POST" });
  const data = await res.json();
  currentSessionId = data.session_id;
  console.log("[sw] Session created:", currentSessionId);

  // Get tab audio stream ID
  console.log("[sw] Getting tab capture stream ID...");
  const streamId = await chrome.tabCapture.getMediaStreamId({
    targetTabId: tab.id,
  });
  console.log("[sw] Got streamId:", streamId.substring(0, 20) + "...");

  // Ensure offscreen document exists
  await ensureOffscreenDocument();
  console.log("[sw] Offscreen document ready");

  // Tell offscreen to start capturing
  await chrome.runtime.sendMessage({
    type: "OFFSCREEN_START",
    streamId,
    sessionId: currentSessionId,
    sourceLang,
    targetLang,
  });
  console.log("[sw] Sent OFFSCREEN_START");

  return { sessionId: currentSessionId };
}

async function handleStopCapture() {
  // Tell offscreen to stop
  chrome.runtime.sendMessage({ type: "OFFSCREEN_STOP" }).catch(() => {});

  // End backend session
  if (currentSessionId) {
    const fd = new FormData();
    fd.append("session_id", currentSessionId);
    await fetch(`${BASE_URL}/session/end`, { method: "POST", body: fd }).catch(
      () => {}
    );
    currentSessionId = null;
  }

  captureTabId = null;
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
