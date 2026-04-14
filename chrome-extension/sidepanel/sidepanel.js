const startStopBtn = document.getElementById("startStopBtn");
const statusBadge = document.getElementById("statusBadge");
const captionsEl = document.getElementById("captions");
const emptyState = document.getElementById("emptyState");
const silenceWarning = document.getElementById("silenceWarning");
const warmingUp = document.getElementById("warmingUp");
const warmingText = document.getElementById("warmingText");
const elapsedTimer = document.getElementById("elapsedTimer");
const sourceLangEl = document.getElementById("sourceLang");
const targetLangEl = document.getElementById("targetLang");
const syncBadge = document.getElementById("syncBadge");
const errorBanner = document.getElementById("errorBanner");
const errorMessage = document.getElementById("errorMessage");
const retryBtn = document.getElementById("retryBtn");
const dismissBtn = document.getElementById("dismissBtn");

let isCapturing = false;
let captions = [];
let connectingStartTime = null;
let connectingTimerInterval = null;

// -- On panel open, clean up any stale capture from a previous session --------
// If the service worker still has an active session (e.g. panel was closed mid-
// warmup), send STOP_CAPTURE so we start fresh.
chrome.runtime.sendMessage({ type: "STOP_CAPTURE" }).catch(() => {});

// -- Load saved preferences ---------------------------------------------------

chrome.storage.local.get(
  ["sourceLang", "targetLang", "captionHistory"],
  (data) => {
    if (data.sourceLang) sourceLangEl.value = data.sourceLang;
    if (data.targetLang) targetLangEl.value = data.targetLang;
    if (data.captionHistory && data.captionHistory.length > 0) {
      captions = data.captionHistory;
      renderCaptions();
    }
  }
);

// Save language preferences on change
sourceLangEl.addEventListener("change", () => {
  chrome.storage.local.set({ sourceLang: sourceLangEl.value });
});
targetLangEl.addEventListener("change", () => {
  chrome.storage.local.set({ targetLang: targetLangEl.value });
});

// -- Error display ------------------------------------------------------------

function showError(message) {
  errorMessage.textContent = message;
  errorBanner.classList.remove("hidden");
}

function hideError() {
  errorBanner.classList.add("hidden");
  errorMessage.textContent = "";
}

retryBtn.addEventListener("click", () => {
  hideError();
  startCapture();
});

dismissBtn.addEventListener("click", () => {
  hideError();
});

// -- Cold start timer ---------------------------------------------------------

function startConnectingTimer() {
  stopConnectingTimer();
  connectingStartTime = Date.now();
  warmingText.textContent = "Warming up GPU...";
  elapsedTimer.textContent = "";

  connectingTimerInterval = setInterval(() => {
    const elapsed = Math.floor((Date.now() - connectingStartTime) / 1000);
    elapsedTimer.textContent = `(${elapsed}s)`;

    if (elapsed >= 5 && elapsed < 90) {
      warmingText.textContent = "GPU is starting up — this usually takes 30-60 seconds on first use...";
    }

    if (elapsed >= 90) {
      stopConnectingTimer();
      warmingUp.classList.add("hidden");
      showError("Connection timed out. The server may be overloaded.");
      // Reset to idle state
      isCapturing = false;
      startStopBtn.textContent = "Start Translating";
      startStopBtn.className = "btn btn-start";
      startStopBtn.disabled = false;
      setStatus("idle");
    }
  }, 1000);
}

function stopConnectingTimer() {
  if (connectingTimerInterval) {
    clearInterval(connectingTimerInterval);
    connectingTimerInterval = null;
  }
  connectingStartTime = null;
  elapsedTimer.textContent = "";
}

// -- Start / Stop -------------------------------------------------------------

startStopBtn.addEventListener("click", async () => {
  if (isCapturing) {
    stopCapture();
  } else {
    startCapture();
  }
});

async function startCapture() {
  startStopBtn.disabled = true;
  silenceWarning.classList.add("hidden");
  hideError();
  warmingUp.classList.remove("hidden");
  setStatus("connecting");
  startConnectingTimer();

  try {
    const response = await chrome.runtime.sendMessage({
      type: "START_CAPTURE",
      sourceLang: sourceLangEl.value,
      targetLang: targetLangEl.value,
    });

    if (response && response.error) {
      console.error("Start error:", response.error);
      stopConnectingTimer();
      setStatus("idle");
      warmingUp.classList.add("hidden");
      startStopBtn.disabled = false;
      showError(response.error);
      return;
    }

    isCapturing = true;
    startStopBtn.textContent = "Stop";
    startStopBtn.className = "btn btn-stop";
    startStopBtn.disabled = false;
  } catch (err) {
    console.error("Start failed:", err);
    stopConnectingTimer();
    setStatus("idle");
    warmingUp.classList.add("hidden");
    startStopBtn.disabled = false;
    showError("Failed to start capture. Please try again.");
  }
}

async function stopCapture() {
  startStopBtn.disabled = true;
  stopConnectingTimer();

  try {
    await chrome.runtime.sendMessage({ type: "STOP_CAPTURE" });
  } catch (err) {
    console.error("Stop error:", err);
  }

  isCapturing = false;
  startStopBtn.textContent = "Start Translating";
  startStopBtn.className = "btn btn-start";
  startStopBtn.disabled = false;
  warmingUp.classList.add("hidden");
  syncBadge.className = "sync-badge hidden";
  setStatus("idle");
}

// -- Status -------------------------------------------------------------------

function setStatus(status) {
  statusBadge.textContent =
    status === "idle"
      ? "Idle"
      : status === "connecting"
      ? "Connecting..."
      : status === "streaming"
      ? "Live"
      : status;
  statusBadge.className = "status-badge " + status;
}

// -- Captions -----------------------------------------------------------------

function addCaption(caption) {
  // Deduplicate: skip if text matches or overlaps with recent captions.
  // Uses substring matching to catch segments where the backend splits the
  // same speech differently on re-capture at the overlap boundary.
  const newText = (caption.translated || caption.text || "").trim();
  const newWords = newText.toLowerCase().replace(/[^\w\s]/g, "").split(/\s+/);
  const recent = captions.slice(-10);
  if (newText && recent.some(c => {
    const oldText = (c.translated || c.text || "").trim();
    if (!oldText) return false;
    // Exact match
    if (oldText === newText) return true;
    // Substring match
    if (oldText.includes(newText) || (newText.includes(oldText) && oldText.length > 20)) return true;
    // Word overlap: if 60%+ of words in the new caption appear in a recent one, it's a duplicate
    if (newWords.length >= 4) {
      const oldWords = new Set(oldText.toLowerCase().replace(/[^\w\s]/g, "").split(/\s+/));
      const overlap = newWords.filter(w => oldWords.has(w)).length;
      if (overlap / newWords.length >= 0.6) return true;
    }
    return false;
  })) {
    return;
  }
  captions.push(caption);
  // Keep last 200 captions
  if (captions.length > 200) captions = captions.slice(-200);
  renderCaptions();
  // Persist
  chrome.storage.local.set({ captionHistory: captions });
}

function renderCaptions() {
  if (captions.length === 0) {
    emptyState.classList.remove("hidden");
    return;
  }
  emptyState.classList.add("hidden");

  // Only render new captions (avoid full re-render)
  const existingCount = captionsEl.querySelectorAll(".caption-item").length;
  for (let i = existingCount; i < captions.length; i++) {
    const cap = captions[i];
    const speakerIdx = extractSpeakerIndex(cap.speaker);

    const div = document.createElement("div");
    div.className = `caption-item speaker-${speakerIdx % 5}`;

    const speakerDiv = document.createElement("div");
    speakerDiv.className = "speaker";
    speakerDiv.textContent = cap.speaker || "Speaker";

    const textDiv = document.createElement("div");
    textDiv.className = "text";
    textDiv.textContent = cap.translated || cap.text || "";

    div.appendChild(speakerDiv);
    div.appendChild(textDiv);

    if (cap.original && cap.original !== (cap.translated || cap.text)) {
      const origDiv = document.createElement("div");
      origDiv.className = "original";
      origDiv.textContent = cap.original;
      div.appendChild(origDiv);
    }

    captionsEl.appendChild(div);
  }

  // Auto-scroll
  captionsEl.scrollTop = captionsEl.scrollHeight;
}

function extractSpeakerIndex(speaker) {
  if (!speaker) return 0;
  const match = speaker.match(/(\d+)/);
  return match ? parseInt(match[1], 10) : 0;
}

// -- Listen for messages from service worker ----------------------------------

chrome.runtime.onMessage.addListener((message) => {
  if (message.type === "CAPTION") {
    warmingUp.classList.add("hidden");
    stopConnectingTimer();
    setStatus("streaming");
    addCaption(message.caption);
  }

  if (message.type === "STATUS") {
    setStatus(message.status);
    if (message.status === "streaming") {
      warmingUp.classList.add("hidden");
      stopConnectingTimer();
    }
    if (message.status === "idle") {
      isCapturing = false;
      startStopBtn.textContent = "Start Translating";
      startStopBtn.className = "btn btn-start";
      startStopBtn.disabled = false;
      warmingUp.classList.add("hidden");
      stopConnectingTimer();
      // Don't auto-hide error banner on idle — keep it visible if present
    }
  }

  if (message.type === "SILENCE_WARNING") {
    silenceWarning.classList.remove("hidden");
  }

  if (message.type === "CAPTURE_ERROR") {
    warmingUp.classList.add("hidden");
    stopConnectingTimer();
    showError(message.error);
    // Reset to idle state
    isCapturing = false;
    startStopBtn.textContent = "Start Translating";
    startStopBtn.className = "btn btn-start";
    startStopBtn.disabled = false;
    setStatus("idle");
  }

  if (message.type === "CHUNK_ERROR") {
    showError(message.error);
  }

  if (message.type === "VIDEO_SYNC_STATUS") {
    if (message.bufferAheadMs !== undefined && message.videoRate !== undefined) {
      // Adaptive rate + buffer display
      const bufSec = (message.bufferAheadMs / 1000).toFixed(1);
      const rateStr = message.videoRate.toFixed(2);
      if (message.synced) {
        syncBadge.textContent = `Synced (${bufSec}s buf, ${rateStr}x)`;
        syncBadge.className = "sync-badge synced";
      } else {
        syncBadge.textContent = `Low buffer (${bufSec}s, ${rateStr}x)`;
        syncBadge.className = "sync-badge drifting";
      }
    } else if (message.status === "syncing") {
      const rateStr = message.videoRate ? message.videoRate.toFixed(2) : "?";
      syncBadge.textContent = `Syncing... (${rateStr}x)`;
      syncBadge.className = "sync-badge waiting";
    } else if (message.latencyMs !== undefined) {
      // Initial sync from FIRST_SEGMENT_READY
      syncBadge.textContent = `Synced (${(message.latencyMs / 1000).toFixed(1)}s delay)`;
      syncBadge.className = "sync-badge synced";
    } else if (message.status === "no_video") {
      syncBadge.textContent = "No video detected";
      syncBadge.className = "sync-badge no-video";
    } else if (message.status === "buffering") {
      syncBadge.textContent = "Buffering...";
      syncBadge.className = "sync-badge waiting";
    } else if (message.status === "waiting") {
      syncBadge.textContent = "Sync waiting...";
      syncBadge.className = "sync-badge waiting";
    }
  }
});
