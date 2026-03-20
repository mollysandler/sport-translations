const startStopBtn = document.getElementById("startStopBtn");
const statusBadge = document.getElementById("statusBadge");
const captionsEl = document.getElementById("captions");
const emptyState = document.getElementById("emptyState");
const silenceWarning = document.getElementById("silenceWarning");
const warmingUp = document.getElementById("warmingUp");
const sourceLangEl = document.getElementById("sourceLang");
const targetLangEl = document.getElementById("targetLang");

let isCapturing = false;
let captions = [];

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
  warmingUp.classList.remove("hidden");
  setStatus("connecting");

  try {
    const response = await chrome.runtime.sendMessage({
      type: "START_CAPTURE",
      sourceLang: sourceLangEl.value,
      targetLang: targetLangEl.value,
    });

    if (response && response.error) {
      console.error("Start error:", response.error);
      setStatus("idle");
      warmingUp.classList.add("hidden");
      startStopBtn.disabled = false;
      return;
    }

    isCapturing = true;
    startStopBtn.textContent = "Stop";
    startStopBtn.className = "btn btn-stop";
    startStopBtn.disabled = false;
  } catch (err) {
    console.error("Start failed:", err);
    setStatus("idle");
    warmingUp.classList.add("hidden");
    startStopBtn.disabled = false;
  }
}

async function stopCapture() {
  startStopBtn.disabled = true;

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

    if (cap.original) {
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
    setStatus("streaming");
    addCaption(message.caption);
  }

  if (message.type === "STATUS") {
    setStatus(message.status);
    if (message.status === "idle") {
      isCapturing = false;
      startStopBtn.textContent = "Start Translating";
      startStopBtn.className = "btn btn-start";
      startStopBtn.disabled = false;
      warmingUp.classList.add("hidden");
    }
  }

  if (message.type === "SILENCE_WARNING") {
    silenceWarning.classList.remove("hidden");
  }
});
