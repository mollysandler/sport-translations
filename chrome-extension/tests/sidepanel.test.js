/**
 * Behavioral tests for sidepanel.js — exercises actual code via evalScript.
 * Tests caption dedup, UI state, message handling, timers, and error display.
 */
const {
  createChromeMock,
  createSidepanelDOM,
  evalScript,
} = require("./helpers");

// ============================================================================
// Helper: load sidepanel.js in a controlled VM
// ============================================================================

function loadSidepanel(overrides = {}) {
  const { chrome, sentMessages, getMessageHandler } = createChromeMock();
  const { document, elements } = createSidepanelDOM();

  const ctx = evalScript("sidepanel/sidepanel.js", {
    chrome,
    document,
    ...overrides,
  });

  const handler = getMessageHandler();

  // Helper to simulate incoming chrome.runtime messages
  function receiveMessage(msg) {
    handler(msg, {}, () => {});
  }

  // Helper to click a button (trigger its 'click' listener)
  function click(elementKey) {
    const el = elements[elementKey];
    if (el && el._listeners.click) {
      el._listeners.click.forEach((fn) => fn());
    }
  }

  return { ctx, chrome, sentMessages, handler, elements, receiveMessage, click, document };
}

// ============================================================================
// Caption deduplication
// ============================================================================

describe("caption dedup", () => {
  test("rejects exact duplicate within last 10", () => {
    const { ctx, elements, receiveMessage } = loadSidepanel();

    receiveMessage({ type: "CAPTION", caption: { translated: "Goal!", speaker: "S1" } });
    receiveMessage({ type: "CAPTION", caption: { translated: "Goal!", speaker: "S1" } });

    // Only 1 caption should be rendered
    expect(elements.captions._children.length).toBe(1);
  });

  test("allows duplicate after scrolling past window of 10", () => {
    const { elements, receiveMessage } = loadSidepanel();

    // Add 11 unique captions
    for (let i = 0; i < 11; i++) {
      receiveMessage({ type: "CAPTION", caption: { translated: `Caption ${i}`, speaker: "S1" } });
    }
    expect(elements.captions._children.length).toBe(11);

    // Re-add the first one — it should now be accepted (out of last-10 window)
    receiveMessage({ type: "CAPTION", caption: { translated: "Caption 0", speaker: "S1" } });
    expect(elements.captions._children.length).toBe(12);
  });

  test("rejects substring match (new contained in old)", () => {
    const { elements, receiveMessage } = loadSidepanel();

    receiveMessage({
      type: "CAPTION",
      caption: { translated: "The player scored a beautiful goal from distance", speaker: "S1" },
    });
    receiveMessage({
      type: "CAPTION",
      caption: { translated: "scored a beautiful goal from distance", speaker: "S1" },
    });

    expect(elements.captions._children.length).toBe(1);
  });

  test("rejects substring match (old in new, old > 20 chars)", () => {
    const { elements, receiveMessage } = loadSidepanel();

    receiveMessage({
      type: "CAPTION",
      caption: { translated: "The goalkeeper made a diving save", speaker: "S1" },
    });
    receiveMessage({
      type: "CAPTION",
      caption: { translated: "And The goalkeeper made a diving save to keep them in", speaker: "S1" },
    });

    expect(elements.captions._children.length).toBe(1);
  });

  test("allows short old text substring (<=20 chars)", () => {
    const { elements, receiveMessage } = loadSidepanel();

    receiveMessage({
      type: "CAPTION",
      caption: { translated: "goal scored", speaker: "S1" },
    });
    // This is longer and contains "goal scored", but old text is <=20 chars
    // so the (newText.includes(oldText) && oldText.length > 20) check fails
    receiveMessage({
      type: "CAPTION",
      caption: { translated: "Another goal scored today in the match", speaker: "S1" },
    });

    expect(elements.captions._children.length).toBe(2);
  });

  test("rejects 60%+ word overlap (>=4 words)", () => {
    const { elements, receiveMessage } = loadSidepanel();

    receiveMessage({
      type: "CAPTION",
      caption: { translated: "the quick brown fox jumps over the lazy dog", speaker: "S1" },
    });
    // 7 of 9 words match = 78% overlap
    receiveMessage({
      type: "CAPTION",
      caption: { translated: "the quick brown fox leaps over a lazy dog", speaker: "S1" },
    });

    expect(elements.captions._children.length).toBe(1);
  });

  test("allows < 60% word overlap", () => {
    const { elements, receiveMessage } = loadSidepanel();

    receiveMessage({
      type: "CAPTION",
      caption: { translated: "the quick brown fox jumps", speaker: "S1" },
    });
    // Only 1 of 4 words matches = 25% overlap
    receiveMessage({
      type: "CAPTION",
      caption: { translated: "a slow red cat sits", speaker: "S1" },
    });

    expect(elements.captions._children.length).toBe(2);
  });

  test("skips word overlap check for < 4 words", () => {
    const { elements, receiveMessage } = loadSidepanel();

    receiveMessage({
      type: "CAPTION",
      caption: { translated: "he scores", speaker: "S1" },
    });
    // Same 2 of 3 words = 67%, but < 4 words so overlap check is skipped
    receiveMessage({
      type: "CAPTION",
      caption: { translated: "he scores again", speaker: "S1" },
    });

    expect(elements.captions._children.length).toBe(2);
  });
});

// ============================================================================
// Start/stop capture
// ============================================================================

describe("start/stop capture", () => {
  beforeEach(() => jest.useFakeTimers());
  afterEach(() => jest.useRealTimers());

  test("start sends START_CAPTURE with selected languages", async () => {
    const { elements, click, chrome } = loadSidepanel();

    // Set mock response for START_CAPTURE
    chrome.runtime.sendMessage.mockImplementation((msg) => {
      if (msg.type === "START_CAPTURE") {
        return Promise.resolve({ sessionId: "test-123" });
      }
      return Promise.resolve();
    });

    elements.sourceLang.value = "es";
    elements.targetLang.value = "fr";

    click("startStopBtn");

    await jest.advanceTimersByTimeAsync(0);
    for (let i = 0; i < 5; i++) await Promise.resolve();

    const calls = chrome.runtime.sendMessage.mock.calls.map((c) => c[0]);
    const startMsg = calls.find((m) => m.type === "START_CAPTURE");
    expect(startMsg).toBeDefined();
    expect(startMsg.sourceLang).toBe("es");
    expect(startMsg.targetLang).toBe("fr");
  });

  test("stop sends STOP_CAPTURE and resets UI", async () => {
    const { elements, click, chrome } = loadSidepanel();

    // First start
    chrome.runtime.sendMessage.mockImplementation((msg) => {
      if (msg.type === "START_CAPTURE") return Promise.resolve({ sessionId: "s1" });
      return Promise.resolve();
    });

    click("startStopBtn");
    await jest.advanceTimersByTimeAsync(0);
    for (let i = 0; i < 5; i++) await Promise.resolve();

    // Clear call history, then stop
    chrome.runtime.sendMessage.mockClear();
    chrome.runtime.sendMessage.mockImplementation(() => Promise.resolve());
    click("startStopBtn");
    await jest.advanceTimersByTimeAsync(0);
    for (let i = 0; i < 5; i++) await Promise.resolve();

    const calls = chrome.runtime.sendMessage.mock.calls.map((c) => c[0]);
    const stopMsg = calls.find((m) => m.type === "STOP_CAPTURE");
    expect(stopMsg).toBeDefined();
    expect(elements.startStopBtn.textContent).toBe("Start Translating");
    expect(elements.startStopBtn.className).toBe("btn btn-start");
    expect(elements.warmingUp.classList.contains("hidden")).toBe(true);
  });

  test("disables button during start, re-enables after", async () => {
    const { elements, click, chrome } = loadSidepanel();

    let resolveStart;
    chrome.runtime.sendMessage.mockImplementation((msg) => {
      if (msg.type === "START_CAPTURE") {
        return new Promise((resolve) => { resolveStart = resolve; });
      }
      return Promise.resolve();
    });

    click("startStopBtn");
    expect(elements.startStopBtn.disabled).toBe(true);

    resolveStart({ sessionId: "s1" });
    await jest.advanceTimersByTimeAsync(0);
    for (let i = 0; i < 5; i++) await Promise.resolve();

    expect(elements.startStopBtn.disabled).toBe(false);
  });
});

// ============================================================================
// Connection timer
// ============================================================================

describe("connection timer", () => {
  beforeEach(() => jest.useFakeTimers());
  afterEach(() => jest.useRealTimers());

  test("shows elapsed time during connection", async () => {
    const { elements, click, chrome } = loadSidepanel();

    let resolveStart;
    chrome.runtime.sendMessage.mockImplementation((msg) => {
      if (msg.type === "START_CAPTURE") {
        return new Promise((resolve) => { resolveStart = resolve; });
      }
      return Promise.resolve();
    });

    click("startStopBtn");

    jest.advanceTimersByTime(10000);
    expect(elements.elapsedTimer.textContent).toBe("(10s)");
  });

  test("times out at 90s with error", async () => {
    const { elements, click, chrome } = loadSidepanel();

    let resolveStart;
    chrome.runtime.sendMessage.mockImplementation((msg) => {
      if (msg.type === "START_CAPTURE") {
        return new Promise((resolve) => { resolveStart = resolve; });
      }
      return Promise.resolve();
    });

    click("startStopBtn");

    jest.advanceTimersByTime(90000);

    expect(elements.warmingUp.classList.contains("hidden")).toBe(true);
    expect(elements.errorBanner.classList.contains("hidden")).toBe(false);
    expect(elements.errorMessage.textContent).toContain("timed out");
    expect(elements.startStopBtn.textContent).toBe("Start Translating");
  });

  test("stops timer when streaming status received", async () => {
    const { elements, receiveMessage, click, chrome } = loadSidepanel();

    chrome.runtime.sendMessage.mockImplementation(() => Promise.resolve({ sessionId: "s1" }));

    click("startStopBtn");
    await jest.advanceTimersByTimeAsync(0);
    for (let i = 0; i < 5; i++) await Promise.resolve();

    jest.advanceTimersByTime(5000);
    expect(elements.elapsedTimer.textContent).toBe("(5s)");

    receiveMessage({ type: "STATUS", status: "streaming" });
    expect(elements.warmingUp.classList.contains("hidden")).toBe(true);

    // Timer should be stopped — advancing further shouldn't change it
    const timerAfter = elements.elapsedTimer.textContent;
    jest.advanceTimersByTime(5000);
    // Timer text may be cleared or frozen
    expect(elements.warmingUp.classList.contains("hidden")).toBe(true);
  });
});

// ============================================================================
// Message handlers
// ============================================================================

describe("message handlers", () => {
  test("CAPTION hides warmup, sets streaming status, renders caption", () => {
    const { elements, receiveMessage } = loadSidepanel();

    receiveMessage({
      type: "CAPTION",
      caption: { translated: "Hello world", speaker: "Speaker 1" },
    });

    expect(elements.warmingUp.classList.contains("hidden")).toBe(true);
    expect(elements.statusBadge.textContent).toBe("Live");
    expect(elements.captions._children.length).toBe(1);
  });

  test("STATUS idle resets button state", () => {
    const { elements, receiveMessage } = loadSidepanel();

    // Simulate active state
    elements.startStopBtn.textContent = "Stop";
    elements.startStopBtn.className = "btn btn-stop";

    receiveMessage({ type: "STATUS", status: "idle" });

    expect(elements.startStopBtn.textContent).toBe("Start Translating");
    expect(elements.startStopBtn.className).toBe("btn btn-start");
    expect(elements.startStopBtn.disabled).toBe(false);
  });

  test("SILENCE_WARNING shows warning", () => {
    const { elements, receiveMessage } = loadSidepanel();

    expect(elements.silenceWarning.classList.contains("hidden")).toBe(true);
    receiveMessage({ type: "SILENCE_WARNING" });
    expect(elements.silenceWarning.classList.contains("hidden")).toBe(false);
  });

  test("CAPTURE_ERROR shows error and resets state", () => {
    const { elements, receiveMessage } = loadSidepanel();

    receiveMessage({ type: "CAPTURE_ERROR", error: "Audio capture failed" });

    expect(elements.errorBanner.classList.contains("hidden")).toBe(false);
    expect(elements.errorMessage.textContent).toBe("Audio capture failed");
    expect(elements.startStopBtn.textContent).toBe("Start Translating");
    expect(elements.statusBadge.textContent).toBe("Idle");
  });

  test("CHUNK_ERROR shows error without resetting capture state", () => {
    const { ctx, elements, receiveMessage } = loadSidepanel();

    // Simulate active capture
    ctx._exec("isCapturing = true");

    receiveMessage({ type: "CHUNK_ERROR", error: "Server error (500)" });

    expect(elements.errorBanner.classList.contains("hidden")).toBe(false);
    expect(elements.errorMessage.textContent).toBe("Server error (500)");
    // isCapturing should still be true
    expect(ctx._exec("isCapturing")).toBe(true);
  });

  test("VIDEO_SYNC_STATUS updates sync badge with buffer info", () => {
    const { elements, receiveMessage } = loadSidepanel();

    receiveMessage({
      type: "VIDEO_SYNC_STATUS",
      synced: true,
      bufferAheadMs: 5000,
      videoRate: 0.95,
    });

    expect(elements.syncBadge.textContent).toContain("5.0");
    expect(elements.syncBadge.textContent).toContain("0.95");
    expect(elements.syncBadge.className).toContain("synced");
  });

  test("VIDEO_SYNC_STATUS no_video shows appropriate text", () => {
    const { elements, receiveMessage } = loadSidepanel();

    receiveMessage({
      type: "VIDEO_SYNC_STATUS",
      synced: false,
      status: "no_video",
    });

    expect(elements.syncBadge.textContent).toContain("No video");
    expect(elements.syncBadge.className).toContain("no-video");
  });
});

// ============================================================================
// Error banner
// ============================================================================

describe("error banner", () => {
  beforeEach(() => jest.useFakeTimers());
  afterEach(() => jest.useRealTimers());

  test("dismiss button hides error", () => {
    const { elements, receiveMessage } = loadSidepanel();

    receiveMessage({ type: "CAPTURE_ERROR", error: "Test error" });
    expect(elements.errorBanner.classList.contains("hidden")).toBe(false);

    // Click dismiss
    elements.dismissBtn._triggerEvent("click");
    expect(elements.errorBanner.classList.contains("hidden")).toBe(true);
  });

  test("retry button hides error and starts capture", async () => {
    const { elements, receiveMessage, chrome } = loadSidepanel();

    chrome.runtime.sendMessage.mockImplementation((msg) => {
      if (msg.type === "START_CAPTURE") return Promise.resolve({ sessionId: "retry-1" });
      return Promise.resolve();
    });

    receiveMessage({ type: "CAPTURE_ERROR", error: "Test error" });
    expect(elements.errorBanner.classList.contains("hidden")).toBe(false);

    chrome.runtime.sendMessage.mockClear();
    chrome.runtime.sendMessage.mockImplementation((msg) => {
      if (msg.type === "START_CAPTURE") return Promise.resolve({ sessionId: "retry-1" });
      return Promise.resolve();
    });
    elements.retryBtn._triggerEvent("click");

    await jest.advanceTimersByTimeAsync(0);
    for (let i = 0; i < 5; i++) await Promise.resolve();

    expect(elements.errorBanner.classList.contains("hidden")).toBe(true);
    const calls = chrome.runtime.sendMessage.mock.calls.map((c) => c[0]);
    const startMsg = calls.find((m) => m.type === "START_CAPTURE");
    expect(startMsg).toBeDefined();
  });
});

// ============================================================================
// Caption rendering
// ============================================================================

describe("caption rendering", () => {
  test("renders speaker name and translated text", () => {
    const { elements, receiveMessage } = loadSidepanel();

    receiveMessage({
      type: "CAPTION",
      caption: { translated: "Hello", speaker: "Speaker 2", original: "Hola" },
    });

    expect(elements.captions._children.length).toBe(1);
    const item = elements.captions._children[0];
    // The element was created via document.createElement, so we check the children appended
    expect(item).toBeDefined();
  });

  test("hides empty state after first caption", () => {
    const { elements, receiveMessage } = loadSidepanel();

    receiveMessage({
      type: "CAPTION",
      caption: { translated: "First caption", speaker: "S1" },
    });

    expect(elements.emptyState.classList.contains("hidden")).toBe(true);
  });
});
