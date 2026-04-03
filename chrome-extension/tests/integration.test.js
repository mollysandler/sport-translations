/**
 * Integration tests for the throughput-matched sync model.
 */
const { createMockVideo, createChromeMock, evalScript } = require("./helpers");

// ============================================================================
// Caption deduplication
// ============================================================================

describe("caption deduplication", () => {
  function createCaptionStore() {
    const captions = [];
    function addCaption(caption) {
      const newText = (caption.translated || caption.text || "").trim();
      const recent = captions.slice(-5);
      if (newText && recent.some(c => (c.translated || c.text || "").trim() === newText)) return false;
      captions.push(caption);
      return true;
    }
    return { captions, addCaption };
  }

  test("rejects exact duplicate of recent caption", () => {
    const { captions, addCaption } = createCaptionStore();
    addCaption({ translated: "Hello" });
    expect(addCaption({ translated: "Hello" })).toBe(false);
    expect(captions).toHaveLength(1);
  });

  test("allows duplicate after scrolling out of last 5", () => {
    const { addCaption } = createCaptionStore();
    "ABCDEF".split("").forEach(c => addCaption({ translated: c }));
    expect(addCaption({ translated: "A" })).toBe(true);
  });
});

// ============================================================================
// Original text display
// ============================================================================

describe("original text display", () => {
  function shouldShowOriginal(cap) {
    return cap.original && cap.original !== (cap.translated || cap.text);
  }

  test("hides when same as translated", () => {
    expect(shouldShowOriginal({ translated: "Hello", original: "Hello" })).toBe(false);
  });

  test("shows when different", () => {
    expect(shouldShowOriginal({ translated: "Hola", original: "Hello" })).toBe(true);
  });
});

// ============================================================================
// Throughput measurement
// ============================================================================

describe("throughput measurement", () => {
  // Now uses 0.9 safety margin (not 0.85), and measures from firstSegmentReceivedAt
  function measureRate(bufferedDurationSec, elapsedMs) {
    const raw = elapsedMs > 0 ? bufferedDurationSec / (elapsedMs / 1000) : 0.75;
    return Math.max(Math.min(raw, 1.0) * 0.9, 0.5);
  }

  test("8s buffered in 8s → rate 0.9", () => {
    expect(measureRate(8, 8000)).toBeCloseTo(0.9);
  });

  test("8s buffered in 6s → rate 0.9 (capped at 1.0 * 0.9)", () => {
    expect(measureRate(8, 6000)).toBeCloseTo(0.9);
  });

  test("8s buffered in 16s → rate 0.5 (floored)", () => {
    expect(measureRate(8, 16000)).toBe(0.5);
  });

  test("8s buffered in 10s → rate ~0.72", () => {
    expect(measureRate(8, 10000)).toBeCloseTo(0.72);
  });
});

// ============================================================================
// Adaptive rate logic
// ============================================================================

describe("adaptive rate simulation", () => {
  function adaptRate(currentRate, bufferSec) {
    if (bufferSec > 8) return Math.min(currentRate + 0.10, 1.0);
    if (bufferSec > 6) return Math.min(currentRate + 0.05, 1.0);
    if (bufferSec < 2) return Math.max(currentRate - 0.05, 0.5);
    return currentRate;
  }

  test("very healthy buffer speeds up fast", () => {
    expect(adaptRate(0.7, 9)).toBeCloseTo(0.80);
  });

  test("healthy buffer speeds up moderately", () => {
    expect(adaptRate(0.7, 7)).toBeCloseTo(0.75);
  });

  test("low buffer slows down", () => {
    expect(adaptRate(0.7, 1)).toBeCloseTo(0.65);
  });

  test("normal buffer holds rate", () => {
    expect(adaptRate(0.7, 4)).toBe(0.7);
  });

  test("rate never exceeds 1.0", () => {
    expect(adaptRate(0.95, 9)).toBe(1.0);
  });

  test("rate never goes below 0.5", () => {
    expect(adaptRate(0.52, 0.5)).toBe(0.5);
  });

  test("convergence: starts at 0.5, with very healthy buffer, reaches 1.0 fast", () => {
    let rate = 0.5;
    let steps = 0;
    while (rate < 1.0) { rate = adaptRate(rate, 10); steps++; }
    expect(rate).toBe(1.0);
    expect(steps).toBeLessThanOrEqual(6); // ~5-6 steps × 0.10 = 0.50-0.60
  });

  test("convergence: starts at 1.0, with low buffer, approaches 0.5", () => {
    let rate = 1.0;
    for (let i = 0; i < 100; i++) rate = adaptRate(rate, 1);
    expect(rate).toBe(0.5);
  });
});

// ============================================================================
// Rate recovery speed
// ============================================================================

describe("adaptive rate recovers quickly when buffer is healthy", () => {
  // Replicate the fixed adaptive rate from service-worker.js
  function adaptRate(currentRate, bufferSec) {
    if (bufferSec > 8) return Math.min(currentRate + 0.10, 1.0);
    if (bufferSec > 6) return Math.min(currentRate + 0.05, 1.0);
    if (bufferSec < 2) return Math.max(currentRate - 0.05, 0.5);
    return currentRate;
  }

  test("recovers from 0.5 to 1.0 in ≤ 10 steps with very healthy buffer", () => {
    let rate = 0.5;
    let steps = 0;
    while (rate < 1.0) {
      rate = adaptRate(rate, 10); // very healthy buffer
      steps++;
    }
    // At +0.10 per step: 5 steps × 2s cooldown = 10 seconds. Acceptable.
    expect(steps).toBeLessThanOrEqual(10);
  });

  test("recovers from 0.5 to 1.0 in ≤ 15 steps with healthy buffer", () => {
    let rate = 0.5;
    let steps = 0;
    while (rate < 1.0) {
      rate = adaptRate(rate, 7); // healthy but not extreme
      steps++;
    }
    // At +0.05 per step: 10 steps × 2s = 20 seconds. OK.
    expect(steps).toBeLessThanOrEqual(15);
  });
});

describe("throughput measurement excludes cold start", () => {
  test("measuredRate uses firstSegmentReceivedAt (backend warm), not captureStartedAt", () => {
    const fs = require("fs");
    const path = require("path");
    const offSource = fs.readFileSync(
      path.resolve(__dirname, "..", "offscreen", "offscreen.js"), "utf-8"
    );
    const match = offSource.match(/function startPlayback\(\)([\s\S]*?)^}/m);
    expect(match[1]).toContain("firstSegmentReceivedAt");
  });

  test("safety margin is 0.9 (not 0.85) to avoid being too conservative", () => {
    const fs = require("fs");
    const path = require("path");
    const offSource = fs.readFileSync(
      path.resolve(__dirname, "..", "offscreen", "offscreen.js"), "utf-8"
    );
    const match = offSource.match(/function startPlayback\(\)([\s\S]*?)^}/m);
    expect(match[1]).toContain("* 0.9");
  });
});

// ============================================================================
// Deadlock prevention
// ============================================================================

describe("deadlock prevention: video never pauses", () => {
  test("handleBufferStatus does NOT pause video", () => {
    const fs = require("fs");
    const path = require("path");
    const swSource = fs.readFileSync(path.resolve(__dirname, "..", "service-worker.js"), "utf-8");
    const match = swSource.match(/function handleBufferStatus([\s\S]*?)^}/m);
    expect(match).not.toBeNull();
    expect(match[1]).not.toContain('"VIDEO_PAUSE"');
  });

  test("handleTransitionToPlayback does NOT pause video", () => {
    const fs = require("fs");
    const path = require("path");
    const swSource = fs.readFileSync(path.resolve(__dirname, "..", "service-worker.js"), "utf-8");
    const match = swSource.match(/function handleTransitionToPlayback([\s\S]*?)^}/m);
    expect(match).not.toBeNull();
    expect(match[1]).not.toContain('"VIDEO_PAUSE"');
  });
});

// ============================================================================
// Seek-back position correctness
// ============================================================================

describe("seek-back position", () => {
  test("VIDEO_SEEK sets video.currentTime", () => {
    const video = createMockVideo();
    video.currentTime = 100;
    const { chrome, getMessageHandler } = createChromeMock();
    const mockDoc = {
      querySelectorAll: jest.fn((sel) => (sel === "video" ? [video] : [])),
      body: {},
    };
    evalScript("content-script.js", { chrome, document: mockDoc });
    const handler = getMessageHandler();
    let response;
    handler({ type: "VIDEO_SEEK", time: 45.2 }, {}, (r) => { response = r; });
    expect(video.currentTime).toBe(45.2);
    expect(response.ok).toBe(true);
  });

  test("seek position is computed from current video pos minus totalAudioSentSec", () => {
    // The seek target should be VIDEO_REPORT_TIME → (currentTime - totalAudioSentSec)
    // NOT captureStartVideoTime (which is stale from VIDEO_FOUND time)
    const fs = require("fs");
    const path = require("path");
    const swSource = fs.readFileSync(path.resolve(__dirname, "..", "service-worker.js"), "utf-8");
    const match = swSource.match(/function handleTransitionToPlayback([\s\S]*?)^}/m);
    expect(match).not.toBeNull();
    const body = match[1];

    // Should query current position via VIDEO_REPORT_TIME
    expect(body).toContain("VIDEO_REPORT_TIME");
    // Should compute seekTarget from currentPos - totalAudioSentSec
    expect(body).toContain("currentPos - totalAudioSentSec");
    // The seek time should use seekTarget, not captureStartVideoTime directly
    expect(body).toContain("time: seekTarget");
  });
});

// ============================================================================
// Buffer-ahead calculation
// ============================================================================

describe("buffer-ahead calculation", () => {
  function bufferAheadMs(nextPlayTime, currentTime) {
    return (nextPlayTime - currentTime) * 1000;
  }

  test("positive = audio scheduled ahead", () => {
    expect(bufferAheadMs(15, 10)).toBe(5000);
  });

  test("zero = about to run out", () => {
    expect(bufferAheadMs(10, 10)).toBe(0);
  });

  test("negative = underrun", () => {
    expect(bufferAheadMs(8, 10)).toBe(-2000);
  });
});
