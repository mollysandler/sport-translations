/**
 * Integration tests: pure-function logic tests and cross-component behavioral tests.
 * Source-text tests and duplicates removed — those are now covered behaviorally in
 * offscreen.test.js, service-worker.test.js, and sidepanel.test.js.
 */
const { createMockVideo, createChromeMock, evalScript } = require("./helpers");

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
// Adaptive rate logic (pure function — mirrors service-worker.js handleBufferStatus)
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
    expect(steps).toBeLessThanOrEqual(6);
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
  function adaptRate(currentRate, bufferSec) {
    if (bufferSec > 8) return Math.min(currentRate + 0.10, 1.0);
    if (bufferSec > 6) return Math.min(currentRate + 0.05, 1.0);
    if (bufferSec < 2) return Math.max(currentRate - 0.05, 0.5);
    return currentRate;
  }

  test("recovers from 0.5 to 1.0 in <= 10 steps with very healthy buffer", () => {
    let rate = 0.5;
    let steps = 0;
    while (rate < 1.0) {
      rate = adaptRate(rate, 10);
      steps++;
    }
    expect(steps).toBeLessThanOrEqual(10);
  });

  test("recovers from 0.5 to 1.0 in <= 15 steps with healthy buffer", () => {
    let rate = 0.5;
    let steps = 0;
    while (rate < 1.0) {
      rate = adaptRate(rate, 7);
      steps++;
    }
    expect(steps).toBeLessThanOrEqual(15);
  });
});

// ============================================================================
// Seek-back position correctness (behavioral)
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
