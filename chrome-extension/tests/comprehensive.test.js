/**
 * Comprehensive tests: cross-component verification and bug detection.
 */
const fs = require("fs");
const path = require("path");
const { createMockVideo, createChromeMock, evalScript } = require("./helpers");

const offscreenSource = fs.readFileSync(
  path.resolve(__dirname, "..", "offscreen", "offscreen.js"), "utf-8"
);
const serviceWorkerSource = fs.readFileSync(
  path.resolve(__dirname, "..", "service-worker.js"), "utf-8"
);

// ============================================================================
// Message contract: all messages are handled
// ============================================================================

describe("message contract", () => {
  const offscreenSends = [
    "FIRST_SEGMENT_LATENCY", "TRANSITION_TO_PLAYBACK", "CAPTION",
    "STATUS", "SILENCE_WARNING", "CAPTURE_ERROR", "CHUNK_ERROR",
    "BUFFER_STATUS",
  ];

  const swHandles = [];
  for (const match of serviceWorkerSource.matchAll(/message\.type === "(\w+)"/g)) {
    swHandles.push(match[1]);
  }

  test("every message offscreen sends is handled by service worker", () => {
    for (const msg of offscreenSends) {
      expect(swHandles).toContain(msg);
    }
  });

  test("every tabs.sendMessage from SW is handled by content script", () => {
    const contentSource = fs.readFileSync(
      path.resolve(__dirname, "..", "content-script.js"), "utf-8"
    );
    const tabMessages = new Set();
    for (const match of serviceWorkerSource.matchAll(
      /chrome\.tabs\.sendMessage\(\w+,\s*\{\s*type:\s*"(\w+)"/g
    )) {
      tabMessages.add(match[1]);
    }
    for (const msg of tabMessages) {
      expect(contentSource).toContain(`"${msg}"`);
    }
  });
});

// ============================================================================
// No stale references in offscreen
// ============================================================================

describe("offscreen: no stale code", () => {
  test("no old drift formula variables", () => {
    // These were from previous iterations and caused ReferenceErrors
    const match = offscreenSource.match(/function checkBuffer\(\)([\s\S]*?)^}/m);
    if (match) {
      expect(match[1]).not.toContain("elapsedSinceCapture");
      expect(match[1]).not.toContain("scheduledPlayback");
      expect(match[1]).not.toContain("videoElapsed");
      expect(match[1]).not.toContain("audioScheduled");
    }
  });

  test("no FIRST_SEGMENT_READY (replaced by TRANSITION_TO_PLAYBACK)", () => {
    expect(offscreenSource).not.toContain('"FIRST_SEGMENT_READY"');
  });

  test("no skip logic (replaced by worklet mute)", () => {
    expect(offscreenSource).not.toContain("skipAudioSec");
    expect(offscreenSource).not.toContain("skippedSoFarSec");
  });
});

// ============================================================================
// Service worker: no deadlock paths
// ============================================================================

describe("service-worker: safety", () => {
  test("NEVER pauses video in any handler", () => {
    // Extract all function bodies and check none send VIDEO_PAUSE
    // (only handleStopCapture should reference VIDEO_CLEANUP, not VIDEO_PAUSE)
    const handlers = [
      /function handleTransitionToPlayback([\s\S]*?)^}/m,
      /function handleBufferStatus([\s\S]*?)^}/m,
    ];
    for (const pattern of handlers) {
      const match = serviceWorkerSource.match(pattern);
      if (match) expect(match[1]).not.toContain('"VIDEO_PAUSE"');
    }
  });

  test("handleStopCapture resets video rate", () => {
    const match = serviceWorkerSource.match(
      /async function handleStopCapture\(\)([\s\S]*?)^}/m
    );
    expect(match[1]).toContain("VIDEO_RESET_RATE");
  });

  test("handleStopCapture stops replay zone polling", () => {
    const match = serviceWorkerSource.match(
      /async function handleStopCapture\(\)([\s\S]*?)^}/m
    );
    expect(match[1]).toContain("stopReplayZonePoll");
  });

  test("adaptive rate has cooldown", () => {
    expect(serviceWorkerSource).toContain("RATE_ADJUST_COOLDOWN_MS");
    expect(serviceWorkerSource).toContain("lastRateAdjustAt");
  });
});

// ============================================================================
// Content script: VIDEO_FOUND includes currentTime
// ============================================================================

describe("content-script: VIDEO_FOUND", () => {
  test("includes currentTime in VIDEO_FOUND message", () => {
    const video = createMockVideo({ paused: false });
    video.currentTime = 77.7;
    const { chrome, getMessageHandler } = createChromeMock();
    const mockDoc = {
      querySelectorAll: jest.fn(() => [video]),
      body: {},
    };
    evalScript("content-script.js", { chrome, document: mockDoc });
    const sentMessages = chrome.runtime.sendMessage.mock.calls.map(c => c[0]);
    const found = sentMessages.find(m => m.type === "VIDEO_FOUND");
    expect(found).toBeDefined();
    expect(found.currentTime).toBe(77.7);
  });
});

// ============================================================================
// WAV encoding correctness
// ============================================================================

describe("encodeWav", () => {
  function encodeWav(samples, sampleRate) {
    const buffer = new ArrayBuffer(44 + samples.length * 2);
    const view = new DataView(buffer);
    const write = (off, str) => {
      for (let i = 0; i < str.length; i++) view.setUint8(off + i, str.charCodeAt(i));
    };
    write(0, "RIFF"); view.setUint32(4, 36 + samples.length * 2, true);
    write(8, "WAVE"); write(12, "fmt "); view.setUint32(16, 16, true);
    view.setUint16(20, 1, true); view.setUint16(22, 1, true);
    view.setUint32(24, sampleRate, true); view.setUint32(28, sampleRate * 2, true);
    view.setUint16(32, 2, true); view.setUint16(34, 16, true);
    write(36, "data"); view.setUint32(40, samples.length * 2, true);
    let offset = 44;
    for (let i = 0; i < samples.length; i++, offset += 2) {
      const s = Math.max(-1, Math.min(1, samples[i]));
      view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7fff, true);
    }
    return buffer;
  }

  test("correct header", () => {
    const buf = encodeWav(new Float32Array(100), 16000);
    const view = new DataView(buf);
    expect(view.getUint32(24, true)).toBe(16000);
    expect(view.getUint32(40, true)).toBe(200);
  });

  test("clamps to [-1, 1]", () => {
    const buf = encodeWav(new Float32Array([2.0, -2.0]), 16000);
    const view = new DataView(buf);
    expect(view.getInt16(44, true)).toBe(0x7FFF);
    expect(view.getInt16(46, true)).toBe(-0x8000);
  });
});
