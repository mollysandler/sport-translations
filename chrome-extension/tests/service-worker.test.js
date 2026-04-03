/**
 * Tests for service-worker.js
 */
const { createChromeMock, evalScript } = require("./helpers");

function loadServiceWorker(overrides = {}) {
  const { chrome, sentMessages, getMessageHandler } = createChromeMock();
  const mockFetch = jest.fn(() =>
    Promise.resolve({
      ok: true,
      json: () => Promise.resolve({ session_id: "test-session-123" }),
    })
  );
  chrome.runtime.getContexts = jest.fn(() => Promise.resolve([]));
  evalScript("service-worker.js", { chrome, fetch: overrides.fetch || mockFetch });
  const handler = getMessageHandler();
  function sendMessage(msg) {
    let response = null;
    handler(msg, {}, (r) => { response = r; });
    return { response, async: response === undefined };
  }
  return { chrome, sentMessages, sendMessage, handler, mockFetch };
}

describe("service-worker", () => {
  beforeEach(() => { jest.useFakeTimers(); });
  afterEach(() => { jest.useRealTimers(); });

  describe("TRANSITION_TO_PLAYBACK (replaces FIRST_SEGMENT_READY)", () => {
    test("queries video position then seeks to (currentPos - totalAudioSent)", async () => {
      const { chrome, sendMessage } = loadServiceWorker();
      chrome.tabs.sendMessage.mockReturnValue(Promise.resolve());

      sendMessage({ type: "START_CAPTURE", sourceLang: "en", targetLang: "hi" });
      await jest.advanceTimersByTimeAsync(0);
      for (let i = 0; i < 10; i++) await Promise.resolve();

      sendMessage({ type: "VIDEO_FOUND", currentTime: 10 });

      // Mock VIDEO_REPORT_TIME to return video at 62s (advanced during cold start + processing)
      chrome.tabs.sendMessage.mockClear();
      chrome.tabs.sendMessage.mockReturnValue(Promise.resolve({ ok: true, currentTime: 62, playbackRate: 1.0 }));

      sendMessage({
        type: "TRANSITION_TO_PLAYBACK",
        totalAudioSentSec: 16,
        measuredRate: 0.65,
      });

      // Wait for the async VIDEO_REPORT_TIME → seek chain
      await jest.advanceTimersByTimeAsync(0);
      for (let i = 0; i < 10; i++) await Promise.resolve();

      // Should seek to 62 - 16 = 46 (NOT 10 which is captureStartVideoTime)
      expect(chrome.tabs.sendMessage).toHaveBeenCalledWith(42, {
        type: "VIDEO_SEEK", time: 46,
      });
      // Should set rate to 0.65
      expect(chrome.tabs.sendMessage).toHaveBeenCalledWith(42, {
        type: "VIDEO_ADJUST_RATE", rate: 0.65,
      });
    });

    test("starts polling VIDEO_REPORT_TIME for replay zone", async () => {
      const { chrome, sendMessage } = loadServiceWorker();
      chrome.tabs.sendMessage.mockReturnValue(Promise.resolve({ ok: true, currentTime: 45.2 }));

      sendMessage({ type: "START_CAPTURE", sourceLang: "en", targetLang: "hi" });
      await jest.advanceTimersByTimeAsync(0);
      for (let i = 0; i < 10; i++) await Promise.resolve();

      sendMessage({ type: "VIDEO_FOUND", currentTime: 45.2 });
      chrome.tabs.sendMessage.mockClear();
      chrome.tabs.sendMessage.mockReturnValue(Promise.resolve({ ok: true, currentTime: 50 }));

      sendMessage({
        type: "TRANSITION_TO_PLAYBACK",
        totalAudioSentSec: 16,
        measuredRate: 0.65,
      });

      // Advance timers to trigger polling
      jest.advanceTimersByTime(500);
      // Should have polled VIDEO_REPORT_TIME
      const reportCalls = chrome.tabs.sendMessage.mock.calls.filter(
        c => c[1].type === "VIDEO_REPORT_TIME"
      );
      expect(reportCalls.length).toBeGreaterThan(0);
    });

    test("sends RESUME_CAPTURE when video passes replay zone", async () => {
      const { chrome, sendMessage, sentMessages } = loadServiceWorker();
      chrome.tabs.sendMessage.mockReturnValue(Promise.resolve({ ok: true, currentTime: 45.2 }));

      sendMessage({ type: "START_CAPTURE", sourceLang: "en", targetLang: "hi" });
      await jest.advanceTimersByTimeAsync(0);
      for (let i = 0; i < 10; i++) await Promise.resolve();

      // Video found at 45.2, totalAudioSentSec=16 → replay zone ends at 61.2 - 3 = 58.2
      sendMessage({ type: "VIDEO_FOUND", currentTime: 45.2 });
      chrome.tabs.sendMessage.mockReturnValue(
        Promise.resolve({ ok: true, currentTime: 59 }) // past the zone
      );

      sendMessage({
        type: "TRANSITION_TO_PLAYBACK",
        totalAudioSentSec: 16,
        measuredRate: 0.65,
      });

      // Trigger poll
      await jest.advanceTimersByTimeAsync(500);
      await Promise.resolve();

      expect(sentMessages).toContainEqual({ type: "RESUME_CAPTURE" });
    });
  });

  describe("adaptive rate from BUFFER_STATUS", () => {
    test("does NOT adjust rate during replay zone (captureResumed=false)", async () => {
      const { chrome, sendMessage } = loadServiceWorker();
      chrome.tabs.sendMessage.mockReturnValue(Promise.resolve({ ok: true, currentTime: 0 }));

      sendMessage({ type: "START_CAPTURE", sourceLang: "en", targetLang: "hi" });
      await jest.advanceTimersByTimeAsync(0);
      for (let i = 0; i < 10; i++) await Promise.resolve();
      sendMessage({ type: "VIDEO_FOUND" });

      sendMessage({ type: "TRANSITION_TO_PLAYBACK", totalAudioSentSec: 8, measuredRate: 0.70 });
      await jest.advanceTimersByTimeAsync(0);
      for (let i = 0; i < 10; i++) await Promise.resolve();
      chrome.tabs.sendMessage.mockClear();
      chrome.tabs.sendMessage.mockReturnValue(Promise.resolve());

      // Buffer looks healthy but capture hasn't resumed → should NOT adjust rate
      jest.advanceTimersByTime(2000);
      sendMessage({ type: "BUFFER_STATUS", bufferAheadMs: 9000 });
      const rateCalls = chrome.tabs.sendMessage.mock.calls.filter(
        c => c[1].type === "VIDEO_ADJUST_RATE"
      );
      expect(rateCalls.length).toBe(0);
    });

    test("speeds up when buffer > 6s AFTER capture resumes", async () => {
      const { chrome, sendMessage, sentMessages } = loadServiceWorker();
      // Return video position past replay zone end to trigger RESUME_CAPTURE
      chrome.tabs.sendMessage.mockReturnValue(Promise.resolve({ ok: true, currentTime: 100 }));

      sendMessage({ type: "START_CAPTURE", sourceLang: "en", targetLang: "hi" });
      await jest.advanceTimersByTimeAsync(0);
      for (let i = 0; i < 10; i++) await Promise.resolve();
      sendMessage({ type: "VIDEO_FOUND" });

      sendMessage({ type: "TRANSITION_TO_PLAYBACK", totalAudioSentSec: 8, measuredRate: 0.70 });
      // Trigger poll to detect replay zone end → sets captureResumed = true
      await jest.advanceTimersByTimeAsync(500);
      for (let i = 0; i < 10; i++) await Promise.resolve();

      chrome.tabs.sendMessage.mockClear();
      chrome.tabs.sendMessage.mockReturnValue(Promise.resolve());

      // Now buffer healthy AND capture resumed → should speed up
      jest.advanceTimersByTime(2000);
      sendMessage({ type: "BUFFER_STATUS", bufferAheadMs: 9000 });
      const rateCalls = chrome.tabs.sendMessage.mock.calls.filter(
        c => c[1].type === "VIDEO_ADJUST_RATE"
      );
      expect(rateCalls.length).toBeGreaterThan(0);
      expect(rateCalls[0][1].rate).toBeGreaterThan(0.70);
    });

    test("slows down when buffer < 2s after capture resumes", async () => {
      const { chrome, sendMessage } = loadServiceWorker();
      chrome.tabs.sendMessage.mockReturnValue(Promise.resolve({ ok: true, currentTime: 100 }));

      sendMessage({ type: "START_CAPTURE", sourceLang: "en", targetLang: "hi" });
      await jest.advanceTimersByTimeAsync(0);
      for (let i = 0; i < 10; i++) await Promise.resolve();
      sendMessage({ type: "VIDEO_FOUND" });

      sendMessage({ type: "TRANSITION_TO_PLAYBACK", totalAudioSentSec: 8, measuredRate: 0.80 });
      await jest.advanceTimersByTimeAsync(500);
      for (let i = 0; i < 10; i++) await Promise.resolve();

      chrome.tabs.sendMessage.mockClear();
      chrome.tabs.sendMessage.mockReturnValue(Promise.resolve());

      jest.advanceTimersByTime(2000);
      sendMessage({ type: "BUFFER_STATUS", bufferAheadMs: 1000 });
      const rateCalls = chrome.tabs.sendMessage.mock.calls.filter(
        c => c[1].type === "VIDEO_ADJUST_RATE"
      );
      expect(rateCalls.length).toBeGreaterThan(0);
      expect(rateCalls[0][1].rate).toBeLessThan(0.80);
    });

    test("never pauses video (no VIDEO_PAUSE in handleBufferStatus)", () => {
      const fs = require("fs");
      const path = require("path");
      const swSource = fs.readFileSync(path.resolve(__dirname, "..", "service-worker.js"), "utf-8");
      const match = swSource.match(/function handleBufferStatus([\s\S]*?)^}/m);
      expect(match[1]).not.toContain("VIDEO_PAUSE");
    });

    test("rate adjustment is gated on captureResumed flag", () => {
      const fs = require("fs");
      const path = require("path");
      const swSource = fs.readFileSync(path.resolve(__dirname, "..", "service-worker.js"), "utf-8");
      const match = swSource.match(/function handleBufferStatus([\s\S]*?)^}/m);
      // Should check captureResumed before adjusting rate
      expect(match[1]).toContain("captureResumed");
    });

    test("rate is capped between 0.5 and 1.0", () => {
      const fs = require("fs");
      const path = require("path");
      const swSource = fs.readFileSync(path.resolve(__dirname, "..", "service-worker.js"), "utf-8");
      const match = swSource.match(/function handleBufferStatus([\s\S]*?)^}/m);
      expect(match[1]).toContain("1.0"); // cap
      expect(match[1]).toContain("0.5"); // floor
    });

    test("buffer status is no-op without videoFound", () => {
      const { chrome, sendMessage } = loadServiceWorker();
      sendMessage({ type: "BUFFER_STATUS", bufferAheadMs: 100 });
      expect(chrome.tabs.sendMessage).not.toHaveBeenCalled();
    });
  });

  describe("stop capture cleanup", () => {
    test("resets video rate on stop", async () => {
      const { chrome, sendMessage } = loadServiceWorker();
      chrome.tabs.sendMessage.mockReturnValue(Promise.resolve());

      sendMessage({ type: "START_CAPTURE", sourceLang: "en", targetLang: "hi" });
      await jest.advanceTimersByTimeAsync(0);
      for (let i = 0; i < 10; i++) await Promise.resolve();

      chrome.tabs.sendMessage.mockClear();
      chrome.tabs.sendMessage.mockReturnValue(Promise.resolve());

      sendMessage({ type: "STOP_CAPTURE" });
      await jest.advanceTimersByTimeAsync(0);

      expect(chrome.tabs.sendMessage).toHaveBeenCalledWith(42, {
        type: "VIDEO_RESET_RATE",
      });
    });
  });

  describe("error relay", () => {
    test("CAPTURE_ERROR is re-broadcast", () => {
      const { sendMessage, sentMessages } = loadServiceWorker();
      sendMessage({ type: "CAPTURE_ERROR", error: "Audio capture failed" });
      expect(sentMessages).toContainEqual({
        type: "CAPTURE_ERROR", error: "Audio capture failed",
      });
    });

    test("CHUNK_ERROR is re-broadcast", () => {
      const { sendMessage, sentMessages } = loadServiceWorker();
      sendMessage({ type: "CHUNK_ERROR", error: "Server error (500)" });
      expect(sentMessages).toContainEqual({
        type: "CHUNK_ERROR", error: "Server error (500)",
      });
    });
  });

  describe("VIDEO_FOUND / VIDEO_NOT_FOUND", () => {
    test("VIDEO_FOUND records currentTime and broadcasts buffering", () => {
      const { sendMessage, sentMessages } = loadServiceWorker();
      sendMessage({ type: "VIDEO_FOUND", currentTime: 99.5 });
      expect(sentMessages).toContainEqual({
        type: "VIDEO_SYNC_STATUS", synced: false, status: "buffering",
      });
    });

    test("VIDEO_NOT_FOUND broadcasts no_video", () => {
      const { sendMessage, sentMessages } = loadServiceWorker();
      sendMessage({ type: "VIDEO_NOT_FOUND" });
      expect(sentMessages).toContainEqual({
        type: "VIDEO_SYNC_STATUS", synced: false, status: "no_video",
      });
    });
  });
});
