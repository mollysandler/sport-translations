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

  describe("buffer-based early resume", () => {
    test("resumes capture early when buffer drops below 5s during replay zone", async () => {
      const { chrome, sendMessage, sentMessages } = loadServiceWorker();
      chrome.tabs.sendMessage.mockReturnValue(Promise.resolve({ ok: true, currentTime: 10 }));

      sendMessage({ type: "START_CAPTURE", sourceLang: "en", targetLang: "hi" });
      await jest.advanceTimersByTimeAsync(0);
      for (let i = 0; i < 10; i++) await Promise.resolve();
      sendMessage({ type: "VIDEO_FOUND" });

      // Enter replay zone (captureResumed = false)
      sendMessage({ type: "TRANSITION_TO_PLAYBACK", totalAudioSentSec: 24, measuredRate: 1.0 });
      await jest.advanceTimersByTimeAsync(0);
      for (let i = 0; i < 10; i++) await Promise.resolve();

      // Buffer is healthy — should NOT resume yet
      sendMessage({ type: "BUFFER_STATUS", bufferAheadMs: 10000 });
      expect(sentMessages.filter(m => m.type === "RESUME_CAPTURE")).toHaveLength(0);

      // Buffer drops below 5s — should trigger early resume
      sendMessage({ type: "BUFFER_STATUS", bufferAheadMs: 4000 });
      expect(sentMessages.filter(m => m.type === "RESUME_CAPTURE")).toHaveLength(1);
    });

    test("negative buffer (underrun) should still trigger RESUME", async () => {
      const { chrome, sendMessage, sentMessages } = loadServiceWorker();
      chrome.tabs.sendMessage.mockReturnValue(Promise.resolve({ ok: true, currentTime: 10 }));

      sendMessage({ type: "START_CAPTURE", sourceLang: "en", targetLang: "hi" });
      await jest.advanceTimersByTimeAsync(0);
      for (let i = 0; i < 10; i++) await Promise.resolve();
      sendMessage({ type: "VIDEO_FOUND" });

      sendMessage({ type: "TRANSITION_TO_PLAYBACK", totalAudioSentSec: 24, measuredRate: 1.0 });
      await jest.advanceTimersByTimeAsync(0);
      for (let i = 0; i < 10; i++) await Promise.resolve();

      // Buffer negative (underrun) — should STILL trigger RESUME so recovery can begin.
      // Currently: bufferSec > 0 check BLOCKS this, which is a bug.
      sendMessage({ type: "BUFFER_STATUS", bufferAheadMs: -500 });
      expect(sentMessages.filter(m => m.type === "RESUME_CAPTURE")).toHaveLength(1);
    });
  });

  describe("content drift detection via video position estimation", () => {
    test("slows video when audio content is ahead of estimated video position", async () => {
      const { chrome, sendMessage } = loadServiceWorker();
      // VIDEO_REPORT_TIME returns video at 100s — so seekTarget = 100-24 = 76
      chrome.tabs.sendMessage.mockReturnValue(Promise.resolve({ ok: true, currentTime: 100 }));

      sendMessage({ type: "START_CAPTURE", sourceLang: "en", targetLang: "hi" });
      await jest.advanceTimersByTimeAsync(0);
      for (let i = 0; i < 10; i++) await Promise.resolve();
      sendMessage({ type: "VIDEO_FOUND" });

      sendMessage({ type: "TRANSITION_TO_PLAYBACK", totalAudioSentSec: 24, measuredRate: 1.0 });
      // Let the .then() in handleTransitionToPlayback run (sets lastSeekTarget=76, lastSeekWallTime)
      await jest.advanceTimersByTimeAsync(0);
      for (let i = 0; i < 10; i++) await Promise.resolve();

      // Trigger zone end so captureResumed = true (video at 100 >= replayZoneEnd(100) - 3)
      await jest.advanceTimersByTimeAsync(500);
      for (let i = 0; i < 10; i++) await Promise.resolve();

      chrome.tabs.sendMessage.mockClear();
      chrome.tabs.sendMessage.mockReturnValue(Promise.resolve());

      // Advance 5 seconds of wall time. Estimated video pos = 76 + 5*1.0 = 81.
      // Send BUFFER_STATUS with totalAudioContentSec = 100 (audio has covered much more).
      // Content drift = 100 - 81 = 19s > 3s → should slow video.
      jest.advanceTimersByTime(5000);
      sendMessage({
        type: "BUFFER_STATUS",
        bufferAheadMs: 6000,
        totalAudioContentSec: 100,
      });

      const rateCalls = chrome.tabs.sendMessage.mock.calls.filter(
        c => c[1].type === "VIDEO_ADJUST_RATE"
      );

      expect(rateCalls.length).toBeGreaterThan(0);
      expect(rateCalls[0][1].rate).toBeLessThan(1.0);
      expect(rateCalls[0][1].rate).toBeGreaterThanOrEqual(0.5);
    });

    test("slows video when video is ahead of audio content (negative drift)", async () => {
      const { chrome, sendMessage } = loadServiceWorker();
      chrome.tabs.sendMessage.mockReturnValue(Promise.resolve({ ok: true, currentTime: 100 }));

      sendMessage({ type: "START_CAPTURE", sourceLang: "en", targetLang: "hi" });
      await jest.advanceTimersByTimeAsync(0);
      for (let i = 0; i < 10; i++) await Promise.resolve();
      sendMessage({ type: "VIDEO_FOUND" });

      sendMessage({ type: "TRANSITION_TO_PLAYBACK", totalAudioSentSec: 24, measuredRate: 1.0 });
      await jest.advanceTimersByTimeAsync(0);
      for (let i = 0; i < 10; i++) await Promise.resolve();

      // Zone end fires (video at 100 >= 100-3)
      await jest.advanceTimersByTimeAsync(500);
      for (let i = 0; i < 10; i++) await Promise.resolve();

      chrome.tabs.sendMessage.mockClear();
      chrome.tabs.sendMessage.mockReturnValue(Promise.resolve());

      // Advance 15 seconds. Estimated video pos = 76 + 15*1.0 = 91.
      // But audio has only sent 32s of content (24 pre-zone + 8 first fresh chunk).
      // Content drift = 32 - 91 = -59. Video is WAY ahead of audio.
      // The system should slow the video to let audio catch up.
      jest.advanceTimersByTime(15000);
      sendMessage({
        type: "BUFFER_STATUS",
        bufferAheadMs: 6000,
        totalAudioContentSec: 32,
      });

      const rateCalls = chrome.tabs.sendMessage.mock.calls.filter(
        c => c[1].type === "VIDEO_ADJUST_RATE"
      );

      // Correct behavior: video should slow down when it's ahead of audio
      expect(rateCalls.length).toBeGreaterThan(0);
      expect(rateCalls[0][1].rate).toBeLessThan(1.0);
    });

    test("does not adjust rate when content drift is within 3s tolerance", async () => {
      const { chrome, sendMessage } = loadServiceWorker();
      chrome.tabs.sendMessage.mockReturnValue(Promise.resolve({ ok: true, currentTime: 100 }));

      sendMessage({ type: "START_CAPTURE", sourceLang: "en", targetLang: "hi" });
      await jest.advanceTimersByTimeAsync(0);
      for (let i = 0; i < 10; i++) await Promise.resolve();
      sendMessage({ type: "VIDEO_FOUND" });

      sendMessage({ type: "TRANSITION_TO_PLAYBACK", totalAudioSentSec: 24, measuredRate: 1.0 });
      await jest.advanceTimersByTimeAsync(0);
      for (let i = 0; i < 10; i++) await Promise.resolve();
      await jest.advanceTimersByTimeAsync(500);
      for (let i = 0; i < 10; i++) await Promise.resolve();

      chrome.tabs.sendMessage.mockClear();
      chrome.tabs.sendMessage.mockReturnValue(Promise.resolve());

      // Advance 5s. Estimated pos = 76 + 5 = 81. totalAudioContentSec = 83.
      // Drift = 83 - 81 = 2s < 3s → no adjustment from drift.
      // Buffer at 4s → in hold range (2-6s) → no adjustment from buffer either.
      jest.advanceTimersByTime(5000);
      sendMessage({
        type: "BUFFER_STATUS",
        bufferAheadMs: 4000,
        totalAudioContentSec: 83,
      });

      const rateCalls = chrome.tabs.sendMessage.mock.calls.filter(
        c => c[1].type === "VIDEO_ADJUST_RATE"
      );
      expect(rateCalls).toHaveLength(0);
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

    test("never sends VIDEO_PAUSE from buffer handling", async () => {
      const { chrome, sendMessage } = loadServiceWorker();
      chrome.tabs.sendMessage.mockReturnValue(Promise.resolve({ ok: true, currentTime: 100 }));

      sendMessage({ type: "START_CAPTURE", sourceLang: "en", targetLang: "hi" });
      await jest.advanceTimersByTimeAsync(0);
      for (let i = 0; i < 10; i++) await Promise.resolve();
      sendMessage({ type: "VIDEO_FOUND" });

      sendMessage({ type: "TRANSITION_TO_PLAYBACK", totalAudioSentSec: 8, measuredRate: 0.70 });
      await jest.advanceTimersByTimeAsync(500);
      for (let i = 0; i < 10; i++) await Promise.resolve();

      chrome.tabs.sendMessage.mockClear();
      chrome.tabs.sendMessage.mockReturnValue(Promise.resolve());

      // Send BUFFER_STATUS at various levels
      for (const ms of [100, 500, 1000, 5000, 9000, 15000]) {
        jest.advanceTimersByTime(2000);
        sendMessage({ type: "BUFFER_STATUS", bufferAheadMs: ms });
      }

      const pauseCalls = chrome.tabs.sendMessage.mock.calls.filter(
        c => c[1].type === "VIDEO_PAUSE"
      );
      expect(pauseCalls).toHaveLength(0);
    });

    test("rate never exceeds 1.0 or drops below 0.5", async () => {
      const { chrome, sendMessage } = loadServiceWorker();
      chrome.tabs.sendMessage.mockReturnValue(Promise.resolve({ ok: true, currentTime: 100 }));

      sendMessage({ type: "START_CAPTURE", sourceLang: "en", targetLang: "hi" });
      await jest.advanceTimersByTimeAsync(0);
      for (let i = 0; i < 10; i++) await Promise.resolve();
      sendMessage({ type: "VIDEO_FOUND" });

      sendMessage({ type: "TRANSITION_TO_PLAYBACK", totalAudioSentSec: 8, measuredRate: 0.70 });
      // Trigger replay zone end
      await jest.advanceTimersByTimeAsync(500);
      for (let i = 0; i < 10; i++) await Promise.resolve();

      chrome.tabs.sendMessage.mockClear();
      chrome.tabs.sendMessage.mockReturnValue(Promise.resolve());

      // Hammer with extreme high buffer (should push rate toward 1.0 but never above)
      for (let i = 0; i < 20; i++) {
        jest.advanceTimersByTime(2100);
        sendMessage({ type: "BUFFER_STATUS", bufferAheadMs: 100000 });
      }
      // Then hammer with extreme low buffer (should push toward 0.5 but never below)
      for (let i = 0; i < 30; i++) {
        jest.advanceTimersByTime(2100);
        sendMessage({ type: "BUFFER_STATUS", bufferAheadMs: 100 });
      }

      const rateCalls = chrome.tabs.sendMessage.mock.calls
        .filter(c => c[1].type === "VIDEO_ADJUST_RATE")
        .map(c => c[1].rate);

      for (const rate of rateCalls) {
        expect(rate).toBeGreaterThanOrEqual(0.5);
        expect(rate).toBeLessThanOrEqual(1.0);
      }
      // Should have made at least some adjustments
      expect(rateCalls.length).toBeGreaterThan(0);
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

  describe("STOP during in-progress START", () => {
    test("cancels the start cleanly", async () => {
      // Create a fetch that hangs until we resolve it
      let resolveFetch;
      const pendingFetch = new Promise((resolve) => { resolveFetch = resolve; });
      const slowFetch = jest.fn(() => pendingFetch);
      const { sendMessage } = loadServiceWorker({ fetch: slowFetch });

      // Start capture — fetch will hang
      sendMessage({ type: "START_CAPTURE", sourceLang: "en", targetLang: "hi" });
      await jest.advanceTimersByTimeAsync(0);
      for (let i = 0; i < 5; i++) await Promise.resolve();

      // Stop while start is pending
      sendMessage({ type: "STOP_CAPTURE" });
      await jest.advanceTimersByTimeAsync(0);
      for (let i = 0; i < 5; i++) await Promise.resolve();

      // Now resolve the fetch — the aborted promise should be caught
      resolveFetch({
        ok: true,
        json: () => Promise.resolve({ session_id: "late-session" }),
      });
      await jest.advanceTimersByTimeAsync(0);
      for (let i = 0; i < 10; i++) await Promise.resolve();

      // The fetch should have been called
      expect(slowFetch).toHaveBeenCalled();
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
