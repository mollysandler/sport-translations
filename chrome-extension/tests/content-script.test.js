/**
 * Tests for content-script.js:
 *  - User pause detection (userPaused flag)
 *  - Extension-initiated pause does NOT set userPaused
 *  - VIDEO_RESUME and VIDEO_MICRO_PAUSE respect userPaused
 *  - VIDEO_CLEANUP resets state
 *  - findVideo prefers playing video, falls back to largest
 */
const { createMockVideo, createChromeMock, evalScript } = require("./helpers");

function loadContentScript(videos = []) {
  const { chrome, sentMessages, getMessageHandler } = createChromeMock();

  const mockDocument = {
    querySelectorAll: jest.fn((sel) => (sel === "video" ? videos : [])),
    body: {},
  };

  evalScript("content-script.js", { chrome, document: mockDocument });

  const handler = getMessageHandler();
  // Helper to send a message and capture the response
  function send(msg) {
    let response = null;
    handler(msg, {}, (r) => { response = r; });
    return response;
  }

  return { chrome, sentMessages, send, mockDocument };
}

describe("content-script", () => {
  beforeEach(() => {
    jest.useFakeTimers();
  });
  afterEach(() => {
    jest.useRealTimers();
  });

  describe("init and video discovery", () => {
    test("sends VIDEO_FOUND with currentTime when a video element exists", () => {
      const video = createMockVideo({ paused: false });
      video.currentTime = 50;
      const { sentMessages } = loadContentScript([video]);
      expect(sentMessages).toContainEqual({ type: "VIDEO_FOUND", currentTime: 50 });
    });

    test("sends VIDEO_NOT_FOUND when no video element exists", () => {
      const { sentMessages } = loadContentScript([]);
      expect(sentMessages).toContainEqual({ type: "VIDEO_NOT_FOUND" });
    });

    test("prefers a currently-playing video over a paused one", () => {
      const paused = createMockVideo({ paused: true, videoWidth: 3840, videoHeight: 2160 });
      const playing = createMockVideo({ paused: false, videoWidth: 640, videoHeight: 480 });
      const { send } = loadContentScript([paused, playing]);

      // Pause the playing video via extension — it should target the playing one
      const result = send({ type: "VIDEO_PAUSE" });
      expect(result.ok).toBe(true);
      expect(playing.paused).toBe(true);
    });

    test("falls back to largest video by area when none are playing", () => {
      const small = createMockVideo({ paused: true, videoWidth: 320, videoHeight: 240 });
      const large = createMockVideo({ paused: true, videoWidth: 1920, videoHeight: 1080 });
      const { send } = loadContentScript([small, large]);

      // VIDEO_ADJUST_RATE should target the largest video
      send({ type: "VIDEO_ADJUST_RATE", rate: 0.5 });
      expect(large.playbackRate).toBe(0.5);
      // small should be unchanged
      expect(small.playbackRate).toBe(1.0);
    });
  });

  describe("user pause detection", () => {
    test("user-initiated pause sets userPaused flag, blocking VIDEO_RESUME", () => {
      const video = createMockVideo({ paused: false });
      const { send } = loadContentScript([video]);

      // User pauses manually (directly calling video.pause, which fires the event)
      video.pause();
      expect(video.paused).toBe(true);

      // Extension tries to resume — should be blocked
      const result = send({ type: "VIDEO_RESUME" });
      expect(result.ok).toBe(false);
      expect(video.paused).toBe(true);
    });

    test("extension-initiated VIDEO_PAUSE does NOT set userPaused", () => {
      const video = createMockVideo({ paused: false });
      const { send } = loadContentScript([video]);

      // Extension pauses
      send({ type: "VIDEO_PAUSE" });
      expect(video.paused).toBe(true);

      // Extension resumes — should work because userPaused is false
      const result = send({ type: "VIDEO_RESUME" });
      expect(result.ok).toBe(true);
      expect(video.paused).toBe(false);
    });

    test("user unpause clears userPaused, allowing extension resume", () => {
      const video = createMockVideo({ paused: false });
      const { send } = loadContentScript([video]);

      // User pauses
      video.pause();
      expect(video.paused).toBe(true);

      // Extension resume blocked
      send({ type: "VIDEO_RESUME" });
      expect(video.paused).toBe(true);

      // User unpauses manually
      video.play();
      expect(video.paused).toBe(false);

      // Extension pauses then resumes — should work now
      send({ type: "VIDEO_PAUSE" });
      const result = send({ type: "VIDEO_RESUME" });
      expect(result.ok).toBe(true);
      expect(video.paused).toBe(false);
    });
  });

  describe("VIDEO_MICRO_PAUSE respects userPaused", () => {
    test("micro-pause resumes after duration when user has not paused", () => {
      const video = createMockVideo({ paused: false });
      const { send } = loadContentScript([video]);

      send({ type: "VIDEO_MICRO_PAUSE", durationMs: 5000 });
      expect(video.paused).toBe(true);

      jest.advanceTimersByTime(5000);
      expect(video.paused).toBe(false);
    });

    test("micro-pause does NOT resume if user paused during the wait", () => {
      const video = createMockVideo({ paused: false });
      const { send } = loadContentScript([video]);

      send({ type: "VIDEO_MICRO_PAUSE", durationMs: 5000 });
      expect(video.paused).toBe(true);

      // User manually pauses (simulated by directly calling pause, triggering listener)
      // The video is already paused by the extension, but the extension has already
      // reset extensionPaused to false. So a new pause event would set userPaused.
      // Simulate: user hits pause button while video is already paused — fire the event.
      video._listeners["pause"].forEach((fn) => fn());

      jest.advanceTimersByTime(5000);
      // Should remain paused because userPaused is now true
      expect(video.paused).toBe(true);
    });

    test("micro-pause resumes normally if user did not interfere", () => {
      const video = createMockVideo({ paused: false });
      const { send } = loadContentScript([video]);

      send({ type: "VIDEO_MICRO_PAUSE", durationMs: 2000 });
      expect(video.paused).toBe(true);

      jest.advanceTimersByTime(1000);
      expect(video.paused).toBe(true); // still paused

      jest.advanceTimersByTime(1000);
      expect(video.paused).toBe(false); // resumed
    });
  });

  describe("VIDEO_ADJUST_RATE and VIDEO_RESET_RATE", () => {
    test("adjusts and resets playback rate", () => {
      const video = createMockVideo();
      const { send } = loadContentScript([video]);

      send({ type: "VIDEO_ADJUST_RATE", rate: 0.95 });
      expect(video.playbackRate).toBe(0.95);

      send({ type: "VIDEO_RESET_RATE" });
      expect(video.playbackRate).toBe(1.0);
    });

    test("rate persistence guard re-applies rate if player resets it", () => {
      const video = createMockVideo();
      const { send } = loadContentScript([video]);

      send({ type: "VIDEO_ADJUST_RATE", rate: 0.90 });
      expect(video.playbackRate).toBe(0.90);

      // Simulate player resetting rate
      video.playbackRate = 1.0;

      // After 200ms, guard should re-apply
      jest.advanceTimersByTime(200);
      expect(video.playbackRate).toBe(0.90);
    });

    test("rate persistence guard stops after 2 seconds", () => {
      const video = createMockVideo();
      const { send } = loadContentScript([video]);

      send({ type: "VIDEO_ADJUST_RATE", rate: 0.90 });

      // Advance past all 10 guard intervals (10 * 200ms = 2s)
      jest.advanceTimersByTime(2000);

      // Now reset the rate — guard should no longer re-apply
      video.playbackRate = 1.0;
      jest.advanceTimersByTime(200);
      expect(video.playbackRate).toBe(1.0);
    });
  });

  describe("VIDEO_SEEK", () => {
    test("seeks video to specified time", () => {
      const video = createMockVideo({ paused: false });
      video.currentTime = 100;
      const { send } = loadContentScript([video]);

      send({ type: "VIDEO_SEEK", time: 42.5 });
      expect(video.currentTime).toBe(42.5);
    });

    test("returns ok: false when no video exists", () => {
      const { send } = loadContentScript([]);
      const result = send({ type: "VIDEO_SEEK", time: 10 });
      expect(result.ok).toBe(false);
    });
  });

  describe("VIDEO_REPORT_TIME", () => {
    test("reports current time and playback rate", () => {
      const video = createMockVideo({ paused: false });
      video.currentTime = 42.5;
      video.playbackRate = 0.95;
      const { send } = loadContentScript([video]);

      const result = send({ type: "VIDEO_REPORT_TIME" });
      expect(result.ok).toBe(true);
      expect(result.currentTime).toBe(42.5);
      expect(result.playbackRate).toBe(0.95);
    });

    test("returns null values when no video exists", () => {
      const { send } = loadContentScript([]);

      const result = send({ type: "VIDEO_REPORT_TIME" });
      expect(result.ok).toBe(false);
      expect(result.currentTime).toBeNull();
      expect(result.playbackRate).toBeNull();
    });
  });

  describe("VIDEO_CLEANUP", () => {
    test("resets all state", () => {
      const video = createMockVideo({ paused: false });
      const { send, mockDocument } = loadContentScript([video]);

      // User pauses
      video.pause();

      // Cleanup should reset everything
      const result = send({ type: "VIDEO_CLEANUP" });
      expect(result.ok).toBe(true);
      expect(video.playbackRate).toBe(1.0);

      // Remove video from DOM mock so re-discovery finds nothing
      mockDocument.querySelectorAll.mockReturnValue([]);

      // After cleanup, sending VIDEO_RESUME with no video should return ok: false
      const resumeResult = send({ type: "VIDEO_RESUME" });
      expect(resumeResult.ok).toBe(false);
    });
  });

  describe("no video element", () => {
    test("all commands return ok: false when no video exists", () => {
      const { send } = loadContentScript([]);

      expect(send({ type: "VIDEO_PAUSE" }).ok).toBe(false);
      expect(send({ type: "VIDEO_RESUME" }).ok).toBe(false);
      expect(send({ type: "VIDEO_ADJUST_RATE", rate: 2 }).ok).toBe(false);
      expect(send({ type: "VIDEO_RESET_RATE" }).ok).toBe(false);
      expect(send({ type: "VIDEO_MICRO_PAUSE", durationMs: 100 }).ok).toBe(false);
    });
  });
});
