/**
 * Tests for offscreen.js — source analysis + logic verification
 */
const fs = require("fs");
const path = require("path");

const offscreenSource = fs.readFileSync(
  path.resolve(__dirname, "..", "offscreen", "offscreen.js"),
  "utf-8"
);

describe("offscreen.js source analysis", () => {
  test("workletNode is NOT connected to audioContext.destination (audio muted)", () => {
    const hasConnectToDestination = /workletNode\.connect\(\s*audioContext\.destination\s*\)/.test(offscreenSource);
    expect(hasConnectToDestination).toBe(false);
    expect(offscreenSource).toContain("Do NOT connect workletNode to audioContext.destination");
  });

  test("source.connect(workletNode) still exists (audio still captured)", () => {
    expect(/source\.connect\(\s*workletNode\s*\)/.test(offscreenSource)).toBe(true);
  });

  test("TRANSITION_TO_PLAYBACK is sent from startPlayback (not FIRST_SEGMENT_READY)", () => {
    const match = offscreenSource.match(/function startPlayback\(\)([\s\S]*?)^}/m);
    expect(match).not.toBeNull();
    expect(match[1]).toContain('"TRANSITION_TO_PLAYBACK"');
    expect(match[1]).not.toContain('"FIRST_SEGMENT_READY"');
  });

  test("startPlayback mutes worklet via SET_CAPTURE_ACTIVE", () => {
    const match = offscreenSource.match(/function startPlayback\(\)([\s\S]*?)^}/m);
    expect(match[1]).toContain("SET_CAPTURE_ACTIVE");
    expect(match[1]).toContain("active: false");
  });

  test("startPlayback measures throughput and computes safeRate", () => {
    const match = offscreenSource.match(/function startPlayback\(\)([\s\S]*?)^}/m);
    expect(match[1]).toContain("measuredRate");
    expect(match[1]).toContain("safeRate");
    expect(match[1]).toContain("* 0.9"); // safety margin
    expect(match[1]).toContain("0.5");  // floor
  });

  test("RESUME_CAPTURE handler unmutes worklet", () => {
    expect(offscreenSource).toContain('"RESUME_CAPTURE"');
    expect(offscreenSource).toContain("SET_CAPTURE_ACTIVE");
    expect(offscreenSource).toContain("active: true");
  });

  test("checkBuffer sends BUFFER_STATUS (not drift)", () => {
    const match = offscreenSource.match(/function checkBuffer\(\)([\s\S]*?)^}/m);
    expect(match).not.toBeNull();
    expect(match[1]).toContain("BUFFER_STATUS");
    expect(match[1]).toContain("bufferAheadMs");
    expect(match[1]).not.toContain("DRIFT_STATUS");
    expect(match[1]).not.toContain("DRIFT_CORRECTION");
  });

  test("no skip logic remains (skipAudioSec removed)", () => {
    expect(offscreenSource).not.toContain("skipAudioSec");
    expect(offscreenSource).not.toContain("skippedSoFarSec");
    expect(offscreenSource).not.toContain("Skipping re-captured");
  });

  test("stopCapture resets all state", () => {
    const match = offscreenSource.match(/async function stopCapture\(\)([\s\S]*?)^}/m);
    const body = match[1];
    const resets = [
      'playbackState = "idle"', "captureStartedAt = null", "playbackStartedAt = null",
      "nextPlayTime = 0", "playbackStartedCtxTime = null", "bufferedDuration = 0",
      "totalAudioSentSec = 0", "captionQueue = [];",
    ];
    for (const r of resets) expect(body).toContain(r);
  });

  test("buffering state machine: idle → buffering → playing", () => {
    expect(offscreenSource).toContain('let playbackState = "idle"');
    expect(offscreenSource).toContain('playbackState = "buffering"');
    expect(offscreenSource).toContain('playbackState = "playing"');
  });

  test("playback starts when buffer reaches TARGET_BUFFER_SEC", () => {
    expect(offscreenSource).toContain("TARGET_BUFFER_SEC = 16");
    expect(offscreenSource).toContain("bufferedDuration >= TARGET_BUFFER_SEC");
  });

  test("playbackCtx created in startPlayback, not during buffering", () => {
    const startMatch = offscreenSource.match(/function startPlayback\(\)([\s\S]*?)^}/m);
    expect(startMatch[1]).toContain("playbackCtx = new AudioContext()");
    const decodeMatch = offscreenSource.match(/function decodeAndQueue\(b64\)([\s\S]*?)^}/m);
    expect(decodeMatch[1]).not.toContain("playbackCtx = new AudioContext()");
  });

  test("captions buffered during buffering, released in startPlayback", () => {
    expect(offscreenSource).toContain("captionQueue.push(item.caption)");
    const match = offscreenSource.match(/function startPlayback\(\)([\s\S]*?)^}/m);
    expect(match[1]).toContain("for (const caption of captionQueue)");
    expect(match[1]).toContain("captionQueue = []");
  });

  test("gapless playback via source.start(nextPlayTime)", () => {
    expect(offscreenSource).toContain("source.start(nextPlayTime)");
    expect(offscreenSource).toContain("nextPlayTime += audioBuffer.duration");
  });

  test("buffer fallback after 45s", () => {
    expect(offscreenSource).toContain("45000");
    expect(offscreenSource).toContain("Buffer fallback");
  });
});

describe("audio-worklet.js capture toggle", () => {
  const workletSource = fs.readFileSync(
    path.resolve(__dirname, "..", "offscreen", "audio-worklet.js"),
    "utf-8"
  );

  test("has _captureActive flag", () => {
    expect(workletSource).toContain("this._captureActive = true");
  });

  test("handles SET_CAPTURE_ACTIVE message", () => {
    expect(workletSource).toContain("SET_CAPTURE_ACTIVE");
    expect(workletSource).toContain("this._captureActive = e.data.active");
  });

  test("resets buffer on toggle", () => {
    expect(workletSource).toContain("this._writeIndex = 0");
    expect(workletSource).toContain("this._buffer.fill(0)");
  });

  test("skips accumulation when captureActive is false", () => {
    expect(workletSource).toContain("if (!this._captureActive) return true");
  });
});
