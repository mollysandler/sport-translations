/**
 * Behavioral tests for offscreen.js — exercises actual code paths via evalScript.
 * Replaces the previous source-text regex tests with real execution.
 */
const fs = require("fs");
const path = require("path");
const {
  createChromeMock,
  createMockMediaStream,
  createMockStreamReader,
  MockAudioContext,
  MockOfflineAudioContext,
  MockAudioWorkletNode,
  MockAudioBuffer,
  MockBlob,
  MockFormData,
  MockTextDecoder,
  MockTextEncoder,
  evalScript,
} = require("./helpers");

// ============================================================================
// Helper: load offscreen.js in a controlled VM
// ============================================================================

function loadOffscreen(overrides = {}) {
  const { chrome, sentMessages, getMessageHandler } = createChromeMock();
  const mockMediaStream = createMockMediaStream();

  // Track AudioContext constructor calls
  const audioContextInstances = [];
  const MockAC = jest.fn(function () {
    const ctx = new MockAudioContext();
    audioContextInstances.push(ctx);
    return ctx;
  });

  const offlineInstances = [];
  const MockOAC = jest.fn(function (...args) {
    const ctx = new MockOfflineAudioContext(...args);
    offlineInstances.push(ctx);
    return ctx;
  });

  const workletInstances = [];
  const MockAWN = jest.fn(function (...args) {
    const node = new MockAudioWorkletNode(...args);
    workletInstances.push(node);
    return node;
  });

  // Default fetch: return an SSE response with one segment
  const defaultFetch = jest.fn(() => {
    const reader = createMockStreamReader([
      'data: {"type":"language_detected","lang":"en"}\n\n',
      'data: {"type":"segment","audio_b64":"AAAA","caption":{"speaker":"Speaker 1","translated":"Hello","text":"Hola","original":"Hola"}}\n\n',
    ]);
    return Promise.resolve({
      ok: true,
      status: 200,
      body: { getReader: () => reader },
      text: () => Promise.resolve(""),
    });
  });

  const ctx = evalScript("offscreen/offscreen.js", {
    chrome,
    AudioContext: MockAC,
    OfflineAudioContext: MockOAC,
    AudioWorkletNode: MockAWN,
    TextDecoder: MockTextDecoder,
    TextEncoder: MockTextEncoder,
    Blob: MockBlob,
    FormData: MockFormData,
    navigator: {
      mediaDevices: {
        getUserMedia: jest.fn(() => Promise.resolve(mockMediaStream)),
      },
    },
    fetch: overrides.fetch || defaultFetch,
    ...overrides,
  });

  const handler = getMessageHandler();

  return {
    ctx,
    chrome,
    sentMessages,
    handler,
    audioContextInstances,
    offlineInstances,
    workletInstances,
    mockMediaStream,
    fetch: overrides.fetch || defaultFetch,
  };
}

// ============================================================================
// Audio capture pipeline
// ============================================================================

describe("audio capture pipeline", () => {
  beforeEach(() => jest.useFakeTimers());
  afterEach(() => jest.useRealTimers());

  test("source connects to workletNode but NOT to audioContext.destination", async () => {
    const { handler, workletInstances } = loadOffscreen();

    handler({
      type: "OFFSCREEN_START",
      streamId: "fake-stream",
      sessionId: "s1",
      sourceLang: "en",
      targetLang: "hi",
    });

    // Wait for async startCapture
    await jest.advanceTimersByTimeAsync(0);
    for (let i = 0; i < 10; i++) await Promise.resolve();

    // The worklet node should exist
    expect(workletInstances.length).toBeGreaterThan(0);
    const worklet = workletInstances[0];

    // Worklet should NOT be connected to audioContext.destination
    const destConnections = worklet._connectCalls.filter(
      (d) => d && d._isDestination
    );
    expect(destConnections).toHaveLength(0);
  });
});

// ============================================================================
// startPlayback
// ============================================================================

describe("startPlayback", () => {
  beforeEach(() => jest.useFakeTimers());
  afterEach(() => jest.useRealTimers());

  function setupForPlayback(overrides = {}) {
    const loaded = loadOffscreen(overrides);
    const { ctx } = loaded;

    // Set up state as if we've been buffering
    ctx._exec('sessionId = "test-session"');
    ctx._exec('playbackState = "buffering"');
    ctx._exec("captureStartedAt = Date.now()");
    ctx._exec("totalAudioSentSec = 16");
    ctx._exec("initialLatencyMs = 2000");

    return loaded;
  }

  test("sends TRANSITION_TO_PLAYBACK with totalAudioSentSec and rate 1.0", () => {
    const { ctx, sentMessages } = setupForPlayback();

    ctx.startPlayback();

    const transition = sentMessages.find(
      (m) => m.type === "TRANSITION_TO_PLAYBACK"
    );
    expect(transition).toBeDefined();
    expect(transition.totalAudioSentSec).toBe(16);
    expect(transition.measuredRate).toBe(1.0);
  });

  test("sets holdChunks = true for replay zone", () => {
    const { ctx } = setupForPlayback();

    expect(ctx._exec("holdChunks")).toBe(false);
    ctx.startPlayback();
    expect(ctx._exec("holdChunks")).toBe(true);
  });

  test("transitions playbackState to playing", () => {
    const { ctx } = setupForPlayback();

    expect(ctx._exec("playbackState")).toBe("buffering");
    ctx.startPlayback();
    expect(ctx._exec("playbackState")).toBe("playing");
  });

  test("creates AudioContext during startPlayback, not before", () => {
    const { ctx, audioContextInstances } = setupForPlayback();

    // During buffering, no playbackCtx AudioContext yet
    const countBefore = audioContextInstances.length;
    ctx.startPlayback();
    // startPlayback creates a new AudioContext
    expect(audioContextInstances.length).toBeGreaterThan(countBefore);
  });

  test("releases buffered captions", () => {
    const { ctx, sentMessages } = setupForPlayback();

    // Queue some captions
    ctx._exec(
      'captionQueue.push({speaker: "S1", translated: "Hello"})'
    );
    ctx._exec(
      'captionQueue.push({speaker: "S2", translated: "World"})'
    );

    ctx.startPlayback();

    const captions = sentMessages.filter((m) => m.type === "CAPTION");
    expect(captions).toHaveLength(2);
    expect(captions[0].caption.translated).toBe("Hello");
    expect(captions[1].caption.translated).toBe("World");
    expect(ctx._exec("captionQueue.length")).toBe(0);
  });

  test("clears buffer fallback timer", () => {
    const { ctx } = setupForPlayback();

    // Simulate the timer being set
    ctx._exec("bufferFallbackTimer = setTimeout(() => {}, 99999)");
    ctx.startPlayback();
    expect(ctx._exec("bufferFallbackTimer")).toBeNull();
  });
});

// ============================================================================
// RESUME_CAPTURE handler
// ============================================================================

describe("RESUME_CAPTURE handler", () => {
  test("sends held chunk but keeps holdChunks true (two-phase resume)", () => {
    const mockFetch = jest.fn(() => {
      const reader = createMockStreamReader([]);
      return Promise.resolve({
        ok: true, status: 200,
        body: { getReader: () => reader },
        text: () => Promise.resolve(""),
      });
    });
    const { ctx, handler } = loadOffscreen({ fetch: mockFetch });

    // Set up state: session active, holdChunks with a held chunk
    ctx._exec('sessionId = "test-session"');
    ctx._exec("holdChunks = true");
    ctx._exec(
      "heldChunk = { samples: new Float32Array(1000), nativeSR: 48000 }"
    );

    handler({ type: "RESUME_CAPTURE" });

    // Held chunk was consumed and sent
    expect(ctx._exec("heldChunk")).toBeNull();
    // But holdChunks stays true — ZONE_ENDED will unlock it
    expect(ctx._exec("holdChunks")).toBe(true);
  });
});

describe("ZONE_ENDED handler", () => {
  test("sets holdChunks to false and sends any interim held chunk", () => {
    const mockFetch = jest.fn(() => {
      const reader = createMockStreamReader([]);
      return Promise.resolve({
        ok: true, status: 200,
        body: { getReader: () => reader },
        text: () => Promise.resolve(""),
      });
    });
    const { ctx, handler } = loadOffscreen({ fetch: mockFetch });

    ctx._exec('sessionId = "test-session"');
    ctx._exec("holdChunks = true");
    ctx._exec(
      "heldChunk = { samples: new Float32Array(1000), nativeSR: 48000 }"
    );

    handler({ type: "ZONE_ENDED" });

    expect(ctx._exec("holdChunks")).toBe(false);
    expect(ctx._exec("heldChunk")).toBeNull();
  });
});

// ============================================================================
// checkBuffer
// ============================================================================

describe("checkBuffer", () => {
  test("sends BUFFER_STATUS with correct bufferAheadMs", () => {
    const { ctx, sentMessages, audioContextInstances } = loadOffscreen();

    // Create a playback context and set state
    ctx._exec('playbackState = "playing"');
    ctx._exec("playbackCtx = new AudioContext()");

    // Set the mock's currentTime
    const playCtx = audioContextInstances[audioContextInstances.length - 1];
    playCtx.currentTime = 5.0;
    ctx._exec("nextPlayTime = 10.0");

    ctx.checkBuffer();

    const status = sentMessages.find((m) => m.type === "BUFFER_STATUS");
    expect(status).toBeDefined();
    expect(status.bufferAheadMs).toBeCloseTo(5000, 0);
  });

  test("does nothing when not in playing state", () => {
    const { ctx, sentMessages } = loadOffscreen();

    ctx._exec('playbackState = "idle"');
    ctx.checkBuffer();

    const status = sentMessages.find((m) => m.type === "BUFFER_STATUS");
    expect(status).toBeUndefined();
  });
});

// ============================================================================
// stopCapture
// ============================================================================

describe("stopCapture", () => {
  test("resets all state variables", async () => {
    const { ctx } = loadOffscreen();

    // Dirty the state
    ctx._exec('sessionId = "dirty"');
    ctx._exec('playbackState = "playing"');
    ctx._exec("captureStartedAt = 12345");
    ctx._exec("playbackStartedAt = 12345");
    ctx._exec("nextPlayTime = 99");
    ctx._exec("playbackStartedCtxTime = 50");
    ctx._exec("bufferedDuration = 30");
    ctx._exec("totalAudioSentSec = 20");
    ctx._exec("holdChunks = true");
    ctx._exec(
      "heldChunk = { samples: new Float32Array(1), nativeSR: 48000 }"
    );
    ctx._exec("chunkCount = 10");
    ctx._exec("scheduledAudioDuration = 42");
    ctx._exec('captionQueue.push({text: "stale"})');

    await ctx.stopCapture();

    expect(ctx._exec("sessionId")).toBeNull();
    expect(ctx._exec("playbackState")).toBe("idle");
    expect(ctx._exec("captureStartedAt")).toBeNull();
    expect(ctx._exec("playbackStartedAt")).toBeNull();
    expect(ctx._exec("nextPlayTime")).toBe(0);
    expect(ctx._exec("playbackStartedCtxTime")).toBeNull();
    expect(ctx._exec("bufferedDuration")).toBe(0);
    expect(ctx._exec("totalAudioSentSec")).toBe(0);
    expect(ctx._exec("holdChunks")).toBe(false);
    expect(ctx._exec("heldChunk")).toBeNull();
    expect(ctx._exec("chunkCount")).toBe(0);
    expect(ctx._exec("scheduledAudioDuration")).toBe(0);
    expect(ctx._exec("captionQueue.length")).toBe(0);
  });
});

// ============================================================================
// Buffering state machine
// ============================================================================

describe("buffering state machine", () => {
  beforeEach(() => jest.useFakeTimers());
  afterEach(() => jest.useRealTimers());

  test("starts as idle", () => {
    const { ctx } = loadOffscreen();
    expect(ctx._exec("playbackState")).toBe("idle");
  });

  test("idle -> buffering on first sendPcmChunk", async () => {
    const { ctx } = loadOffscreen();

    ctx._exec('sessionId = "test"');
    expect(ctx._exec("playbackState")).toBe("idle");

    // Trigger sendPcmChunk
    ctx.sendPcmChunk(new Float32Array(1000), 48000);

    // Allow async OfflineAudioContext.startRendering to complete
    await jest.advanceTimersByTimeAsync(0);
    for (let i = 0; i < 5; i++) await Promise.resolve();

    expect(ctx._exec("playbackState")).toBe("buffering");
  });

  test("buffer fallback starts playback after 45s", async () => {
    const { ctx } = loadOffscreen();

    ctx._exec('sessionId = "test"');

    // Send first chunk to enter buffering
    ctx.sendPcmChunk(new Float32Array(1000), 48000);
    await jest.advanceTimersByTimeAsync(0);
    for (let i = 0; i < 5; i++) await Promise.resolve();
    expect(ctx._exec("playbackState")).toBe("buffering");

    // Push a decoded buffer so fallback has something to play
    ctx._exec(
      'decodedQueue.push({ duration: 3.0, numberOfChannels: 1, sampleRate: 16000, length: 48000, getChannelData: () => new Float32Array(48000) })'
    );

    // Advance past fallback timer
    await jest.advanceTimersByTimeAsync(45000);
    for (let i = 0; i < 5; i++) await Promise.resolve();

    expect(ctx._exec("playbackState")).toBe("playing");
  });
});

// ============================================================================
// Gapless playback scheduling
// ============================================================================

describe("gapless playback scheduling", () => {
  test("schedules buffers at incrementing nextPlayTime", () => {
    const { ctx, audioContextInstances } = loadOffscreen();

    // Set up playing state with a playback context
    ctx._exec('playbackState = "playing"');
    ctx._exec("playbackCtx = new AudioContext()");
    const playCtx = audioContextInstances[audioContextInstances.length - 1];
    playCtx.currentTime = 0;
    ctx._exec("nextPlayTime = 0");

    // Push mock audio buffers
    ctx._exec(
      'decodedQueue.push({ duration: 3.0, numberOfChannels: 1, sampleRate: 16000, length: 48000, getChannelData: () => new Float32Array(1) })'
    );
    ctx._exec(
      'decodedQueue.push({ duration: 2.0, numberOfChannels: 1, sampleRate: 16000, length: 32000, getChannelData: () => new Float32Array(1) })'
    );

    ctx.scheduleBuffers();

    // Verify sources were started at the right times
    expect(playCtx._sources.length).toBe(2);
    expect(playCtx._sources[0].start).toHaveBeenCalledWith(0);
    expect(playCtx._sources[1].start).toHaveBeenCalledWith(3.0);
    expect(ctx._exec("nextPlayTime")).toBe(5.0);
  });

  test("recovers from queue underrun by resetting nextPlayTime", () => {
    const { ctx, audioContextInstances } = loadOffscreen();

    ctx._exec('playbackState = "playing"');
    ctx._exec("playbackCtx = new AudioContext()");
    const playCtx = audioContextInstances[audioContextInstances.length - 1];

    // currentTime has advanced past nextPlayTime (underrun)
    playCtx.currentTime = 20.0;
    ctx._exec("nextPlayTime = 10.0");

    ctx._exec(
      'decodedQueue.push({ duration: 1.0, numberOfChannels: 1, sampleRate: 16000, length: 16000, getChannelData: () => new Float32Array(1) })'
    );

    ctx.scheduleBuffers();

    // Should have reset nextPlayTime to currentTime (20.0) then scheduled at 20.0
    expect(playCtx._sources[0].start).toHaveBeenCalledWith(20.0);
    expect(ctx._exec("nextPlayTime")).toBe(21.0);
  });
});

// ============================================================================
// SSE parser (readSseEvents)
// ============================================================================

describe("readSseEvents", () => {
  test("parses multiple SSE events", async () => {
    const { ctx } = loadOffscreen();

    const reader = createMockStreamReader([
      'data: {"type":"segment","caption":"hello"}\n\n',
      'data: {"type":"segment","caption":"world"}\n\n',
    ]);

    const mockResponse = { body: { getReader: () => reader } };
    const events = [];
    for await (const event of ctx.readSseEvents(mockResponse)) {
      events.push(event);
    }

    expect(events).toHaveLength(2);
    expect(events[0].caption).toBe("hello");
    expect(events[1].caption).toBe("world");
  });

  test("handles events split across chunks", async () => {
    const { ctx } = loadOffscreen();

    // Split a single event across two read() calls
    const reader = createMockStreamReader([
      'data: {"type":"seg',
      'ment","value":1}\n\ndata: {"type":"complete"}\n\n',
    ]);

    const mockResponse = { body: { getReader: () => reader } };
    const events = [];
    for await (const event of ctx.readSseEvents(mockResponse)) {
      events.push(event);
    }

    expect(events).toHaveLength(2);
    expect(events[0].type).toBe("segment");
    expect(events[1].type).toBe("complete");
  });

  test("silently skips malformed JSON", async () => {
    const { ctx } = loadOffscreen();

    const reader = createMockStreamReader([
      "data: not-valid-json\n\n",
      'data: {"type":"ok"}\n\n',
    ]);

    const mockResponse = { body: { getReader: () => reader } };
    const events = [];
    for await (const event of ctx.readSseEvents(mockResponse)) {
      events.push(event);
    }

    // Only the valid event should be yielded
    expect(events).toHaveLength(1);
    expect(events[0].type).toBe("ok");
  });

  test("handles multiple events in a single chunk", async () => {
    const { ctx } = loadOffscreen();

    const reader = createMockStreamReader([
      'data: {"a":1}\ndata: {"b":2}\ndata: {"c":3}\n',
    ]);

    const mockResponse = { body: { getReader: () => reader } };
    const events = [];
    for await (const event of ctx.readSseEvents(mockResponse)) {
      events.push(event);
    }

    expect(events).toHaveLength(3);
  });
});

// ============================================================================
// encodeWav
// ============================================================================

describe("encodeWav", () => {
  test("produces valid WAV header", () => {
    const { ctx } = loadOffscreen();
    const buf = ctx.encodeWav(new Float32Array(100), 16000);
    const view = new DataView(buf);

    // RIFF header
    expect(String.fromCharCode(view.getUint8(0), view.getUint8(1), view.getUint8(2), view.getUint8(3))).toBe("RIFF");
    // WAVE format
    expect(String.fromCharCode(view.getUint8(8), view.getUint8(9), view.getUint8(10), view.getUint8(11))).toBe("WAVE");
    // Sample rate
    expect(view.getUint32(24, true)).toBe(16000);
    // Data size
    expect(view.getUint32(40, true)).toBe(200); // 100 samples * 2 bytes
  });

  test("clamps samples to [-1, 1]", () => {
    const { ctx } = loadOffscreen();
    const buf = ctx.encodeWav(new Float32Array([2.0, -2.0]), 16000);
    const view = new DataView(buf);

    expect(view.getInt16(44, true)).toBe(0x7fff);  // +2.0 clamped to +1.0
    expect(view.getInt16(46, true)).toBe(-0x8000);  // -2.0 clamped to -1.0
  });

  test("NaN input is treated as silence (zero)", () => {
    const { ctx } = loadOffscreen();
    const buf = ctx.encodeWav(new Float32Array([NaN, 0.5]), 16000);
    const view = new DataView(buf);

    // NaN should be treated as 0 (silence)
    expect(view.getInt16(44, true)).toBe(0);
    // Normal value should still work
    expect(view.getInt16(46, true)).toBeGreaterThan(0);
  });

  test("handles empty array", () => {
    const { ctx } = loadOffscreen();
    const buf = ctx.encodeWav(new Float32Array(0), 16000);
    // Should be exactly the 44-byte header
    expect(buf.byteLength).toBe(44);
  });
});

// ============================================================================
// Bug-catching tests
// ============================================================================

describe("OFFSCREEN_START without OFFSCREEN_STOP resets stale state", () => {
  test("second OFFSCREEN_START resets playback state from previous session", async () => {
    jest.useFakeTimers();
    const { ctx, handler } = loadOffscreen();

    // First start
    handler({
      type: "OFFSCREEN_START",
      streamId: "stream-1",
      sessionId: "session-1",
      sourceLang: "en",
      targetLang: "hi",
    });
    await jest.advanceTimersByTimeAsync(0);
    for (let i = 0; i < 10; i++) await Promise.resolve();

    // Dirty the state as if we've been playing
    ctx._exec('playbackState = "playing"');
    ctx._exec("bufferedDuration = 25");
    ctx._exec("totalAudioSentSec = 40");

    // Second start WITHOUT stopping first
    handler({
      type: "OFFSCREEN_START",
      streamId: "stream-2",
      sessionId: "session-2",
      sourceLang: "es",
      targetLang: "fr",
    });
    await jest.advanceTimersByTimeAsync(0);
    for (let i = 0; i < 10; i++) await Promise.resolve();

    // Correct behavior: state should be fully reset for the new session
    expect(ctx._exec("sessionId")).toBe("session-2");
    expect(ctx._exec("playbackState")).toBe("idle");
    expect(ctx._exec("bufferedDuration")).toBe(0);
    expect(ctx._exec("totalAudioSentSec")).toBe(0);

    jest.useRealTimers();
  });
});

// ============================================================================
// Replay zone → live transition: prevent repeated audio
// ============================================================================

describe("replay zone: prevent repeated audio", () => {
  beforeEach(() => jest.useFakeTimers());
  afterEach(() => jest.useRealTimers());

  test("RESUME_CAPTURE should discard the held chunk (it contains replayed audio)", async () => {
    const mockFetch = jest.fn(() => {
      const reader = createMockStreamReader([]);
      return Promise.resolve({
        ok: true, status: 200,
        body: { getReader: () => reader },
        text: () => Promise.resolve(""),
      });
    });
    const { ctx, handler } = loadOffscreen({ fetch: mockFetch });

    ctx._exec('sessionId = "test-session"');
    ctx._exec('playbackState = "playing"');
    ctx._exec("holdChunks = true");
    ctx._exec("captureStartedAt = Date.now()");
    // The held chunk was captured during replay — it's replayed audio
    ctx._exec(
      "heldChunk = { samples: new Float32Array(8000), nativeSR: 48000 }"
    );

    mockFetch.mockClear();
    handler({ type: "RESUME_CAPTURE" });

    await jest.advanceTimersByTimeAsync(0);
    for (let i = 0; i < 10; i++) await Promise.resolve();

    // Correct behavior: held chunk should be DISCARDED, not sent.
    // It contains replayed audio that would produce duplicate translations.
    expect(mockFetch).not.toHaveBeenCalled();
    expect(ctx._exec("heldChunk")).toBeNull();
  });

  test("segments with startTime before replay boundary should not be scheduled", async () => {
    const { ctx, audioContextInstances } = loadOffscreen();

    ctx._exec('sessionId = "test-session"');
    ctx._exec('playbackState = "playing"');
    ctx._exec("playbackCtx = new AudioContext()");
    ctx._exec("nextPlayTime = 0");
    const playCtx = audioContextInstances[audioContextInstances.length - 1];
    const sourcesBefore = playCtx._sources.length;

    // Mark that we translated up to 40s of content before the replay zone
    ctx._exec("replayContentBoundary = 40");

    // Simulate a segment from the backend for replayed content (startTime=35 < 40)
    ctx.decodeAndQueue("AAAA", 35, 38);

    // Allow async decode + scheduleBuffers to complete
    await jest.advanceTimersByTimeAsync(0);
    for (let i = 0; i < 10; i++) await Promise.resolve();

    // Correct behavior: segment is replayed content (35 < 40).
    // It should NOT be scheduled on the playback context.
    // No new audio sources should have been created.
    expect(playCtx._sources.length).toBe(sourcesBefore);
    // bufferedDuration should not have increased
    expect(ctx._exec("bufferedDuration")).toBe(0);
  });

  test("segments with startTime after replay boundary should be scheduled normally", async () => {
    const { ctx, audioContextInstances } = loadOffscreen();

    ctx._exec('sessionId = "test-session"');
    ctx._exec('playbackState = "playing"');
    ctx._exec("playbackCtx = new AudioContext()");
    ctx._exec("nextPlayTime = 0");
    const playCtx = audioContextInstances[audioContextInstances.length - 1];
    const sourcesBefore = playCtx._sources.length;

    // Boundary at 40s, segment at 42s — this is fresh content
    ctx._exec("replayContentBoundary = 40");

    ctx.decodeAndQueue("AAAA", 42, 45);

    await jest.advanceTimersByTimeAsync(0);
    for (let i = 0; i < 10; i++) await Promise.resolve();

    // Fresh content SHOULD be decoded and scheduled
    expect(playCtx._sources.length).toBeGreaterThan(sourcesBefore);
  });
});

// ============================================================================
// Edge cases in replay zone transition
// ============================================================================

describe("replay zone edge cases", () => {
  beforeEach(() => jest.useFakeTimers());
  afterEach(() => jest.useRealTimers());

  test("segment at exact boundary (sourceStart === boundary) should pass through", async () => {
    const { ctx, audioContextInstances } = loadOffscreen();

    ctx._exec('playbackState = "playing"');
    ctx._exec("playbackCtx = new AudioContext()");
    ctx._exec("nextPlayTime = 0");
    const playCtx = audioContextInstances[audioContextInstances.length - 1];
    const sourcesBefore = playCtx._sources.length;

    ctx._exec("replayContentBoundary = 40");

    // Segment at EXACTLY the boundary — this is the first fresh content
    ctx.decodeAndQueue("AAAA", 40, 45);

    await jest.advanceTimersByTimeAsync(0);
    for (let i = 0; i < 10; i++) await Promise.resolve();

    // Should pass (40 is NOT < 40, so filter doesn't trigger)
    expect(playCtx._sources.length).toBeGreaterThan(sourcesBefore);
  });

  test("ZONE_ENDED sends held chunk that gets filtered by content boundary", async () => {
    const mockFetch = jest.fn(() => {
      const reader = createMockStreamReader([
        // The held chunk is replayed audio — backend returns a segment for it
        'data: {"type":"segment","audio_b64":"AAAA","caption":{"startTime":30,"endTime":35,"translated":"replayed"}}\n\n',
      ]);
      return Promise.resolve({
        ok: true, status: 200,
        body: { getReader: () => reader },
        text: () => Promise.resolve(""),
      });
    });
    const { ctx, handler, audioContextInstances } = loadOffscreen({ fetch: mockFetch });

    ctx._exec('sessionId = "test-session"');
    ctx._exec('playbackState = "playing"');
    ctx._exec("playbackCtx = new AudioContext()");
    ctx._exec("nextPlayTime = 0");
    ctx._exec("captureStartedAt = Date.now()");
    ctx._exec("replayContentBoundary = 40");
    ctx._exec("holdChunks = true");
    ctx._exec("heldChunk = { samples: new Float32Array(8000), nativeSR: 48000 }");

    const playCtx = audioContextInstances[audioContextInstances.length - 1];
    const sourcesBefore = playCtx._sources.length;

    // ZONE_ENDED sends the held chunk to backend
    handler({ type: "ZONE_ENDED" });

    await jest.advanceTimersByTimeAsync(0);
    for (let i = 0; i < 20; i++) await Promise.resolve();

    // The backend returned a segment with startTime=30 < boundary=40
    // It should be FILTERED — no new audio source created
    expect(playCtx._sources.length).toBe(sourcesBefore);
  });

  test("after RESUME_CAPTURE, holdChunks stays true (chunks still blocked)", () => {
    const { ctx, handler } = loadOffscreen();

    ctx._exec('sessionId = "test-session"');
    ctx._exec("holdChunks = true");
    ctx._exec("heldChunk = { samples: new Float32Array(1000), nativeSR: 48000 }");

    handler({ type: "RESUME_CAPTURE" });

    // holdChunks should remain true — only ZONE_ENDED unlocks it
    expect(ctx._exec("holdChunks")).toBe(true);
    // Held chunk discarded
    expect(ctx._exec("heldChunk")).toBeNull();
  });

  test("source timing pacing makes buffer survive the full replay zone", () => {
    const { ctx, audioContextInstances } = loadOffscreen();

    ctx._exec('playbackState = "playing"');
    ctx._exec("playbackCtx = new AudioContext()");
    const playCtx = audioContextInstances[audioContextInstances.length - 1];
    playCtx.currentTime = 0;
    ctx._exec("nextPlayTime = 0");

    // Simulate: 5 chunks × 8s source each = 40s totalAudioSentSec
    // TTS produces 4.8s per chunk (compression ratio 0.6)
    // With source timing: slotDuration = max(4.8, 8.0) = 8.0 per segment
    // Total timeline = 5 × 8.0 = 40.0s
    for (let i = 0; i < 5; i++) {
      ctx._exec(
        `decodedQueue.push({ audioBuffer: { duration: 4.8, numberOfChannels: 1, sampleRate: 16000, length: 76800, getChannelData: () => new Float32Array(1) }, sourceStart: ${i * 8}, sourceEnd: ${(i + 1) * 8} })`
      );
    }

    ctx.scheduleBuffers();

    // Buffer metric (nextPlayTime) should be 40s — matching a 40s replay zone
    expect(ctx._exec("nextPlayTime")).toBe(40.0);

    // Even though only 24s of actual audio was scheduled
    expect(ctx._exec("scheduledAudioDuration")).toBeCloseTo(24.0, 1);
  });
});

// ============================================================================
// Edge cases in service-worker buffer handling
// ============================================================================

describe("service-worker buffer edge cases", () => {
  // Note: these use the service-worker test infrastructure from service-worker.test.js
  // but are placed here to group with other edge case tests.
  // They test offscreen.js behavior directly.

  test("replayContentBoundary resets on stopCapture", async () => {
    const { ctx } = loadOffscreen();

    ctx._exec("replayContentBoundary = 40");
    await ctx.stopCapture();
    expect(ctx._exec("replayContentBoundary")).toBe(0);
  });

  test("replayContentBoundary is set from totalAudioSentSec at startPlayback", () => {
    const { ctx } = loadOffscreen();

    ctx._exec('sessionId = "test"');
    ctx._exec('playbackState = "buffering"');
    ctx._exec("captureStartedAt = Date.now()");
    ctx._exec("totalAudioSentSec = 35");
    ctx._exec("bufferedDuration = 25");

    ctx.startPlayback();

    expect(ctx._exec("replayContentBoundary")).toBe(35);
  });

  test("segments during buffering phase are never filtered (boundary is 0)", async () => {
    const { ctx, audioContextInstances } = loadOffscreen();

    // During buffering, replayContentBoundary = 0 (default)
    expect(ctx._exec("replayContentBoundary")).toBe(0);

    // Even segments with low startTime should pass through during buffering
    ctx._exec('playbackState = "buffering"');

    ctx.decodeAndQueue("AAAA", 5, 10);

    await jest.advanceTimersByTimeAsync(0);
    for (let i = 0; i < 10; i++) await Promise.resolve();

    // Should be decoded (boundary is 0, so filter doesn't trigger)
    expect(ctx._exec("bufferedDuration")).toBeGreaterThan(0);
  });
});

// ============================================================================
// Bug: TTS shorter than source causes audio to race ahead of video
// ============================================================================

describe("source-timing-based segment pacing", () => {
  test("short TTS segments are spaced by source slot duration (natural pauses)", () => {
    const { ctx, audioContextInstances } = loadOffscreen();

    ctx._exec('playbackState = "playing"');
    ctx._exec("playbackCtx = new AudioContext()");
    const playCtx = audioContextInstances[audioContextInstances.length - 1];
    playCtx.currentTime = 0;
    ctx._exec("nextPlayTime = 0");
    ctx._exec("scheduledAudioDuration = 0");

    // 3 segments: 3s TTS each, but source slots are 8s each.
    // Segments should be spaced by 8s (the source duration), not 3s.
    const mkBuf = (dur, ss, se) => `{ audioBuffer: { duration: ${dur}, numberOfChannels: 1, sampleRate: 16000, length: ${dur * 16000}, getChannelData: () => new Float32Array(1) }, sourceStart: ${ss}, sourceEnd: ${se} }`;
    ctx._exec(`decodedQueue.push(${mkBuf(3.0, 0, 8)})`);
    ctx._exec(`decodedQueue.push(${mkBuf(3.0, 8, 16)})`);
    ctx._exec(`decodedQueue.push(${mkBuf(3.0, 16, 24)})`);

    ctx.scheduleBuffers();

    // Each segment gets an 8s slot (source duration > TTS duration)
    // So segments start at 0, 8, 16 — matching the source timeline
    expect(playCtx._sources.length).toBe(3);
    expect(playCtx._sources[0].start).toHaveBeenCalledWith(0);
    expect(playCtx._sources[1].start).toHaveBeenCalledWith(8.0);
    expect(playCtx._sources[2].start).toHaveBeenCalledWith(16.0);
    expect(ctx._exec("nextPlayTime")).toBe(24.0);

    // scheduledAudioDuration tracks actual audio, not slot durations
    expect(ctx._exec("scheduledAudioDuration")).toBe(9.0);
  });

  test("TTS longer than source slot overflows gracefully (minor drift)", () => {
    const { ctx, audioContextInstances } = loadOffscreen();

    ctx._exec('playbackState = "playing"');
    ctx._exec("playbackCtx = new AudioContext()");
    const playCtx = audioContextInstances[audioContextInstances.length - 1];
    playCtx.currentTime = 0;
    ctx._exec("nextPlayTime = 0");

    // Segment: 2s source slot, but 3.5s TTS (verbose translation)
    ctx._exec(
      'decodedQueue.push({ audioBuffer: { duration: 3.5, numberOfChannels: 1, sampleRate: 16000, length: 56000, getChannelData: () => new Float32Array(1) }, sourceStart: 0, sourceEnd: 2 })'
    );

    ctx.scheduleBuffers();

    // slotDuration = max(2, 3.5) = 3.5 — TTS overflows, no overlap
    expect(ctx._exec("nextPlayTime")).toBe(3.5);
  });

  test("segments without timing info fall back to gapless scheduling", () => {
    const { ctx, audioContextInstances } = loadOffscreen();

    ctx._exec('playbackState = "playing"');
    ctx._exec("playbackCtx = new AudioContext()");
    const playCtx = audioContextInstances[audioContextInstances.length - 1];
    playCtx.currentTime = 0;
    ctx._exec("nextPlayTime = 0");

    // No sourceStart/sourceEnd — should schedule gaplessly
    ctx._exec(
      'decodedQueue.push({ duration: 3.0, numberOfChannels: 1, sampleRate: 16000, length: 48000, getChannelData: () => new Float32Array(1) })'
    );
    ctx._exec(
      'decodedQueue.push({ duration: 2.0, numberOfChannels: 1, sampleRate: 16000, length: 32000, getChannelData: () => new Float32Array(1) })'
    );

    ctx.scheduleBuffers();

    expect(ctx._exec("nextPlayTime")).toBe(5.0);
  });
});

// ============================================================================
// audio-worklet.js source verification
// (AudioWorkletProcessor can't run in Node VM — these verify source structure)
// ============================================================================

describe("audio-worklet.js (source verification — not executable in Node)", () => {
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
