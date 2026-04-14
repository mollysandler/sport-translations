/**
 * Shared test helpers: Chrome API mocks, audio mocks, DOM mocks,
 * and script evaluation utilities.
 */
const vm = require("vm");
const fs = require("fs");
const path = require("path");

// ============================================================================
// Mock video element
// ============================================================================

function createMockVideo(opts = {}) {
  const listeners = {};
  return {
    paused: opts.paused ?? false,
    ended: opts.ended ?? false,
    playbackRate: opts.playbackRate ?? 1.0,
    videoWidth: opts.videoWidth ?? 1920,
    videoHeight: opts.videoHeight ?? 1080,
    currentTime: opts.currentTime ?? 0,
    pause() {
      this.paused = true;
      (listeners["pause"] || []).forEach((fn) => fn());
    },
    play() {
      this.paused = false;
      (listeners["play"] || []).forEach((fn) => fn());
    },
    addEventListener(event, fn) {
      if (!listeners[event]) listeners[event] = [];
      listeners[event].push(fn);
    },
    removeEventListener(event, fn) {
      if (listeners[event]) {
        listeners[event] = listeners[event].filter((f) => f !== fn);
      }
    },
    _listeners: listeners,
  };
}

// ============================================================================
// Chrome API mock
// ============================================================================

function createChromeMock() {
  const sentMessages = [];
  let messageHandler = null;

  const chrome = {
    runtime: {
      sendMessage: jest.fn((msg) => {
        sentMessages.push(msg);
        return Promise.resolve();
      }),
      onMessage: {
        addListener: jest.fn((handler) => {
          messageHandler = handler;
        }),
      },
      getContexts: jest.fn(() => Promise.resolve([])),
    },
    tabs: {
      sendMessage: jest.fn(() => Promise.resolve()),
      query: jest.fn(() => Promise.resolve([{ id: 42, url: "https://youtube.com" }])),
      onRemoved: { addListener: jest.fn() },
      onUpdated: { addListener: jest.fn() },
    },
    scripting: {
      executeScript: jest.fn(() => Promise.resolve()),
    },
    action: {
      onClicked: { addListener: jest.fn() },
    },
    sidePanel: {
      open: jest.fn(),
    },
    offscreen: {
      createDocument: jest.fn(() => Promise.resolve()),
      closeDocument: jest.fn(() => Promise.resolve()),
    },
    tabCapture: {
      getMediaStreamId: jest.fn(() => Promise.resolve("fake-stream-id-1234567890")),
    },
    storage: {
      local: {
        get: jest.fn((keys, cb) => cb({})),
        set: jest.fn(),
      },
    },
  };

  return { chrome, sentMessages, getMessageHandler: () => messageHandler };
}

// ============================================================================
// Audio API mocks
// ============================================================================

class MockAudioBuffer {
  constructor(duration = 1.0, sampleRate = 16000) {
    this.duration = duration;
    this.sampleRate = sampleRate;
    this.numberOfChannels = 1;
    this.length = Math.ceil(duration * sampleRate);
    this._channelData = new Float32Array(this.length);
  }
  getChannelData() {
    return this._channelData;
  }
}

class MockBufferSource {
  constructor() {
    this.buffer = null;
    this.connect = jest.fn();
    this.start = jest.fn();
    this.stop = jest.fn();
    this.onended = null;
  }
}

class MockAudioContext {
  constructor() {
    this.currentTime = 0;
    this.state = "running";
    this.sampleRate = 48000;
    this.destination = { _isDestination: true };
    this._sources = [];
    this._listeners = {};
    this._closed = false;
    this.audioWorklet = {
      addModule: jest.fn(() => Promise.resolve()),
    };
  }

  createBufferSource() {
    const source = new MockBufferSource();
    this._sources.push(source);
    return source;
  }

  createMediaStreamSource() {
    return { connect: jest.fn(), disconnect: jest.fn() };
  }

  createBuffer(channels, length, sampleRate) {
    return new MockAudioBuffer(length / sampleRate, sampleRate);
  }

  decodeAudioData(arrayBuffer) {
    // Default: 1-second buffer. Tests can override via mockImplementation.
    return Promise.resolve(new MockAudioBuffer(1.0, 16000));
  }

  resume() {
    this.state = "running";
    return Promise.resolve();
  }

  close() {
    this._closed = true;
    return Promise.resolve();
  }

  addEventListener(event, fn) {
    if (!this._listeners[event]) this._listeners[event] = [];
    this._listeners[event].push(fn);
  }

  removeEventListener(event, fn) {
    if (this._listeners[event]) {
      this._listeners[event] = this._listeners[event].filter((f) => f !== fn);
    }
  }
}

class MockOfflineAudioContext {
  constructor(channels, length, sampleRate) {
    this._channels = channels;
    this._length = length;
    this.sampleRate = sampleRate;
    this.destination = { _isDestination: true };
  }

  createBuffer(channels, length, sampleRate) {
    return new MockAudioBuffer(length / sampleRate, sampleRate);
  }

  createBufferSource() {
    return new MockBufferSource();
  }

  startRendering() {
    const buf = new MockAudioBuffer(this._length / this.sampleRate, this.sampleRate);
    return Promise.resolve(buf);
  }
}

class MockAudioWorkletNode {
  constructor(context, name, options) {
    this._name = name;
    this._options = options;
    this._connectCalls = [];
    this.port = {
      onmessage: null,
      postMessage: jest.fn(),
    };
  }

  connect(dest) {
    this._connectCalls.push(dest);
  }

  disconnect() {}
}

// ============================================================================
// Media stream mock
// ============================================================================

function createMockMediaStream() {
  const tracks = [{ stop: jest.fn(), kind: "audio" }];
  return {
    getTracks: () => tracks,
    _tracks: tracks,
  };
}

// ============================================================================
// Web API mocks for offscreen.js
// ============================================================================

class MockBlob {
  constructor(parts, opts) {
    this._parts = parts || [];
    this.type = (opts && opts.type) || "";
    this.size = this._parts.reduce((sum, p) => {
      if (p instanceof ArrayBuffer) return sum + p.byteLength;
      if (p && p.byteLength !== undefined) return sum + p.byteLength;
      if (typeof p === "string") return sum + p.length;
      return sum;
    }, 0);
  }
}

class MockFormData {
  constructor() {
    this._data = {};
  }
  append(key, value, filename) {
    this._data[key] = { value, filename };
  }
  get(key) {
    return this._data[key] ? this._data[key].value : undefined;
  }
}

class MockTextDecoder {
  decode(value, opts) {
    if (!value) return "";
    return Buffer.from(value).toString("utf-8");
  }
}

class MockTextEncoder {
  encode(str) {
    return new Uint8Array(Buffer.from(str, "utf-8"));
  }
}

/**
 * Creates a mock ReadableStream reader that yields the given chunks in order.
 */
function createMockStreamReader(chunks) {
  let index = 0;
  return {
    read: jest.fn(async () => {
      if (index >= chunks.length) return { done: true, value: undefined };
      const value = typeof chunks[index] === "string"
        ? new MockTextEncoder().encode(chunks[index])
        : chunks[index];
      index++;
      return { done: false, value };
    }),
  };
}

// ============================================================================
// Side panel DOM mock
// ============================================================================

function createMockElement(tag, defaults = {}) {
  const classList = new Set(defaults.classes || []);
  const listeners = {};
  const children = [];

  return {
    tagName: tag.toUpperCase(),
    textContent: defaults.textContent || "",
    className: defaults.className || "",
    value: defaults.value || "",
    disabled: defaults.disabled || false,
    scrollTop: 0,
    scrollHeight: 0,
    classList: {
      add: (...cls) => cls.forEach((c) => classList.add(c)),
      remove: (...cls) => cls.forEach((c) => classList.delete(c)),
      contains: (c) => classList.has(c),
      _set: classList,
    },
    addEventListener(event, fn) {
      if (!listeners[event]) listeners[event] = [];
      listeners[event].push(fn);
    },
    querySelectorAll(sel) {
      if (sel === ".caption-item") return children.filter((c) => c._isCaptionItem);
      return [];
    },
    appendChild(child) {
      child._isCaptionItem = true;
      children.push(child);
    },
    _listeners: listeners,
    _children: children,
    _triggerEvent(event, ...args) {
      (listeners[event] || []).forEach((fn) => fn(...args));
    },
  };
}

function createSidepanelDOM() {
  const elements = {
    startStopBtn: createMockElement("button", { textContent: "Start Translating", className: "btn btn-start" }),
    statusBadge: createMockElement("span", { textContent: "Idle", className: "status-badge" }),
    captions: createMockElement("div"),
    emptyState: createMockElement("div"),
    silenceWarning: createMockElement("div", { classes: ["hidden"] }),
    warmingUp: createMockElement("div", { classes: ["hidden"] }),
    warmingText: createMockElement("span", { textContent: "Warming up GPU..." }),
    elapsedTimer: createMockElement("span"),
    sourceLang: createMockElement("select", { value: "en" }),
    targetLang: createMockElement("select", { value: "hi" }),
    syncBadge: createMockElement("span", { className: "sync-badge hidden" }),
    errorBanner: createMockElement("div", { classes: ["hidden"] }),
    errorMessage: createMockElement("span"),
    retryBtn: createMockElement("button"),
    dismissBtn: createMockElement("button"),
  };

  const document = {
    getElementById: jest.fn((id) => elements[id] || null),
    createElement: jest.fn((tag) => createMockElement(tag)),
    body: {},
    querySelectorAll: jest.fn(() => []),
  };

  return { document, elements };
}

// ============================================================================
// Script evaluation via VM
// ============================================================================

/**
 * Evaluates a script file inside a vm context with the given globals.
 * Returns the context with an `_exec(code)` method for reading/writing
 * `let`/`const` variables that aren't on the context object.
 */
function evalScript(filename, contextGlobals) {
  const filePath = path.resolve(__dirname, "..", filename);
  const code = fs.readFileSync(filePath, "utf-8");
  const context = vm.createContext({
    console,
    setTimeout,
    clearTimeout,
    setInterval,
    clearInterval,
    Array,
    Object,
    Set,
    RegExp,
    Math,
    Date,
    Promise,
    DataView,
    ArrayBuffer,
    Uint8Array,
    Float32Array,
    Int16Array,
    AbortController,
    TextDecoder: MockTextDecoder,
    TextEncoder: MockTextEncoder,
    Blob: MockBlob,
    FormData: MockFormData,
    atob: (s) => Buffer.from(s, "base64").toString("binary"),
    btoa: (s) => Buffer.from(s, "binary").toString("base64"),
    navigator: { mediaDevices: { getUserMedia: jest.fn() } },
    AudioContext: jest.fn(() => new MockAudioContext()),
    OfflineAudioContext: jest.fn((...args) => new MockOfflineAudioContext(...args)),
    AudioWorkletNode: jest.fn((...args) => new MockAudioWorkletNode(...args)),
    MutationObserver: class MutationObserver {
      constructor(cb) { this._cb = cb; }
      observe() {}
      disconnect() {}
    },
    document: {
      querySelectorAll: jest.fn(() => []),
      body: {},
    },
    fetch: jest.fn(),
    ...contextGlobals,
  });
  vm.runInContext(code, context);

  // Wrap context so tests can read/write let/const variables
  context._exec = (expr) => vm.runInContext(expr, context);

  return context;
}

module.exports = {
  createMockVideo,
  createChromeMock,
  createMockMediaStream,
  createSidepanelDOM,
  createMockElement,
  createMockStreamReader,
  MockAudioContext,
  MockOfflineAudioContext,
  MockAudioWorkletNode,
  MockAudioBuffer,
  MockBufferSource,
  MockBlob,
  MockFormData,
  MockTextDecoder,
  MockTextEncoder,
  evalScript,
};
