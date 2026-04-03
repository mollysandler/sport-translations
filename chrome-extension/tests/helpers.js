/**
 * Shared test helpers: Chrome API mocks and script evaluation utilities.
 */
const vm = require("vm");
const fs = require("fs");
const path = require("path");

/**
 * Creates a mock video element with controllable state.
 */
function createMockVideo(opts = {}) {
  const listeners = {};
  return {
    paused: opts.paused ?? false,
    ended: opts.ended ?? false,
    playbackRate: opts.playbackRate ?? 1.0,
    videoWidth: opts.videoWidth ?? 1920,
    videoHeight: opts.videoHeight ?? 1080,
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

/**
 * Creates a minimal mock of the chrome.* APIs used by the extension scripts.
 * Returns { chrome, sentMessages, messageHandler } so tests can inspect
 * outgoing messages and simulate incoming ones.
 */
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

/**
 * Evaluates a script file inside a vm context with the given globals.
 * This lets us run the IIFE-wrapped extension scripts in a controlled env.
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
    Math,
    Date,
    Promise,
    DataView,
    ArrayBuffer,
    Uint8Array,
    Float32Array,
    AbortController,
    Blob: class Blob {},
    FormData: class FormData {
      constructor() { this._data = {}; }
      append(k, v) { this._data[k] = v; }
    },
    atob: (s) => Buffer.from(s, "base64").toString("binary"),
    btoa: (s) => Buffer.from(s, "binary").toString("base64"),
    navigator: { mediaDevices: { getUserMedia: jest.fn() } },
    AudioContext: jest.fn(),
    OfflineAudioContext: jest.fn(),
    AudioWorkletNode: jest.fn(),
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
  return context;
}

module.exports = { createMockVideo, createChromeMock, evalScript };
