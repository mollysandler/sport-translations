/**
 * Content script: canvas frame-delay overlay with seek-back fallback.
 *
 * Primary mode (canvas overlay):
 *   1. Find <video>, hide it (opacity:0), overlay a <canvas>
 *   2. requestVideoFrameCallback captures frames into a ring buffer
 *   3. Draw frames delayed by pipeline latency — user sees delayed video
 *   4. Translated audio plays in sync with the delayed frames
 *
 * Fallback mode (seek-back):
 *   Used when DRM is detected (canvas draws black) or requestVideoFrameCallback
 *   is unavailable. Falls back to the simpler seek-back + rate adjustment approach.
 */
(function () {
  // Kill any previous instance (stale scripts survive extension reloads).
  // The previous instance's teardown function cleans up its canvas, observers,
  // and requestVideoFrameCallback loop so it stops interfering.
  if (window.__liveTranslatorV2Teardown) {
    try { window.__liveTranslatorV2Teardown(); } catch (e) {}
  }

  // -------------------------------------------------------------------------
  // State
  // -------------------------------------------------------------------------

  let video = null;
  let observer = null;
  let pendingOverlayText = null;

  // Overlay / buffering UI
  let overlayEl = null;
  let overlayTextEl = null;
  let overlayProgressEl = null;

  // Canvas overlay mode
  let syncMode = null; // "canvas" or "seekback" — determined by DRM detection
  let canvasEl = null;
  let canvasCtx = null;
  let frameBuffer = []; // [{bitmap: ImageBitmap, time: number}]
  let delayFrames = 0; // how many frames to buffer before drawing
  let rvfcId = null; // requestVideoFrameCallback handle
  let resizeObs = null;
  let canvasActive = false;  // capturing frames into buffer
  let drawingActive = false; // drawing delayed frames (starts when audio playback begins)
  let drmVerified = false;
  let drmBlackFrames = 0; // consecutive all-black frames seen during DRM check
  let targetDelaySec = 3;

  // Seek-back fallback mode
  let extensionPaused = false;
  let userPaused = false;

  // Guards to distinguish extension-triggered pause/play from user-triggered
  let extensionTriggeredPause = false;
  let extensionTriggeredPlay = false;

  const MAX_BUFFER_FRAMES = 300; // hard cap (~10s at 30fps — must accommodate full pipeline delay)
  const CANVAS_SCALE = 480; // capture height (lower = less memory, 300 frames at 480p ≈ 500MB)

  // Canvas pool: reuse capture canvases instead of creating/GC-ing 30/sec
  let canvasPool = [];

  // -------------------------------------------------------------------------
  // Video discovery
  // -------------------------------------------------------------------------

  function findVideo() {
    const videos = Array.from(document.querySelectorAll("video"));
    if (videos.length === 0) return null;
    const playing = videos.find((v) => !v.paused && !v.ended);
    if (playing) return playing;
    return videos.sort(
      (a, b) => b.videoWidth * b.videoHeight - a.videoWidth * a.videoHeight
    )[0];
  }

  function init() {
    video = findVideo();
    if (video) {
      onVideoFound();
    } else {
      observer = new MutationObserver(() => {
        video = findVideo();
        if (video) {
          if (observer) { observer.disconnect(); observer = null; }
          onVideoFound();
        }
      });
      observer.observe(document.body, { childList: true, subtree: true });
      sendMsg({ type: "VIDEO_NOT_FOUND" });
    }
  }

  function onVideoFound() {
    bindPauseDetection(video);
    if (pendingOverlayText) createOverlay(pendingOverlayText);
    sendMsg({ type: "VIDEO_FOUND", currentTime: video.currentTime });
  }

  function bindPauseDetection(v) {
    v.addEventListener("pause", () => {
      if (extensionTriggeredPause) {
        extensionTriggeredPause = false;
        return;
      }
      if (!extensionPaused) {
        userPaused = true;
        sendMsg({ type: "USER_PAUSED_VIDEO" });
      }
    });
    v.addEventListener("play", () => {
      if (extensionTriggeredPlay) {
        extensionTriggeredPlay = false;
        userPaused = false;
        return;
      }
      // User manually resumed (either overriding extension pause or their own pause)
      if (userPaused || extensionPaused) {
        userPaused = false;
        extensionPaused = false;
        sendMsg({ type: "USER_RESUMED_VIDEO" });
      }
      userPaused = false;
    });
  }

  function sendMsg(msg) {
    try { chrome.runtime.sendMessage(msg).catch(() => {}); } catch (e) {}
  }

  // -------------------------------------------------------------------------
  // DRM detection
  // -------------------------------------------------------------------------

  function detectDRM() {
    if (!video) return true;
    if (!video.requestVideoFrameCallback) return true;
    if (video.mediaKeys) return true;
    return false;
  }

  // -------------------------------------------------------------------------
  // Canvas overlay mode
  // -------------------------------------------------------------------------

  let hideStyleEl = null;

  function startCanvasMode() {
    syncMode = "canvas";
    sendMsg({ type: "SYNC_MODE", mode: "canvas" });

    const parent = video.parentElement || document.body;

    // Create canvas, sized to match video display area
    canvasEl = document.createElement("canvas");
    canvasEl.id = "__live-translator-canvas";
    canvasCtx = canvasEl.getContext("2d");

    if (getComputedStyle(parent).position === "static") {
      parent.style.position = "relative";
    }

    // Size canvas drawing buffer AND CSS display to match the video's layout.
    // CRITICAL: Do NOT use CSS `width: 100%; height: 100%` — on YouTube,
    // the parent's content box can differ from the video's display area,
    // causing the drawing buffer and CSS size to mismatch. This makes
    // drawn content appear at wrong positions or be invisible.
    const videoRect = video.getBoundingClientRect();
    const parentRect = parent.getBoundingClientRect();
    const cssW = Math.round(videoRect.width);
    const cssH = Math.round(videoRect.height);
    canvasEl.width = cssW;
    canvasEl.height = cssH;
    canvasEl.style.cssText = `
      position: absolute;
      top: ${Math.round(videoRect.top - parentRect.top)}px;
      left: ${Math.round(videoRect.left - parentRect.left)}px;
      width: ${cssW}px;
      height: ${cssH}px;
      z-index: 999998;
      pointer-events: none;
    `;

    parent.appendChild(canvasEl);
    resizeObs = new ResizeObserver(resizeCanvas);
    resizeObs.observe(video);
    document.addEventListener("fullscreenchange", resizeCanvas);

    // Hide original video via CSS rule with !important — YouTube's player JS
    // periodically resets inline styles on the video element, so a plain
    // video.style.opacity = "0" gets overridden within milliseconds.
    video.classList.add("__lt-hidden");
    if (!hideStyleEl) {
      hideStyleEl = document.createElement("style");
      hideStyleEl.id = "__live-translator-hide-style";
      hideStyleEl.textContent = "video.__lt-hidden { opacity: 0 !important; }";
      (document.head || document.documentElement).appendChild(hideStyleEl);
    }

    // Compute frame delay from target latency
    updateDelayFrames();

    // Start capturing frames
    canvasActive = true;
    requestFrame();
  }

  function resizeCanvas() {
    if (!canvasEl || !video) return;
    const videoRect = video.getBoundingClientRect();
    const w = Math.round(Math.min(videoRect.width, 1920));
    const h = Math.round(Math.min(videoRect.height, 1080));
    if (w === 0 || h === 0) return;
    if (canvasEl.width === w && canvasEl.height === h) return;
    canvasEl.width = w;
    canvasEl.height = h;
    // Also update CSS dimensions to match (keeps drawing buffer and display in sync)
    canvasEl.style.width = w + "px";
    canvasEl.style.height = h + "px";
    const parent = canvasEl.parentElement;
    if (parent) {
      const parentRect = parent.getBoundingClientRect();
      canvasEl.style.top = Math.round(videoRect.top - parentRect.top) + "px";
      canvasEl.style.left = Math.round(videoRect.left - parentRect.left) + "px";
    }
  }

  function updateDelayFrames() {
    // Estimate FPS from video (default 30)
    const fps = 30;
    delayFrames = Math.max(1, Math.ceil(targetDelaySec * fps));
    // Don't exceed hard cap
    if (delayFrames > MAX_BUFFER_FRAMES) delayFrames = MAX_BUFFER_FRAMES;
  }

  function drawBufferingOverlay() {
    if (!canvasCtx || !canvasEl) return;
    const w = canvasEl.width;
    const h = canvasEl.height;
    canvasCtx.fillStyle = "rgba(0, 0, 0, 0.5)";
    canvasCtx.fillRect(0, 0, w, h);
    const fontSize = Math.max(16, Math.round(h / 25));
    canvasCtx.fillStyle = "white";
    canvasCtx.font = `600 ${fontSize}px -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif`;
    canvasCtx.textAlign = "center";
    canvasCtx.textBaseline = "middle";
    canvasCtx.fillText("Buffering translation\u2026", w / 2, h / 2);
  }

  function requestFrame() {
    if (!canvasActive || !video) return;
    rvfcId = video.requestVideoFrameCallback(onVideoFrame);
  }

  // Frame counter for periodic logging
  let frameCount = 0;

  function onVideoFrame(now, metadata) {
    if (!canvasActive || !video) { requestFrame(); return; }
    if (video.videoWidth === 0 || video.videoHeight === 0) { requestFrame(); return; }
    if (!canvasCtx || !canvasEl || canvasEl.width === 0 || canvasEl.height === 0) { requestFrame(); return; }

    frameCount++;

    // ---- Step 1: Draw video DIRECTLY to display canvas ----
    // This is the primary render path. Drawing directly from the video element
    // to the display canvas avoids intermediate-canvas issues (GPU texture sync,
    // canvas clearing, pool recycling). The user always sees the video.
    canvasCtx.drawImage(video, 0, 0, canvasEl.width, canvasEl.height);

    // ---- Step 2: Buffer frames for delay mechanism ----
    // Capture a snapshot for the frame buffer using createImageBitmap (GPU-safe)
    // or a pooled canvas as fallback. The buffer enables showing delayed frames.
    const aspect = video.videoWidth / video.videoHeight;
    const captureH = Math.min(video.videoHeight, CANVAS_SCALE);
    const captureW = Math.round(aspect * captureH);

    const frameCanvas = canvasPool.length > 0 ? canvasPool.pop() : document.createElement("canvas");
    if (frameCanvas.width !== captureW || frameCanvas.height !== captureH) {
      frameCanvas.width = captureW;
      frameCanvas.height = captureH;
    }
    const frameCtx = frameCanvas._cachedCtx || (frameCanvas._cachedCtx = frameCanvas.getContext("2d"));
    frameCtx.drawImage(video, 0, 0, captureW, captureH);
    frameBuffer.push({ canvas: frameCanvas, time: metadata.mediaTime });

    // ---- Step 3: If drawing delayed frames, overwrite the direct draw ----
    if (drawingActive && frameBuffer.length > 1) {
      const old = frameBuffer.shift();
      // Draw the delayed frame on top of the direct draw.
      // If the buffer frame has real content, it covers the live frame.
      // If it's blank (GPU issue), the live video from Step 1 shows through.
      canvasCtx.drawImage(old.canvas, 0, 0, canvasEl.width, canvasEl.height);
      canvasPool.push(old.canvas);
    } else if (!drawingActive) {
      drawBufferingOverlay();
    }

    // ---- Step 4: Hard cap ----
    while (frameBuffer.length > MAX_BUFFER_FRAMES) {
      const discarded = frameBuffer.shift();
      canvasPool.push(discarded.canvas);
    }

    // ---- Step 5: Verify canvas is actually rendering (frame 15) ----
    // Check the DISPLAY canvas for real pixels. If blank after 15 frames,
    // drawImage(video) is broken (GPU decode issue) — fall back to seekback.
    if (frameCount === 15 && !drmVerified) {
      try {
        const cx = Math.floor(canvasEl.width / 2);
        const cy = Math.floor(canvasEl.height / 2);
        const px = canvasCtx.getImageData(cx - 4, cy - 4, 8, 8).data;
        let hasContent = false;
        let allBlack = true;
        for (let i = 0; i < px.length; i += 4) {
          if (px[i + 3] > 0) hasContent = true; // not transparent
          if (px[i] > 16 || px[i + 1] > 16 || px[i + 2] > 16) allBlack = false;
          if (hasContent && !allBlack) break;
        }
        if (!hasContent) {
          // Canvas is transparent — drawImage(video) is broken
          console.log("[content] Canvas is transparent after 15 frames — drawImage not working, trying seekback");
          drmVerified = true;
          stopCanvasMode();
          startSeekbackMode();
          return;
        }
        if (hasContent && !allBlack) {
          drmVerified = true;
          console.log("[content] Canvas rendering verified — video frames OK");
        }
        // If allBlack, continue checking on subsequent frames (might be dark scene)
      } catch (e) {
        // getImageData threw — canvas is tainted (cross-origin). Pixels ARE there.
        drmVerified = true;
        console.log("[content] Canvas tainted (cross-origin) — rendering OK");
      }
    }

    // ---- Step 6: DRM check (frames 15-30, only if not yet verified) ----
    if (frameCount > 15 && frameCount <= 30 && !drmVerified) {
      try {
        const cx = Math.floor(canvasEl.width / 2);
        const cy = Math.floor(canvasEl.height / 2);
        const px = canvasCtx.getImageData(cx - 4, cy - 4, 8, 8).data;
        let allBlack = true;
        for (let i = 0; i < px.length; i += 4) {
          if (px[i] > 16 || px[i + 1] > 16 || px[i + 2] > 16) { allBlack = false; break; }
        }
        if (allBlack) {
          drmBlackFrames++;
          if (drmBlackFrames >= 5) {
            drmVerified = true;
            console.log("[content] DRM detected (5 consecutive black frames) — falling back to seekback");
            stopCanvasMode();
            startSeekbackMode();
            return;
          }
        } else {
          drmVerified = true;
          drmBlackFrames = 0;
          console.log("[content] Canvas rendering verified — not DRM");
        }
      } catch (e) {
        drmVerified = true;
      }
    }

    // After frame 30, stop checking regardless
    if (frameCount > 30 && !drmVerified) {
      drmVerified = true;
    }

    requestFrame();
  }

  function stopCanvasMode() {
    canvasActive = false;
    drawingActive = false;
    drmVerified = false;
    drmBlackFrames = 0;
    frameCount = 0;
    if (rvfcId !== null && video && video.cancelVideoFrameCallback) {
      video.cancelVideoFrameCallback(rvfcId);
      rvfcId = null;
    }
    frameBuffer = [];
    canvasPool = [];
    if (canvasEl) { canvasEl.remove(); canvasEl = null; canvasCtx = null; }
    if (resizeObs) { resizeObs.disconnect(); resizeObs = null; }
    document.removeEventListener("fullscreenchange", resizeCanvas);
    if (video) {
      video.classList.remove("__lt-hidden");
      video.style.opacity = "";
    }
    if (hideStyleEl) { hideStyleEl.remove(); hideStyleEl = null; }
  }

  // -------------------------------------------------------------------------
  // Seek-back fallback mode
  // -------------------------------------------------------------------------

  function startSeekbackMode() {
    syncMode = "seekback";
    sendMsg({ type: "SYNC_MODE", mode: "seekback" });
  }

  function handleSeekback(seekBackSec) {
    if (!video) return;
    extensionPaused = true;
    userPaused = false;
    video.currentTime = Math.max(0, video.currentTime - seekBackSec);
    const onSeeked = () => {
      video.removeEventListener("seeked", onSeeked);
      userPaused = false;
      video.play().catch(() => {});
    };
    video.addEventListener("seeked", onSeeked);
    let attempts = 0;
    const guard = setInterval(() => {
      attempts++;
      if (!video) { clearInterval(guard); return; }
      userPaused = false;
      if (video.paused) video.play().catch(() => {});
      if (!video.paused || attempts >= 10) {
        clearInterval(guard);
        extensionPaused = false;
      }
    }, 500);
  }

  // -------------------------------------------------------------------------
  // Overlay management (used during buffer phase in both modes)
  // -------------------------------------------------------------------------

  function createOverlay(text) {
    removeOverlay();
    if (!video) { pendingOverlayText = text; return; }
    pendingOverlayText = null;

    const parent = video.parentElement || document.body;
    overlayEl = document.createElement("div");
    overlayEl.id = "__live-translator-overlay";
    overlayEl.style.cssText = `
      position: absolute;
      top: 0; left: 0; right: 0; bottom: 0;
      display: flex; flex-direction: column;
      align-items: center; justify-content: center;
      background: rgba(0, 0, 0, 0.6);
      z-index: 999999;
      pointer-events: none;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    `;
    overlayTextEl = document.createElement("div");
    overlayTextEl.style.cssText = `
      color: white; font-size: 18px; font-weight: 600;
      text-shadow: 0 2px 4px rgba(0,0,0,0.5);
      margin-bottom: 12px;
    `;
    overlayTextEl.textContent = text || "Buffering translation...";
    overlayProgressEl = document.createElement("div");
    overlayProgressEl.style.cssText = `
      width: 200px; height: 4px;
      background: rgba(255,255,255,0.2);
      border-radius: 2px; overflow: hidden;
    `;
    const bar = document.createElement("div");
    bar.id = "__live-translator-progress-bar";
    bar.style.cssText = `
      width: 0%; height: 100%;
      background: #3b82f6; border-radius: 2px;
      transition: width 0.3s ease;
    `;
    overlayProgressEl.appendChild(bar);
    overlayEl.appendChild(overlayTextEl);
    overlayEl.appendChild(overlayProgressEl);

    if (getComputedStyle(parent).position === "static") {
      parent.style.position = "relative";
    }
    parent.appendChild(overlayEl);
  }

  function updateOverlay(text, progress) {
    if (overlayTextEl) overlayTextEl.textContent = text;
    const bar = document.getElementById("__live-translator-progress-bar");
    if (bar) bar.style.width = `${progress}%`;
  }

  function removeOverlay() {
    if (overlayEl) { overlayEl.remove(); overlayEl = null; overlayTextEl = null; overlayProgressEl = null; }
  }

  // -------------------------------------------------------------------------
  // Message handling
  // -------------------------------------------------------------------------

  chrome.runtime.onMessage.addListener((msg, _sender, sendResponse) => {
    if (!video && msg.type !== "VIDEO_CLEANUP") {
      video = findVideo();
    }

    switch (msg.type) {
      // --- Sync mode initialization ---
      case "START_SYNC":
        // Offscreen tells us to begin. Detect DRM and choose mode.
        if (video) {
          const isDRM = detectDRM();
          if (isDRM) {
            startSeekbackMode();
          } else {
            startCanvasMode();
          }
        }
        sendResponse({ ok: !!video, mode: syncMode });
        break;

      case "SET_DELAY":
        // Offscreen measured pipeline latency — update canvas delay
        targetDelaySec = msg.delaySec || 3;
        updateDelayFrames();
        sendResponse({ ok: true });
        break;

      // --- Pause / Resume from side panel ---
      case "PAUSE_ALL":
        extensionPaused = true;
        if (video && !video.paused) {
          extensionTriggeredPause = true;
          video.pause();
        }
        sendResponse({ ok: true });
        break;

      case "RESUME_ALL":
        extensionPaused = false;
        userPaused = false;
        if (video && video.paused) {
          extensionTriggeredPlay = true;
          video.play().catch(() => {});
        }
        sendResponse({ ok: true });
        break;

      // --- Seek-back fallback commands ---
      case "VIDEO_SEEK_BACK":
        handleSeekback(msg.seekBackSec);
        sendResponse({ ok: !!video });
        break;

      case "VIDEO_ADJUST_RATE":
        if (video && syncMode === "seekback") {
          extensionPaused = true;
          video.playbackRate = msg.rate;
          let count = 0;
          const guard = setInterval(() => {
            count++;
            if (video && video.playbackRate !== msg.rate) video.playbackRate = msg.rate;
            if (count >= 10) { clearInterval(guard); extensionPaused = false; }
          }, 200);
          setTimeout(() => {
            if (video && video.paused) { userPaused = false; video.play().catch(() => {}); }
          }, 300);
        }
        sendResponse({ ok: !!video });
        break;

      case "VIDEO_REPORT_TIME":
        sendResponse({
          ok: !!video,
          currentTime: video ? video.currentTime : null,
          playbackRate: video ? video.playbackRate : null,
          paused: video ? video.paused : true,
        });
        break;

      // --- Overlay commands (both modes) ---
      case "SHOW_OVERLAY":
        createOverlay(msg.text);
        sendResponse({ ok: true });
        break;

      case "UPDATE_OVERLAY":
        updateOverlay(msg.text, msg.progress);
        sendResponse({ ok: true });
        break;

      case "HIDE_OVERLAY":
        removeOverlay();
        sendResponse({ ok: true });
        break;

      case "PLAYBACK_STARTED":
        // Audio playback has begun — start drawing delayed frames.
        // The full buffer IS the delay — the oldest frame corresponds to the
        // video that was playing when capture started, which is when the first
        // audio was captured and sent to the backend for translation.
        if (syncMode === "canvas") {
          drawingActive = true;
          console.log(
            `[content] Canvas drawing activated.`,
            `Buffer: ${frameBuffer.length} frames`,
            `(~${(frameBuffer.length / 30).toFixed(1)}s delay)`
          );
        }
        sendResponse({ ok: true });
        break;

      // --- Cleanup ---
      case "VIDEO_CLEANUP":
        if (window.__liveTranslatorV2Teardown) {
          window.__liveTranslatorV2Teardown();
        }
        userPaused = false;
        extensionPaused = false;
        sendResponse({ ok: true });
        break;
    }
  });

  // Register teardown so the NEXT injection can kill this instance cleanly.
  // This handles: extension reloads (old VMs survive), re-injection on SPA nav,
  // and multiple Start clicks without Stop in between.
  window.__liveTranslatorV2Teardown = function () {
    stopCanvasMode();
    removeOverlay();
    if (observer) { observer.disconnect(); observer = null; }
    if (video) {
      video.playbackRate = 1.0;
      video.classList.remove("__lt-hidden");
      video.style.opacity = "";
    }
    video = null;
    syncMode = null;
    canvasActive = false;
    drawingActive = false;
    extensionTriggeredPause = false;
    extensionTriggeredPlay = false;
  };

  init();
})();
