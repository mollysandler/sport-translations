// Content script injected into the active tab to control video playback
// for sync with translated audio.
(function () {
  let video = null;
  let observer = null;
  let extensionPaused = false; // true when the extension initiated a pause
  let userPaused = false;      // true when the user paused manually

  function bindPauseDetection(v) {
    v.addEventListener("pause", () => {
      if (!extensionPaused) userPaused = true;
    });
    v.addEventListener("play", () => {
      userPaused = false;
    });
  }

  function findVideo() {
    const videos = Array.from(document.querySelectorAll("video"));
    if (videos.length === 0) return null;
    // Prefer a currently-playing video
    const playing = videos.find((v) => !v.paused && !v.ended);
    if (playing) return playing;
    // Fallback: largest by area
    return videos.sort(
      (a, b) => b.videoWidth * b.videoHeight - a.videoWidth * a.videoHeight
    )[0];
  }

  function init() {
    video = findVideo();
    if (video) {
      bindPauseDetection(video);
      chrome.runtime.sendMessage({ type: "VIDEO_FOUND", currentTime: video.currentTime });
    } else {
      // Watch for video to appear (SPAs, lazy-loaded players)
      observer = new MutationObserver(() => {
        video = findVideo();
        if (video) {
          bindPauseDetection(video);
          observer.disconnect();
          observer = null;
          chrome.runtime.sendMessage({ type: "VIDEO_FOUND", currentTime: video.currentTime });
        }
      });
      observer.observe(document.body, { childList: true, subtree: true });
      chrome.runtime.sendMessage({ type: "VIDEO_NOT_FOUND" });
    }
  }

  chrome.runtime.onMessage.addListener((msg, _sender, sendResponse) => {
    // Re-discover video if we lost it
    if (!video && msg.type !== "VIDEO_CLEANUP") {
      video = findVideo();
    }

    switch (msg.type) {
      case "VIDEO_PAUSE":
        if (video) {
          extensionPaused = true;
          video.pause();
          extensionPaused = false;
        }
        sendResponse({ ok: !!video });
        break;

      case "VIDEO_RESUME":
        if (video && !userPaused) video.play();
        sendResponse({ ok: !!video && !userPaused });
        break;

      case "VIDEO_ADJUST_RATE":
        if (video) {
          video.playbackRate = msg.rate;
          // Guard against players that reset playbackRate on their own tick
          // Re-apply the rate every 200ms for 2 seconds
          let rateGuardCount = 0;
          const rateGuard = setInterval(() => {
            rateGuardCount++;
            if (video && video.playbackRate !== msg.rate) {
              video.playbackRate = msg.rate;
            }
            if (rateGuardCount >= 10) clearInterval(rateGuard);
          }, 200);
        }
        sendResponse({ ok: !!video });
        break;

      case "VIDEO_RESET_RATE":
        if (video) video.playbackRate = 1.0;
        sendResponse({ ok: !!video });
        break;

      case "VIDEO_MICRO_PAUSE":
        if (video) {
          extensionPaused = true;
          video.pause();
          extensionPaused = false;
          setTimeout(() => {
            if (video && !userPaused) video.play();
          }, msg.durationMs);
        }
        sendResponse({ ok: !!video });
        break;

      case "VIDEO_SEEK":
        if (video) video.currentTime = msg.time;
        sendResponse({ ok: !!video });
        break;

      case "VIDEO_REPORT_TIME":
        sendResponse({
          ok: !!video,
          currentTime: video ? video.currentTime : null,
          playbackRate: video ? video.playbackRate : null,
        });
        break;

      case "VIDEO_CLEANUP":
        if (observer) {
          observer.disconnect();
          observer = null;
        }
        if (video) video.playbackRate = 1.0;
        video = null;
        userPaused = false;
        extensionPaused = false;
        sendResponse({ ok: true });
        break;
    }
  });

  init();
})();
