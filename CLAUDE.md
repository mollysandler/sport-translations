# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Project Does

Real-time sports commentary translation system. Audio goes through: diarization (pyannote) → ASR (Faster-Whisper) → translation (deep-translator/Google) → TTS (ElevenLabs stock voices, Qwen3-TTS, or XTTS voice cloning) → composed output audio. Supports both batch (full file) and live streaming (chunked SSE) modes.

The primary interface is a **Chrome extension** that captures tab audio for live translation. There is also a React web app for file upload and URL-based translation.

## Commands

### Python backend
```bash
# Install dependencies (use Python 3.10+ venv)
pip install -r requirements.txt          # CPU/Mac
pip install -r requirements_gpu.txt      # Modal GPU container overrides

# Run tests
pytest tests/ -q                         # all tests
pytest tests/test_utils.py -q            # single test file
pytest tests/test_utils.py::test_name -q # single test

# Lint
ruff check .                             # lint Python
ruff format .                            # format Python

# Deploy to Modal
modal deploy modal_app.py
```

### React frontend (`sports/`)
```bash
cd sports
npm install
npm run dev      # Vite dev server
npm run build    # production build
npm run lint     # ESLint
```

### Chrome extension (`chrome-extension/`)
```bash
cd chrome-extension
npm install
npm test         # Jest tests
```
Load as unpacked extension in `chrome://extensions` → Developer mode → Load unpacked → select `chrome-extension/`.

### Environment variables
Requires `.env` with `GOOGLE_APPLICATION_CREDENTIALS`, `HUGGING_FACE_TOKEN`, and `ELEVENLABS_API_KEY`. Modal secrets are in `sports-secrets`.

## Architecture

### Two execution modes
- **Batch mode**: Upload a file → `TranslatorService.translate_wav_bytes_local()` (on Modal) or `.remote()` (off Modal) → returns translated audio + captions.
- **Live streaming mode**: Audio chunks → `POST /translate-live` → SSE stream of `{type, audio_b64, caption, progress}` per segment. Sessions are UUID-keyed dicts on `TranslatorService`.

### Chrome Extension (`chrome-extension/`) — primary interface

Captures tab audio and streams it through the backend for live translation. Uses Chrome's Manifest V3 APIs.

**Message flow:**
```
Side Panel (UI) → Service Worker (orchestrator) → Offscreen Document (audio capture/playback)
                                                 → Content Script (video sync)
                                                 → Backend API (/session/start, /translate-live, /session/end)
```

**Files:**
- **manifest.json** — MV3 config. Permissions: `tabCapture`, `offscreen`, `sidePanel`, `storage`, `activeTab`, `scripting`. Host permission scoped to the Modal API URL.
- **service-worker.js** — Central orchestrator. Handles `START_CAPTURE`/`STOP_CAPTURE` from the side panel, creates backend sessions, gets tab stream IDs via `chrome.tabCapture.getMediaStreamId()`, creates the offscreen document, injects the content script, relays messages between all contexts, and manages drift correction (0.95x/1.05x rate adjustment or micro-pause, with 30s cooldown).
- **content-script.js** — Injected into the active tab. Finds the video element (prefers playing, falls back to largest by area), uses a `MutationObserver` for SPAs/lazy-loaded players. Handles `VIDEO_PAUSE`, `VIDEO_RESUME`, `VIDEO_ADJUST_RATE`, `VIDEO_MICRO_PAUSE`, `VIDEO_CLEANUP`. Tracks `extensionPaused` vs `userPaused` so user-initiated pauses are never overridden.
- **offscreen/offscreen.js** — Runs headless in an offscreen document. Captures tab audio via `getUserMedia` + `AudioWorkletNode`, resamples to 16kHz, encodes as WAV, POSTs chunks to `/translate-live` (SSE), queues and plays back translated audio segments. Mutes original audio by not connecting the worklet to `audioContext.destination`. Monitors drift between original and translated audio duration, sends `DRIFT_CORRECTION` messages when drift exceeds 500ms. Per-chunk timeout: 60s. SSE stall timeout: 45s.
- **offscreen/audio-worklet.js** — `ChunkAccumulator` AudioWorkletProcessor. Accumulates samples into 8-second chunks, detects silence (RMS < 0.001) in the first 3 seconds.
- **sidepanel/sidepanel.js** — UI controller. Language selection (persisted to `chrome.storage.local`), start/stop capture, caption display (last 200, color-coded by speaker), connection timer with 90s timeout, error/silence warnings.
- **sidepanel/sidepanel.html** / **sidepanel/sidepanel.css** — Side panel markup and styles.
- **offscreen/offscreen.html** — Minimal HTML shell for the offscreen document (no UI).

**Video synchronization strategy:**
1. Measure initial latency (first chunk sent → first segment received)
2. Micro-pause video for that duration to let translated audio queue up
3. Monitor drift every 5 segments: `(originalAudioSent - translatedAudioPlayed)`
4. Correct with rate adjustment (small drift) or micro-pause (large drift), 30s cooldown

**Tests** (`chrome-extension/tests/`):
- `content-script.test.js` — Video discovery, user pause tracking, rate adjustment, cleanup
- `service-worker.test.js` — Latency tracking, drift correction logic, message relay, cooldown
- `offscreen.test.js` — Audio muting verification, SSE parsing, abort/timeout behavior, drift calc
- `helpers.js` — `createMockVideo()`, `createChromeMock()`, `evalScript()` (VM-based script evaluation)

### Backend files (root)

- **main.py** — `DynamicSpeakerTranslator`: core pipeline class. Contains `SmartVoiceManager` for pitch/gender-based ElevenLabs voice matching, `VoiceProfile` and `TranslationSegment` dataclasses. Public methods: `translate_audio_file_no_playback()`, `translate_audio_file_stream()`, `translate_chunk_stream()`, `translate_video_streaming()`. TTS engines: ElevenLabs (primary), Qwen3-TTS and XTTS v2 (optional voice cloning, lazy-loaded).
- **api_server.py** — FastAPI app factory (`make_app(service)`). Endpoints: `POST /translate-audio` (single file), `POST /translate-stream` (URL via ffmpeg/silence chunking), `POST /translate-live` (SSE streaming), `POST /session/start`, `POST /session/end`, `POST /feedback`. Helper: `wav_bytes_16k_mono()` for format normalization.
- **modal_app.py** — Modal deployment. `TranslatorService` class with `@modal.enter()` model loading. Config: A10G GPU, 4 CPU, 16GB RAM, 60min timeout, max 1 container. Volumes: `/cache/hf`, `/cache/torch`, `/feedback`. Caches translator instances by `(source_lang, target_lang)` tuple. Sessions stored in `_sessions` dict.
- **diarizer.py** — `SpeakerDiarizer` / `SportsDiarizer` using pyannote 3.1. Pipeline: diarize → filter short segments (<300ms) → merge close segments (gap ≤0.3s) → split long segments (>25s) → consolidate speakers via SpeechBrain embeddings → re-attribute to closest centroid. Configurable via `SpeakerMergeConfig`.
- **utils.py** — `TTSConfig`, `SpeakerMergeConfig` dataclasses. Helpers: `estimate_pitch_yin()` (pure numpy, no numba), `find_silence_split()` (RMS-based silence detection), `gender_from_pitch()`.

### Frontend (`sports/`)
React 19 + Vite 7 app. Components: `AudioInput` (file upload, URL input, live mic with silence-based chunking), `CommentaryPlayer` (playback + scrollable captions + download SRT/audio), `LanguageSelector` (auto-detect + 10 languages, swap button), `FeedbackPanel` (star rating + category + comments), `Toast` (auto-dismiss notifications). State persisted to localStorage. Keyboard shortcuts: Space (play/pause), D (download audio), M (mute), S (download SRT). Communicates with backend via fetch + ReadableStream for SSE.

### Key patterns
- `_running_on_modal()` checks `MODAL_TASK_ID` / `MODAL_IS_REMOTE` env vars to determine execution context.
- Pitch estimation uses pure-numpy autocorrelation (`estimate_pitch_yin`), not librosa.yin or numba, to avoid SIGSEGV in GPU containers.
- `speaker_voice_ids` is set per-chunk; `session_state` dict persists speaker mappings across chunks for voice continuity (30 Hz pitch threshold).
- Chrome extension mutes original audio by NOT connecting the worklet node to the audio destination — only translated audio plays.
- `old-interations/` contains previous iteration code — not actively used.
