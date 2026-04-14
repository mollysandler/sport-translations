# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Product Requirements (MUST satisfy — every change should move toward these)

1. **Sync**: Audio and video are less than 2 seconds out of sync at all times.
2. **Buffer**: Initial buffer of max 60 seconds (ideally ~30s) before translated audio plays.
3. **Video manipulation**: Video can be seeked/rate-adjusted to align audio and video.
4. **No pauses**: After the initial buffer, there should be no further pauses in playback.
5. **Human voice quality**: Translated audio must sound human with natural flow and cadence.
6. **Speaker diarization**: Speakers must be identified and kept consistent throughout a session.
7. **Speed (Modal)**: Must run on Modal for speed. GPU for batch mode; CPU-only streaming via external APIs (Deepgram, ElevenLabs).

## What This Project Does

Real-time sports commentary translation system. Two architectures:

**v2 (primary, `chrome-extension-v2/`)**: Streaming pipeline — tab audio → WebSocket → Deepgram Nova-3 (streaming ASR + diarization) → Google Translate → ElevenLabs Flash v2.5 (streaming TTS with voice cloning) → WebSocket → playback. ~1-2s per-utterance latency.

**v1 (legacy, `chrome-extension/`)**: Batch-per-chunk pipeline — 8s audio chunks → HTTP POST → Faster-Whisper → Google Translate → ElevenLabs REST TTS → SSE. 13-18s per-chunk latency.

There is also a React web app (`sports/`) for file upload and URL-based translation.

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

### Chrome extension v2 (`chrome-extension-v2/`)
```bash
cd chrome-extension-v2
npm install
npm test         # Jest tests
```
Load as unpacked extension in `chrome://extensions` → Developer mode → Load unpacked → select `chrome-extension-v2/`.

### Streaming backend deploy
```bash
modal deploy modal_app_v2.py    # deploys both StreamingService (CPU) and BatchService (GPU)
```

### Environment variables
Requires `.env` with `DEEPGRAM_API_KEY`, `ELEVENLABS_API_KEY`, `GOOGLE_APPLICATION_CREDENTIALS`, and `HUGGING_FACE_TOKEN`. Modal secrets are in `sports-secrets`.

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

### Streaming backend v2 (root — new files)

- **protocol.py** — WebSocket message types and constants shared between server and extension.
- **deepgram_client.py** — Async Deepgram Nova-3 WebSocket wrapper. Streaming ASR + diarization. Emits `Utterance` objects (text, speaker_id, timestamps).
- **tts_client.py** — ElevenLabs Flash v2.5 WebSocket TTS (via raw `websockets` lib, not SDK). Stock voice assignment. Voice cloning via SDK `client.voices.ivc.create()`.
- **speaker_manager.py** — Maps Deepgram speaker_id → ElevenLabs voice_id. Accumulates audio per speaker, triggers async voice cloning after ~15s. Falls back to stock voices until clone is ready.
- **translator.py** — Async Google Translate wrapper (runs `deep_translator.GoogleTranslator` in executor).
- **session.py** — Pipeline orchestrator per WebSocket connection: Deepgram → translate → TTS → forward audio. Up to 3 concurrent utterances via `asyncio.Semaphore`.
- **streaming_server.py** — FastAPI WebSocket endpoint at `/ws/translate?source=en&target=es`. Binary frames = PCM16 audio in, MP3 audio out. JSON frames = control messages.
- **modal_app_v2.py** — Modal deployment. `StreamingService` (CPU-only, `min_containers=1`) + `BatchService` (GPU A10G, existing pipeline).

### Chrome Extension v2 (`chrome-extension-v2/`) — primary interface

WebSocket-based streaming. Continuous 200ms audio frames (not 8s chunks).

**Message flow:**
```
Side Panel (UI) → Service Worker (orchestrator) → Offscreen Document (WebSocket + audio capture/playback)
                                                 → Content Script (video sync + overlay)
                                                 → Backend WebSocket (ws://.../ws/translate)
```

**Files:**
- **service-worker.js** — Opens side panel via `action.onClicked` (grants `activeTab`). Relays messages between offscreen/content-script/sidepanel.
- **content-script.js** — Video control, rate guard for YouTube, seek-back with aggressive play retry, loading overlay during buffer phase.
- **offscreen/offscreen.js** — WebSocket to backend, AudioWorklet capture (200ms frames), 16kHz PCM16 resampling, MP3 decode + gapless playback scheduling, buffer management (30s target), drift correction via video position polling.
- **offscreen/stream-processor.js** — AudioWorkletProcessor emitting continuous 200ms Float32 frames.
- **sidepanel/** — UI (adapted from v1).

**Video synchronization strategy:**
1. Buffer ~30s of translated audio while showing overlay on video
2. When buffer is ready: seek video back by `totalAudioCapturedSec`, remove overlay, start audio playback
3. Aggressive play retry (every 200ms for 6s) to overcome YouTube's player fighting the seek-back
4. After transition: poll actual video position every 2s, adjust rate (0.85x-1.10x) or micro-pause for drift >3s

**Known issue (2026-04-08):** Drift correction can fire during the seek-back transition (video is buffering/paused) and fight the play retry. Need a "transition state" that suppresses drift correction until video is confirmed playing.

### Key patterns
- `_running_on_modal()` checks `MODAL_TASK_ID` / `MODAL_IS_REMOTE` env vars to determine execution context.
- Pitch estimation uses pure-numpy autocorrelation (`estimate_pitch_yin`), not librosa.yin or numba, to avoid SIGSEGV in GPU containers.
- Chrome extension mutes original audio by NOT connecting the worklet node to the audio destination — only translated audio plays.
- `old-interations/` contains previous iteration code — not actively used.
