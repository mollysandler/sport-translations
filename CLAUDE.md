# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Project Does

Real-time sports commentary translation system. Audio goes through: diarization (pyannote) → ASR (Faster-Whisper) → translation (deep-translator/Google) → TTS (ElevenLabs stock voices, Qwen3-TTS, or XTTS voice cloning) → composed output audio. Supports both batch (full file) and live streaming (chunked SSE) modes.

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

### Environment variables
Requires `.env` with `GOOGLE_APPLICATION_CREDENTIALS`, `HUGGING_FACE_TOKEN`, and `ELEVENLABS_API_KEY`. Modal secrets are in `sports-secrets`.

## Architecture

### Two execution modes
- **Batch mode**: Upload a file → `TranslatorService.translate_wav_bytes_local()` (on Modal) or `.remote()` (off Modal) → returns translated audio + captions.
- **Live streaming mode**: Frontend sends audio chunks → `POST /translate-live` → SSE stream of `{type, audio_b64, caption, progress}` per segment. Sessions are UUID-keyed dicts on `TranslatorService`.

### Backend files (root)
- **main.py** — `DynamicSpeakerTranslator`: core pipeline class. Handles voice matching via pitch analysis, TTS engine selection, and audio composition. `translate_chunk_stream()` is the live-mode generator.
- **api_server.py** — FastAPI app factory (`make_app(service)`). Endpoints: `/translate` (batch), `/translate-live` (SSE streaming), `/session/start|end`, `/feedback`. Also has `stream_chunks_from_url()` for URL-based input.
- **modal_app.py** — Modal deployment. `TranslatorService` class with `@modal.enter()` model loading, GPU container config (A10G), and volume mounts for HF/torch caches. Serves the FastAPI app via `@modal.asgi_app()`.
- **diarizer.py** — `SpeakerDiarizer` / `SportsDiarizer` using pyannote. `SpeakerSegment` dataclass. Embedding-based speaker merging across chunks.
- **utils.py** — `TTSConfig`, `SpeakerMergeConfig` dataclasses, `gender_from_pitch()` helper.

### Frontend (`sports/`)
React + Vite app. Components: `AudioInput` (file/URL/live mic input), `CommentaryPlayer` (audio playback + captions), `LanguageSelector`, `FeedbackPanel`, `Toast`. Communicates with backend via fetch + ReadableStream for SSE (not EventSource, which is GET-only).

### Chrome Extension (`chrome-extension/`)
Captures tab audio for live translation via the same backend API. Uses offscreen documents for audio capture.

### Key patterns
- `_running_on_modal()` checks `MODAL_TASK_ID` / `MODAL_IS_REMOTE` env vars to determine execution context.
- Pitch estimation uses pure-numpy autocorrelation (`_estimate_pitch_safe`), not librosa.yin or numba, to avoid SIGSEGV in GPU containers.
- `speaker_voice_ids` is set per-chunk; `session_state` dict persists speaker mappings across chunks for voice continuity (30 Hz pitch threshold).
- `old-interations/` contains previous iteration code — not actively used.
