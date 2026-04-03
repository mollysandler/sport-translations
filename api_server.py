import os
import traceback
import base64
import json
import subprocess
import io
from typing import Any
import numpy as np
import torch
import torchaudio
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from utils import find_silence_split

def wav_bytes_16k_mono(upload: UploadFile) -> bytes:
    data = upload.file.read()
    wav, sr = torchaudio.load(io.BytesIO(data))  # wav always ok; mp3 depends on ffmpeg backend
    if wav.shape[0] > 1:
        wav = torch.mean(wav, dim=0, keepdim=True)
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)

    out = io.BytesIO()
    torchaudio.save(out, wav, 16000, format="wav")
    return out.getvalue()


def stream_chunks_from_url(
    url: str,
    min_chunk_seconds: float = 5.0,
    max_chunk_seconds: float = 20.0,
    overlap_seconds: float = 3.0,
):
    """Yield (WAV bytes, overlap_duration_sec) from a URL using ffmpeg.

    Reads audio incrementally and splits at silence boundaries between
    ``min_chunk_seconds`` and ``max_chunk_seconds``.  Falls back to a hard
    split at ``max_chunk_seconds`` if no silence is found.  The last
    ``overlap_seconds`` of PCM from the previous chunk is prepended to the
    next one so that Whisper gets full sentence context at chunk boundaries.
    The first chunk has ``overlap_duration_sec=0.0``.
    """
    import struct

    sample_rate = 16000
    read_seconds = 1  # read 1 second at a time
    read_bytes = sample_rate * 2 * read_seconds
    min_chunk_samples = int(sample_rate * min_chunk_seconds)
    max_chunk_samples = int(sample_rate * max_chunk_seconds)
    overlap_samples = int(sample_rate * overlap_seconds)

    cmd = [
        "ffmpeg",
        "-i", url,
        "-ac", "1",
        "-ar", str(sample_rate),
        "-f", "s16le",  # raw 16-bit little-endian PCM to stdout
        "-v", "error",
        "pipe:1",
    ]

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    pcm_buffer = np.empty(0, dtype=np.int16)
    trailing_samples = np.empty(0, dtype=np.int16)

    def _pack_wav(int16_data: np.ndarray) -> bytes:
        raw = int16_data.tobytes()
        data_len = len(raw)
        header = struct.pack(
            "<4sI4s4sIHHIIHH4sI",
            b"RIFF", 36 + data_len, b"WAVE",
            b"fmt ", 16, 1, 1,  # PCM, mono
            sample_rate, sample_rate * 2, 2, 16,
            b"data", data_len,
        )
        return header + raw

    def _emit_chunk(chunk_int16: np.ndarray):
        """Prepend overlap from previous chunk and return (wav_bytes, overlap_sec)."""
        nonlocal trailing_samples
        if trailing_samples.size > 0:
            combined = np.concatenate([trailing_samples, chunk_int16])
            overlap_sec = overlap_seconds
        else:
            combined = chunk_int16
            overlap_sec = 0.0
        # Save trailing portion for next iteration
        trailing_samples = chunk_int16[-overlap_samples:] if chunk_int16.size >= overlap_samples else chunk_int16.copy()
        return (_pack_wav(combined), overlap_sec)

    try:
        while True:
            raw = proc.stdout.read(read_bytes)
            if not raw:
                break

            new_samples = np.frombuffer(raw, dtype=np.int16)
            pcm_buffer = np.concatenate([pcm_buffer, new_samples])

            if pcm_buffer.size < min_chunk_samples:
                continue

            # Convert to float32 for silence detection
            float_buf = pcm_buffer.astype(np.float32) / 32768.0
            split_idx = find_silence_split(
                float_buf, sample_rate, min_chunk_samples, max_chunk_samples,
            )

            if split_idx > 0:
                chunk = pcm_buffer[:split_idx]
                pcm_buffer = pcm_buffer[split_idx:]
                yield _emit_chunk(chunk)
            elif pcm_buffer.size >= max_chunk_samples:
                # No silence found — hard split at max
                chunk = pcm_buffer[:max_chunk_samples]
                pcm_buffer = pcm_buffer[max_chunk_samples:]
                yield _emit_chunk(chunk)

        # Flush remaining buffer
        if pcm_buffer.size >= sample_rate:  # at least 1 second
            yield _emit_chunk(pcm_buffer)
    finally:
        proc.stdout.close()
        proc.terminate()
        proc.wait()


def _running_on_modal() -> bool:
    return bool(os.environ.get("MODAL_TASK_ID") or os.environ.get("MODAL_IS_REMOTE"))


def make_app(service: Any) -> FastAPI:
    app = FastAPI(redirect_slashes=False)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", "https://sport-translations.vercel.app"],
        allow_origin_regex=r"^chrome-extension://.*$",
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
        max_age=86400,
    )

    @app.post("/translate-audio")
    def translate_audio(
        audio: UploadFile = File(...),
        source_lang: str = Form("en"),
        target_lang: str = Form("hi"),
    ):
        wav_bytes = wav_bytes_16k_mono(audio)

        if _running_on_modal():
            result = service.translate_wav_bytes_local(wav_bytes, source_lang, target_lang)
        else:
            result = service.translate_wav_bytes.remote(wav_bytes, source_lang, target_lang)

        mp3_bytes, captions = result[0], result[1]
        detected_language = result[2] if len(result) > 2 else None

        return {
            "audio_base64": base64.b64encode(mp3_bytes).decode("utf-8"),
            "captions": captions,
            "detected_language": detected_language,
        }

    @app.post("/session/start")
    def session_start():
        return {"session_id": service.create_session()}

    @app.post("/session/end")
    def session_end(session_id: str = Form(...)):
        service.delete_session(session_id)
        return {"ok": True}

    @app.post("/translate-stream")
    def translate_stream(
        url: str = Form(...),
        session_id: str = Form(""),
        source_lang: str = Form("en"),
        target_lang: str = Form("hi"),
    ):
        def event_stream():
            try:
                for wav_chunk, overlap_sec in stream_chunks_from_url(url):
                    try:
                        for item in service.process_live_chunk(wav_chunk, session_id, source_lang, target_lang, overlap_duration_sec=overlap_sec):
                            yield f"data: {json.dumps(item)}\n\n"
                    except Exception as e:
                        yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
            except Exception as e:
                yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
            yield f"data: {json.dumps({'type': 'done'})}\n\n"

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.post("/feedback")
    def submit_feedback(
        rating: int = Form(0),
        feedbackType: str = Form("general"),
        comments: str = Form(""),
        sourceLanguage: str = Form(""),
        targetLanguage: str = Form(""),
        audioName: str = Form(""),
    ):
        try:
            service.store_feedback({
                "rating": rating,
                "feedbackType": feedbackType,
                "comments": comments,
                "sourceLanguage": sourceLanguage,
                "targetLanguage": targetLanguage,
                "audioName": audioName,
            })
            return {"ok": True}
        except Exception as e:
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/translate-live")
    def translate_live(
        audio: UploadFile = File(...),
        session_id: str = Form(""),
        source_lang: str = Form("en"),
        target_lang: str = Form("hi"),
    ):
        wav_bytes = wav_bytes_16k_mono(audio)

        def event_stream():
            try:
                for item in service.process_live_chunk(wav_bytes, session_id, source_lang, target_lang):
                    yield f"data: {json.dumps(item)}\n\n"
            except Exception as e:
                yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
            yield f"data: {json.dumps({'type': 'done'})}\n\n"

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    return app
