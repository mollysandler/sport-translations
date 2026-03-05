import os, tempfile, traceback, base64, json, subprocess
import io
from typing import Any
import torch, torchaudio
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse
from pydub import AudioSegment

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


def stream_chunks_from_url(url: str, chunk_seconds: int = 8):
    """Yield WAV bytes chunks from a URL using ffmpeg piped output.

    ffmpeg streams audio from the URL and pipes raw PCM to stdout.
    We read chunk_seconds worth of samples at a time, wrap in a WAV
    header, and yield. Generator stops when ffmpeg ends (stream over)
    or on error.
    """
    import struct

    sample_rate = 16000
    samples_per_chunk = sample_rate * chunk_seconds
    bytes_per_chunk = samples_per_chunk * 2  # 16-bit mono

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
    try:
        while True:
            raw = proc.stdout.read(bytes_per_chunk)
            if not raw or len(raw) < sample_rate * 2:
                # Less than 1 second of audio — stream ended or too short
                break

            # Wrap raw PCM in a WAV header
            data_len = len(raw)
            header = struct.pack(
                "<4sI4s4sIHHIIHH4sI",
                b"RIFF", 36 + data_len, b"WAVE",
                b"fmt ", 16, 1, 1,  # PCM, mono
                sample_rate, sample_rate * 2, 2, 16,
                b"data", data_len,
            )
            yield header + raw
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
            mp3_bytes, captions = service.translate_wav_bytes_local(wav_bytes, source_lang, target_lang)
        else:
            mp3_bytes, captions = service.translate_wav_bytes.remote(wav_bytes, source_lang, target_lang)

        return {
            "audio_base64": base64.b64encode(mp3_bytes).decode("utf-8"),
            "captions": captions,
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
                for wav_chunk in stream_chunks_from_url(url):
                    try:
                        for item in service.process_live_chunk(wav_chunk, session_id, source_lang, target_lang):
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
