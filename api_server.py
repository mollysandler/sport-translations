# api_server.py
import base64
import io
import os
import tempfile
from typing import Optional

import torchaudio
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydub import AudioSegment

from main import DynamicSpeakerTranslator
from utils import TTSConfig, SpeakerMergeConfig

app = FastAPI()

# Allow your Vite dev server to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5173", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def _save_upload_to_temp(upload: UploadFile) -> str:
    suffix = os.path.splitext(upload.filename or "")[1] or ".wav"
    fd, path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    with open(path, "wb") as f:
        f.write(upload.file.read())
    return path

def _ensure_wav_16k_mono(in_path: str) -> str:
    """
    Convert any audio input to 16kHz mono WAV so torchaudio + diarizer behave predictably.
    """
    audio = AudioSegment.from_file(in_path)
    audio = audio.set_frame_rate(16000).set_channels(1)
    fd, out_path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    audio.export(out_path, format="wav")
    return out_path

@app.post("/translate-audio")
def translate_audio(
    audio: UploadFile = File(...),
    source_lang: str = Form("en"),
    target_lang: str = Form("hi"),

    # Advanced args (mirror your CLI defaults)
    buffer_duration_sec: int = Form(30),
    max_workers: int = Form(3),

    use_voice_cloning: bool = Form(True),

    tts_backend: str = Form("qwen"),
    qwen_tts_enable: int = Form(1),
    qwen_tts_model_id: str = Form("Qwen/Qwen3-TTS-12Hz-0.6B-Base"),
    qwen_tts_device: str = Form("mps"),
    xtts_enable: int = Form(1),
    xtts_model_id: str = Form("tts_models/multilingual/multi-dataset/xtts_v2"),
    xtts_device: str = Form("cpu"),

    speaker_merge_enable: int = Form(1),
    speaker_merge_sim: float = Form(0.74),
    speaker_tiny_total_ms: int = Form(6000),
    speaker_emb_min_chunk_ms: int = Form(250),
    speaker_merge_ref_sec: float = Form(20.0),
):
    # 1) Save upload, normalize format
    raw_path = _save_upload_to_temp(audio)
    wav_path = _ensure_wav_16k_mono(raw_path)

    # 2) Build configs from "CLI-like" params
    tts_cfg = TTSConfig(
        tts_backend=tts_backend,
        qwen_enable=bool(qwen_tts_enable),
        qwen_model_id=qwen_tts_model_id,
        qwen_device=qwen_tts_device,
        xtts_enable=bool(xtts_enable),
        xtts_model_id=xtts_model_id,
        xtts_device=xtts_device,
    )

    speaker_cfg = SpeakerMergeConfig(
        merge_enable=bool(speaker_merge_enable),
        merge_sim=speaker_merge_sim,
        tiny_total_ms=speaker_tiny_total_ms,
        emb_min_chunk_ms=speaker_emb_min_chunk_ms,
        merge_ref_sec=float(speaker_merge_ref_sec),
    )

    # 3) Run translation WITHOUT playback (important for servers)
    translator = DynamicSpeakerTranslator(
        source_lang=source_lang,
        target_lang=target_lang,
        buffer_duration_sec=buffer_duration_sec,
        max_workers=max_workers,
        use_voice_cloning=use_voice_cloning,
        tts_config=tts_cfg,
        speaker_merge=speaker_cfg,
    )

    # --- key change: call a "no-playback" method ---
    mp3_bytes, captions = translator.translate_audio_file_no_playback(wav_path)

    return {
        "audio_base64": base64.b64encode(mp3_bytes).decode("utf-8"),
        "captions": captions,
    }
