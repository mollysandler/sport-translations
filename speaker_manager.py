"""
Speaker tracking, voice assignment, and voice cloning lifecycle.

Maintains a mapping from Deepgram speaker_id -> ElevenLabs voice_id.
Accumulates audio per speaker and triggers voice cloning when enough
audio is collected.
"""

from __future__ import annotations

import asyncio
import io
import logging
import struct
from typing import Optional

import numpy as np

from protocol import VOICE_CLONE_TARGET_SEC, VOICE_CLONE_MIN_SEC
from tts_client import TTSClient
from utils import estimate_pitch_yin, gender_from_pitch

logger = logging.getLogger(__name__)


def _pcm16_to_wav(
    pcm_bytes: bytes, sample_rate: int = 16000, channels: int = 1
) -> bytes:
    """Wrap raw PCM16 bytes in a WAV header."""
    bits_per_sample = 16
    byte_rate = sample_rate * channels * bits_per_sample // 8
    block_align = channels * bits_per_sample // 8
    data_size = len(pcm_bytes)

    buf = io.BytesIO()
    # RIFF header
    buf.write(b"RIFF")
    buf.write(struct.pack("<I", 36 + data_size))
    buf.write(b"WAVE")
    # fmt chunk
    buf.write(b"fmt ")
    buf.write(struct.pack("<I", 16))  # chunk size
    buf.write(struct.pack("<H", 1))  # PCM format
    buf.write(struct.pack("<H", channels))
    buf.write(struct.pack("<I", sample_rate))
    buf.write(struct.pack("<I", byte_rate))
    buf.write(struct.pack("<H", block_align))
    buf.write(struct.pack("<H", bits_per_sample))
    # data chunk
    buf.write(b"data")
    buf.write(struct.pack("<I", data_size))
    buf.write(pcm_bytes)
    return buf.getvalue()


class SpeakerInfo:
    """State for a single tracked speaker."""

    def __init__(self, speaker_id: int):
        self.speaker_id = speaker_id
        self.voice_id: Optional[str] = None
        self.is_cloned = False
        self.clone_in_progress = False
        self.audio_samples: bytearray = bytearray()  # raw PCM16 @ 16kHz
        self.audio_sec: float = 0.0
        self.gender: str = "male"
        self.avg_pitch: float = 150.0


class SpeakerManager:
    """Manages speaker -> voice mapping with cloning lifecycle."""

    def __init__(self, tts_client: TTSClient, enable_cloning: bool = True):
        self._tts = tts_client
        self._enable_cloning = enable_cloning
        self._speakers: dict[int, SpeakerInfo] = {}
        self._cloned_voice_ids: list[str] = []  # for cleanup
        self._clone_tasks: dict[int, asyncio.Task] = {}

    def get_voice_id(self, speaker_id: int) -> str:
        """Get the current voice_id for a speaker. Assigns stock voice if new."""
        info = self._speakers.get(speaker_id)
        if info is None:
            info = SpeakerInfo(speaker_id)
            # Assign a stock voice immediately
            info.voice_id = self._tts.assign_stock_voice(
                gender=info.gender, avg_pitch=info.avg_pitch
            )
            self._speakers[speaker_id] = info
            logger.info(
                "Speaker %d: assigned stock voice %s", speaker_id, info.voice_id
            )
        return info.voice_id

    def add_audio(self, speaker_id: int, pcm16_bytes: bytes):
        """Add captured audio for a speaker. May trigger voice cloning."""
        info = self._speakers.get(speaker_id)
        if info is None:
            # Auto-register
            self.get_voice_id(speaker_id)
            info = self._speakers[speaker_id]

        if info.is_cloned or info.clone_in_progress:
            return  # Already cloned or cloning

        info.audio_samples.extend(pcm16_bytes)
        info.audio_sec = len(info.audio_samples) / (16000 * 2)  # 16kHz 16-bit

        # Analyze pitch/gender once we have enough audio (~3 seconds)
        if info.audio_sec >= 3.0 and info.gender == "male" and info.avg_pitch == 150.0:
            self._analyze_speaker(info)

        # Trigger cloning when we have enough audio
        if (
            self._enable_cloning
            and info.audio_sec >= VOICE_CLONE_MIN_SEC
            and not info.clone_in_progress
        ):
            info.clone_in_progress = True
            task = asyncio.create_task(self._clone_speaker(info))
            self._clone_tasks[speaker_id] = task

    def _analyze_speaker(self, info: SpeakerInfo):
        """Analyze pitch/gender from accumulated audio."""
        try:
            pcm = (
                np.frombuffer(info.audio_samples, dtype=np.int16).astype(np.float32)
                / 32768.0
            )
            pitch = estimate_pitch_yin(pcm, 16000)
            if pitch is not None:
                info.avg_pitch = pitch
                info.gender = gender_from_pitch(pitch)
                # Re-assign stock voice with better pitch info
                new_voice = self._tts.assign_stock_voice(
                    gender=info.gender, avg_pitch=info.avg_pitch
                )
                if new_voice != info.voice_id and not info.is_cloned:
                    info.voice_id = new_voice
                    logger.info(
                        "Speaker %d: re-assigned stock voice %s (gender=%s, pitch=%.0f)",
                        info.speaker_id,
                        info.voice_id,
                        info.gender,
                        info.avg_pitch,
                    )
        except Exception as e:
            logger.warning("Speaker %d pitch analysis failed: %s", info.speaker_id, e)

    async def _clone_speaker(self, info: SpeakerInfo):
        """Clone a speaker's voice from accumulated audio."""
        try:
            # Use up to VOICE_CLONE_TARGET_SEC of audio
            max_bytes = int(VOICE_CLONE_TARGET_SEC * 16000 * 2)
            pcm_bytes = bytes(info.audio_samples[:max_bytes])
            wav_bytes = _pcm16_to_wav(pcm_bytes)

            name = f"live-speaker-{info.speaker_id}"
            voice_id = await self._tts.clone_voice(
                name=name,
                audio_bytes=wav_bytes,
                description=f"Cloned from live speaker {info.speaker_id}",
            )

            info.voice_id = voice_id
            info.is_cloned = True
            self._cloned_voice_ids.append(voice_id)
            logger.info("Speaker %d: voice cloned -> %s", info.speaker_id, voice_id)

        except Exception as e:
            logger.error("Speaker %d: voice cloning failed: %s", info.speaker_id, e)
            # Check for permission/auth errors — disable cloning permanently
            status = getattr(e, "status_code", None)
            if status in (401, 403):
                logger.warning(
                    "Voice cloning disabled: API key lacks permission (HTTP %s)", status
                )
                self._enable_cloning = False
            info.clone_in_progress = False

    def is_speaker_cloned(self, speaker_id: int) -> bool:
        info = self._speakers.get(speaker_id)
        return info is not None and info.is_cloned

    async def cleanup(self):
        """Cancel pending clones and delete cloned voices."""
        for task in self._clone_tasks.values():
            task.cancel()
        self._clone_tasks.clear()

        for voice_id in self._cloned_voice_ids:
            await self._tts.delete_voice(voice_id)
        self._cloned_voice_ids.clear()
        self._speakers.clear()
