"""
Async wrapper around ElevenLabs WebSocket streaming TTS + voice cloning.

Uses raw websockets for the streaming TTS API (SDK doesn't expose WS input
streaming). Uses the elevenlabs SDK for voice cloning via REST.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import time
from typing import AsyncIterator, Optional

import websockets

logger = logging.getLogger(__name__)

# Stock voices from the existing SmartVoiceManager
STOCK_VOICES = {
    # Female
    "21m00Tcm4TlvDq8ikWAM": {"gender": "female", "pitch": "medium", "style": "warm_calm"},
    "AZnzlk1XvdvUeBnXmlld": {"gender": "female", "pitch": "medium_low", "style": "strong_confident"},
    "EXAVITQu4vr4xnSDxMaL": {"gender": "female", "pitch": "medium_high", "style": "soft_friendly"},
    "MF3mGyEYCl7XYWbV9V6O": {"gender": "female", "pitch": "medium", "style": "emotional_expressive"},
    # Male
    "ErXwobaYiN019PkySvjV": {"gender": "male", "pitch": "medium", "style": "well_rounded"},
    "TxGEqnHWrfWFTfGW9XjX": {"gender": "male", "pitch": "medium_low", "style": "deep_authoritative"},
    "VR6AewLTigWG4xSOukaG": {"gender": "male", "pitch": "low", "style": "crisp_strong"},
    "pNInz6obpgDQGcFmaJgB": {"gender": "male", "pitch": "medium_low", "style": "deep_calm"},
    "yoZ06aMxZJJ28mfd3POQ": {"gender": "male", "pitch": "medium", "style": "energetic_young"},
    "IKne3meq5aSn9XLyUdCD": {"gender": "male", "pitch": "medium_high", "style": "casual_conversational"},
}


class TTSConnection:
    """A single ElevenLabs WebSocket TTS connection for one voice."""

    WS_BASE = "wss://api.elevenlabs.io/v1/text-to-speech"

    def __init__(
        self,
        api_key: str,
        voice_id: str,
        model_id: str = "eleven_flash_v2_5",
        output_format: str = "mp3_44100_64",
    ):
        self._api_key = api_key
        self._voice_id = voice_id
        self._model_id = model_id
        self._output_format = output_format
        self._ws = None
        self._last_used = time.monotonic()

    @property
    def idle_seconds(self) -> float:
        return time.monotonic() - self._last_used

    async def connect(self):
        """Open WebSocket and send BOS message."""
        url = (
            f"{self.WS_BASE}/{self._voice_id}/stream-input"
            f"?model_id={self._model_id}"
            f"&output_format={self._output_format}"
            f"&inactivity_timeout=30"
        )
        self._ws = await websockets.connect(
            url,
            additional_headers={"xi-api-key": self._api_key},
        )
        # BOS: space + voice settings
        # stability=0.35 for more expressive output (default 0.5 is too monotone)
        # speaker_boost=False to save latency in real-time streaming
        bos = {
            "text": " ",
            "voice_settings": {
                "stability": 0.35,
                "similarity_boost": 0.75,
                "style": 0.0,
                "use_speaker_boost": False,
                "speed": 1.0,
            },
            "generation_config": {
                "chunk_length_schedule": [100, 140, 200, 260],
            },
        }
        await self._ws.send(json.dumps(bos))
        self._last_used = time.monotonic()

    async def synthesize(self, text: str) -> AsyncIterator[bytes]:
        """Send text and yield MP3 audio chunks as they arrive.

        Opens a fresh connection for each utterance (ElevenLabs WS is
        designed for one text -> close pattern).
        """
        if self._ws is None:
            await self.connect()

        self._last_used = time.monotonic()

        # Send text with flush
        msg = {
            "text": text + " ",
            "try_trigger_generation": True,
            "flush": True,
        }
        await self._ws.send(json.dumps(msg))

        # Send EOS
        await self._ws.send(json.dumps({"text": ""}))

        # Receive audio chunks until isFinal
        while True:
            try:
                raw = await asyncio.wait_for(self._ws.recv(), timeout=30.0)
                data = json.loads(raw)
                if data.get("isFinal"):
                    break
                audio_b64 = data.get("audio")
                if audio_b64:
                    yield base64.b64decode(audio_b64)
            except (websockets.exceptions.ConnectionClosed, asyncio.TimeoutError):
                break

        # Connection is consumed after EOS; close and reset for next use
        await self.close()

    async def close(self):
        if self._ws:
            try:
                await self._ws.close()
            except Exception:
                pass
            self._ws = None


class TTSClient:
    """Manages TTS connections and voice cloning for a session."""

    def __init__(self, api_key: Optional[str] = None):
        self._api_key = api_key or os.environ["ELEVENLABS_API_KEY"]
        self._used_stock_voices: set[str] = set()

    # ------------------------------------------------------------------
    # TTS synthesis
    # ------------------------------------------------------------------

    async def synthesize(self, text: str, voice_id: str) -> AsyncIterator[bytes]:
        """Synthesize text with the given voice, yielding MP3 chunks."""
        conn = TTSConnection(self._api_key, voice_id)
        async for chunk in conn.synthesize(text):
            yield chunk

    # ------------------------------------------------------------------
    # Stock voice assignment
    # ------------------------------------------------------------------

    def assign_stock_voice(self, gender: str = "male", avg_pitch: float = 150.0) -> str:
        """Pick a stock voice based on gender/pitch. Avoids reuse when possible."""
        candidates = {
            vid: props for vid, props in STOCK_VOICES.items()
            if props["gender"] == gender and vid not in self._used_stock_voices
        }
        if not candidates:
            candidates = {
                vid: props for vid, props in STOCK_VOICES.items()
                if vid not in self._used_stock_voices
            }
        if not candidates:
            candidates = {
                vid: props for vid, props in STOCK_VOICES.items()
                if props["gender"] == gender
            }
        if not candidates:
            candidates = STOCK_VOICES

        # Score by pitch match
        best_vid = None
        best_score = -1
        for vid, props in candidates.items():
            score = 0
            if avg_pitch < 140 and props["pitch"] == "low":
                score += 3
            elif 140 <= avg_pitch < 180 and props["pitch"] in ("medium_low", "medium"):
                score += 3
            elif avg_pitch >= 180 and props["pitch"] in ("medium_high", "medium"):
                score += 3
            if props["style"] in ("deep_authoritative", "energetic_young", "strong_confident"):
                score += 1
            if score > best_score:
                best_score = score
                best_vid = vid

        if best_vid is None:
            best_vid = list(candidates.keys())[0]

        self._used_stock_voices.add(best_vid)
        return best_vid

    # ------------------------------------------------------------------
    # Voice cloning
    # ------------------------------------------------------------------

    async def clone_voice(
        self,
        name: str,
        audio_bytes: bytes,
        description: str = "",
    ) -> str:
        """Clone a voice from audio bytes via ElevenLabs IVC. Returns voice_id."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, self._clone_voice_sync, name, audio_bytes, description
        )

    def _clone_voice_sync(self, name: str, audio_bytes: bytes, description: str) -> str:
        from elevenlabs.client import ElevenLabs

        client = ElevenLabs(api_key=self._api_key)
        voice = client.voices.ivc.create(
            name=name,
            files=[("sample.wav", audio_bytes, "audio/wav")],
            description=description,
            remove_background_noise=False,
        )
        voice_id = voice.voice_id
        logger.info("Voice cloned: %s -> %s", name, voice_id)
        return voice_id

    async def delete_voice(self, voice_id: str):
        """Delete a cloned voice to clean up."""
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._delete_voice_sync, voice_id)

    def _delete_voice_sync(self, voice_id: str):
        try:
            from elevenlabs.client import ElevenLabs
            client = ElevenLabs(api_key=self._api_key)
            client.voices.delete(voice_id=voice_id)
            logger.info("Voice deleted: %s", voice_id)
        except Exception as e:
            logger.warning("Failed to delete voice %s: %s", voice_id, e)

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    async def close(self):
        """Clean up all resources."""
        pass  # Connections are per-utterance, no persistent state
