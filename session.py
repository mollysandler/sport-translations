"""
Session orchestrator: ties together Deepgram, translation, TTS, and
speaker management for one WebSocket connection.

One Session per connected client. Manages the full pipeline:
  Audio in -> Deepgram ASR -> Translate -> TTS -> Audio out

Handles late-arriving speakers with a re-buffer phase: when a new
speaker appears after playback has started, utterances from that speaker
are queued while their voice characteristics are analyzed.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from typing import Callable, Optional

from protocol import (
    Utterance,
    UtteranceStartMsg,
    UtteranceEndMsg,
    CaptionMsg,
    RebufferStartMsg,
    RebufferProgressMsg,
    RebufferEndMsg,
    ErrorMsg,
    encode_msg,
    MAX_CONCURRENT_UTTERANCES,
)
from deepgram_client import DeepgramStream
from translator import translate_text
from tts_client import TTSClient
from tts_provider import VoiceDirective
from voice_catalog import VoiceMatcher
from speaker_manager import SpeakerManager

logger = logging.getLogger(__name__)


class Session:
    """Pipeline orchestrator for one streaming translation session."""

    def __init__(
        self,
        source_lang: str,
        target_lang: str,
        send_text: Callable[[str], asyncio.coroutines],  # send JSON text frame
        send_bytes: Callable[[bytes], asyncio.coroutines],  # send binary frame
        tts_provider: Optional[str] = None,
    ):
        self.session_id = str(uuid.uuid4())
        self._source_lang = source_lang
        self._target_lang = target_lang
        self._send_text = send_text
        self._send_bytes = send_bytes

        self._tts = TTSClient(provider=tts_provider)
        self._matcher = VoiceMatcher(provider=self._tts.provider_name)
        self._speaker_mgr = SpeakerManager(self._matcher)
        self._deepgram: Optional[DeepgramStream] = None

        self._seq = 0  # utterance sequence counter
        self._semaphore = asyncio.Semaphore(MAX_CONCURRENT_UTTERANCES)
        self._utterance_tasks: list[asyncio.Task] = []
        self._running = False

        # Ordered output: utterances process concurrently but send in seq order.
        # This prevents interleaved binary chunks on the WebSocket.
        self._send_lock = asyncio.Lock()
        self._next_send_seq = 1  # next seq to send
        self._pending_results: dict[int, dict] = {}  # seq -> result dict

        # Dedup: reject duplicate utterances before creating tasks
        self._recent_utterance_keys: set[str] = set()
        self._recent_utterance_list: list[str] = []  # FIFO for eviction
        self._last_utterance_end_sec: float = 0.0

        # Audio buffer: attributed to the speaker of the next utterance
        self._audio_buffer: bytearray = bytearray()

        # Re-buffer state for late-arriving speakers
        self._rebuffering_speaker: Optional[int] = None
        self._rebuffer_queue: list[Utterance] = []
        self._rebuffer_task: Optional[asyncio.Task] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self):
        """Start the Deepgram stream and begin processing."""
        self._running = True
        self._deepgram = DeepgramStream(
            language=self._source_lang,
            on_utterance=self._on_utterance,
            on_utterance_end=self._on_utterance_end,
        )
        await self._deepgram.start()
        logger.info(
            "Session %s started (%s -> %s, provider=%s)",
            self.session_id,
            self._source_lang,
            self._target_lang,
            self._tts.provider_name,
        )

    async def stop(self):
        """Stop the session and clean up all resources."""
        self._running = False

        # Cancel rebuffer task
        if self._rebuffer_task:
            self._rebuffer_task.cancel()
            self._rebuffer_task = None

        # Cancel pending utterance tasks
        for task in self._utterance_tasks:
            task.cancel()
        if self._utterance_tasks:
            await asyncio.gather(*self._utterance_tasks, return_exceptions=True)
        self._utterance_tasks.clear()

        # Stop Deepgram
        if self._deepgram:
            await self._deepgram.stop()

        # Clean up speaker manager and TTS
        await self._speaker_mgr.cleanup()
        await self._tts.close()
        logger.info("Session %s stopped", self.session_id)

    # ------------------------------------------------------------------
    # Audio input
    # ------------------------------------------------------------------

    async def receive_audio(self, pcm16_bytes: bytes):
        """Receive raw PCM16 audio from the extension and forward to Deepgram.

        Also buffers audio for speaker voice analysis. During re-buffer,
        feeds audio directly to the new speaker for faster analysis.
        """
        if not self._running or self._deepgram is None:
            return

        # Forward to Deepgram
        await self._deepgram.send_audio(pcm16_bytes)

        # Buffer for speaker attribution (assigned when utterance arrives)
        self._audio_buffer.extend(pcm16_bytes)

        # During re-buffer, also feed audio directly to the new speaker
        # so they accumulate audio even before Deepgram emits an utterance.
        if self._rebuffering_speaker is not None:
            self._speaker_mgr.add_audio(self._rebuffering_speaker, pcm16_bytes)

    # ------------------------------------------------------------------
    # Deepgram callbacks
    # ------------------------------------------------------------------

    def _on_utterance(self, utterance: Utterance):
        """Called by DeepgramStream when a final utterance is ready."""
        if not self._running:
            return

        # Dedup: reject utterances that duplicate recent content or time ranges
        key = f"{utterance.text.lower().strip()}|{round(utterance.start_sec, 1)}"
        if key in self._recent_utterance_keys:
            logger.debug("Session dedup: skipping duplicate '%s'", utterance.text[:40])
            return
        if utterance.end_sec <= self._last_utterance_end_sec + 0.05:
            logger.debug(
                "Session dedup: skipping utterance ending at %.1f (last=%.1f)",
                utterance.end_sec,
                self._last_utterance_end_sec,
            )
            return
        self._recent_utterance_keys.add(key)
        self._recent_utterance_list.append(key)
        if len(self._recent_utterance_list) > 50:
            old_key = self._recent_utterance_list.pop(0)
            self._recent_utterance_keys.discard(old_key)
        self._last_utterance_end_sec = max(
            self._last_utterance_end_sec, utterance.end_sec
        )

        # Feed audio to speaker manager for voice analysis
        if self._audio_buffer:
            self._speaker_mgr.add_audio(utterance.speaker_id, bytes(self._audio_buffer))
            self._audio_buffer.clear()

        # Check for late-arriving speaker requiring re-buffer
        if (
            self._speaker_mgr.playback_started
            and self._speaker_mgr.is_new_speaker(utterance.speaker_id)
            and self._rebuffering_speaker is None
        ):
            # New speaker after playback — start re-buffer
            self._rebuffering_speaker = utterance.speaker_id
            self._speaker_mgr.get_voice_id(utterance.speaker_id)  # register provisional
            self._rebuffer_queue.append(utterance)
            self._rebuffer_task = asyncio.create_task(
                self._run_rebuffer(utterance.speaker_id)
            )
            logger.info(
                "Late speaker %d detected — starting re-buffer",
                utterance.speaker_id,
            )
            return

        # If currently re-buffering and this is the new speaker, queue it
        if self._rebuffering_speaker == utterance.speaker_id:
            self._rebuffer_queue.append(utterance)
            return

        # Normal processing
        task = asyncio.create_task(self._process_utterance(utterance))
        self._utterance_tasks.append(task)
        task.add_done_callback(
            lambda t: (
                self._utterance_tasks.remove(t) if t in self._utterance_tasks else None
            )
        )

    def _on_utterance_end(self):
        """Called when Deepgram detects an utterance boundary (silence gap)."""
        pass  # The utterance callback handles everything

    # ------------------------------------------------------------------
    # Re-buffer for late-arriving speakers
    # ------------------------------------------------------------------

    async def _run_rebuffer(self, speaker_id: int):
        """Wait for voice analysis of a late speaker, then drain queued utterances."""
        try:
            # Notify client to pause and show overlay
            await self._send_text(encode_msg(RebufferStartMsg(speaker_id=speaker_id)))

            # Poll analysis progress and send updates
            info = self._speaker_mgr._speakers.get(speaker_id)
            target_sec = 3.0

            while info and not info.analysis_complete.is_set():
                progress = min(100, int((info.audio_sec / target_sec) * 100))
                await self._send_text(
                    encode_msg(
                        RebufferProgressMsg(
                            speaker_id=speaker_id,
                            progress=progress,
                        )
                    )
                )
                try:
                    await asyncio.wait_for(
                        asyncio.shield(info.analysis_complete.wait()),
                        timeout=0.5,
                    )
                    break  # Analysis complete
                except asyncio.TimeoutError:
                    continue  # Keep polling

            # If analysis didn't complete, wait_for_analysis handles the timeout
            if info and not info.analysis_complete.is_set():
                await self._speaker_mgr.wait_for_analysis(speaker_id, timeout=5.0)

            # Notify client to resume
            await self._send_text(encode_msg(RebufferEndMsg(speaker_id=speaker_id)))

            logger.info(
                "Re-buffer complete for speaker %d — draining %d queued utterances",
                speaker_id,
                len(self._rebuffer_queue),
            )

            # Drain queued utterances into the normal pipeline
            self._rebuffering_speaker = None
            queued = list(self._rebuffer_queue)
            self._rebuffer_queue.clear()

            for utt in queued:
                if not self._running:
                    break
                task = asyncio.create_task(self._process_utterance(utt))
                self._utterance_tasks.append(task)
                task.add_done_callback(
                    lambda t: (
                        self._utterance_tasks.remove(t)
                        if t in self._utterance_tasks
                        else None
                    )
                )

        except asyncio.CancelledError:
            self._rebuffering_speaker = None
            self._rebuffer_queue.clear()
        except Exception as e:
            logger.error("Re-buffer failed for speaker %d: %s", speaker_id, e)
            self._rebuffering_speaker = None
            # Drain queue anyway with whatever voice we have
            queued = list(self._rebuffer_queue)
            self._rebuffer_queue.clear()
            for utt in queued:
                if not self._running:
                    break
                task = asyncio.create_task(self._process_utterance(utt))
                self._utterance_tasks.append(task)
                task.add_done_callback(
                    lambda t: (
                        self._utterance_tasks.remove(t)
                        if t in self._utterance_tasks
                        else None
                    )
                )

    # ------------------------------------------------------------------
    # Utterance processing pipeline
    # ------------------------------------------------------------------

    async def _process_utterance(self, utterance: Utterance):
        """Full pipeline: translate -> TTS -> buffer -> send in seq order.

        Multiple utterances can translate/TTS concurrently (via semaphore),
        but results are sent to the client strictly in sequence order.
        This prevents interleaved binary chunks on the WebSocket.
        """
        async with self._semaphore:
            if not self._running:
                return

            self._seq += 1
            seq = self._seq
            speaker_id = utterance.speaker_id

            try:
                # 1. Get voice for this speaker
                voice_id = self._speaker_mgr.get_voice_id(speaker_id)

                # 2. Translate
                translated = await translate_text(
                    utterance.text, self._source_lang, self._target_lang
                )

                if not translated.strip():
                    self._pending_results[seq] = None
                    await self._flush_pending()
                    return

                translated = _add_emotion_cues(utterance.text, translated)

                logger.info(
                    "Utterance %d [speaker %d]: '%s' -> '%s'",
                    seq,
                    speaker_id,
                    utterance.text[:50],
                    translated[:50],
                )

                # 3. Derive emotion directive from source utterance
                directive = _derive_directive(utterance, translated)

                # 4. TTS -> collect all audio chunks (don't send yet)
                audio_chunks = []
                async for chunk in self._tts.synthesize(
                    translated, voice_id, directive
                ):
                    audio_chunks.append(chunk)

                total_audio_bytes = sum(len(c) for c in audio_chunks)
                duration_sec = (
                    total_audio_bytes * 8 / 64000 if total_audio_bytes > 0 else 0
                )

                # 5. Store result for ordered sending
                self._pending_results[seq] = {
                    "seq": seq,
                    "speaker_id": speaker_id,
                    "audio_chunks": audio_chunks,
                    "duration_sec": duration_sec,
                    "utterance": utterance,
                    "translated": translated,
                }

            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.error("Utterance %d failed: %s", seq, e, exc_info=True)
                self._pending_results[seq] = None

        # After releasing the semaphore, flush any ready results in order
        await self._flush_pending()

    async def _flush_pending(self):
        """Send buffered utterance results to the client in strict seq order."""
        async with self._send_lock:
            while self._next_send_seq in self._pending_results:
                result = self._pending_results.pop(self._next_send_seq)
                self._next_send_seq += 1

                if result is None:
                    continue  # empty or failed utterance

                if not self._running:
                    return

                seq = result["seq"]
                speaker_id = result["speaker_id"]
                utterance = result["utterance"]
                translated = result["translated"]

                try:
                    # Send utterance_start
                    await self._send_text(
                        encode_msg(
                            UtteranceStartMsg(
                                seq=seq,
                                speaker_id=speaker_id,
                            )
                        )
                    )

                    # Send audio chunks
                    for chunk in result["audio_chunks"]:
                        await self._send_bytes(chunk)

                    # Send utterance_end
                    await self._send_text(
                        encode_msg(
                            UtteranceEndMsg(
                                seq=seq,
                                duration_sec=round(result["duration_sec"], 2),
                                original_start_sec=round(utterance.start_sec, 3),
                                original_end_sec=round(utterance.end_sec, 3),
                            )
                        )
                    )

                    # Send caption
                    await self._send_text(
                        encode_msg(
                            CaptionMsg(
                                seq=seq,
                                speaker_id=speaker_id,
                                original=utterance.text,
                                translated=translated,
                                start_time_sec=utterance.start_sec,
                                end_time_sec=utterance.end_sec,
                            )
                        )
                    )

                    # Track that an utterance was sent with this voice
                    self._speaker_mgr.mark_utterance_sent(speaker_id)

                    # Mark playback as started after first utterance is delivered
                    if not self._speaker_mgr.playback_started:
                        self._speaker_mgr.mark_playback_started()

                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    logger.error("Failed to send utterance %d: %s", seq, e)
                    try:
                        await self._send_text(
                            encode_msg(
                                ErrorMsg(
                                    message=f"Utterance send failed: {e}",
                                    recoverable=True,
                                )
                            )
                        )
                    except Exception:
                        pass


# ---------------------------------------------------------------------------
# Emotion analysis
# ---------------------------------------------------------------------------


def _derive_directive(utterance: Utterance, translated: str) -> VoiceDirective:
    """Derive a VoiceDirective from the source utterance for TTS styling."""
    has_exclamation = "!" in utterance.text
    is_question = utterance.text.rstrip().endswith("?")
    is_short = len(utterance.text.split()) <= 5

    if has_exclamation and is_short:
        return VoiceDirective(emotion="excited", energy=0.9, speed=1.05)
    elif has_exclamation:
        return VoiceDirective(emotion="enthusiastic", energy=0.7, speed=1.0)
    elif is_question:
        return VoiceDirective(emotion="curious", energy=0.5, speed=0.95)
    else:
        return VoiceDirective(emotion="neutral", energy=0.4, speed=1.0)


def _add_emotion_cues(original_text: str, translated_text: str) -> str:
    """Enhance translated text with emotion cues for more expressive TTS.

    Works alongside VoiceDirective — text cues + voice settings = better results.
    """
    original_has_exclamation = "!" in original_text
    original_is_short = len(original_text.split()) <= 5
    original_is_question = original_text.rstrip().endswith("?")

    text = translated_text.strip()

    if original_has_exclamation and not text.endswith("!") and not text.endswith("?"):
        text = text.rstrip(".") + "!"

    if original_is_short and original_has_exclamation and len(text.split()) <= 6:
        words = text.split()
        if len(words) >= 2:
            words[0] = words[0].upper()
            text = " ".join(words)

    if original_is_question and not text.endswith("?"):
        text = text.rstrip(".!") + "?"

    return text
