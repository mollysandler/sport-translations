"""
Test v7c voice assignment on live radio audio (France Info).
Fetches real audio, diarizes in 8s chunks, runs _assign_chunk_voices.
"""
import sys, os, time
import numpy as np
import torch

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import torchaudio
if not hasattr(torchaudio, 'list_audio_backends'):
    torchaudio.list_audio_backends = lambda: ['soundfile']

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(_project_root, ".env"))
except ImportError:
    pass


def _load_real_diarizer():
    """Load real SportsDiarizer bypassing conftest stubs."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "real_diarizer", os.path.join(_project_root, "diarizer.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.SportsDiarizer()


def test_live_v7c_voice_assignment():
    """Run v7c voice assignment on 60s of live radio in 8s chunks."""
    import soundfile as sf
    import pytest

    audio_path = "/tmp/live_test_60s.wav"
    if not os.path.exists(audio_path):
        pytest.skip("Run: ffmpeg -i <stream_url> -ac 1 -ar 16000 -t 60 -y /tmp/live_test_60s.wav")

    audio_np, sr = sf.read(audio_path)
    audio_tensor = torch.tensor(audio_np, dtype=torch.float32).unsqueeze(0)
    total_sec = len(audio_np) / sr
    print(f"\n  Audio: {total_sec:.1f}s, {sr}Hz (live radio)")

    # Load real diarizer
    diarizer = _load_real_diarizer()
    print(f"  Diarizer loaded.")

    # Build a minimal translator with the real diarizer's ECAPA model
    # We'll call _assign_chunk_voices directly via the class
    from main import DynamicSpeakerTranslator, _estimate_pitch_safe
    from utils import gender_from_pitch

    # We need a translator instance with voice_manager and diarizer
    # but skip full initialization — just set up what _assign_chunk_voices needs
    translator = object.__new__(DynamicSpeakerTranslator)
    translator.diarizer = diarizer

    # Minimal voice_manager mock
    class FakeVoiceManager:
        def _match_best_voice(self, gender, pitch, pitch_range):
            return f"voice_{gender}_{pitch:.0f}"
    translator.voice_manager = FakeVoiceManager()

    CHUNK_SEC = 8.0
    num_chunks = int(total_sec / CHUNK_SEC)

    # Session state (persists across chunks)
    session_state = {
        "speaker_voice_ids": {},
        "speaker_pitches": {},
        "speaker_pitch_histories": {},
        "speaker_embeddings": {},
    }

    print(f"\n  Processing {num_chunks} chunks of {CHUNK_SEC}s...\n")

    all_assignments = []
    total_diar_ms = 0
    total_assign_ms = 0

    for chunk_idx in range(num_chunks):
        chunk_start_sec = chunk_idx * CHUNK_SEC
        chunk_start = int(chunk_start_sec * sr)
        chunk_end = int((chunk_start_sec + CHUNK_SEC) * sr)
        chunk_audio = audio_tensor[:, chunk_start:chunk_end]

        # Diarize
        t0 = time.perf_counter()
        segments = diarizer.diarize(chunk_audio, sr)
        diar_ms = (time.perf_counter() - t0) * 1000
        total_diar_ms += diar_ms

        if not segments:
            print(f"  Chunk {chunk_idx:2d} [{chunk_start_sec:5.1f}-{chunk_start_sec+CHUNK_SEC:5.1f}s]: "
                  f"no speech detected")
            continue

        n_spk = len(set(s.speaker_id for s in segments))
        seg_info = ", ".join(f"{s.speaker_id}[{s.start_sec:.1f}-{s.end_sec:.1f}s]" for s in segments)

        # Assign voices using v7c
        t0 = time.perf_counter()
        chunk_voice_map = translator._assign_chunk_voices(chunk_audio, segments, sr, session_state)
        assign_ms = (time.perf_counter() - t0) * 1000
        total_assign_ms += assign_ms

        print(f"  Chunk {chunk_idx:2d} [{chunk_start_sec:5.1f}-{chunk_start_sec+CHUNK_SEC:5.1f}s]: "
              f"{len(segments)} segs, {n_spk} spk, "
              f"diar={diar_ms:.0f}ms, assign={assign_ms:.0f}ms")
        for spk_id, voice_id in chunk_voice_map.items():
            print(f"    {spk_id} → {voice_id}")

        all_assignments.append({
            "chunk": chunk_idx,
            "time": f"{chunk_start_sec:.0f}-{chunk_start_sec+CHUNK_SEC:.0f}s",
            "map": dict(chunk_voice_map),
            "n_spk": n_spk,
        })

    # Summary
    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")

    # Count unique session voices
    session_voices = set(session_state["speaker_voice_ids"].values())
    print(f"\n  Session voices created: {len(session_voices)}")
    for key, voice in session_state["speaker_voice_ids"].items():
        pitch = session_state["speaker_pitches"].get(key, "?")
        print(f"    {key}: {voice} (pitch={pitch})")

    # Diar history
    diar_history = session_state.get("diar_history", {})
    if diar_history:
        print(f"\n  Diarizer label history:")
        for label, counter in sorted(diar_history.items()):
            print(f"    {label}: {dict(counter)}")

    # Voice assignment consistency per diarizer label
    print(f"\n  Voice consistency per diarizer label:")
    label_voices = {}
    for a in all_assignments:
        for spk_id, voice_id in a["map"].items():
            label_voices.setdefault(spk_id, []).append(voice_id)

    for label, voices in sorted(label_voices.items()):
        from collections import Counter
        counts = Counter(voices)
        most_common, mc_count = counts.most_common(1)[0]
        consistency = mc_count / len(voices)
        detail = ", ".join(f"{v}:{c}" for v, c in counts.most_common())
        print(f"    {label}: {len(voices)} chunks → [{detail}] ({consistency:.0%} consistent)")

    print(f"\n  Total diarization: {total_diar_ms:.0f}ms ({total_diar_ms/num_chunks:.0f}ms/chunk)")
    print(f"  Total assignment:  {total_assign_ms:.0f}ms ({total_assign_ms/num_chunks:.0f}ms/chunk)")


if __name__ == "__main__":
    test_live_v7c_voice_assignment()
