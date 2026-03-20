"""
Test embedding-based speaker continuity with REAL sports commentary.

Uses extracted audio from sample-videos/ to test:
1. How well ECAPA discriminates real commentators
2. Cross-chunk continuity with embeddings vs pitch-only
3. Actual latency on real audio
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

# Load .env for HF token
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(_project_root, ".env"))
except ImportError:
    pass


def _cosine_sim(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def _load_ecapa():
    from speechbrain.inference.speaker import EncoderClassifier
    device = torch.device("cpu")
    model = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        run_opts={"device": str(device)},
    )
    return model, device


def _load_diarizer():
    """Try to load the REAL SportsDiarizer with HF token (bypasses conftest stub)."""
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "real_diarizer",
            os.path.join(_project_root, "diarizer.py")
        )
        real_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(real_mod)
        d = real_mod.SportsDiarizer()
        return d
    except Exception as e:
        print(f"  ⚠️  Could not load SportsDiarizer: {e}")
        return None


def _simple_vad_segments(audio_np, sr, min_dur=0.5, energy_threshold=0.01):
    """Simple energy-based VAD as fallback when pyannote isn't available.
    Returns list of (start_sec, end_sec) tuples."""
    frame_len = int(0.025 * sr)  # 25ms frames
    hop = int(0.010 * sr)  # 10ms hop

    energies = []
    for i in range(0, len(audio_np) - frame_len, hop):
        frame = audio_np[i:i + frame_len]
        energies.append(np.sqrt(np.mean(frame ** 2)))

    energies = np.array(energies)
    threshold = max(energy_threshold, np.percentile(energies, 30))

    # Find speech regions
    is_speech = energies > threshold
    segments = []
    in_seg = False
    seg_start = 0

    for i, active in enumerate(is_speech):
        t = i * hop / sr
        if active and not in_seg:
            seg_start = t
            in_seg = True
        elif not active and in_seg:
            if t - seg_start >= min_dur:
                segments.append((seg_start, t))
            in_seg = False

    if in_seg:
        t = len(audio_np) / sr
        if t - seg_start >= min_dur:
            segments.append((seg_start, t))

    return segments


class TestRealSportsEmbeddings:

    def test_short_clip_diarize_and_embed(self):
        """Diarize the short clip and extract embeddings per speaker."""
        import soundfile as sf
        audio_path = "/tmp/short_audio.wav"
        if not os.path.exists(audio_path):
            # Try to extract from sample video
            import subprocess
            video = os.path.join(_project_root, "sample-videos", "short.mp4")
            if not os.path.exists(video):
                import pytest; pytest.skip("No sample video available")
            subprocess.run([
                "ffmpeg", "-y", "-i", video,
                "-ac", "1", "-ar", "16000", "-t", "10", audio_path
            ], capture_output=True)
            if not os.path.exists(audio_path):
                import pytest; pytest.skip("ffmpeg extraction failed")

        audio_np, sr = sf.read(audio_path)
        audio_tensor = torch.tensor(audio_np, dtype=torch.float32).unsqueeze(0)
        print(f"\n  Short clip: {len(audio_np)/sr:.1f}s, {sr}Hz")

        # Try pyannote diarization first, fall back to simple VAD
        diarizer = _load_diarizer()

        if diarizer is not None:
            segments = diarizer.diarize(audio_tensor, sr)
            seg_list = [(s.speaker_id, s.start_sec, s.end_sec) for s in segments]
            print(f"  Pyannote found {len(seg_list)} segments, "
                  f"{len(set(s[0] for s in seg_list))} speakers")
        else:
            # Fallback: simple VAD, label all as SPEAKER_0
            vad_segs = _simple_vad_segments(audio_np, sr)
            seg_list = [(f"SPEAKER_0", s, e) for s, e in vad_segs]
            print(f"  VAD fallback: {len(seg_list)} speech segments")

        for spk, start, end in seg_list:
            print(f"    {spk}: {start:.1f}-{end:.1f}s ({end - start:.1f}s)")

        # Extract embeddings per speaker
        model, device = _load_ecapa()
        from main import _estimate_pitch_safe

        speaker_data = {}
        for spk, start_sec, end_sec in seg_list:
            if end_sec - start_sec < 0.5:
                continue
            s = int(start_sec * sr)
            e = int(end_sec * sr)
            seg_audio = audio_tensor[:, s:e]

            wav = seg_audio.to(device).float()
            with torch.no_grad():
                emb = model.encode_batch(wav).squeeze(0).squeeze(0).cpu().numpy()

            seg_np = seg_audio.squeeze().numpy().astype(np.float32)
            try:
                pitch, _ = _estimate_pitch_safe(seg_np, sr)
            except:
                pitch = 0.0

            speaker_data.setdefault(spk, []).append({
                "emb": emb, "pitch": pitch,
                "start": start_sec, "end": end_sec,
            })

        print(f"\n  Embeddings extracted per speaker:")
        for spk, data in speaker_data.items():
            pitches = [d["pitch"] for d in data]
            print(f"    {spk}: {len(data)} segments, "
                  f"pitch range {min(pitches):.0f}-{max(pitches):.0f}Hz")

        # Cross-speaker similarity
        spk_ids = list(speaker_data.keys())
        if len(spk_ids) >= 2:
            print(f"\n  Cross-speaker embedding similarities:")
            for i in range(len(spk_ids)):
                for j in range(i + 1, len(spk_ids)):
                    sims = []
                    for d1 in speaker_data[spk_ids[i]]:
                        for d2 in speaker_data[spk_ids[j]]:
                            sims.append(_cosine_sim(d1["emb"], d2["emb"]))
                    print(f"    {spk_ids[i]} <-> {spk_ids[j]}: "
                          f"mean={np.mean(sims):.3f}, "
                          f"min={np.min(sims):.3f}, max={np.max(sims):.3f}")

            # Within-speaker
            print(f"\n  Within-speaker embedding similarities:")
            for spk in spk_ids:
                if len(speaker_data[spk]) < 2:
                    continue
                sims = []
                embs = [d["emb"] for d in speaker_data[spk]]
                for ii in range(len(embs)):
                    for jj in range(ii + 1, len(embs)):
                        sims.append(_cosine_sim(embs[ii], embs[jj]))
                if sims:
                    print(f"    {spk}: mean={np.mean(sims):.3f}, "
                          f"min={np.min(sims):.3f}, max={np.max(sims):.3f}")

    def test_long_clip_chunked_continuity(self):
        """Split 60s of the long clip into chunks, simulate live streaming,
        and compare pitch-only vs embedding-enhanced continuity."""
        import soundfile as sf
        audio_path = "/tmp/long_audio_60s.wav"
        if not os.path.exists(audio_path):
            import subprocess
            video = os.path.join(_project_root, "sample-videos", "long.mp4")
            if not os.path.exists(video):
                import pytest; pytest.skip("No sample video available")
            subprocess.run([
                "ffmpeg", "-y", "-i", video,
                "-ac", "1", "-ar", "16000", "-t", "60", audio_path
            ], capture_output=True)
            if not os.path.exists(audio_path):
                import pytest; pytest.skip("ffmpeg extraction failed")

        audio_np, sr = sf.read(audio_path)
        audio_tensor = torch.tensor(audio_np, dtype=torch.float32).unsqueeze(0)
        total_sec = len(audio_np) / sr
        print(f"\n  Long clip (first 60s): {total_sec:.1f}s, {sr}Hz")

        from main import _estimate_pitch_safe

        diarizer = _load_diarizer()
        model, device = _load_ecapa()

        CHUNK_SEC = 10.0
        num_chunks = int(total_sec / CHUNK_SEC)

        # ---- Helper: diarize a chunk ----
        def diarize_chunk(chunk_audio, chunk_sr):
            if diarizer is not None:
                segments = diarizer.diarize(chunk_audio, chunk_sr)
                return [(s.speaker_id, s.start_sec, s.end_sec) for s in segments]
            else:
                chunk_np = chunk_audio.squeeze().numpy()
                vad = _simple_vad_segments(chunk_np, chunk_sr, min_dur=0.5)
                return [(f"SPK_0", s, e) for s, e in vad]

        # ---- Run 1: Pitch-only matching ----
        print(f"\n{'='*70}")
        print(f"  PITCH-ONLY MATCHING ({num_chunks} chunks of {CHUNK_SEC}s)")
        print(f"{'='*70}")

        pitch_session = {"speaker_pitches": {}}
        pitch_assignments = []

        for chunk_idx in range(num_chunks):
            chunk_start_sec = chunk_idx * CHUNK_SEC
            chunk_start = int(chunk_start_sec * sr)
            chunk_end = int((chunk_start_sec + CHUNK_SEC) * sr)
            chunk_audio = audio_tensor[:, chunk_start:chunk_end]

            t0 = time.perf_counter()
            segments = diarize_chunk(chunk_audio, sr)
            diarize_ms = (time.perf_counter() - t0) * 1000

            for spk, seg_start, seg_end in segments:
                if seg_end - seg_start < 0.5:
                    continue
                s = int(seg_start * sr)
                e = int(seg_end * sr)
                seg_np = chunk_audio[:, s:e].squeeze().numpy().astype(np.float32)

                try:
                    pitch, _ = _estimate_pitch_safe(seg_np, sr)
                except:
                    pitch = 150.0

                # Simple pitch matching (baseline)
                best_match = None
                best_dist = float("inf")
                for stored_spk, stored_pitch in pitch_session["speaker_pitches"].items():
                    dist = abs(pitch - stored_pitch)
                    if dist < best_dist:
                        best_dist = dist
                        best_match = stored_spk

                if best_match is not None and best_dist <= 30.0:
                    assigned = best_match
                else:
                    assigned = f"SPK_{len(pitch_session['speaker_pitches'])}"
                    pitch_session["speaker_pitches"][assigned] = pitch

                pitch_assignments.append((chunk_idx, spk, assigned, pitch))

            print(f"  Chunk {chunk_idx} [{chunk_start_sec:.0f}-{chunk_start_sec+CHUNK_SEC:.0f}s]: "
                  f"{len(segments)} segs, diarize={diarize_ms:.0f}ms")

        # ---- Run 2: Embedding-enhanced matching ----
        print(f"\n{'='*70}")
        print(f"  EMBEDDING + PITCH MATCHING ({num_chunks} chunks of {CHUNK_SEC}s)")
        print(f"{'='*70}")

        emb_session = {
            "speaker_pitches": {},
            "speaker_embeddings": {},
        }
        emb_assignments = []

        EMB_THRESHOLD = 0.40  # tuned lower for real speech
        EMB_MARGIN = 0.03

        for chunk_idx in range(num_chunks):
            chunk_start_sec = chunk_idx * CHUNK_SEC
            chunk_start = int(chunk_start_sec * sr)
            chunk_end = int((chunk_start_sec + CHUNK_SEC) * sr)
            chunk_audio = audio_tensor[:, chunk_start:chunk_end]

            t0 = time.perf_counter()
            segments = diarize_chunk(chunk_audio, sr)
            diarize_ms = (time.perf_counter() - t0) * 1000

            embed_total_ms = 0
            for spk, seg_start, seg_end in segments:
                if seg_end - seg_start < 0.5:
                    continue
                s = int(seg_start * sr)
                e = int(seg_end * sr)
                seg_audio = chunk_audio[:, s:e]
                seg_np = seg_audio.squeeze().numpy().astype(np.float32)

                try:
                    pitch, _ = _estimate_pitch_safe(seg_np, sr)
                except:
                    pitch = 150.0

                # Extract embedding
                t_emb = time.perf_counter()
                wav = seg_audio.to(device).float()
                with torch.no_grad():
                    emb = model.encode_batch(wav).squeeze(0).squeeze(0).cpu().numpy()
                embed_total_ms += (time.perf_counter() - t_emb) * 1000

                # Embedding match
                assigned = None
                method = "NEW"

                if emb_session["speaker_embeddings"]:
                    sims = {k: _cosine_sim(emb, v)
                            for k, v in emb_session["speaker_embeddings"].items()}
                    sorted_sims = sorted(sims.items(), key=lambda x: x[1], reverse=True)
                    best_k, best_sim = sorted_sims[0]
                    second_sim = sorted_sims[1][1] if len(sorted_sims) > 1 else 0.0

                    if best_sim >= EMB_THRESHOLD and (best_sim - second_sim) >= EMB_MARGIN:
                        assigned = best_k
                        method = f"EMB({best_sim:.2f})"

                # Pitch fallback
                if assigned is None:
                    best_dist = float("inf")
                    best_match_p = None
                    for stored_spk, stored_pitch in emb_session["speaker_pitches"].items():
                        dist = abs(pitch - stored_pitch)
                        if dist < best_dist:
                            best_dist = dist
                            best_match_p = stored_spk
                    if best_match_p is not None and best_dist <= 30.0:
                        assigned = best_match_p
                        method = f"PITCH({best_dist:.0f}Hz)"

                if assigned is None:
                    assigned = f"SPK_{len(emb_session['speaker_pitches'])}"
                    emb_session["speaker_pitches"][assigned] = pitch
                    emb_session["speaker_embeddings"][assigned] = emb
                    method = "NEW"
                else:
                    # Update embedding with EMA
                    if assigned in emb_session["speaker_embeddings"]:
                        old = emb_session["speaker_embeddings"][assigned]
                        emb_session["speaker_embeddings"][assigned] = 0.7 * old + 0.3 * emb

                emb_assignments.append((chunk_idx, spk, assigned, pitch, method))

            print(f"  Chunk {chunk_idx} [{chunk_start_sec:.0f}-{chunk_start_sec+CHUNK_SEC:.0f}s]: "
                  f"{len(segments)} segs, diarize={diarize_ms:.0f}ms, embed={embed_total_ms:.0f}ms")

        # ---- Compare results ----
        print(f"\n{'='*70}")
        print(f"  COMPARISON")
        print(f"{'='*70}")

        def analyze_assignments(assignments, label):
            print(f"\n  {label}:")
            by_diarizer = {}
            for chunk, diar_spk, session_spk, *rest in assignments:
                by_diarizer.setdefault(diar_spk, []).append(session_spk)

            session_spks = set()
            for diar_spk, session_ids in sorted(by_diarizer.items()):
                most_common = max(set(session_ids), key=session_ids.count)
                consistency = session_ids.count(most_common) / len(session_ids)
                session_spks.update(session_ids)
                print(f"    {diar_spk}: -> {set(session_ids)} "
                      f"(primary={most_common}, {consistency:.0%} consistent, "
                      f"{len(session_ids)} segments)")

            print(f"    Total session speakers created: {len(session_spks)}")
            return session_spks

        pitch_spks = analyze_assignments(pitch_assignments, "PITCH-ONLY")
        emb_spks = analyze_assignments(emb_assignments, "EMBEDDING + PITCH")

        # Show methods used
        methods = [m for _, _, _, _, m in emb_assignments]
        emb_count = sum(1 for m in methods if m.startswith("EMB"))
        pitch_count = sum(1 for m in methods if m.startswith("PITCH"))
        new_count = sum(1 for m in methods if m == "NEW")
        print(f"\n  Embedding match methods: {emb_count} embedding, "
              f"{pitch_count} pitch, {new_count} new "
              f"(out of {len(methods)} total)")

        print(f"\n  Pitch-only created {len(pitch_spks)} session speakers")
        print(f"  Embedding+pitch created {len(emb_spks)} session speakers")
        print(f"  (Fewer = better consolidation of same speakers across chunks)")
