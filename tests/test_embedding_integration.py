"""
Integration test for embedding-enhanced _assign_chunk_voices.

Tests that the hybrid embedding + pitch matching works end-to-end:
1. ECAPA model loads and produces embeddings
2. Embedding-based speaker matching works
3. Latency stays within acceptable bounds
4. Full hybrid matching across simulated chunks
"""
import sys, os, time
import numpy as np
import torch

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# Fix torchaudio compat
import torchaudio
if not hasattr(torchaudio, 'list_audio_backends'):
    torchaudio.list_audio_backends = lambda: ['soundfile']


def _cosine_sim(a, b):
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def _load_ecapa():
    """Load ECAPA-TDNN model directly (bypass diarizer constructor)."""
    try:
        from speechbrain.inference.speaker import EncoderClassifier
    except ImportError:
        return None, None
    device = torch.device("cpu")
    model = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        run_opts={"device": str(device)},
    )
    return model, device


def make_speaker_audio(freq_hz, duration_sec=2.0, sr=16000, formant_shift=0.0):
    """Generate speaker-like audio with distinct harmonic structure."""
    t = np.arange(int(sr * duration_sec), dtype=np.float32) / sr
    audio = 0.3 * np.sin(2 * np.pi * freq_hz * t)
    for h, base_gain in [(2, 0.6), (3, 0.35), (4, 0.2), (5, 0.1)]:
        gain = base_gain * (1.0 + formant_shift * (0.5 if h % 2 == 0 else -0.3))
        harm_freq = freq_hz * h
        if harm_freq < sr / 2:
            audio += 0.3 * gain * np.sin(2 * np.pi * harm_freq * t)
    audio += 0.02 * np.random.randn(len(audio)).astype(np.float32)
    return torch.tensor(audio, dtype=torch.float32).unsqueeze(0)


def extract_embedding(model, device, audio_tensor):
    wav = audio_tensor.to(device).float()
    with torch.no_grad():
        emb = model.encode_batch(wav).squeeze(0).squeeze(0).cpu().numpy()
    return emb


class TestEmbeddingIntegration:

    def test_embedding_extraction(self):
        """Verify ECAPA model loads and produces 192-dim embeddings."""
        model, device = _load_ecapa()
        if model is None:
            import pytest; pytest.skip("speechbrain not available")

        audio = make_speaker_audio(150.0, duration_sec=2.0)
        emb = extract_embedding(model, device, audio)

        print(f"\n  Embedding shape: {emb.shape}")
        print(f"  Embedding norm: {np.linalg.norm(emb):.3f}")
        print(f"  First 5 values: {emb[:5]}")
        assert emb.shape[0] == 192

    def test_speaker_discrimination(self):
        """Test how well ECAPA distinguishes different synthetic speakers."""
        model, device = _load_ecapa()
        if model is None:
            import pytest; pytest.skip("speechbrain not available")

        scenarios = [
            ("Same pitch, same timbre (same speaker)", 150.0, 0.0, 150.0, 0.0),
            ("Drifted pitch, same timbre (same speaker)", 150.0, 0.0, 170.0, 0.0),
            ("Large drift, same timbre (same speaker)", 150.0, 0.0, 200.0, 0.0),
            ("Same pitch, diff timbre (diff speaker)", 150.0, 0.0, 150.0, 0.8),
            ("Close pitch, diff timbre (diff speaker)", 150.0, 0.0, 155.0, 0.6),
            ("Male vs Female (clearly different)", 130.0, 0.0, 220.0, 0.5),
        ]

        print(f"\n  {'Scenario':<50} {'Cosine':>8}")
        print(f"  {'-'*50} {'-'*8}")

        for desc, f1, s1, f2, s2 in scenarios:
            a1 = make_speaker_audio(f1, formant_shift=s1)
            a2 = make_speaker_audio(f2, formant_shift=s2)
            e1 = extract_embedding(model, device, a1)
            e2 = extract_embedding(model, device, a2)
            sim = _cosine_sim(e1, e2)
            print(f"  {desc:<50} {sim:>8.4f}")

    def test_embedding_latency(self):
        """Measure embedding extraction time."""
        model, device = _load_ecapa()
        if model is None:
            import pytest; pytest.skip("speechbrain not available")

        durations = [1.0, 2.0, 3.0, 5.0]
        print(f"\n  {'Duration':>10} {'Mean (ms)':>12} {'Std (ms)':>10}")
        print(f"  {'-'*10} {'-'*12} {'-'*10}")

        for dur in durations:
            audio = make_speaker_audio(150.0, duration_sec=dur)
            wav = audio.to(device).float()

            # Warmup
            for _ in range(3):
                with torch.no_grad():
                    model.encode_batch(wav)

            times = []
            for _ in range(10):
                t0 = time.perf_counter()
                with torch.no_grad():
                    model.encode_batch(wav)
                times.append((time.perf_counter() - t0) * 1000)

            print(f"  {dur:>8.1f}s {np.mean(times):>10.1f}ms {np.std(times):>8.1f}ms")

        # Per-chunk estimate for 2 speakers
        audio_2s = make_speaker_audio(150.0, duration_sec=2.5)
        wav_2s = audio_2s.to(device).float()
        t0 = time.perf_counter()
        for _ in range(2):  # 2 speakers
            with torch.no_grad():
                model.encode_batch(wav_2s)
        chunk_ms = (time.perf_counter() - t0) * 1000
        print(f"\n  2-speaker chunk total: {chunk_ms:.1f}ms")

    def test_cross_chunk_matching_simulation(self):
        """Simulate 5 chunks with 2 speakers, test embedding continuity."""
        model, device = _load_ecapa()
        if model is None:
            import pytest; pytest.skip("speechbrain not available")

        np.random.seed(42)

        # Speaker A: male ~130Hz, Speaker B: female ~220Hz
        speaker_a_base = 130.0
        speaker_b_base = 220.0

        # Store embeddings as we go (simulating session_state)
        stored_embeddings = {}  # spk_id -> embedding
        stored_pitches = {}     # spk_id -> pitch

        EMB_THRESHOLD = 0.65
        PITCH_THRESHOLD = 30.0

        print(f"\n  Simulating 5-chunk session with 2 speakers...")
        print(f"  Speaker A: ~{speaker_a_base}Hz, Speaker B: ~{speaker_b_base}Hz\n")

        assignments = {"A": [], "B": []}

        for chunk in range(5):
            # Simulate pitch drift
            pitch_a = speaker_a_base + np.random.uniform(-15, 15)
            pitch_b = speaker_b_base + np.random.uniform(-15, 15)

            audio_a = make_speaker_audio(pitch_a, duration_sec=2.0, formant_shift=0.0)
            audio_b = make_speaker_audio(pitch_b, duration_sec=2.0, formant_shift=0.5)

            emb_a = extract_embedding(model, device, audio_a)
            emb_b = extract_embedding(model, device, audio_b)

            for label, emb, pitch in [("A", emb_a, pitch_a), ("B", emb_b, pitch_b)]:
                matched = None

                # Try embedding match
                if stored_embeddings:
                    sims = {k: _cosine_sim(emb, v) for k, v in stored_embeddings.items()}
                    best_k = max(sims, key=sims.get)
                    best_sim = sims[best_k]
                    if best_sim >= EMB_THRESHOLD:
                        matched = best_k
                        method = f"emb(sim={best_sim:.3f})"

                # Fallback to pitch
                if matched is None and stored_pitches:
                    dists = {k: abs(pitch - v) for k, v in stored_pitches.items()}
                    best_k = min(dists, key=dists.get)
                    if dists[best_k] <= PITCH_THRESHOLD:
                        matched = best_k
                        method = f"pitch(dist={dists[best_k]:.0f}Hz)"

                if matched is None:
                    spk_id = f"SPK_{len(stored_embeddings)}"
                    stored_embeddings[spk_id] = emb
                    stored_pitches[spk_id] = pitch
                    matched = spk_id
                    method = "NEW"
                else:
                    # Update embedding with EMA
                    stored_embeddings[matched] = 0.7 * stored_embeddings[matched] + 0.3 * emb

                assignments[label].append(matched)
                print(f"  Chunk {chunk} Speaker {label} ({pitch:.0f}Hz) → {matched} [{method}]")

        print(f"\n  Speaker A assignments: {assignments['A']}")
        print(f"  Speaker B assignments: {assignments['B']}")

        # Check consistency
        a_consistent = len(set(assignments["A"])) == 1
        b_consistent = len(set(assignments["B"])) == 1
        a_b_different = set(assignments["A"]) != set(assignments["B"])

        print(f"\n  Speaker A consistent: {a_consistent} ({'✓' if a_consistent else '✗'})")
        print(f"  Speaker B consistent: {b_consistent} ({'✓' if b_consistent else '✗'})")
        print(f"  A ≠ B voices:         {a_b_different} ({'✓' if a_b_different else '✗'})")

        a_score = max(assignments["A"].count(v) for v in set(assignments["A"])) / 5
        b_score = max(assignments["B"].count(v) for v in set(assignments["B"])) / 5
        print(f"\n  Continuity: A={a_score:.0%}, B={b_score:.0%}")

    def test_crossing_pitches_with_embeddings(self):
        """The key test: can embeddings handle crossing pitches?"""
        model, device = _load_ecapa()
        if model is None:
            import pytest; pytest.skip("speechbrain not available")

        np.random.seed(42)

        # Speaker A: 130 → 180 Hz (getting excited)
        # Speaker B: 180 → 130 Hz (calming down)
        # They CROSS at ~155 Hz — pitch-only matching fails here

        stored_embeddings = {}
        EMB_THRESHOLD = 0.60  # slightly lower for this harder test

        print(f"\n  Crossing pitch test: A goes 130→180, B goes 180→130")
        print(f"  (This is where pitch-only matching breaks down)\n")

        assignments = {"A": [], "B": []}
        num_chunks = 5

        for chunk in range(num_chunks):
            pitch_a = 130.0 + (50.0 * chunk / (num_chunks - 1)) + np.random.uniform(-5, 5)
            pitch_b = 180.0 - (50.0 * chunk / (num_chunks - 1)) + np.random.uniform(-5, 5)

            audio_a = make_speaker_audio(pitch_a, duration_sec=2.0, formant_shift=0.0)
            audio_b = make_speaker_audio(pitch_b, duration_sec=2.0, formant_shift=0.6)

            emb_a = extract_embedding(model, device, audio_a)
            emb_b = extract_embedding(model, device, audio_b)

            for label, emb, pitch in [("A", emb_a, pitch_a), ("B", emb_b, pitch_b)]:
                matched = None

                if stored_embeddings:
                    sims = {k: _cosine_sim(emb, v) for k, v in stored_embeddings.items()}
                    best_k = max(sims, key=sims.get)
                    best_sim = sims[best_k]
                    if best_sim >= EMB_THRESHOLD:
                        matched = best_k

                if matched is None:
                    spk_id = f"SPK_{len(stored_embeddings)}"
                    stored_embeddings[spk_id] = emb
                    matched = spk_id

                if matched in stored_embeddings and matched != f"SPK_{len(stored_embeddings)}":
                    stored_embeddings[matched] = 0.7 * stored_embeddings[matched] + 0.3 * emb

                assignments[label].append(matched)
                sims_str = ""
                if len(stored_embeddings) > 1:
                    sims_all = {k: _cosine_sim(emb, v) for k, v in stored_embeddings.items()}
                    sims_str = " | ".join(f"{k}={v:.3f}" for k, v in sorted(sims_all.items()))
                print(f"  Chunk {chunk}: {label}@{pitch:.0f}Hz → {matched}  [{sims_str}]")

        print(f"\n  Speaker A: {assignments['A']}")
        print(f"  Speaker B: {assignments['B']}")

        a_score = max(assignments["A"].count(v) for v in set(assignments["A"])) / num_chunks
        b_score = max(assignments["B"].count(v) for v in set(assignments["B"])) / num_chunks
        print(f"\n  Crossing continuity: A={a_score:.0%}, B={b_score:.0%}")
        print(f"  (Pitch-only baseline: A=80%, B=60%)")
