"""
Experiment G: ECAPA-TDNN Embedding Latency Benchmark

Measures the overhead of adding speaker embeddings to the live streaming
pipeline. The live path currently uses pitch estimation (~5ms) per speaker
per chunk. This experiment quantifies how much latency embedding extraction
and cosine similarity matching would add.

Benchmarked operations:
  1. ECAPA model cold-start load time
  2. encode_batch latency vs audio duration (0.5s - 10s)
  3. Cosine similarity computation vs number of stored speakers
  4. Pitch estimation (current baseline) for comparison
  5. Projected per-chunk overhead for realistic scenarios

Run:
    python -m pytest tests/experiment_g_latency.py -v -s
"""

import sys
import os
import time
import statistics

import numpy as np
import torch
import torchaudio
# Workaround: torchaudio 2.10 removed list_audio_backends, speechbrain expects it
if not hasattr(torchaudio, 'list_audio_backends'):
    torchaudio.list_audio_backends = lambda: ['soundfile']

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SAMPLE_RATE = 16000
DURATIONS_SEC = [0.5, 1.0, 2.0, 3.0, 5.0, 10.0]
SPEAKER_COUNTS = [1, 2, 4, 8, 16]
WARMUP_ITERS = 3
BENCH_ITERS = 10


def _generate_tone(duration_sec: float, sr: int = SAMPLE_RATE, freq: float = 220.0) -> torch.Tensor:
    """Generate a sine-wave tone as a [1, N] torch tensor (simulates voiced speech)."""
    n_samples = int(duration_sec * sr)
    t = np.linspace(0, duration_sec, n_samples, dtype=np.float32)
    # Add a bit of noise so it's not a perfect sine (more realistic for embeddings)
    tone = np.sin(2.0 * np.pi * freq * t) * 0.5
    tone += np.random.default_rng(42).normal(0, 0.02, size=n_samples).astype(np.float32)
    return torch.from_numpy(tone).unsqueeze(0)  # shape [1, N]


def _bench(fn, warmup: int = WARMUP_ITERS, iters: int = BENCH_ITERS):
    """Run fn() with warmup, then time `iters` calls. Return (mean_sec, std_sec)."""
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return statistics.mean(times), statistics.stdev(times) if len(times) > 1 else 0.0


# ---------------------------------------------------------------------------
# Test class
# ---------------------------------------------------------------------------
class TestExperimentGLatency:
    """Benchmark ECAPA-TDNN embedding latency for the live streaming pipeline."""

    def test_latency_report(self):
        results = {}

        # ==================================================================
        # 1. Model cold-start
        # ==================================================================
        print()
        print("=" * 80)
        print("  EXPERIMENT G: ECAPA-TDNN Embedding Latency Benchmark")
        print("=" * 80)
        print()
        print("[1/6] Loading ECAPA model (cold start)...")

        t0 = time.perf_counter()
        from speechbrain.inference.speaker import EncoderClassifier
        model = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            run_opts={"device": "cpu"},
        )
        load_time = time.perf_counter() - t0
        results["model_load_sec"] = load_time
        print(f"       Model loaded in {load_time:.3f}s")

        # ==================================================================
        # 2. Embedding extraction latency vs audio duration
        # ==================================================================
        print()
        print("[2/6] Embedding extraction latency (CPU)...")
        emb_results = {}  # duration -> (mean_ms, std_ms)

        for dur in DURATIONS_SEC:
            wav = _generate_tone(dur)

            def _extract():
                with torch.no_grad():
                    model.encode_batch(wav)

            mean_s, std_s = _bench(_extract)
            mean_ms = mean_s * 1000
            std_ms = std_s * 1000
            emb_results[dur] = (mean_ms, std_ms)
            print(f"       {dur:5.1f}s audio -> {mean_ms:7.2f} +/- {std_ms:5.2f} ms")

        results["embedding_ms"] = emb_results

        # Check GPU availability
        gpu_emb_results = None
        if torch.cuda.is_available():
            print()
            print("       [GPU available - also benchmarking CUDA]")
            gpu_model = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                run_opts={"device": "cuda"},
            )
            gpu_emb_results = {}
            for dur in DURATIONS_SEC:
                wav_gpu = _generate_tone(dur).cuda()

                def _extract_gpu(w=wav_gpu):
                    with torch.no_grad():
                        gpu_model.encode_batch(w)
                    torch.cuda.synchronize()

                mean_s, std_s = _bench(_extract_gpu)
                mean_ms = mean_s * 1000
                std_ms = std_s * 1000
                gpu_emb_results[dur] = (mean_ms, std_ms)
                print(f"       {dur:5.1f}s audio -> {mean_ms:7.2f} +/- {std_ms:5.2f} ms (GPU)")

            results["embedding_gpu_ms"] = gpu_emb_results
        else:
            print("       [No CUDA - skipping GPU benchmark]")

        # ==================================================================
        # 3. Cosine similarity computation time
        # ==================================================================
        print()
        print("[3/6] Cosine similarity computation...")

        # Extract a reference embedding for similarity tests
        ref_wav = _generate_tone(2.0)
        with torch.no_grad():
            ref_emb = model.encode_batch(ref_wav).squeeze(0).squeeze(0).cpu().numpy()

        emb_dim = ref_emb.shape[0]
        print(f"       Embedding dimension: {emb_dim}")

        # Import the project's cosine similarity function
        def _cosine_sim(a, b):
            a = a.astype(np.float32)
            b = b.astype(np.float32)
            return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

        sim_results = {}  # N -> (mean_ms, std_ms)

        for n_speakers in SPEAKER_COUNTS:
            # Generate N random stored embeddings (normalized)
            rng = np.random.default_rng(123)
            stored = []
            for _ in range(n_speakers):
                e = rng.standard_normal(emb_dim).astype(np.float32)
                e /= np.linalg.norm(e) + 1e-9
                stored.append(e)

            def _compare(stored_embs=stored):
                for emb in stored_embs:
                    _cosine_sim(ref_emb, emb)

            mean_s, std_s = _bench(_compare, warmup=WARMUP_ITERS, iters=BENCH_ITERS)
            mean_ms = mean_s * 1000
            std_ms = std_s * 1000
            sim_results[n_speakers] = (mean_ms, std_ms)
            print(f"       vs {n_speakers:2d} stored speakers -> {mean_ms:7.4f} +/- {std_ms:5.4f} ms")

        results["cosine_sim_ms"] = sim_results

        # ==================================================================
        # 4. Pitch estimation (current baseline) for comparison
        # ==================================================================
        print()
        print("[4/6] Pitch estimation (current baseline)...")

        from main import _estimate_pitch_safe

        pitch_results = {}  # duration -> (mean_ms, std_ms)

        for dur in DURATIONS_SEC:
            wav_np = _generate_tone(dur).squeeze(0).numpy()

            def _pitch(audio=wav_np):
                _estimate_pitch_safe(audio, SAMPLE_RATE)

            mean_s, std_s = _bench(_pitch)
            mean_ms = mean_s * 1000
            std_ms = std_s * 1000
            pitch_results[dur] = (mean_ms, std_ms)
            print(f"       {dur:5.1f}s audio -> {mean_ms:7.2f} +/- {std_ms:5.2f} ms")

        results["pitch_ms"] = pitch_results

        # ==================================================================
        # 5. End-to-end per-chunk estimate
        # ==================================================================
        print()
        print("[5/6] End-to-end per-chunk estimate (5s chunk, 2 speakers)...")

        # Scenario: 5s chunk, 2 speakers each with ~2.5s audio
        # Need: 2 embedding extractions + 2 * (cosine sim vs 3 stored) + 2 pitch estimates
        emb_2_5s_ms = emb_results[3.0][0]  # closest to 2.5s
        sim_vs_3_ms = sim_results[4][0]     # closest to 3 stored
        pitch_2_5s_ms = pitch_results[3.0][0]

        e2e_embedding_ms = 2 * emb_2_5s_ms + 2 * sim_vs_3_ms
        e2e_pitch_ms = 2 * pitch_2_5s_ms
        e2e_total_added_ms = e2e_embedding_ms  # embeddings are the NEW cost

        print(f"       Current pitch cost:     {e2e_pitch_ms:.2f} ms (2 x pitch estimate)")
        print(f"       New embedding cost:     {e2e_embedding_ms:.2f} ms (2 x encode + 2 x sim_vs_3)")
        print(f"       Total added latency:    {e2e_total_added_ms:.2f} ms")
        print(f"       New total (pitch+emb):  {e2e_pitch_ms + e2e_embedding_ms:.2f} ms")

        results["e2e"] = {
            "pitch_ms": e2e_pitch_ms,
            "embedding_ms": e2e_embedding_ms,
            "total_added_ms": e2e_total_added_ms,
            "combined_ms": e2e_pitch_ms + e2e_embedding_ms,
        }

        # ==================================================================
        # 6. Comprehensive latency report
        # ==================================================================
        print()
        print("=" * 80)
        print("  COMPREHENSIVE LATENCY REPORT")
        print("=" * 80)

        # --- Model load ---
        print()
        print("  Model Load (cold start)")
        print("  " + "-" * 40)
        print(f"    ECAPA-TDNN load:  {results['model_load_sec']:.3f} s")
        print(f"    (one-time cost, amortized over session)")

        # --- Embedding extraction table ---
        print()
        print("  Embedding Extraction (CPU)")
        print("  " + "-" * 60)
        header = f"    {'Duration':>10}  {'Mean (ms)':>12}  {'Std (ms)':>10}"
        if gpu_emb_results:
            header += f"  {'GPU Mean':>12}  {'GPU Std':>10}"
        print(header)
        print("    " + "-" * (56 if not gpu_emb_results else 80))

        for dur in DURATIONS_SEC:
            mean_ms, std_ms = emb_results[dur]
            line = f"    {dur:>8.1f} s  {mean_ms:>10.2f} ms  {std_ms:>8.2f} ms"
            if gpu_emb_results:
                g_mean, g_std = gpu_emb_results[dur]
                line += f"  {g_mean:>10.2f} ms  {g_std:>8.2f} ms"
            print(line)

        # --- Cosine similarity table ---
        print()
        print("  Cosine Similarity Matching")
        print("  " + "-" * 50)
        print(f"    {'N stored':>10}  {'Mean (ms)':>12}  {'Std (ms)':>10}")
        print("    " + "-" * 40)
        for n in SPEAKER_COUNTS:
            mean_ms, std_ms = sim_results[n]
            print(f"    {n:>10}  {mean_ms:>10.4f} ms  {std_ms:>8.4f} ms")

        # --- Pitch estimation table ---
        print()
        print("  Pitch Estimation (current baseline)")
        print("  " + "-" * 50)
        print(f"    {'Duration':>10}  {'Mean (ms)':>12}  {'Std (ms)':>10}")
        print("    " + "-" * 40)
        for dur in DURATIONS_SEC:
            mean_ms, std_ms = pitch_results[dur]
            print(f"    {dur:>8.1f} s  {mean_ms:>10.2f} ms  {std_ms:>8.2f} ms")

        # --- Per-chunk overhead projections ---
        print()
        print("  Per-Chunk Overhead Projections")
        print("  " + "-" * 76)
        print(f"    {'Scenario':<35}  {'Pitch (ms)':>11}  {'Embed (ms)':>11}  {'Added (ms)':>11}  {'Total (ms)':>11}")
        print("    " + "-" * 72)

        scenarios = [
            ("2 speakers, 3 stored (typical)", 2, 2.5, 3),
            ("3 speakers, 4 stored", 3, 1.7, 4),
            ("5 speakers, 6 stored", 5, 1.0, 6),
        ]

        for label, n_spk, approx_dur, n_stored in scenarios:
            # Find closest benchmarked duration for embedding
            closest_dur = min(DURATIONS_SEC, key=lambda d: abs(d - approx_dur))
            emb_per_spk_ms = emb_results[closest_dur][0]

            # Find closest benchmarked N for similarity
            closest_n = min(SPEAKER_COUNTS, key=lambda n: abs(n - n_stored))
            sim_per_spk_ms = sim_results[closest_n][0]

            # Pitch cost
            pitch_per_spk_ms = pitch_results[closest_dur][0]
            total_pitch_ms = n_spk * pitch_per_spk_ms

            # Embedding cost = extraction + matching
            total_embed_ms = n_spk * (emb_per_spk_ms + sim_per_spk_ms)
            total_added_ms = total_embed_ms
            total_combined_ms = total_pitch_ms + total_embed_ms

            print(
                f"    {label:<35}  {total_pitch_ms:>9.2f} ms"
                f"  {total_embed_ms:>9.2f} ms"
                f"  {total_added_ms:>9.2f} ms"
                f"  {total_combined_ms:>9.2f} ms"
            )

        # --- Summary ---
        print()
        print("  " + "=" * 76)
        print("  SUMMARY")
        print("  " + "=" * 76)
        typical_emb_ms = emb_results[3.0][0]
        typical_pitch_ms = pitch_results[3.0][0]
        ratio = typical_emb_ms / max(typical_pitch_ms, 0.001)
        print(f"    Typical embedding extraction (3s audio):  {typical_emb_ms:.2f} ms")
        print(f"    Typical pitch estimation (3s audio):      {typical_pitch_ms:.2f} ms")
        print(f"    Embedding / Pitch ratio:                  {ratio:.1f}x")
        print()

        budget_ms = 200.0  # reasonable latency budget for live streaming
        print(f"    Latency budget for live path:             {budget_ms:.0f} ms")
        print(f"    Typical 2-speaker chunk embedding cost:   {results['e2e']['embedding_ms']:.2f} ms")
        fits = results["e2e"]["embedding_ms"] < budget_ms
        verdict = "WITHIN BUDGET" if fits else "EXCEEDS BUDGET"
        print(f"    Verdict:                                  {verdict}")
        print()
        print("=" * 80)

        # The test always passes -- this is a benchmark, not a correctness test.
        # We just assert that the model loaded and produced valid embeddings.
        assert results["model_load_sec"] > 0
        assert all(m > 0 for m, _ in emb_results.values())
        assert ref_emb.shape == (192,), f"Expected 192-dim embedding, got {ref_emb.shape}"
