"""
Test embedding-based continuity with REAL speech (pyannote sample).

Uses the 30s pyannote sample.wav with 2 known speakers (speaker90, speaker91)
to test whether ECAPA embeddings can distinguish real speakers and maintain
voice continuity across simulated chunks.
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

SAMPLE_WAV = os.path.join(
    _project_root, ".venv/lib/python3.10/site-packages/pyannote/audio/sample/sample.wav"
)

# Ground truth from sample.rttm (2 speakers alternating)
GROUND_TRUTH = [
    ("speaker90", 6.690, 7.120),
    ("speaker91", 7.550, 8.350),
    ("speaker90", 8.320, 10.020),
    ("speaker91", 9.920, 11.030),
    ("speaker90", 10.570, 14.700),
    ("speaker91", 14.490, 17.920),
    ("speaker90", 18.050, 21.490),
    ("speaker91", 18.150, 18.590),  # overlap
    ("speaker91", 21.780, 28.500),
    ("speaker90", 27.850, 30.000),
]


def _cosine_sim(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def _load_ecapa():
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


class TestRealSpeechEmbeddings:

    def test_speaker_discrimination(self):
        """Can ECAPA distinguish the two real speakers in the sample?"""
        model, device = _load_ecapa()
        if model is None:
            import pytest; pytest.skip("speechbrain not available")

        import soundfile as sf
        audio, sr = sf.read(SAMPLE_WAV)
        audio_tensor = torch.tensor(audio, dtype=torch.float32)

        # Extract embeddings for each speaker's segments
        speaker_embeddings = {"speaker90": [], "speaker91": []}

        for spk, start, end in GROUND_TRUTH:
            s = int(start * sr)
            e = int(end * sr)
            if e - s < sr * 0.5:  # skip segments < 0.5s
                continue
            seg = audio_tensor[s:e].unsqueeze(0).to(device)
            with torch.no_grad():
                emb = model.encode_batch(seg).squeeze(0).squeeze(0).cpu().numpy()
            speaker_embeddings[spk].append(emb)

        print(f"\n  Speaker90 segments: {len(speaker_embeddings['speaker90'])}")
        print(f"  Speaker91 segments: {len(speaker_embeddings['speaker91'])}")

        # Compute within-speaker and between-speaker similarities
        within_sims = []
        between_sims = []

        embs_90 = speaker_embeddings["speaker90"]
        embs_91 = speaker_embeddings["speaker91"]

        # Within speaker90
        for i in range(len(embs_90)):
            for j in range(i + 1, len(embs_90)):
                within_sims.append(("90↔90", _cosine_sim(embs_90[i], embs_90[j])))

        # Within speaker91
        for i in range(len(embs_91)):
            for j in range(i + 1, len(embs_91)):
                within_sims.append(("91↔91", _cosine_sim(embs_91[i], embs_91[j])))

        # Between speakers
        for e90 in embs_90:
            for e91 in embs_91:
                between_sims.append(("90↔91", _cosine_sim(e90, e91)))

        within_vals = [s[1] for s in within_sims]
        between_vals = [s[1] for s in between_sims]

        print(f"\n  Within-speaker similarity:")
        for label, sim in within_sims:
            print(f"    {label}: {sim:.4f}")
        print(f"    Mean: {np.mean(within_vals):.4f}, Min: {np.min(within_vals):.4f}")

        print(f"\n  Between-speaker similarity:")
        for label, sim in between_sims:
            print(f"    {label}: {sim:.4f}")
        print(f"    Mean: {np.mean(between_vals):.4f}, Max: {np.max(between_vals):.4f}")

        margin = np.min(within_vals) - np.max(between_vals)
        print(f"\n  Discrimination margin (min_within - max_between): {margin:.4f}")
        print(f"  {'✓ SEPARABLE' if margin > 0 else '✗ OVERLAP — threshold tuning needed'}")

    def test_cross_chunk_continuity_with_embeddings(self):
        """Simulate chunked streaming: split the 30s sample into 5s chunks
        and test if embeddings maintain speaker identity across chunks."""
        model, device = _load_ecapa()
        if model is None:
            import pytest; pytest.skip("speechbrain not available")

        import soundfile as sf
        audio, sr = sf.read(SAMPLE_WAV)
        audio_tensor = torch.tensor(audio, dtype=torch.float32)

        from main import _estimate_pitch_safe
        from utils import gender_from_pitch

        CHUNK_SEC = 6.0
        EMB_THRESHOLD = 0.65
        PITCH_THRESHOLD = 30.0

        # Session state
        stored_embeddings = {}
        stored_pitches = {}
        assignment_log = []  # (chunk_idx, true_speaker, assigned_id, method)

        num_chunks = int(len(audio) / sr / CHUNK_SEC)

        print(f"\n  Processing {num_chunks} chunks of {CHUNK_SEC}s each...")
        print(f"  Using ground truth segments for speaker extraction\n")

        for chunk_idx in range(num_chunks):
            chunk_start = chunk_idx * CHUNK_SEC
            chunk_end = chunk_start + CHUNK_SEC

            # Find ground truth segments in this chunk
            chunk_segments = []
            for spk, start, end in GROUND_TRUTH:
                seg_start = max(start, chunk_start)
                seg_end = min(end, chunk_end)
                if seg_end - seg_start >= 0.5:
                    chunk_segments.append((spk, seg_start, seg_end))

            for true_spk, seg_start, seg_end in chunk_segments:
                s = int(seg_start * sr)
                e = int(seg_end * sr)
                seg_audio = audio_tensor[s:e]

                # Extract embedding
                wav = seg_audio.unsqueeze(0).to(device)
                with torch.no_grad():
                    emb = model.encode_batch(wav).squeeze(0).squeeze(0).cpu().numpy()

                # Extract pitch
                audio_np = seg_audio.numpy().astype(np.float32)
                try:
                    pitch, pitch_range = _estimate_pitch_safe(audio_np, sr)
                except:
                    pitch, pitch_range = 150.0, 20.0

                # Try embedding match
                matched = None
                method = "NEW"

                if stored_embeddings:
                    sims = {k: _cosine_sim(emb, v) for k, v in stored_embeddings.items()}
                    sorted_sims = sorted(sims.items(), key=lambda x: x[1], reverse=True)
                    best_k, best_sim = sorted_sims[0]
                    second_sim = sorted_sims[1][1] if len(sorted_sims) > 1 else 0.0

                    if best_sim >= EMB_THRESHOLD and (best_sim - second_sim) >= 0.03:
                        matched = best_k
                        method = f"EMB(sim={best_sim:.3f},margin={best_sim-second_sim:.3f})"

                # Pitch fallback
                if matched is None and stored_pitches:
                    dists = {k: abs(pitch - v) for k, v in stored_pitches.items()}
                    best_k = min(dists, key=dists.get)
                    if dists[best_k] <= PITCH_THRESHOLD:
                        matched = best_k
                        method = f"PITCH(dist={dists[best_k]:.0f}Hz)"

                if matched is None:
                    spk_id = f"SPK_{len(stored_embeddings)}"
                    stored_embeddings[spk_id] = emb
                    stored_pitches[spk_id] = pitch
                    matched = spk_id
                    method = "NEW"
                else:
                    stored_embeddings[matched] = 0.7 * stored_embeddings[matched] + 0.3 * emb

                assignment_log.append((chunk_idx, true_spk, matched, method, pitch))
                print(f"  Chunk {chunk_idx} [{seg_start:.1f}-{seg_end:.1f}s] "
                      f"{true_spk} pitch={pitch:.0f}Hz → {matched} [{method}]")

        # Analyze results
        print(f"\n{'='*70}")
        print(f"  CONTINUITY RESULTS")
        print(f"{'='*70}")

        # Group by true speaker
        by_true = {}
        for chunk_idx, true_spk, assigned, method, pitch in assignment_log:
            by_true.setdefault(true_spk, []).append(assigned)

        for true_spk, assigned_ids in sorted(by_true.items()):
            most_common = max(set(assigned_ids), key=assigned_ids.count)
            consistency = assigned_ids.count(most_common) / len(assigned_ids)
            unique_ids = len(set(assigned_ids))
            print(f"\n  {true_spk}:")
            print(f"    Assignments: {assigned_ids}")
            print(f"    Most common: {most_common} ({consistency:.0%})")
            print(f"    Unique IDs: {unique_ids}")

        # Are the two speakers mapped to different IDs?
        if len(by_true) >= 2:
            spks = list(by_true.keys())
            primary_0 = max(set(by_true[spks[0]]), key=by_true[spks[0]].count)
            primary_1 = max(set(by_true[spks[1]]), key=by_true[spks[1]].count)
            distinct = primary_0 != primary_1
            print(f"\n  Speakers mapped to distinct voices: {distinct} "
                  f"({'✓' if distinct else '✗'})")

        # Count methods used
        methods = [m for _, _, _, m, _ in assignment_log]
        emb_matches = sum(1 for m in methods if m.startswith("EMB"))
        pitch_matches = sum(1 for m in methods if m.startswith("PITCH"))
        new_matches = sum(1 for m in methods if m == "NEW")
        print(f"\n  Match methods: {emb_matches} embedding, "
              f"{pitch_matches} pitch fallback, {new_matches} new")

    def test_pitch_vs_embedding_comparison(self):
        """Run both pitch-only and hybrid matching, compare results."""
        model, device = _load_ecapa()
        if model is None:
            import pytest; pytest.skip("speechbrain not available")

        import soundfile as sf
        audio, sr = sf.read(SAMPLE_WAV)
        audio_tensor = torch.tensor(audio, dtype=torch.float32)
        from main import _estimate_pitch_safe

        # Get long segments (>1s) for reliable comparison
        long_segments = [(spk, s, e) for spk, s, e in GROUND_TRUTH if e - s >= 1.0]

        print(f"\n  Testing {len(long_segments)} segments (>1s each)\n")

        # Extract all embeddings and pitches
        seg_data = []
        for spk, start, end in long_segments:
            s_samp = int(start * sr)
            e_samp = int(end * sr)
            seg = audio_tensor[s_samp:e_samp]

            wav = seg.unsqueeze(0).to(device)
            with torch.no_grad():
                emb = model.encode_batch(wav).squeeze(0).squeeze(0).cpu().numpy()

            audio_np = seg.numpy().astype(np.float32)
            try:
                pitch, _ = _estimate_pitch_safe(audio_np, sr)
            except:
                pitch = 150.0

            seg_data.append({"spk": spk, "start": start, "end": end,
                            "emb": emb, "pitch": pitch})
            print(f"  {spk} [{start:.1f}-{end:.1f}s] pitch={pitch:.0f}Hz")

        # Build similarity matrix
        print(f"\n  Embedding cosine similarity matrix:")
        print(f"  {'':>25}", end="")
        for j, d in enumerate(seg_data):
            print(f"  {d['spk'][-2:]}@{d['start']:.0f}s", end="")
        print()

        for i, di in enumerate(seg_data):
            print(f"  {di['spk']}@{di['start']:4.1f}s  ", end="")
            for j, dj in enumerate(seg_data):
                sim = _cosine_sim(di["emb"], dj["emb"])
                marker = " " if di["spk"] == dj["spk"] else "*"
                print(f"  {sim:.3f}{marker}", end="")
            print()

        print(f"\n  (* = between different speakers)")

        # Pitch differences
        print(f"\n  Pitch comparison:")
        for d in seg_data:
            print(f"    {d['spk']} [{d['start']:.1f}s]: {d['pitch']:.0f}Hz")
