#!/usr/bin/env python3
"""
Stress test: run silence-aware chunking on real audio streams.

Downloads audio from live streams, simulates the streaming chunker,
and checks whether splits land mid-word by measuring energy around
each split boundary.

A "word split" is flagged when the 200ms window centered on the split
point has high RMS (> 50% of the stream's overall RMS), meaning we
cut through active speech.
"""

import subprocess
import sys
import time
import numpy as np

sys.path.insert(0, ".")
from utils import find_silence_split

SAMPLE_RATE = 16000
MIN_CHUNK_SEC = 5.0
MAX_CHUNK_SEC = 20.0
MIN_CHUNK_SAMPLES = int(SAMPLE_RATE * MIN_CHUNK_SEC)
MAX_CHUNK_SAMPLES = int(SAMPLE_RATE * MAX_CHUNK_SEC)
CAPTURE_SECONDS = 120
ROUNDS = 3

# Wider set of streams — different languages, content types
STREAMS = [
    ("BBC World Service (EN)",   "https://stream.live.vc.bbcmedia.co.uk/bbc_world_service"),
    ("France Info (FR)",         "https://stream.radiofrance.fr/franceinfo/franceinfo_hifi.m3u8"),
    ("Deutsche Welle (DE)",      "https://rbmn-live.akamaized.net/hls/live/590198/dwstream5/index.m3u8"),
    ("BBC 5 Live Sports (EN)",   "http://a.files.bbci.co.uk/media/live/manifesto/audio/simulcast/hls/uk/sbr_high/ak/bbc_radio_five_live.m3u8"),
    ("BBC 5 Sports Extra (EN)",  "http://a.files.bbci.co.uk/media/live/manifesto/audio/simulcast/hls/uk/sbr_high/ak/bbc_radio_five_live_sports_extra.m3u8"),
    ("NPR News (EN)",            "https://npr-ice.streamguys1.com/live.mp3"),
    ("RNE Radio Nacional (ES)",  "https://rtvelivestream.akamaized.net/rne_r1_main.mp3"),
    ("NHK World Radio (JP)",     "https://nhkworld.webcdn.stream.ne.jp/www11/nhkworld/def/live/radio/r-news/a.m3u8"),
    ("RAI Radio 1 (IT)",         "https://icestreaming.rai.it/1.mp3"),
    ("CBC Radio One (EN)",       "https://cbcliveradio-lh.akamaihd.net/i/CBCR1_TOR@118420/master.m3u8"),
]


def download_audio(url: str, duration: int = CAPTURE_SECONDS) -> np.ndarray | None:
    """Download `duration` seconds of audio from a URL as float32 PCM."""
    cmd = [
        "ffmpeg",
        "-i", url,
        "-ac", "1",
        "-ar", str(SAMPLE_RATE),
        "-f", "s16le",
        "-t", str(duration),
        "-v", "error",
        "pipe:1",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=duration + 30)
        if result.returncode != 0 or len(result.stdout) < SAMPLE_RATE * 2:
            return None
        samples = np.frombuffer(result.stdout, dtype=np.int16)
        return samples.astype(np.float32) / 32768.0
    except (subprocess.TimeoutExpired, Exception) as e:
        print(f"      Download failed: {e}")
        return None


def simulate_streaming(samples: np.ndarray, overall_rms: float):
    """Simulate incremental chunking. Returns list of split-info dicts."""
    results = []
    offset = 0
    word_split_threshold = overall_rms * 0.5  # flag if boundary > 50% of overall

    while offset < len(samples):
        remaining = samples[offset:]
        if len(remaining) < MIN_CHUNK_SAMPLES:
            if len(remaining) >= SAMPLE_RATE:
                results.append({
                    "offset_sec": offset / SAMPLE_RATE,
                    "chunk_sec": len(remaining) / SAMPLE_RATE,
                    "split_type": "tail",
                    "boundary_rms": 0.0,
                    "word_split": False,
                    "scan_ms": 0.0,
                })
            break

        t0 = time.perf_counter()
        idx = find_silence_split(remaining, SAMPLE_RATE, MIN_CHUNK_SAMPLES, MAX_CHUNK_SAMPLES)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        if idx > 0:
            split_type = "silence"
            chunk_end = idx
        elif len(remaining) >= MAX_CHUNK_SAMPLES:
            split_type = "HARD"
            chunk_end = MAX_CHUNK_SAMPLES
        else:
            split_type = "tail"
            chunk_end = len(remaining)

        # Measure RMS in 100ms window BEFORE the split point
        # (the region that should be silent if we split correctly)
        pre_window = int(0.1 * SAMPLE_RATE)  # 100ms before split
        abs_split = offset + chunk_end
        w_start = max(0, abs_split - pre_window)
        boundary_window = samples[w_start:abs_split]
        boundary_rms = float(np.sqrt(np.mean(boundary_window ** 2))) if len(boundary_window) > 0 else 0.0

        is_word_split = (
            split_type != "tail"
            and boundary_rms > word_split_threshold
        )

        results.append({
            "offset_sec": offset / SAMPLE_RATE,
            "chunk_sec": chunk_end / SAMPLE_RATE,
            "split_type": split_type,
            "boundary_rms": boundary_rms,
            "word_split": is_word_split,
            "scan_ms": elapsed_ms,
        })
        offset += chunk_end

    return results


def run_one_capture(name: str, url: str, round_num: int):
    """Run a single capture + analysis. Returns summary dict or None."""
    print(f"    Round {round_num}: downloading {CAPTURE_SECONDS}s...")
    samples = download_audio(url)
    if samples is None:
        print(f"    Round {round_num}: FAILED to download")
        return None

    duration = len(samples) / SAMPLE_RATE
    overall_rms = float(np.sqrt(np.mean(samples ** 2)))
    print(f"    Round {round_num}: got {duration:.1f}s, RMS={overall_rms:.4f}")

    results = simulate_streaming(samples, overall_rms)

    non_tail = [r for r in results if r["split_type"] != "tail"]
    silence_splits = [r for r in results if r["split_type"] == "silence"]
    hard_splits = [r for r in results if r["split_type"] == "HARD"]
    word_splits = [r for r in non_tail if r["word_split"]]
    chunk_durations = [r["chunk_sec"] for r in results]
    boundary_rms_vals = [r["boundary_rms"] for r in non_tail]
    scan_times = [r["scan_ms"] for r in non_tail]

    # Per-chunk detail
    for i, r in enumerate(results):
        flag = ""
        if r["word_split"]:
            flag = " !! WORD SPLIT"
        elif r["split_type"] == "HARD":
            flag = " ** HARD"
        print(f"      chunk {i}: {r['offset_sec']:6.1f}s + {r['chunk_sec']:5.1f}s  "
              f"[{r['split_type']:7s}]  bnd_rms={r['boundary_rms']:.4f}  "
              f"scan={r['scan_ms']:.2f}ms{flag}")

    n_non_tail = len(non_tail)
    return {
        "duration": duration,
        "n_chunks": len(results),
        "n_silence": len(silence_splits),
        "n_hard": len(hard_splits),
        "n_word_splits": len(word_splits),
        "n_non_tail": n_non_tail,
        "silence_pct": len(silence_splits) / max(1, n_non_tail) * 100,
        "hard_pct": len(hard_splits) / max(1, n_non_tail) * 100,
        "word_split_pct": len(word_splits) / max(1, n_non_tail) * 100,
        "mean_boundary_rms": np.mean(boundary_rms_vals) if boundary_rms_vals else 0,
        "max_boundary_rms": max(boundary_rms_vals) if boundary_rms_vals else 0,
        "overall_rms": overall_rms,
        "mean_chunk_sec": np.mean(chunk_durations),
        "min_chunk_sec": min(chunk_durations),
        "max_chunk_sec": max(chunk_durations),
        "median_scan_ms": np.median(scan_times) if scan_times else 0,
        "max_scan_ms": max(scan_times) if scan_times else 0,
    }


def run_stream_test(name: str, url: str):
    """Run multiple rounds for one stream."""
    print(f"\n{'='*64}")
    print(f"  {name}")
    print(f"  {url}")
    print(f"{'='*64}")

    round_results = []
    for r in range(1, ROUNDS + 1):
        result = run_one_capture(name, url, r)
        if result:
            round_results.append(result)
        else:
            # If first round fails, skip remaining rounds
            if r == 1:
                print("  SKIPPED — stream unreachable")
                return None

    if not round_results:
        return None

    # Aggregate across rounds
    total_splits = sum(r["n_non_tail"] for r in round_results)
    total_silence = sum(r["n_silence"] for r in round_results)
    total_hard = sum(r["n_hard"] for r in round_results)
    total_word_splits = sum(r["n_word_splits"] for r in round_results)
    total_audio_sec = sum(r["duration"] for r in round_results)

    agg = {
        "name": name,
        "rounds": len(round_results),
        "total_audio_sec": total_audio_sec,
        "total_splits": total_splits,
        "total_silence": total_silence,
        "total_hard": total_hard,
        "total_word_splits": total_word_splits,
        "silence_pct": total_silence / max(1, total_splits) * 100,
        "hard_pct": total_hard / max(1, total_splits) * 100,
        "word_split_pct": total_word_splits / max(1, total_splits) * 100,
        "mean_boundary_rms": np.mean([r["mean_boundary_rms"] for r in round_results]),
        "overall_rms": np.mean([r["overall_rms"] for r in round_results]),
        "median_scan_ms": np.median([r["median_scan_ms"] for r in round_results]),
    }

    print(f"\n  --- {name} aggregate ({len(round_results)} rounds, {total_audio_sec:.0f}s audio) ---")
    print(f"  Total splits:   {total_splits}")
    print(f"  Silence splits: {total_silence} ({agg['silence_pct']:.0f}%)")
    print(f"  Hard splits:    {total_hard} ({agg['hard_pct']:.0f}%)")
    print(f"  Word splits:    {total_word_splits} ({agg['word_split_pct']:.0f}%) "
          f"{'  !! PROBLEM' if total_word_splits > 0 else '  OK'}")
    ratio = agg["mean_boundary_rms"] / agg["overall_rms"] if agg["overall_rms"] > 0 else 0
    print(f"  Boundary/overall RMS: {ratio:.2f}x")
    print(f"  Scan latency median:  {agg['median_scan_ms']:.3f}ms")

    return agg


def main():
    print("=" * 64)
    print("  Silence-Aware Chunk Splitting — Extended Stress Test")
    print("=" * 64)
    print(f"  Config: min={MIN_CHUNK_SEC}s, max={MAX_CHUNK_SEC}s, sr={SAMPLE_RATE}")
    print(f"  Capture: {CAPTURE_SECONDS}s x {ROUNDS} rounds per stream")
    print(f"  Streams: {len(STREAMS)}")
    print("  Word-split threshold: boundary RMS > 50% of overall RMS")

    summaries = []
    for name, url in STREAMS:
        summary = run_stream_test(name, url)
        if summary:
            summaries.append(summary)

    # Final report
    print(f"\n\n{'='*64}")
    print("  FINAL REPORT")
    print(f"{'='*64}")

    if not summaries:
        print("  No streams were reachable. Check network connectivity.")
        return

    grand_total_splits = sum(s["total_splits"] for s in summaries)
    grand_total_silence = sum(s["total_silence"] for s in summaries)
    grand_total_hard = sum(s["total_hard"] for s in summaries)
    grand_total_word_splits = sum(s["total_word_splits"] for s in summaries)
    grand_total_audio = sum(s["total_audio_sec"] for s in summaries)

    print(f"\n  Streams tested:    {len(summaries)}")
    print(f"  Total audio:       {grand_total_audio:.0f}s ({grand_total_audio/60:.1f} min)")
    print(f"  Total splits:      {grand_total_splits}")
    print()

    header = f"  {'Stream':30s}  {'Splits':>6s}  {'Silence':>8s}  {'Hard':>6s}  {'WordCut':>8s}  {'Bnd/RMS':>8s}"
    print(header)
    print(f"  {'-'*30}  {'-'*6}  {'-'*8}  {'-'*6}  {'-'*8}  {'-'*8}")
    for s in summaries:
        ratio = s["mean_boundary_rms"] / s["overall_rms"] if s["overall_rms"] > 0 else 0
        wc_marker = "!!" if s["total_word_splits"] > 0 else "ok"
        print(f"  {s['name']:30s}  {s['total_splits']:6d}  "
              f"{s['silence_pct']:7.0f}%  {s['hard_pct']:5.0f}%  "
              f"{s['total_word_splits']:3d} {wc_marker:>3s}  "
              f"{ratio:7.2f}x")

    print()
    if grand_total_splits > 0:
        print(f"  Overall silence rate:    {grand_total_silence/grand_total_splits*100:.1f}%")
        print(f"  Overall hard split rate: {grand_total_hard/grand_total_splits*100:.1f}%")
        print(f"  Overall word-cut rate:   {grand_total_word_splits/grand_total_splits*100:.1f}%")

    print()
    if grand_total_word_splits == 0:
        print("  PASS: Zero word splits detected across all streams and rounds.")
    elif grand_total_word_splits <= 2:
        print(f"  WARN: {grand_total_word_splits} possible word split(s) — review the flagged chunks above.")
    else:
        print(f"  FAIL: {grand_total_word_splits} word splits detected — chunker needs tuning.")


if __name__ == "__main__":
    main()
