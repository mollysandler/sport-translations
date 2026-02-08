import main

SpeakerSegment = main.SpeakerSegment

def seg(speaker, start_ms, end_ms):
    # Use the SpeakerSegment class that main imports from diarizer
    return main.SpeakerSegment(
        speaker_id=speaker,
        start_ms=start_ms,
        end_ms=end_ms,
        start_sec=start_ms / 1000.0,
        end_sec=end_ms / 1000.0,
    )


def test_merge_adjacent_turns_merges_same_speaker_with_small_gap():
    t = main.DynamicSpeakerTranslator()

    segments = [
        seg("S1", 0, 1000),
        seg("S1", 1100, 2000),  # 100ms gap
        seg("S2", 2100, 2600),
    ]

    merged = t._merge_adjacent_turns(segments, max_gap_ms=400, min_keep_ms=250)

    assert len(merged) == 2
    assert merged[0].speaker_id == "S1"
    assert merged[0].start_ms == 0
    assert merged[0].end_ms == 2000
    assert merged[1].speaker_id == "S2"


def test_merge_adjacent_turns_absorbs_tiny_turns():
    t = main.DynamicSpeakerTranslator()

    segments = [
        seg("S1", 0, 100),
        seg("S1", 200, 800),
    ]

    merged = t._merge_adjacent_turns(segments, max_gap_ms=400, min_keep_ms=250)

    assert len(merged) == 1
    assert merged[0].speaker_id == "S1"
    assert merged[0].start_ms == 0
    assert merged[0].end_ms == 800



def test_make_exclusive_turns_bounded_overlap():
    t = main.DynamicSpeakerTranslator()

    segments = [
        seg("A", 0, 2000),
        seg("B", 1500, 3000),
    ]

    merge_gap_ms = 120
    turns = t._make_exclusive_turns(
        segments,
        solo_keep_ms=250,
        ignore_interruptions_ms=900,
        min_turn_ms=350,
        merge_gap_ms=merge_gap_ms,
    )

    turns = sorted(turns, key=lambda s: (s.start_ms, s.end_ms))
    for prev, nxt in zip(turns, turns[1:]):
        overlap = prev.end_ms - nxt.start_ms
        assert overlap <= merge_gap_ms + 50  # allow small padding/smoothing
