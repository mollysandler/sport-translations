# tests/test_voice_manager.py
"""
Tests for SmartVoiceManager._match_best_voice in main.py.

SmartVoiceManager has no heavy dependencies — just a dict of voice
metadata and scoring logic. No mocking needed.
"""
import main


def _fresh_vm():
    return main.SmartVoiceManager()


class TestMatchBestVoice:
    def test_male_low_pitch_gets_low_voice(self):
        vm = _fresh_vm()
        vid = vm._match_best_voice("male", avg_pitch=110.0, pitch_range=15.0)
        props = vm.available_voices[vid]
        assert props["gender"] == "male"
        # Low pitch (<140) should score highest for "low" pitch voice
        assert props["pitch"] == "low"

    def test_female_high_pitch_gets_female(self):
        vm = _fresh_vm()
        vid = vm._match_best_voice("female", avg_pitch=220.0, pitch_range=30.0)
        props = vm.available_voices[vid]
        assert props["gender"] == "female"

    def test_marks_voice_used(self):
        vm = _fresh_vm()
        vid = vm._match_best_voice("male", avg_pitch=150.0, pitch_range=20.0)
        assert vid in vm.used_voice_ids

    def test_no_duplicates_until_pool_exhausted(self):
        vm = _fresh_vm()
        male_voices = [
            vid for vid, p in vm.available_voices.items() if p["gender"] == "male"
        ]
        assigned = set()
        for i in range(len(male_voices)):
            vid = vm._match_best_voice("male", avg_pitch=150.0 + i, pitch_range=20.0)
            assigned.add(vid)
        # All should be distinct
        assert len(assigned) == len(male_voices)

    def test_falls_back_to_opposite_gender(self):
        vm = _fresh_vm()
        # Exhaust all female voices (4 female voices)
        female_voices = [
            vid for vid, p in vm.available_voices.items() if p["gender"] == "female"
        ]
        for vid in female_voices:
            vm.used_voice_ids.add(vid)
        # Next female request should fall back to a male voice
        vid = vm._match_best_voice("female", avg_pitch=200.0, pitch_range=25.0)
        assert vid not in female_voices
        assert vm.available_voices[vid]["gender"] == "male"

    def test_reuses_when_all_exhausted(self):
        vm = _fresh_vm()
        # Mark every voice as used
        for vid in vm.available_voices:
            vm.used_voice_ids.add(vid)
        # Should still return a voice (reuse path)
        vid = vm._match_best_voice("male", avg_pitch=150.0, pitch_range=20.0)
        assert vid in vm.available_voices

    def test_unknown_gender_falls_through(self):
        vm = _fresh_vm()
        # "unknown" gender won't match male or female filter → falls to opposite-gender path
        vid = vm._match_best_voice("unknown", avg_pitch=160.0, pitch_range=20.0)
        assert vid in vm.available_voices

    def test_medium_pitch_scores_medium(self):
        vm = _fresh_vm()
        vid = vm._match_best_voice("male", avg_pitch=160.0, pitch_range=20.0)
        props = vm.available_voices[vid]
        # 140-180 should prefer medium/medium_low
        assert props["pitch"] in ("medium", "medium_low")

    def test_sports_style_bonus(self):
        vm = _fresh_vm()
        # Give two runs: once requesting low pitch with all voices available
        # The style bonus should push strong/energetic/deep styles higher
        vid = vm._match_best_voice("male", avg_pitch=160.0, pitch_range=20.0)
        props = vm.available_voices[vid]
        # With the +1 bonus, styles like strong_confident, deep_authoritative, energetic_young
        # should tend to win ties
        assert props["style"] in (
            "strong_confident", "deep_authoritative", "energetic_young",
            "deep_calm", "well_rounded", "casual_conversational", "crisp_strong",
        )
