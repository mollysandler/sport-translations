def get_speaker_for_segment(whisper_start, whisper_end, diarization_segments):
    """
    Find which speaker was most active during this Whisper segment.
    Uses overlap calculation to handle speaker changes mid-sentence.
    """
    max_overlap = 0
    best_speaker = "SPEAKER_00"  # Default fallback
    
    for diag_seg in diarization_segments:
        # Calculate overlap between Whisper segment and diarization segment
        overlap_start = max(whisper_start, diag_seg.start_sec)
        overlap_end = min(whisper_end, diag_seg.end_sec)
        overlap_duration = max(0, overlap_end - overlap_start)
        
        if overlap_duration > max_overlap:
            max_overlap = overlap_duration
            best_speaker = diag_seg.speaker_id
    
    return best_speaker

from v3.main import get_speaker_for_segment
from v3.translator import AccurateTranslator
from diarizer import SpeakerDiarizer

def test():
    # Load previous results (mock or real) to save time, 
    # OR run the full classes again just to print the alignment map
    # ...
    print(f"Segment '{text}' (0.0-5.0) assigned to -> {speaker_id}") 