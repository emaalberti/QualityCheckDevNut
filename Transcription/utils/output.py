from typing import List, Dict
from transcriber import TranscriptionSegment
import json


def format_output(transcription: List[TranscriptionSegment]) -> Dict:
    """
    Format the transcription segments into the final output.

    Args:
        transcription: List of transcription segments

    Returns:
        Dictionary containing the formatted output with just the text
    """
    # Combine all text segments
    full_text = ""

    for trans_seg in transcription:
        # Add a space between segments
        if full_text and not full_text.endswith((".", "!", "?")):
            full_text += " "
        full_text += trans_seg.text

    # Create final output
    output = {
        "transcription": full_text,
        "metadata": {"num_segments": len(transcription)},
    }

    return output
