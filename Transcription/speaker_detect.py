from dataclasses import dataclass
from typing import List
import torch
from pyannote.audio import Pipeline
import numpy as np
from tqdm import tqdm


@dataclass
class SpeakerSegment:
    """Data class for storing speaker segments."""

    start: float
    end: float
    speaker: str
    confidence: float


class SpeakerDiarization:
    """Class for handling speaker diarization using pyannote.audio."""

    def __init__(
        self,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        hf_token: str = None,
    ):
        """
        Initialize the speaker diarization pipeline.

        Args:
            device: Device to run the model on (cuda/cpu)
            hf_token: HuggingFace token for accessing the model (required)

        Raises:
            ValueError: If no HuggingFace token is provided
            RuntimeError: If authentication with the provided token fails
        """
        if not hf_token:
            raise ValueError(
                "HuggingFace token is required. Get it from https://huggingface.co/settings/tokens"
            )

        self.device = device
        print("Loading speaker diarization pipeline...")

        try:
            # Test token validity by trying to get model info first
            from huggingface_hub import HfApi

            api = HfApi()
            api.model_info("pyannote/speaker-diarization", token=hf_token)

            # If successful, initialize the pipeline
            self.pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization",
                use_auth_token=hf_token,
            ).to(device)

        except Exception as e:
            raise RuntimeError(
                "Failed to initialize speaker diarization pipeline. "
                "Please ensure your HuggingFace token is valid and you have accepted "
                "the user agreement at https://huggingface.co/pyannote/speaker-diarization"
            ) from e

    def diarize(
        self, audio: np.ndarray, sample_rate: int = 16000
    ) -> List[SpeakerSegment]:
        """
        Perform speaker diarization on the audio.

        Args:
            audio: Numpy array of audio samples
            sample_rate: Audio sample rate

        Returns:
            List of SpeakerSegment objects
        """
        # Convert numpy array to waveform dictionary format required by pyannote
        waveform = {
            "waveform": torch.from_numpy(audio).unsqueeze(0),
            "sample_rate": sample_rate,
        }

        print("Performing speaker diarization...")
        diarization = self.pipeline(waveform)

        segments = []
        # Process each speaker turn
        for turn, _, speaker in tqdm(
            diarization.itertracks(yield_label=True), desc="Processing speakers"
        ):
            segment = SpeakerSegment(
                start=float(turn.start),
                end=float(turn.end),
                speaker=str(speaker),
                confidence=0.9,  # TODO: Implement proper confidence scoring
            )
            segments.append(segment)

        return segments
