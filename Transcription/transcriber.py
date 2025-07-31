from transformers import WhisperProcessor, WhisperForConditionalGeneration
from dataclasses import dataclass
from typing import List, Dict
import torch
import numpy as np
from tqdm import tqdm


@dataclass
class TranscriptionSegment:
    """Data class for storing transcription segments."""

    start: float
    end: float
    text: str
    confidence: float


class WhisperTranscriber:
    """Class for handling transcription using Whisper model."""

    def __init__(
        self,
        model_size: str = "large-v2",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        language: str = "it",
    ):
        """
        Initialize the Whisper transcriber.

        Args:
            model_size: Size of the Whisper model to use
            device: Device to run the model on (cuda/cpu)
            language: Language code for transcription
        """
        # Initialize model and processor
        model_name = f"openai/whisper-{model_size}"
        self.device = device
        self.language = language

        print(f"Loading Whisper model: {model_name}")
        self.processor = WhisperProcessor.from_pretrained(model_name)
        self.model = WhisperForConditionalGeneration.from_pretrained(model_name).to(
            device
        )

        # Set generation parameters
        self.max_length = 448
        self.chunk_length_s = 30  # Process 30 seconds chunks

    def _process_audio_chunk(self, audio_chunk: np.ndarray) -> Dict:
        """
        Process a single chunk of audio.

        Args:
            audio_chunk: Numpy array of audio samples

        Returns:
            Dictionary containing transcription information
        """
        # Process audio
        input_features = self.processor(
            audio_chunk, sampling_rate=16000, return_tensors="pt"
        ).input_features.to(self.device)

        # Generate token ids
        forced_decoder_ids = self.processor.get_decoder_prompt_ids(
            language=self.language, task="transcribe"
        )

        # Generate predictions
        predicted_ids = self.model.generate(
            input_features,
            forced_decoder_ids=forced_decoder_ids,
            max_length=self.max_length,
        )

        # Decode the output
        transcription = self.processor.batch_decode(
            predicted_ids, skip_special_tokens=True
        )[0]

        return {
            "text": transcription.strip(),
            "confidence": 0.95,  # TODO: Implement proper confidence scoring
        }

    def transcribe(self, audio: np.ndarray) -> List[TranscriptionSegment]:
        """
        Transcribe the given audio.

        Args:
            audio: Numpy array of audio samples (16kHz mono)

        Returns:
            List of TranscriptionSegment objects
        """
        segments = []
        chunk_samples = int(self.chunk_length_s * 16000)

        # Process audio in chunks
        num_chunks = len(audio) // chunk_samples + (
            1 if len(audio) % chunk_samples else 0
        )
        for i in tqdm(range(num_chunks), desc="Transcribing"):
            # Extract chunk
            start_idx = i * chunk_samples
            end_idx = min(start_idx + chunk_samples, len(audio))
            chunk = audio[start_idx:end_idx]

            # Process chunk
            result = self._process_audio_chunk(chunk)

            # Create segment
            start_time = start_idx / 16000  # Convert samples to seconds
            end_time = end_idx / 16000

            segment = TranscriptionSegment(
                start=start_time,
                end=end_time,
                text=result["text"],
                confidence=result["confidence"],
            )

            segments.append(segment)

        return segments
