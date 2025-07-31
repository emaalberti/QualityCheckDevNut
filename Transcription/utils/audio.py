import numpy as np
import librosa
import soundfile as sf
import io
from typing import Union, BinaryIO
from pathlib import Path


class AudioProcessor:
    """Class for handling audio file processing."""

    def __init__(self):
        """Initialize the audio processor."""
        self.target_sample_rate = 16000

    def process(self, audio_path: Union[str, Path, BinaryIO]) -> np.ndarray:
        """
        Process audio file and convert it to the required format.

        Args:
            audio_path: Path to the audio file or file-like object

        Returns:
            Numpy array of processed audio samples
        """
        print("Processing audio file...")

        # Load audio file with librosa
        if isinstance(audio_path, (str, Path)):
            # librosa gestisce automaticamente la conversione di formato
            audio, original_sr = librosa.load(str(audio_path), sr=None)
        else:
            # Per file-like objects, salva temporaneamente
            import tempfile
            import os
            
            with tempfile.NamedTemporaryFile(delete=False, suffix='.tmp') as tmp_file:
                tmp_file.write(audio_path.read())
                tmp_file.flush()
                audio, original_sr = librosa.load(tmp_file.name, sr=None)
            
            # Pulisci il file temporaneo
            os.unlink(tmp_file.name)

        # Convert to mono if stereo (librosa carica già in mono di default)
        if len(audio.shape) > 1:
            print("Converting to mono...")
            audio = librosa.to_mono(audio)

        # Convert sample rate
        if original_sr != self.target_sample_rate:
            print(f"Converting sample rate from {original_sr}Hz to {self.target_sample_rate}Hz...")
            audio = librosa.resample(audio, orig_sr=original_sr, target_sr=self.target_sample_rate)

        # librosa restituisce già float32 normalizzato tra -1 e 1
        samples = audio.astype(np.float32)

        print(f"Audio processed: {len(samples)/self.target_sample_rate:.2f} seconds")
        return samples

    @staticmethod
    def get_duration(audio: np.ndarray, sample_rate: int = 16000) -> float:
        """
        Get the duration of the audio in seconds.

        Args:
            audio: Numpy array of audio samples
            sample_rate: Audio sample rate

        Returns:
            Duration in seconds
        """
        return len(audio) / sample_rate