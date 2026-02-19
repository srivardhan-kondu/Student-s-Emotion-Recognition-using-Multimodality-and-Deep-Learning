"""
Voice Recorder Utility for Real-time Speech Emotion Analysis.
Captures microphone audio and returns it as a WAV file or numpy array.
Used by the Streamlit dashboard for live voice input.
"""

import numpy as np
import tempfile
import os
import logging
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


class VoiceRecorder:
    """
    Records audio from the system microphone using sounddevice.
    Saves to a temporary WAV file for downstream processing.
    """

    def __init__(self, sample_rate: int = 22050, channels: int = 1):
        """
        Args:
            sample_rate: Recording sample rate in Hz (22050 matches librosa default)
            channels: Number of audio channels (1 = mono)
        """
        self.sample_rate = sample_rate
        self.channels = channels
        self._recording = None
        logger.info(f"VoiceRecorder initialized: {sample_rate}Hz, {channels}ch")

    def record(self, duration: float = 5.0) -> Tuple[np.ndarray, str]:
        """
        Record audio from the default microphone.

        Args:
            duration: Recording duration in seconds

        Returns:
            Tuple of (audio_array, temp_wav_path)
            audio_array: float32 numpy array (samples,)
            temp_wav_path: Path to saved WAV file

        Raises:
            RuntimeError: If sounddevice is not available
        """
        try:
            import sounddevice as sd
        except ImportError:
            raise RuntimeError(
                "sounddevice not installed. "
                "Run: pip install sounddevice"
            )

        logger.info(f"Recording {duration:.1f}s at {self.sample_rate}Hz...")

        audio = sd.rec(
            int(duration * self.sample_rate),
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype='float32'
        )
        sd.wait()   # block until done

        # Flatten to 1D mono
        audio_mono = audio.flatten()
        self._recording = audio_mono

        # Save to temp WAV
        wav_path = self._save_wav(audio_mono)
        logger.info(f"Recording saved: {wav_path}")
        return audio_mono, wav_path

    def _save_wav(self, audio: np.ndarray) -> str:
        """Save float32 numpy array to a temporary WAV file."""
        try:
            import scipy.io.wavfile as wav

            # scipy expects int16 or float32 — we use float32
            tmp = tempfile.NamedTemporaryFile(
                suffix='.wav', delete=False, prefix='emotion_rec_'
            )
            tmp.close()

            # Normalize to prevent clipping
            max_val = np.max(np.abs(audio))
            if max_val > 0:
                audio_normalized = audio / max_val * 0.95
            else:
                audio_normalized = audio

            wav.write(tmp.name, self.sample_rate, audio_normalized.astype('float32'))
            return tmp.name

        except ImportError:
            # Fallback: write raw numpy file if scipy missing
            tmp = tempfile.NamedTemporaryFile(
                suffix='.npy', delete=False, prefix='emotion_rec_'
            )
            tmp.close()
            np.save(tmp.name, audio)
            logger.warning("scipy not found — saved as .npy (wav unavailable)")
            return tmp.name

    def get_last_recording(self) -> Optional[np.ndarray]:
        """Return the last recorded audio array."""
        return self._recording

    def cleanup(self, wav_path: str):
        """Delete a temporary WAV file."""
        try:
            if os.path.exists(wav_path):
                os.remove(wav_path)
                logger.debug(f"Deleted temp file: {wav_path}")
        except Exception as e:
            logger.warning(f"Could not delete {wav_path}: {e}")

    @staticmethod
    def list_devices():
        """List available audio input devices."""
        try:
            import sounddevice as sd
            return sd.query_devices()
        except ImportError:
            return "sounddevice not installed"

    @staticmethod
    def is_available() -> bool:
        """Check if sounddevice is installed and a microphone is accessible."""
        try:
            import sounddevice as sd
            devices = sd.query_devices()
            # Check for at least one input device
            input_devices = [d for d in devices if d['max_input_channels'] > 0]
            return len(input_devices) > 0
        except Exception:
            return False
