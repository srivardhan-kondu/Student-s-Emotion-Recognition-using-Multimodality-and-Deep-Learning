"""
Audio Feature Extraction for Speech Emotion Recognition — Upgraded.
Key improvements:
  - Delta + delta-delta MFCCs (rate-of-change features)
  - Raw MFCC time-series extraction (not flattened stats)
  - Audio augmentation: noise, pitch shift, time stretch
  - Mel + MFCC combined feature matrix for Attention-BiLSTM
"""

import librosa
import librosa.feature
import numpy as np
from typing import Tuple, Dict, Optional
import logging
import os

logger = logging.getLogger(__name__)

# ─── Constants ─────────────────────────────────────────────────────────────── #
N_MFCC = 40          # base MFCC coefficients
N_MELS = 128         # mel filterbanks
TARGET_SR = 22050    # sample rate
MAX_TIME_STEPS = 216 # ~5 seconds at hop_length=512
FEATURE_DIM = N_MFCC * 3  # MFCC + delta + delta-delta = 120


class AudioFeatureExtractor:
    """
    Extract audio features for emotion recognition.
    Supports both flattened stat vectors (legacy) and
    raw time-series matrices (for Attention-BiLSTM).
    """

    def __init__(self, sample_rate: int = TARGET_SR):
        self.sample_rate = sample_rate

    # ------------------------------------------------------------------ #
    #  Loading
    # ------------------------------------------------------------------ #

    def load_audio(self, audio_path: str) -> Tuple[np.ndarray, int]:
        """Load and resample audio file."""
        audio, sr = librosa.load(audio_path, sr=self.sample_rate, mono=True)
        logger.debug(f"Loaded {audio_path}: {len(audio)/sr:.2f}s")
        return audio, sr

    # ------------------------------------------------------------------ #
    #  Individual feature extractors
    # ------------------------------------------------------------------ #

    def extract_mfcc(self, audio: np.ndarray, n_mfcc: int = N_MFCC) -> np.ndarray:
        """Extract MFCC features (n_mfcc x T)."""
        return librosa.feature.mfcc(y=audio, sr=self.sample_rate, n_mfcc=n_mfcc)

    def extract_delta_mfcc(self, mfcc: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute delta and delta-delta MFCCs.
        Returns (delta, delta2) each of shape (n_mfcc x T).
        """
        delta = librosa.feature.delta(mfcc, order=1)
        delta2 = librosa.feature.delta(mfcc, order=2)
        return delta, delta2

    def extract_mel_spectrogram(self, audio: np.ndarray) -> np.ndarray:
        """Extract log-mel spectrogram (n_mels x T)."""
        mel = librosa.feature.melspectrogram(
            y=audio, sr=self.sample_rate, n_mels=N_MELS
        )
        return librosa.power_to_db(mel, ref=np.max)

    def extract_chroma(self, audio: np.ndarray) -> np.ndarray:
        """Extract chroma features (12 x T)."""
        return librosa.feature.chroma_stft(y=audio, sr=self.sample_rate)

    def extract_spectral_contrast(self, audio: np.ndarray) -> np.ndarray:
        """Extract spectral contrast (7 x T)."""
        return librosa.feature.spectral_contrast(y=audio, sr=self.sample_rate)

    def extract_zero_crossing_rate(self, audio: np.ndarray) -> np.ndarray:
        """Extract zero crossing rate (1 x T)."""
        return librosa.feature.zero_crossing_rate(audio)

    def extract_pitch(self, audio: np.ndarray) -> np.ndarray:
        """Extract fundamental frequency (pitch) using YIN."""
        return librosa.yin(
            audio,
            fmin=librosa.note_to_hz('C2'),
            fmax=librosa.note_to_hz('C7'),
            sr=self.sample_rate
        )

    # ------------------------------------------------------------------ #
    #  NEW: Raw time-series MFCC matrix (for Attention-BiLSTM)
    # ------------------------------------------------------------------ #

    def extract_mfcc_sequence(self, audio: np.ndarray,
                               max_steps: int = MAX_TIME_STEPS) -> np.ndarray:
        """
        Extract a 2D MFCC+delta+delta2 matrix suitable for sequential models.

        Output shape: (max_steps, FEATURE_DIM) where FEATURE_DIM = N_MFCC * 3 = 120
        - Columns 0:40    → MFCC
        - Columns 40:80   → delta MFCC (velocity)
        - Columns 80:120  → delta-delta MFCC (acceleration)

        Sequences are zero-padded or truncated to max_steps.

        Args:
            audio: Audio signal
            max_steps: Fixed number of time frames

        Returns:
            (max_steps, FEATURE_DIM) float32 array
        """
        mfcc = self.extract_mfcc(audio)            # (40, T)
        delta, delta2 = self.extract_delta_mfcc(mfcc)  # (40, T) each

        # Stack along feature axis → (120, T) then transpose → (T, 120)
        combined = np.vstack([mfcc, delta, delta2]).T   # (T, 120)

        T = combined.shape[0]
        if T < max_steps:
            # Zero-pad to max_steps
            pad = np.zeros((max_steps - T, FEATURE_DIM), dtype='float32')
            combined = np.vstack([combined, pad])
        else:
            # Truncate
            combined = combined[:max_steps]

        # Normalise each feature dimension
        mean = combined.mean(axis=0, keepdims=True)
        std = combined.std(axis=0, keepdims=True) + 1e-8
        combined = (combined - mean) / std

        return combined.astype('float32')

    def extract_feature_vector_from_sequence(self, audio: np.ndarray) -> np.ndarray:
        """
        Legacy: flatten sequence stats into a 1D vector.
        Kept for backward compatibility with old models.
        """
        mfcc = self.extract_mfcc(audio)
        mel = self.extract_mel_spectrogram(audio)
        chroma = self.extract_chroma(audio)
        contrast = self.extract_spectral_contrast(audio)
        zcr = self.extract_zero_crossing_rate(audio)

        def stats(x):
            return np.array([
                np.mean(x, axis=1),
                np.std(x, axis=1),
                np.min(x, axis=1),
                np.max(x, axis=1),
            ]).flatten()

        return np.concatenate([
            stats(mfcc), stats(mel), stats(chroma), stats(contrast), stats(zcr)
        ])

    # ------------------------------------------------------------------ #
    #  Audio augmentation
    # ------------------------------------------------------------------ #

    def augment_audio(self, audio: np.ndarray,
                      sr: Optional[int] = None,
                      noise_factor: float = 0.005,
                      pitch_steps: int = 2,
                      time_stretch_rate: float = 0.9) -> list:
        """
        Generate augmented versions of an audio clip.

        Returns a list of (augmented_audio, sr) tuples:
          [0] Original
          [1] + Gaussian noise
          [2] + Pitch shift up
          [3] + Time stretch (slower)
          [4] + Pitch shift down
        """
        sr = sr or self.sample_rate
        augmented = [audio]  # original

        # Gaussian noise
        noise = noise_factor * np.random.randn(len(audio))
        augmented.append(audio + noise)

        # Pitch shift up
        try:
            augmented.append(librosa.effects.pitch_shift(
                audio, sr=sr, n_steps=pitch_steps
            ))
        except Exception:
            augmented.append(audio)

        # Time stretch (slower)
        try:
            stretched = librosa.effects.time_stretch(audio, rate=time_stretch_rate)
            # Pad/trim to original length
            if len(stretched) < len(audio):
                stretched = np.pad(stretched, (0, len(audio) - len(stretched)))
            else:
                stretched = stretched[:len(audio)]
            augmented.append(stretched)
        except Exception:
            augmented.append(audio)

        # Pitch shift down
        try:
            augmented.append(librosa.effects.pitch_shift(
                audio, sr=sr, n_steps=-pitch_steps
            ))
        except Exception:
            augmented.append(audio)

        return augmented

    # ------------------------------------------------------------------ #
    #  High-level: extract sequence from file
    # ------------------------------------------------------------------ #

    def extract_sequence_from_file(self, audio_path: str,
                                    max_steps: int = MAX_TIME_STEPS) -> np.ndarray:
        """
        Load audio file and return MFCC+delta+delta2 sequence.

        Args:
            audio_path: Path to audio file
            max_steps: Number of time frames

        Returns:
            (max_steps, FEATURE_DIM) float32 array
        """
        audio, _ = self.load_audio(audio_path)
        return self.extract_mfcc_sequence(audio, max_steps=max_steps)

    def extract_feature_vector(self, audio_path: str) -> np.ndarray:
        """Legacy flat feature vector from file path (for backward compatibility)."""
        audio, _ = self.load_audio(audio_path)
        return self.extract_feature_vector_from_sequence(audio)

    def compute_statistics(self, features: np.ndarray) -> np.ndarray:
        """Legacy stat aggregation (for backward compatibility)."""
        return np.array([
            np.mean(features, axis=1),
            np.std(features, axis=1),
            np.min(features, axis=1),
            np.max(features, axis=1),
        ]).flatten()
