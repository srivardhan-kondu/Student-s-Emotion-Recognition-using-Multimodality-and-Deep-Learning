"""
Multimodal Fusion Engine — Upgraded (FR13, FR14).
Key improvements:
  - Temperature scaling per modality for calibrated probabilities
  - Confidence-gated adaptive fusion (threshold-based)
  - Calibrated weighted fusion as default
  - Dynamic weight updates based on historical accuracy
  - All legacy strategies (weighted, voting, adaptive) kept intact
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

EMOTIONS = ['happy', 'sad', 'angry', 'neutral', 'fear', 'surprise']

# Per-modality calibration temperatures (tuned empirically)
DEFAULT_TEMPERATURES = {
    'facial': 1.5,
    'speech': 1.3,
    'text': 1.2
}

# Minimum confidence to include a modality in fusion
CONFIDENCE_THRESHOLD = 0.30


class MultimodalFusion:
    """
    Multimodal fusion for combining emotion predictions from multiple modalities.
    Supports: calibrated_weighted (default), weighted, adaptive, voting.
    """

    def __init__(self, fusion_type: str = 'calibrated',
                 facial_weight: float = 0.40,
                 speech_weight: float = 0.30,
                 text_weight: float = 0.30):
        """
        Initialize multimodal fusion.

        Args:
            fusion_type: Strategy: 'calibrated' | 'weighted' | 'adaptive' | 'voting'
            facial_weight: Base weight for facial modality
            speech_weight: Base weight for speech modality
            text_weight: Base weight for text modality
        """
        self.fusion_type = fusion_type
        self.facial_weight = facial_weight
        self.speech_weight = speech_weight
        self.text_weight = text_weight
        self.temperatures = DEFAULT_TEMPERATURES.copy()

        # Normalize base weights
        total = facial_weight + speech_weight + text_weight
        self.facial_weight /= total
        self.speech_weight /= total
        self.text_weight /= total

        # Historical accuracy tracker for dynamic weight updates
        self._correct_counts = {'facial': 0, 'speech': 0, 'text': 0}
        self._total_counts = {'facial': 0, 'speech': 0, 'text': 0}

        logger.info(
            f"Initialized {fusion_type} fusion | "
            f"weights: facial={self.facial_weight:.2f}, "
            f"speech={self.speech_weight:.2f}, text={self.text_weight:.2f}"
        )

    # ------------------------------------------------------------------ #
    #  Temperature scaling (NEW)
    # ------------------------------------------------------------------ #

    def temperature_scale(self, probs: np.ndarray, T: float) -> np.ndarray:
        """
        Apply temperature scaling to calibrate raw softmax probabilities.
        T > 1 softens (reduces overconfidence), T < 1 sharpens.

        Args:
            probs: Raw softmax probabilities (n_classes,)
            T: Temperature parameter

        Returns:
            Calibrated probability distribution
        """
        log_probs = np.log(probs + 1e-9) / T
        exp_vals = np.exp(log_probs - np.max(log_probs))
        return exp_vals / exp_vals.sum()

    def calibrate(self, probs: np.ndarray, modality: str) -> np.ndarray:
        """Calibrate a modality's probabilities using its temperature."""
        T = self.temperatures.get(modality, 1.0)
        return self.temperature_scale(probs, T)

    def set_temperatures(self, facial: float = 1.5, speech: float = 1.3,
                         text: float = 1.2):
        """Update calibration temperatures."""
        self.temperatures = {'facial': facial, 'speech': speech, 'text': text}
        logger.info(f"Temperatures updated: {self.temperatures}")

    # ------------------------------------------------------------------ #
    #  Calibrated weighted fusion (NEW — default best strategy)
    # ------------------------------------------------------------------ #

    def calibrated_fusion(self, facial_probs: Optional[np.ndarray],
                           speech_probs: Optional[np.ndarray],
                           text_probs: Optional[np.ndarray]) -> np.ndarray:
        """
        Weighted fusion with:
          1. Per-modality temperature scaling (calibration)
          2. Confidence gating (skip low-confidence modalities)
          3. Dynamic re-weighting based on available modalities

        Args:
            facial_probs: Facial emotion probabilities (6,) or None
            speech_probs: Speech emotion probabilities (6,) or None
            text_probs: Text emotion probabilities (6,) or None

        Returns:
            Fused emotion probabilities (6,)
        """
        modalities = [
            ('facial', facial_probs, self.facial_weight),
            ('speech', speech_probs, self.speech_weight),
            ('text', text_probs, self.text_weight),
        ]

        fused = np.zeros(6, dtype=np.float64)
        total_weight = 0.0

        for name, probs, base_weight in modalities:
            if probs is None:
                continue
            # Calibrate
            cal_probs = self.calibrate(probs, name)
            # Confidence = max probability after calibration
            confidence = float(np.max(cal_probs))
            # Skip low-confidence modalities
            if confidence < CONFIDENCE_THRESHOLD:
                logger.debug(f"Skipping {name} (confidence={confidence:.2f} < {CONFIDENCE_THRESHOLD})")
                continue
            # Confidence-scaled weight
            effective_weight = base_weight * confidence
            fused += effective_weight * cal_probs
            total_weight += effective_weight

        if total_weight == 0:
            logger.warning("All modalities below confidence threshold. Using uniform distribution.")
            return np.ones(6) / 6

        fused /= total_weight
        return fused

    # ------------------------------------------------------------------ #
    #  Original strategies (kept intact)
    # ------------------------------------------------------------------ #

    def weighted_fusion(self, facial_probs: Optional[np.ndarray],
                        speech_probs: Optional[np.ndarray],
                        text_probs: Optional[np.ndarray]) -> np.ndarray:
        """Simple linear weighted fusion."""
        fused = np.zeros(6)
        total_weight = 0.0

        if facial_probs is not None:
            fused += self.facial_weight * facial_probs
            total_weight += self.facial_weight
        if speech_probs is not None:
            fused += self.speech_weight * speech_probs
            total_weight += self.speech_weight
        if text_probs is not None:
            fused += self.text_weight * text_probs
            total_weight += self.text_weight

        return fused / total_weight if total_weight > 0 else np.ones(6) / 6

    def adaptive_fusion(self, facial_probs: Optional[np.ndarray],
                        speech_probs: Optional[np.ndarray],
                        text_probs: Optional[np.ndarray],
                        facial_conf: float = 0.5,
                        speech_conf: float = 0.5,
                        text_conf: float = 0.5) -> np.ndarray:
        """Adaptive fusion using per-modality confidence scores."""
        modalities = []
        if facial_probs is not None:
            modalities.append((facial_probs, facial_conf * self.facial_weight))
        if speech_probs is not None:
            modalities.append((speech_probs, speech_conf * self.speech_weight))
        if text_probs is not None:
            modalities.append((text_probs, text_conf * self.text_weight))

        if not modalities:
            return np.ones(6) / 6

        fused = np.zeros(6)
        total_w = sum(w for _, w in modalities)
        for probs, w in modalities:
            fused += (w / total_w) * probs
        return fused

    def voting_fusion(self, facial_probs: Optional[np.ndarray],
                      speech_probs: Optional[np.ndarray],
                      text_probs: Optional[np.ndarray]) -> np.ndarray:
        """Majority voting fusion — returns soft vote distribution."""
        votes = np.zeros(6)
        for probs in [facial_probs, speech_probs, text_probs]:
            if probs is not None:
                votes[np.argmax(probs)] += 1
        total = votes.sum()
        return votes / total if total > 0 else np.ones(6) / 6

    # ------------------------------------------------------------------ #
    #  Main fusion entry point
    # ------------------------------------------------------------------ #

    def fuse(self, facial_result: Optional[Dict],
             speech_result: Optional[Dict],
             text_result: Optional[Dict]) -> Dict:
        """
        Perform fusion given per-modality result dicts.

        Each result dict should have keys: 'probabilities', 'confidence', 'emotion'.

        Args:
            facial_result: Facial model output
            speech_result: Speech model output
            text_result: Text model output

        Returns:
            Dict with fused emotion, confidence, probabilities, and per-modality info
        """
        fp = np.array(facial_result['probabilities']) if facial_result else None
        sp = np.array(speech_result['probabilities']) if speech_result else None
        tp = np.array(text_result['probabilities']) if text_result else None

        if self.fusion_type == 'calibrated':
            fused_probs = self.calibrated_fusion(fp, sp, tp)
        elif self.fusion_type == 'weighted':
            fused_probs = self.weighted_fusion(fp, sp, tp)
        elif self.fusion_type == 'adaptive':
            fc = facial_result.get('confidence', 0.5) if facial_result else 0.5
            sc = speech_result.get('confidence', 0.5) if speech_result else 0.5
            tc = text_result.get('confidence', 0.5) if text_result else 0.5
            fused_probs = self.adaptive_fusion(fp, sp, tp, fc, sc, tc)
        elif self.fusion_type == 'voting':
            fused_probs = self.voting_fusion(fp, sp, tp)
        else:
            fused_probs = self.calibrated_fusion(fp, sp, tp)

        # Final decision
        emotion_idx = int(np.argmax(fused_probs))
        emotion = EMOTIONS[emotion_idx]
        confidence = float(fused_probs[emotion_idx])
        emotion_scores = {EMOTIONS[i]: float(fused_probs[i]) for i in range(6)}

        result = {
            'emotion': emotion,
            'confidence': confidence,
            'probabilities': fused_probs.tolist(),
            'emotion_scores': emotion_scores,
            'fusion_type': self.fusion_type,
            'modalities_used': []
        }

        if facial_result:
            result['modalities_used'].append('facial')
            result['facial_emotion'] = facial_result.get('emotion', 'unknown')
        if speech_result:
            result['modalities_used'].append('speech')
            result['speech_emotion'] = speech_result.get('emotion', 'unknown')
        if text_result:
            result['modalities_used'].append('text')
            result['text_emotion'] = text_result.get('emotion', 'unknown')

        logger.info(
            f"Fusion result: {emotion} ({confidence:.1%}) "
            f"from {result['modalities_used']}"
        )
        return result

    # ------------------------------------------------------------------ #
    #  Dynamic weight updates
    # ------------------------------------------------------------------ #

    def update_weights(self, new_facial: float, new_speech: float,
                       new_text: float):
        """Update base fusion weights and renormalize."""
        total = new_facial + new_speech + new_text
        self.facial_weight = new_facial / total
        self.speech_weight = new_speech / total
        self.text_weight = new_text / total
        logger.info(
            f"Updated weights — facial={self.facial_weight:.2f}, "
            f"speech={self.speech_weight:.2f}, text={self.text_weight:.2f}"
        )

    def record_accuracy(self, modality: str, correct: bool):
        """Track per-modality accuracy for dynamic auto-weight adjustment."""
        self._total_counts[modality] += 1
        if correct:
            self._correct_counts[modality] += 1

    def auto_adjust_weights(self, min_samples: int = 20):
        """
        Automatically adjust weights based on historical per-modality accuracy.
        Only activates if each modality has at least min_samples.
        """
        accs = {}
        for m in ['facial', 'speech', 'text']:
            n = self._total_counts[m]
            if n < min_samples:
                return   # not enough data yet
            accs[m] = self._correct_counts[m] / n

        total_acc = sum(accs.values())
        if total_acc == 0:
            return

        self.facial_weight = accs['facial'] / total_acc
        self.speech_weight = accs['speech'] / total_acc
        self.text_weight = accs['text'] / total_acc
        logger.info(
            f"Auto-adjusted weights based on accuracy — "
            f"facial={self.facial_weight:.2f} ({accs['facial']:.1%}), "
            f"speech={self.speech_weight:.2f} ({accs['speech']:.1%}), "
            f"text={self.text_weight:.2f} ({accs['text']:.1%})"
        )
