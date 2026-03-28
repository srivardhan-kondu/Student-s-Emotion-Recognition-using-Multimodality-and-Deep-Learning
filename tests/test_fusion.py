"""
Unit tests for multimodal fusion module.
"""

import unittest
import numpy as np
from fusion.multimodal_fusion import MultimodalFusion


class TestMultimodalFusion(unittest.TestCase):
    """Test cases for MultimodalFusion class."""

    def setUp(self):
        """Set up test fixtures."""
        self.fusion = MultimodalFusion(
            fusion_type='weighted',
            facial_weight=0.4,
            speech_weight=0.3,
            text_weight=0.3
        )

    def test_initialization(self):
        """Test fusion initialization."""
        self.assertIsNotNone(self.fusion)
        self.assertEqual(self.fusion.fusion_type, 'weighted')
        self.assertAlmostEqual(
            self.fusion.facial_weight +
            self.fusion.speech_weight +
            self.fusion.text_weight,
            1.0
        )

    def test_weighted_fusion(self):
        """Test weighted fusion."""
        facial_probs = np.array([0.8, 0.1, 0.05, 0.03, 0.01, 0.01])
        speech_probs = np.array([0.7, 0.15, 0.05, 0.05, 0.03, 0.02])
        text_probs = np.array([0.75, 0.1, 0.05, 0.05, 0.03, 0.02])

        fused = self.fusion.weighted_fusion(facial_probs, speech_probs, text_probs)

        self.assertEqual(len(fused), 6)
        self.assertAlmostEqual(np.sum(fused), 1.0, places=5)

    def test_adaptive_fusion(self):
        """Test adaptive fusion."""
        facial_probs = np.array([0.8, 0.1, 0.05, 0.03, 0.01, 0.01])
        speech_probs = np.array([0.7, 0.15, 0.05, 0.05, 0.03, 0.02])
        text_probs = np.array([0.75, 0.1, 0.05, 0.05, 0.03, 0.02])

        fused = self.fusion.adaptive_fusion(
            facial_probs, speech_probs, text_probs,
            facial_conf=0.9, speech_conf=0.7, text_conf=0.8
        )

        self.assertEqual(len(fused), 6)
        self.assertAlmostEqual(np.sum(fused), 1.0, places=5)

    def test_voting_fusion(self):
        """Test voting fusion returns a distribution; majority class index wins."""
        facial = np.array([0.9, 0.05, 0.02, 0.01, 0.01, 0.01])
        speech = np.array([0.85, 0.05, 0.05, 0.02, 0.02, 0.01])
        text = np.array([0.1, 0.7, 0.1, 0.05, 0.03, 0.02])
        fused = self.fusion.voting_fusion(facial, speech, text)
        self.assertEqual(len(fused), 6)
        self.assertAlmostEqual(float(fused.sum()), 1.0, places=5)
        # Two modalities peak at index 0 (happy), one at index 1 (sad)
        self.assertEqual(int(np.argmax(fused)), 0)

    def test_fuse_with_missing_modality(self):
        """Test fusion with only facial modality."""
        facial_result = {
            'emotion': 'happy',
            'confidence': 0.9,
            'probabilities': np.array([0.9, 0.05, 0.02, 0.01, 0.01, 0.01])
        }

        result = self.fusion.fuse(facial_result, None, None)

        self.assertIsNotNone(result)
        self.assertEqual(result['modalities_used'], ['facial'])
        self.assertIn('emotion', result)
        self.assertIn('confidence', result)

    def test_update_weights(self):
        """Test fusion weight updating."""
        self.fusion.update_weights(0.5, 0.3, 0.2)
        self.assertAlmostEqual(self.fusion.facial_weight, 0.5)
        self.assertAlmostEqual(self.fusion.speech_weight, 0.3)
        self.assertAlmostEqual(self.fusion.text_weight, 0.2)


if __name__ == '__main__':
    unittest.main()
