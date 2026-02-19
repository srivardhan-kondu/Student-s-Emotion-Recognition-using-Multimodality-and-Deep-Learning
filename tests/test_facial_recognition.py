"""
Unit tests for facial emotion recognition module.
"""

import unittest
import numpy as np
import cv2
from src.facial_recognition.face_detector import FaceDetector
from src.facial_recognition.emotion_model import EmotionCNN


class TestFaceDetector(unittest.TestCase):
    """Test cases for FaceDetector class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.detector = FaceDetector(method='haar')
    
    def test_initialization(self):
        """Test detector initialization."""
        self.assertIsNotNone(self.detector)
        self.assertEqual(self.detector.method, 'haar')
    
    def test_detect_faces_empty_image(self):
        """Test face detection on empty image."""
        empty_image = np.zeros((100, 100, 3), dtype=np.uint8)
        faces = self.detector.detect_faces(empty_image)
        self.assertIsInstance(faces, list)
    
    def test_extract_face(self):
        """Test face extraction."""
        image = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        bbox = (50, 50, 100, 100)
        face = self.detector.extract_face(image, bbox, target_size=(48, 48))
        self.assertEqual(face.shape, (48, 48, 3))


class TestEmotionCNN(unittest.TestCase):
    """Test cases for EmotionCNN class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.model = EmotionCNN(input_shape=(48, 48, 1), num_classes=6)
    
    def test_initialization(self):
        """Test model initialization."""
        self.assertIsNotNone(self.model)
        self.assertEqual(self.model.num_classes, 6)
    
    def test_build_custom_cnn(self):
        """Test custom CNN building."""
        model = self.model.build_model(architecture='custom')
        self.assertIsNotNone(model)
        self.assertEqual(model.output_shape[-1], 6)
    
    def test_predict_shape(self):
        """Test prediction output shape."""
        self.model.build_model()
        test_image = np.random.rand(48, 48, 1)
        emotion, confidence, probabilities = self.model.predict(test_image)
        
        self.assertIsInstance(emotion, str)
        self.assertIsInstance(confidence, (float, np.floating))
        self.assertEqual(len(probabilities), 6)


if __name__ == '__main__':
    unittest.main()
