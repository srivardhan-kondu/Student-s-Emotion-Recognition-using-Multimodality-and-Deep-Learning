"""
Multimodal Emotion Predictor.
Integrates all three modalities for complete emotion prediction.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from typing import Optional, Dict
import cv2
import logging

from facial_recognition.face_detector import FaceDetector
from facial_recognition.emotion_model import EmotionCNN
from speech_analysis.audio_features import (
    AudioFeatureExtractor,
    MAX_TIME_STEPS,
    FEATURE_DIM,
)
from speech_analysis.speech_recognition import SpeechRecognizer
from speech_analysis.emotion_model import SpeechEmotionModel
from text_analysis.text_preprocessing import TextPreprocessor
from text_analysis.emotion_model import TextEmotionAnalyzer
from fusion.multimodal_fusion import MultimodalFusion
from config import MODEL_SAVE_PATHS, FUSION_CONFIG, EMOTIONS

logger = logging.getLogger(__name__)


class MultimodalEmotionPredictor:
    """
    Complete multimodal emotion prediction system.
    """
    
    def __init__(self):
        """Initialize all components."""
        logger.info("Initializing Multimodal Emotion Predictor...")
        
        # Initialize facial components
        self.face_detector = FaceDetector(method='haar')
        self.facial_model = EmotionCNN(input_shape=(48, 48, 1), num_classes=6)
        
        # Initialize speech components
        self.audio_extractor = AudioFeatureExtractor()
        self.speech_recognizer = SpeechRecognizer(method='google')
        self.speech_model = SpeechEmotionModel(
            input_shape=(MAX_TIME_STEPS, FEATURE_DIM),
            num_classes=6,
        )
        
        # Initialize text components
        self.text_preprocessor = TextPreprocessor()
        self.text_analyzer = TextEmotionAnalyzer(model_type='roberta')
        
        # Initialize fusion
        self.fusion = MultimodalFusion(
            fusion_type=FUSION_CONFIG['fusion_type'],
            facial_weight=FUSION_CONFIG['facial_weight'],
            speech_weight=FUSION_CONFIG['speech_weight'],
            text_weight=FUSION_CONFIG['text_weight']
        )
        
        logger.info("Multimodal Emotion Predictor initialized")
    
    def load_models(self):
        """Load all trained models."""
        # Load each modality independently so one failure doesn't block others.
        if os.path.exists(MODEL_SAVE_PATHS['facial']):
            try:
                self.facial_model.load_model(str(MODEL_SAVE_PATHS['facial']))
                logger.info("Loaded facial emotion model")
            except Exception as e:
                logger.error("Failed to load facial model: %s", e)
        else:
            logger.warning(f"Facial model not found at {MODEL_SAVE_PATHS['facial']}")

        if os.path.exists(MODEL_SAVE_PATHS['speech']):
            try:
                self.speech_model.load_model(str(MODEL_SAVE_PATHS['speech']))
                logger.info("Loaded speech emotion model")
            except Exception as e:
                logger.error("Failed to load speech model: %s", e)
        else:
            logger.warning(f"Speech model not found at {MODEL_SAVE_PATHS['speech']}")

        # Text model is already loaded by TextEmotionAnalyzer.__init__
        # from saved_models/text_bert_model/ directory
        if self.text_analyzer.model is not None:
            logger.info("Text emotion model loaded via TextEmotionAnalyzer")
        else:
            logger.warning("Text model could not be loaded")
    
    def predict_from_image(self, image_path: str) -> Optional[Dict]:
        """
        Predict emotion from image.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Facial emotion result dict
        """
        try:
            image = cv2.imread(image_path)
            faces_bbox = self.face_detector.detect_faces(image)
            
            if not faces_bbox:
                logger.warning("No faces detected in image")
                return None
            
            # Extract face region at full resolution for analysis
            x, y, w, h = faces_bbox[0]
            face = image[y:y+h, x:x+w]
            emotion, confidence, probabilities = self.facial_model.predict(face)
            
            return {
                'emotion': emotion,
                'confidence': confidence,
                'probabilities': probabilities
            }
        except Exception as e:
            logger.error(f"Error predicting from image: {e}")
            return None
    
    def predict_from_audio(self, audio_path: str) -> Optional[Dict]:
        """
        Predict emotion from audio.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Speech emotion result dict
        """
        try:
            if self.speech_model.model is None:
                speech_path = str(MODEL_SAVE_PATHS['speech'])
                if not os.path.exists(speech_path):
                    logger.error("Speech model missing at %s", speech_path)
                    return None
                try:
                    self.speech_model.load_model(speech_path)
                    logger.info("Loaded speech emotion model on demand")
                except Exception as e:
                    logger.error("Failed to load speech model on demand: %s", e)
                    return None

            features = self.audio_extractor.extract_sequence_from_file(audio_path)
            features = np.expand_dims(features, axis=0)

            emotion, confidence, probabilities = self.speech_model.predict(features)
            
            return {
                'emotion': emotion,
                'confidence': confidence,
                'probabilities': probabilities
            }
        except Exception as e:
            logger.error(f"Error predicting from audio: {e}")
            return None
    
    def predict_from_text(self, text: str) -> Optional[Dict]:
        """
        Predict emotion from text.
        
        Args:
            text: Input text
            
        Returns:
            Text emotion result dict
        """
        try:
            preprocessed_text = self.text_preprocessor.preprocess(text)
            result = self.text_analyzer.analyze_emotion(preprocessed_text)
            
            probabilities = np.array([result['emotion_scores'][e] for e in EMOTIONS])
            
            return {
                'emotion': result['emotion'],
                'confidence': result['confidence'],
                'probabilities': probabilities
            }
        except Exception as e:
            logger.error(f"Error predicting from text: {e}")
            return None
    
    def predict_multimodal(self, image_path: Optional[str] = None,
                          audio_path: Optional[str] = None,
                          text: Optional[str] = None) -> Dict:
        """
        Predict emotion using multiple modalities.
        
        Args:
            image_path: Optional path to image
            audio_path: Optional path to audio
            text: Optional text input
            
        Returns:
            Fused emotion result
        """
        # Get predictions from each modality
        facial_result = self.predict_from_image(image_path) if image_path else None
        speech_result = self.predict_from_audio(audio_path) if audio_path else None
        text_result = self.predict_from_text(text) if text else None
        
        # Fuse results
        fused_result = self.fusion.fuse(facial_result, speech_result, text_result)
        
        return fused_result
