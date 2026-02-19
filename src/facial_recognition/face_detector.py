"""
Face Detection Module
Implements automatic face detection from images and video frames (FR6).
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class FaceDetector:
    """
    Face detector using OpenCV's Haar Cascade or DNN-based methods.
    """
    
    def __init__(self, method='haar'):
        """
        Initialize face detector.
        
        Args:
            method: Detection method ('haar', 'dnn', or 'mtcnn')
        """
        self.method = method
        
        if method == 'haar':
            # Load Haar Cascade classifier
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
            logger.info("Initialized Haar Cascade face detector")
            
        elif method == 'dnn':
            # Load DNN-based face detector
            model_file = "models/facial_model/opencv_face_detector_uint8.pb"
            config_file = "models/facial_model/opencv_face_detector.pbtxt"
            try:
                self.net = cv2.dnn.readNetFromTensorflow(model_file, config_file)
                logger.info("Initialized DNN face detector")
            except:
                logger.warning("DNN model not found, falling back to Haar Cascade")
                self.method = 'haar'
                cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
                self.face_cascade = cv2.CascadeClassifier(cascade_path)
    
    def detect_faces(self, image: np.ndarray, 
                     min_confidence: float = 0.5) -> List[Tuple[int, int, int, int]]:
        """
        Detect faces in an image.
        
        Args:
            image: Input image (BGR format)
            min_confidence: Minimum confidence threshold for detection
            
        Returns:
            List of bounding boxes [(x, y, w, h), ...]
        """
        if self.method == 'haar':
            return self._detect_haar(image)
        elif self.method == 'dnn':
            return self._detect_dnn(image, min_confidence)
    
    def _detect_haar(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect faces using Haar Cascade."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        return [tuple(face) for face in faces]
    
    def _detect_dnn(self, image: np.ndarray, 
                    min_confidence: float) -> List[Tuple[int, int, int, int]]:
        """Detect faces using DNN."""
        h, w = image.shape[:2]
        blob = cv2.dnn.blobFromImage(
            cv2.resize(image, (300, 300)), 1.0,
            (300, 300), (104.0, 177.0, 123.0)
        )
        
        self.net.setInput(blob)
        detections = self.net.forward()
        
        faces = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > min_confidence:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x, y, x2, y2) = box.astype("int")
                faces.append((x, y, x2-x, y2-y))
        
        return faces
    
    def extract_face(self, image: np.ndarray, 
                     bbox: Tuple[int, int, int, int],
                     target_size: Tuple[int, int] = (48, 48)) -> np.ndarray:
        """
        Extract and resize face region from image.
        
        Args:
            image: Input image
            bbox: Bounding box (x, y, w, h)
            target_size: Target size for resized face
            
        Returns:
            Extracted and resized face image
        """
        x, y, w, h = bbox
        face = image[y:y+h, x:x+w]
        face = cv2.resize(face, target_size)
        return face
    
    def detect_and_extract_faces(self, image: np.ndarray,
                                  target_size: Tuple[int, int] = (48, 48),
                                  grayscale: bool = True) -> List[np.ndarray]:
        """
        Detect faces and extract face regions.
        
        Args:
            image: Input image
            target_size: Target size for extracted faces
            grayscale: Convert to grayscale if True
            
        Returns:
            List of extracted face images
        """
        faces_bbox = self.detect_faces(image)
        extracted_faces = []
        
        for bbox in faces_bbox:
            face = self.extract_face(image, bbox, target_size)
            if grayscale and len(face.shape) == 3:
                face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            extracted_faces.append(face)
        
        logger.info(f"Detected and extracted {len(extracted_faces)} faces")
        return extracted_faces
    
    def draw_faces(self, image: np.ndarray, 
                   faces: List[Tuple[int, int, int, int]],
                   color: Tuple[int, int, int] = (0, 255, 0),
                   thickness: int = 2) -> np.ndarray:
        """
        Draw bounding boxes around detected faces.
        
        Args:
            image: Input image
            faces: List of bounding boxes
            color: Box color (BGR)
            thickness: Box line thickness
            
        Returns:
            Image with drawn bounding boxes
        """
        output = image.copy()
        for (x, y, w, h) in faces:
            cv2.rectangle(output, (x, y), (x+w, y+h), color, thickness)
        return output
