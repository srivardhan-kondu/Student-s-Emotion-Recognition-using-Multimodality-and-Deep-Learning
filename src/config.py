"""
Configuration file for the Multimodal Emotion Recognition System.
Contains all project-wide settings and constants.
"""

import os
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
SAVED_MODELS_DIR = PROJECT_ROOT / "saved_models"

# Emotion labels
EMOTIONS = ["happy", "sad", "angry", "neutral", "fear", "surprise"]
NUM_EMOTIONS = len(EMOTIONS)

# Facial Recognition Settings
FACIAL_CONFIG = {
    "image_size": (48, 48),  # Input image size for CNN
    "color_mode": "grayscale",  # or "rgb"
    "batch_size": 32,
    "epochs": 50,
    "learning_rate": 0.001,
    "model_architecture": "custom_cnn",  # or "vgg16", "resnet50"
}

# Speech Analysis Settings
SPEECH_CONFIG = {
    "sample_rate": 22050,
    "n_mfcc": 40,
    "n_fft": 2048,
    "hop_length": 512,
    "batch_size": 32,
    "epochs": 50,
    "learning_rate": 0.001,
}

# Text Analysis Settings
TEXT_CONFIG = {
    "max_length": 128,
    "model_name": "bert-base-uncased",  # or "roberta-base"
    "batch_size": 16,
    "epochs": 10,
    "learning_rate": 2e-5,
}

# Multimodal Fusion Settings
FUSION_CONFIG = {
    "fusion_type": "weighted",  # "weighted", "adaptive", "late"
    "facial_weight": 0.4,
    "speech_weight": 0.3,
    "text_weight": 0.3,
    "confidence_threshold": 0.5,
}

# Dashboard Settings
DASHBOARD_CONFIG = {
    "title": "Student Emotion Recognition System",
    "theme": "light",
    "port": 8501,
}

# Dataset paths
DATASET_PATHS = {
    "facial": {
        "fer2013": DATA_DIR / "facial" / "fer2013",
        "ck_plus": DATA_DIR / "facial" / "ck_plus",
    },
    "speech": {
        "ravdess": DATA_DIR / "speech" / "ravdess",
        "tess": DATA_DIR / "speech" / "tess",
    },
    "text": {
        "goemotions": DATA_DIR / "text" / "goemotions",
        "emocontext": DATA_DIR / "text" / "emocontext",
    },
}

# Model save paths
MODEL_SAVE_PATHS = {
    "facial": SAVED_MODELS_DIR / "facial_emotion_model.h5",
    "speech": SAVED_MODELS_DIR / "speech_emotion_model.h5",
    "text": SAVED_MODELS_DIR / "text_emotion_model.pt",
    "fusion": SAVED_MODELS_DIR / "fusion_weights.pkl",
}

# Training settings
TRAIN_CONFIG = {
    "test_size": 0.15,
    "val_size": 0.15,
    "random_state": 42,
    "shuffle": True,
}

# Logging
LOG_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "log_file": PROJECT_ROOT / "logs" / "app.log",
}
