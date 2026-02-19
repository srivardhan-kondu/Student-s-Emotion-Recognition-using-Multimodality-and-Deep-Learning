"""
CNN Model for Facial Emotion Classification — Upgraded.
Supports:
  - Custom CNN (legacy)
  - VGG16 / ResNet50 transfer learning
  - EfficientNetV2-B0 (NEW — best performance)
Implements FR7, FR8.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from typing import Tuple, Dict, Optional
import logging
import cv2

from facial_recognition.model_architecture import (
    MiniXception, build_efficientnet_model
)

logger = logging.getLogger(__name__)

EMOTIONS = ['happy', 'sad', 'angry', 'neutral', 'fear', 'surprise']
EFFICIENTNET_INPUT_SIZE = (96, 96)   # RGB 96x96 for EfficientNet
MINIXCEPTION_INPUT_SIZE = (48, 48)   # Grayscale 48x48 for MiniXception


class EmotionCNN:
    """
    CNN model for facial emotion recognition.
    Classifies emotions: happy, sad, angry, neutral, fear, surprise (FR8).
    
    Default architecture: EfficientNetV2-B0 (highest accuracy).
    """

    def __init__(self, input_shape: Tuple[int, int, int] = (96, 96, 3),
                 num_classes: int = 6):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None
        self.history = None
        self.architecture = None
        self.temperature = 1.5   # calibration temperature — tuned after training

    # ------------------------------------------------------------------ #
    #  Model builders
    # ------------------------------------------------------------------ #

    def build_model(self, architecture: str = 'efficientnet') -> keras.Model:
        """
        Build CNN architecture.

        Args:
            architecture: 'efficientnet' (recommended) | 'custom' | 'vgg16' | 'resnet50'
        """
        self.architecture = architecture

        if architecture == 'efficientnet':
            self.model = build_efficientnet_model(
                num_classes=self.num_classes,
                input_shape=self.input_shape,
                trainable_base_layers=30
            )
            self.model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=1e-4),
                loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
                metrics=['accuracy']
            )
        elif architecture == 'custom':
            self.model = self._build_custom_cnn()
        elif architecture == 'vgg16':
            self.model = self._build_vgg16()
        elif architecture == 'resnet50':
            self.model = self._build_resnet50()
        else:
            raise ValueError(f"Unknown architecture: {architecture}")

        logger.info(
            f"Built {architecture} model with "
            f"{self.model.count_params():,} parameters"
        )
        return self.model

    def _build_custom_cnn(self) -> keras.Model:
        """Baseline custom CNN (48x48 grayscale)."""
        model = models.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=self.input_shape),
            layers.BatchNormalization(),
            layers.Conv2D(32, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),

            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),

            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),

            layers.Flatten(),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        return model

    def _build_vgg16(self) -> keras.Model:
        """VGG16-based model with transfer learning."""
        base_model = keras.applications.VGG16(
            weights='imagenet', include_top=False,
            input_shape=(96, 96, 3)
        )
        base_model.trainable = False
        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-4),
            loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
            metrics=['accuracy']
        )
        return model

    def _build_resnet50(self) -> keras.Model:
        """ResNet50-based model with transfer learning."""
        base_model = keras.applications.ResNet50(
            weights='imagenet', include_top=False,
            input_shape=(96, 96, 3)
        )
        base_model.trainable = False
        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-4),
            loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
            metrics=['accuracy']
        )
        return model

    # ------------------------------------------------------------------ #
    #  Training
    # ------------------------------------------------------------------ #

    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray, y_val: np.ndarray,
              epochs: int = 60, batch_size: int = 32,
              use_augmentation: bool = True,
              class_weights: Optional[Dict] = None) -> Dict:
        """
        Train the model with optional augmentation and class balancing.

        Args:
            X_train: Training images
            y_train: One-hot encoded labels
            X_val: Validation images
            y_val: Validation labels
            epochs: Max epochs (early stopping applies)
            batch_size: Batch size
            use_augmentation: Apply data augmentation
            class_weights: Dict mapping class index to weight

        Returns:
            Training history dictionary
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")

        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=12,
                restore_best_weights=True, verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss', factor=0.5,
                patience=5, min_lr=1e-7, verbose=1
            ),
            keras.callbacks.ModelCheckpoint(
                'saved_models/facial_best.h5',
                monitor='val_accuracy',
                save_best_only=True, verbose=1
            )
        ]

        if use_augmentation:
            datagen = ImageDataGenerator(
                rotation_range=20,
                width_shift_range=0.2,
                height_shift_range=0.2,
                horizontal_flip=True,
                zoom_range=0.2,
                shear_range=0.1,
                brightness_range=[0.7, 1.3],
                fill_mode='nearest',
                preprocessing_function=self._apply_cutout
            )
            datagen.fit(X_train)

            self.history = self.model.fit(
                datagen.flow(X_train, y_train, batch_size=batch_size),
                validation_data=(X_val, y_val),
                epochs=epochs,
                callbacks=callbacks,
                class_weight=class_weights,
                verbose=1
            )
        else:
            self.history = self.model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks,
                class_weight=class_weights,
                verbose=1
            )

        logger.info("Training completed")
        return self.history.history

    @staticmethod
    def _apply_cutout(img: np.ndarray, length: int = 12) -> np.ndarray:
        """Apply cutout (random erasing) augmentation to a single image."""
        h, w = img.shape[:2]
        y = np.random.randint(0, h)
        x = np.random.randint(0, w)
        y1 = max(0, y - length // 2)
        y2 = min(h, y + length // 2)
        x1 = max(0, x - length // 2)
        x2 = min(w, x + length // 2)
        img_copy = img.copy()
        img_copy[y1:y2, x1:x2] = 0.0
        return img_copy

    # ------------------------------------------------------------------ #
    #  Inference with temperature scaling
    # ------------------------------------------------------------------ #

    def _softmax_with_temperature(self, logits: np.ndarray, T: float) -> np.ndarray:
        """Apply temperature scaling to soften/sharpen predictions."""
        scaled = logits / T
        exp_vals = np.exp(scaled - np.max(scaled))
        return exp_vals / exp_vals.sum()

    def predict(self, image: np.ndarray) -> Tuple[str, float, np.ndarray]:
        """
        Predict emotion from a face image with temperature-calibrated confidence.

        Args:
            image: Face image (preprocessed, any of the supported shapes)

        Returns:
            Tuple of (emotion_label, confidence, probabilities)
        """
        if self.model is None:
            raise ValueError("Model not loaded")

        # Prepare image according to architecture
        if self.architecture == 'efficientnet':
            image = self._prepare_efficientnet_input(image)
        else:
            image = self._prepare_grayscale_input(image)

        # Raw softmax probabilities from model
        raw_probs = self.model.predict(image, verbose=0)[0]

        # Apply temperature scaling for calibrated confidence
        probabilities = self._softmax_with_temperature(raw_probs, self.temperature)

        emotion_idx = np.argmax(probabilities)
        confidence = float(probabilities[emotion_idx])
        emotion_label = EMOTIONS[emotion_idx]

        return emotion_label, confidence, probabilities

    def _prepare_efficientnet_input(self, image: np.ndarray) -> np.ndarray:
        """Resize to 96x96 RGB and normalise for EfficientNet."""
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[-1] == 1:
            image = cv2.cvtColor(image[:, :, 0], cv2.COLOR_GRAY2RGB)

        if image.shape[:2] != EFFICIENTNET_INPUT_SIZE:
            image = cv2.resize(image, EFFICIENTNET_INPUT_SIZE)

        image = image.astype('float32') / 255.0
        return np.expand_dims(image, axis=0)

    def _prepare_grayscale_input(self, image: np.ndarray) -> np.ndarray:
        """Resize to 48x48 grayscale and normalise."""
        if len(image.shape) == 3 and image.shape[-1] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        if image.shape[:2] != MINIXCEPTION_INPUT_SIZE:
            image = cv2.resize(image, MINIXCEPTION_INPUT_SIZE)
        image = np.expand_dims(image, axis=-1)
        image = np.expand_dims(image, axis=0)
        image = image.astype('float32') / 255.0
        return image

    # ------------------------------------------------------------------ #
    #  Persistence
    # ------------------------------------------------------------------ #

    def save_model(self, filepath: str):
        """Save model to file."""
        if self.model is None:
            raise ValueError("No model to save")
        self.model.save(filepath)
        logger.info(f"Model saved to {filepath}")

    def load_model(self, filepath: str, architecture: str = 'efficientnet'):
        """Load model from file."""
        self.model = keras.models.load_model(filepath)
        self.architecture = architecture
        logger.info(f"Model loaded from {filepath} (architecture: {architecture})")
