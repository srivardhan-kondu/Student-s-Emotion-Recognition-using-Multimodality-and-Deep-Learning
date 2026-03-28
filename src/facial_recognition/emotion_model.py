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
from typing import Tuple, Dict, Optional, List
import logging
import cv2

import config as project_config
from facial_recognition.model_architecture import build_efficientnet_model

logger = logging.getLogger(__name__)

# Class index order from facial_recognition/train.py (FER2013 folder layout)
FER_OUTPUT_EMOTIONS: List[str] = [
    'angry', 'fear', 'happy', 'neutral', 'sad', 'surprise'
]
# canonical[i] = raw[FER_TRAIN_PY_TO_CANONICAL[i]] — train.py folder order
FER_TRAIN_PY_TO_CANONICAL = np.array([2, 4, 0, 3, 1, 5], dtype=np.intp)
# Kaggle 6-class: angry, fear, happy, sad, surprise, neutral
FER_KAGGLE_6_TO_CANONICAL = np.array([2, 3, 0, 5, 1, 4], dtype=np.intp)

# predict() returns labels/probs aligned with config.EMOTIONS (speech/text/fusion)
EMOTIONS = project_config.EMOTIONS

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
        self.temperature = 1.2   # post-softmax calibration (see _apply_temperature)

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
        """Apply temperature to logits (linear layer output), then softmax."""
        scaled = np.asarray(logits, dtype=np.float64) / T
        exp_vals = np.exp(scaled - np.max(scaled))
        return exp_vals / exp_vals.sum()

    @staticmethod
    def _seven_to_six_kaggle_probs(vec: np.ndarray) -> np.ndarray:
        """
        FER-2013 7-class order: angry, disgust, fear, happy, sad, surprise, neutral.
        Drop disgust → 6 probs in Kaggle-6 order: angry, fear, happy, sad, surprise, neutral.
        """
        v = np.asarray(vec, dtype=np.float64).reshape(-1)
        if len(v) != 7:
            raise ValueError("internal: expected 7-class vector")
        if v.min() >= -1e-4 and v.max() <= 1.0 + 1e-4 and abs(v.sum() - 1.0) < 0.08:
            p7 = v
        else:
            e = np.exp(v - np.max(v))
            p7 = e / (e.sum() + 1e-12)
        six = np.concatenate([p7[0:1], p7[2:7]])
        six = six / (six.sum() + 1e-12)
        return six

    def _apply_temperature(self, raw: np.ndarray, T: float) -> np.ndarray:
        """
        MiniXception ends with softmax — model.predict returns probabilities, not logits.
        Temperature must be applied in log-probability space, not exp(prob/T).
        """
        raw = np.asarray(raw, dtype=np.float64)
        if abs(T - 1.0) < 1e-6:
            return raw / (raw.sum() + 1e-12)
        looks_like_softmax = (
            raw.min() >= -1e-5
            and raw.max() <= 1.0 + 1e-5
            and abs(raw.sum() - 1.0) < 0.05
        )
        if looks_like_softmax:
            log_p = np.log(np.clip(raw, 1e-12, 1.0))
            z = log_p / T
            exp_vals = np.exp(z - np.max(z))
            return exp_vals / exp_vals.sum()
        return self._softmax_with_temperature(raw, T)

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

        raw_probs = np.asarray(self.model.predict(image, verbose=0)[0], dtype=np.float64).reshape(-1)

        # Many public FER weights use 7 outputs (incl. disgust). Using only 6 breaks softmax (~1/7 each).
        if raw_probs.size == 7:
            six_kag = self._seven_to_six_kaggle_probs(raw_probs)
            probabilities_raw = self._apply_temperature(six_kag, self.temperature)
            probabilities = probabilities_raw[FER_KAGGLE_6_TO_CANONICAL]
        elif raw_probs.size == 6:
            probabilities_raw = self._apply_temperature(raw_probs, self.temperature)
            order = project_config.FACIAL_MODEL_SOFTMAX_ORDER
            if order == "identity":
                probabilities = np.asarray(probabilities_raw, dtype=np.float64)
            elif order == "fer_train_py":
                probabilities = probabilities_raw[FER_TRAIN_PY_TO_CANONICAL]
            elif order == "fer_kaggle_6":
                probabilities = probabilities_raw[FER_KAGGLE_6_TO_CANONICAL]
            else:
                raise ValueError(
                    f"Unknown FACIAL_MODEL_SOFTMAX_ORDER={order!r}; "
                    'use "fer_train_py", "fer_kaggle_6", or "identity"'
                )
        else:
            raise ValueError(
                f"Facial model has {raw_probs.size} outputs; expected 6 or 7."
            )

        emotion_idx = int(np.argmax(probabilities))
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
        """Resize to 48x48 grayscale and normalise (OpenCV uses BGR)."""
        if len(image.shape) == 3 and image.shape[-1] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        elif len(image.shape) == 3 and image.shape[-1] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)

        if getattr(project_config, "FACIAL_EQUALIZE_HISTOGRAM", True):
            if image.dtype != np.uint8:
                img_u8 = np.clip(image, 0, 255).astype(np.uint8)
            else:
                img_u8 = image
            image = cv2.equalizeHist(img_u8)

        if image.shape[:2] != MINIXCEPTION_INPUT_SIZE:
            image = cv2.resize(
                image, MINIXCEPTION_INPUT_SIZE, interpolation=cv2.INTER_AREA
            )
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

    def load_model(self, filepath: str, architecture: Optional[str] = None):
        """
        Load model from file.

        If architecture is None, infer from input shape: 48x48x1 → grayscale (MiniXception
        / custom CNN), else EfficientNet-style RGB preprocessing.
        """
        try:
            self.model = keras.models.load_model(filepath, compile=False)
        except Exception as e:
            logger.warning("Retrying facial load_model with safe_mode=False: %s", e)
            try:
                self.model = keras.models.load_model(
                    filepath, compile=False, safe_mode=False
                )
            except TypeError:
                self.model = keras.models.load_model(filepath, compile=False)
        if architecture is not None:
            self.architecture = architecture
        else:
            inp = self.model.input_shape
            if inp is not None and len(inp) == 4 and inp[-1] == 1:
                self.architecture = 'custom'
            else:
                self.architecture = 'efficientnet'
        n_out = self.model.output_shape[-1]
        logger.info(
            f"Model loaded from {filepath} (architecture: {self.architecture}, "
            f"output_classes={n_out})"
        )
