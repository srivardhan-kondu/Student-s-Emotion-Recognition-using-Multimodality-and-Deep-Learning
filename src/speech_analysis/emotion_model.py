"""
Speech Emotion Recognition Model — Upgraded.
Key improvements:
  - Attention-based Bidirectional LSTM (best sequential model)
  - Supports raw MFCC time-series input (not flattened)
  - Temperature scaling for calibrated confidence
  - Kept legacy LSTM/CNN models for backward compatibility
Implements FR10.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import numpy as np
from typing import Tuple, Dict, Optional
import logging

logger = logging.getLogger(__name__)

EMOTIONS = ['happy', 'sad', 'angry', 'neutral', 'fear', 'surprise']


def build_attention_layer(input_tensor):
    """
    Self-attention mechanism for temporal feature weighting.
    Computes attention weights over the time axis.

    Args:
        input_tensor: Shape (batch, time_steps, features)

    Returns:
        Weighted context vector (batch, features)
    """
    # Attention score per time step
    score = layers.Dense(1, activation='tanh')(input_tensor)   # (batch, T, 1)
    score = layers.Flatten()(score)                              # (batch, T)
    attention_weights = layers.Activation('softmax')(score)     # (batch, T)
    attention_weights = layers.RepeatVector(
        input_tensor.shape[-1]
    )(attention_weights)                                         # (batch, F, T)
    attention_weights = layers.Permute([2, 1])(attention_weights) # (batch, T, F)

    # Weighted sum
    context = layers.Multiply()([input_tensor, attention_weights])
    context = layers.Lambda(lambda x: tf.reduce_sum(x, axis=1))(context)
    return context


class SpeechEmotionModel:
    """
    Deep learning model for speech emotion recognition.
    Default: Attention-based BiLSTM for maximum accuracy.
    """

    def __init__(self, input_shape: Tuple[int, ...], num_classes: int = 6):
        """
        Args:
            input_shape: (time_steps, feature_dim) for sequence models
                         e.g. (216, 120) for MFCC+delta+delta2
            num_classes: Number of emotion classes
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None
        self.history = None
        self.temperature = 1.3   # calibration temperature

    # ------------------------------------------------------------------ #
    #  Model builders
    # ------------------------------------------------------------------ #

    def build_attention_bilstm(self) -> keras.Model:
        """
        Attention-based Bidirectional LSTM — highest accuracy model.

        Architecture:
            Input (T, F)
            → Conv1D(64) + BN + MaxPool → feature extraction shortcut
            → BiLSTM(128, return_sequences=True) → Self-Attention
            → BiLSTM(64, return_sequences=True) → GlobalMaxPooling
            → Dense(256, relu) + BN + Dropout(0.4)
            → Dense(128, relu) + Dropout(0.3)
            → Dense(num_classes, softmax)
        """
        inp = layers.Input(shape=self.input_shape)

        # Short-range feature extraction with 1D convolutions
        x = layers.Conv1D(64, 3, activation='relu', padding='same')(inp)
        x = layers.BatchNormalization()(x)
        x = layers.Conv1D(64, 3, activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling1D(2)(x)
        x = layers.Dropout(0.2)(x)

        # Bidirectional LSTM layers
        x = layers.Bidirectional(
            layers.LSTM(128, return_sequences=True)
        )(x)
        x = layers.Dropout(0.3)(x)

        x = layers.Bidirectional(
            layers.LSTM(64, return_sequences=True)
        )(x)
        x = layers.Dropout(0.3)(x)

        # Self-attention over time axis
        x = build_attention_layer(x)

        # Classification head
        x = layers.Dense(256, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.4)(x)

        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.3)(x)

        output = layers.Dense(self.num_classes, activation='softmax')(x)

        model = keras.Model(inputs=inp, outputs=output)
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
            metrics=['accuracy']
        )

        self.model = model
        logger.info(f"Built Attention-BiLSTM model: {model.count_params():,} parameters")
        return model

    def build_lstm_model(self) -> keras.Model:
        """Legacy LSTM model (kept for backward compatibility)."""
        model = models.Sequential([
            layers.LSTM(128, return_sequences=True, input_shape=self.input_shape),
            layers.Dropout(0.3),
            layers.LSTM(64, return_sequences=True),
            layers.Dropout(0.3),
            layers.LSTM(32),
            layers.Dropout(0.3),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.4),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        self.model = model
        return model

    def build_cnn_model(self) -> keras.Model:
        """Legacy 1D CNN model (kept for backward compatibility)."""
        model = models.Sequential([
            layers.Conv1D(64, 3, activation='relu', input_shape=self.input_shape),
            layers.BatchNormalization(),
            layers.MaxPooling1D(2),
            layers.Dropout(0.3),

            layers.Conv1D(128, 3, activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling1D(2),
            layers.Dropout(0.3),

            layers.Conv1D(256, 3, activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling1D(2),
            layers.Dropout(0.3),

            layers.GlobalAveragePooling1D(),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.4),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        self.model = model
        return model

    def build_hybrid_model(self) -> keras.Model:
        """Legacy hybrid CNN-LSTM model (kept for backward compatibility)."""
        model = models.Sequential([
            layers.Conv1D(64, 3, activation='relu', input_shape=self.input_shape),
            layers.BatchNormalization(),
            layers.MaxPooling1D(2),
            layers.Dropout(0.3),

            layers.Conv1D(128, 3, activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling1D(2),
            layers.Dropout(0.3),

            layers.LSTM(128, return_sequences=True),
            layers.Dropout(0.3),
            layers.LSTM(64),
            layers.Dropout(0.3),

            layers.Dense(128, activation='relu'),
            layers.Dropout(0.4),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        self.model = model
        return model

    # ------------------------------------------------------------------ #
    #  Training
    # ------------------------------------------------------------------ #

    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray, y_val: np.ndarray,
              epochs: int = 80, batch_size: int = 32,
              class_weight: Optional[Dict] = None) -> Dict:
        """
        Train the model.

        Args:
            X_train: (samples, time_steps, features) training sequences
            y_train: One-hot encoded labels
            X_val: Validation sequences
            y_val: Validation labels
            epochs: Max epochs
            batch_size: Batch size
            class_weight: Optional class weights dict

        Returns:
            Training history dictionary
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_*() first.")

        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=15,
                restore_best_weights=True, verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss', factor=0.5,
                patience=6, min_lr=1e-7, verbose=1
            ),
            keras.callbacks.ModelCheckpoint(
                'saved_models/speech_best.h5',
                monitor='val_accuracy', save_best_only=True, verbose=1
            )
        ]

        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            class_weight=class_weight,
            verbose=1
        )

        logger.info("Training completed")
        return self.history.history

    # ------------------------------------------------------------------ #
    #  Inference with temperature scaling
    # ------------------------------------------------------------------ #

    def _softmax_with_temperature(self, probs: np.ndarray, T: float) -> np.ndarray:
        """Soften predictions with temperature T > 1."""
        log_probs = np.log(probs + 1e-9) / T
        exp_vals = np.exp(log_probs - np.max(log_probs))
        return exp_vals / exp_vals.sum()

    def predict(self, features: np.ndarray) -> Tuple[str, float, np.ndarray]:
        """
        Predict emotion from audio feature sequence.

        Args:
            features: (time_steps, feature_dim) or (1, time_steps, feature_dim)

        Returns:
            (emotion_label, confidence, probabilities)
        """
        if self.model is None:
            raise ValueError("Model not loaded")

        if len(features.shape) == 2:
            features = np.expand_dims(features, axis=0)

        raw_probs = self.model.predict(features, verbose=0)[0]
        probabilities = self._softmax_with_temperature(raw_probs, self.temperature)

        emotion_idx = np.argmax(probabilities)
        confidence = float(probabilities[emotion_idx])
        emotion_label = EMOTIONS[emotion_idx]

        return emotion_label, confidence, probabilities

    # ------------------------------------------------------------------ #
    #  Persistence
    # ------------------------------------------------------------------ #

    def save_model(self, filepath: str):
        """Save model weights."""
        if self.model is None:
            raise ValueError("No model to save")
        self.model.save(filepath)
        logger.info(f"Model saved to {filepath}")

    def load_model(self, filepath: str):
        """Load saved model."""
        self.model = keras.models.load_model(filepath)
        logger.info(f"Model loaded from {filepath}")
