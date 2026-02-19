"""
Training script for facial emotion recognition model — Upgraded.
Key improvements:
  - EfficientNetV2-B0 with ImageNet weights (96x96 RGB)
  - Class-weight balancing for FER2013 imbalance
  - Label smoothing (0.1)
  - Cutout augmentation via preprocessing_function
  - Two-phase training: frozen base → unfreeze top layers
Implements FR19 (training) and FR20 (evaluation).
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
import tensorflow as tf

from utils.helpers import (
    plot_confusion_matrix, plot_training_history,
    calculate_metrics, print_metrics, save_model_info
)
from config import FACIAL_CONFIG, MODEL_SAVE_PATHS
from facial_recognition.emotion_model import EmotionCNN
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Emotions: matches FER2013 folder names we keep
EMOTIONS = ['angry', 'fear', 'happy', 'neutral', 'sad', 'surprise']
INPUT_SIZE_EFFICIENTNET = (96, 96)


def load_fer2013_folders(base_dir: str, img_size=(96, 96), rgb=True):
    """
    Load FER2013 from folder structure.
    Expected layout:
        base_dir/train/{emotion}/image.jpg
        base_dir/test/{emotion}/image.jpg

    IMPORTANT: Returns images as float32 in range [0, 255].
    EfficientNetV2B0 has its own internal Rescaling layer and
    expects raw pixel values — DO NOT divide by 255 here.

    Args:
        base_dir: Root dataset directory
        img_size: Target (H, W)
        rgb: If True, convert to 3-channel RGB for EfficientNet

    Returns:
        (images, labels) numpy arrays
    """
    emotion_map = {
        'angry': 0,
        'fear': 1,
        'happy': 2,
        'neutral': 3,
        'sad': 4,
        'surprise': 5
    }

    images, labels = [], []

    for split in ['train', 'test']:
        split_dir = os.path.join(base_dir, split)
        if not os.path.exists(split_dir):
            logger.warning(f"Directory not found: {split_dir}")
            continue

        for emotion_name in os.listdir(split_dir):
            if emotion_name not in emotion_map:
                logger.info(f"Skipping emotion folder: {emotion_name}")
                continue

            emotion_dir = os.path.join(split_dir, emotion_name)
            if not os.path.isdir(emotion_dir):
                continue

            label = emotion_map[emotion_name]
            count = 0

            for img_file in os.listdir(emotion_dir):
                img_path = os.path.join(emotion_dir, img_file)
                try:
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    if img is None:
                        continue
                    img = cv2.resize(img, img_size)
                    if rgb:
                        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                    images.append(img)
                    labels.append(label)
                    count += 1
                except Exception:
                    pass

            logger.info(f"  {split}/{emotion_name}: {count} images")

    # Keep as 0-255 float32 — EfficientNet rescales internally
    images = np.array(images, dtype='float32')
    labels = np.array(labels)

    logger.info(f"Total: {len(images)} images across {len(set(labels))} classes")
    return images, labels


def get_class_weights(labels: np.ndarray) -> dict:
    """Compute balanced class weights to handle FER2013 imbalance."""
    classes = np.unique(labels)
    weights = compute_class_weight('balanced', classes=classes, y=labels)
    cw = {int(c): float(w) for c, w in zip(classes, weights)}
    logger.info(f"Class weights: {cw}")
    return cw


def apply_cutout(img: np.ndarray, length: int = 14) -> np.ndarray:
    """
    Cutout augmentation — zeros out a random square patch.
    Works for both 0-1 and 0-255 image ranges.
    """
    h, w = img.shape[:2]
    y = np.random.randint(0, h)
    x = np.random.randint(0, w)
    y1 = max(0, y - length // 2)
    y2 = min(h, y + length // 2)
    x1 = max(0, x - length // 2)
    x2 = min(w, x + length // 2)
    img_copy = img.copy()
    # Zero fill (works with any pixel range)
    img_copy[y1:y2, x1:x2] = 0
    return img_copy


def main():
    """Main training function — uses EfficientNetV2-B0 by default."""
    logger.info("=" * 60)
    logger.info("  FACIAL EMOTION RECOGNITION - TRAINING (EfficientNetV2-B0)")
    logger.info("=" * 60)

    dataset_path = "data/facial/fer2013"
    model_save_path = str(MODEL_SAVE_PATHS['facial'])

    if not os.path.exists(os.path.join(dataset_path, 'train')):
        logger.error(f"Dataset not found at {dataset_path}/train")
        logger.info("Download FER2013 from Kaggle: https://www.kaggle.com/datasets/msambare/fer2013")
        return

    # ------------------------------------------------------------------ #
    #  Load & split data
    # ------------------------------------------------------------------ #
    logger.info("Loading FER2013 dataset (96x96 RGB for EfficientNet)...")
    images, labels = load_fer2013_folders(
        dataset_path, img_size=INPUT_SIZE_EFFICIENTNET, rgb=True
    )

    if len(images) == 0:
        logger.error("No images loaded!")
        return

    # Stratified 80/10/10 split
    X_train, X_temp, y_train, y_temp = train_test_split(
        images, labels, test_size=0.2, random_state=42, stratify=labels
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    logger.info(f"Train: {X_train.shape} | Val: {X_val.shape} | Test: {X_test.shape}")

    # Compute class weights BEFORE one-hot encoding
    class_weights = get_class_weights(y_train)

    # One-hot encode
    y_train_cat = to_categorical(y_train, 6)
    y_val_cat = to_categorical(y_val, 6)
    y_test_cat = to_categorical(y_test, 6)

    # ------------------------------------------------------------------ #
    #  Build model — EfficientNetV2-B0
    # ------------------------------------------------------------------ #
    logger.info("Building EfficientNetV2-B0 model...")
    model_wrapper = EmotionCNN(input_shape=(96, 96, 3), num_classes=6)
    model_wrapper.build_model(architecture='efficientnet')
    model_wrapper.model.summary()

    # Use legacy Adam on Apple Silicon for ~3x speedup
    try:
        optimizer_phase1 = tf.keras.optimizers.legacy.Adam(learning_rate=1e-4)
        optimizer_phase2 = tf.keras.optimizers.legacy.Adam(learning_rate=2e-5)
        logger.info("Using legacy Adam optimizer (M1/M2 optimized)")
    except AttributeError:
        optimizer_phase1 = Adam(learning_rate=1e-4)
        optimizer_phase2 = Adam(learning_rate=2e-5)

    model_wrapper.model.compile(
        optimizer=optimizer_phase1,
        loss=CategoricalCrossentropy(label_smoothing=0.1),
        metrics=['accuracy']
    )

    # ------------------------------------------------------------------ #
    #  Phase 1: Train classification head (base frozen)
    # ------------------------------------------------------------------ #
    os.makedirs('saved_models', exist_ok=True)
    os.makedirs('docs', exist_ok=True)

    # EfficientNet input is 0-255; augmentation ranges must match
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        zoom_range=0.2,
        shear_range=0.1,
        # brightness_range is a multiplier (works on any scale)
        brightness_range=[0.7, 1.3],
        fill_mode='nearest',
        preprocessing_function=apply_cutout
    )
    datagen.fit(X_train)

    callbacks_phase1 = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-6),
        ModelCheckpoint(model_save_path, monitor='val_accuracy', save_best_only=True)
    ]

    logger.info("Phase 1: Training head with frozen base (30 epochs max)...")
    history1 = model_wrapper.model.fit(
        datagen.flow(X_train, y_train_cat, batch_size=32),
        validation_data=(X_val, y_val_cat),
        epochs=30,
        callbacks=callbacks_phase1,
        class_weight=class_weights,
        verbose=1
    )

    # ------------------------------------------------------------------ #
    #  Phase 2: Fine-tune top 30 layers of EfficientNet base
    # ------------------------------------------------------------------ #
    logger.info("Phase 2: Fine-tuning top 30 base layers (lower LR)...")
    base_model = model_wrapper.model.layers[1]   # EfficientNetV2B0 sub-model
    base_model.trainable = True
    # Freeze all but the top 30 layers
    for layer in base_model.layers[:-30]:
        layer.trainable = False

    model_wrapper.model.compile(
        optimizer=optimizer_phase2,
        loss=CategoricalCrossentropy(label_smoothing=0.1),
        metrics=['accuracy']
    )

    callbacks_phase2 = [
        EarlyStopping(monitor='val_loss', patience=12, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-8),
        ModelCheckpoint(model_save_path, monitor='val_accuracy', save_best_only=True)
    ]

    history2 = model_wrapper.model.fit(
        datagen.flow(X_train, y_train_cat, batch_size=16),
        validation_data=(X_val, y_val_cat),
        epochs=30,
        callbacks=callbacks_phase2,
        class_weight=class_weights,
        verbose=1
    )

    # ------------------------------------------------------------------ #
    #  Evaluate
    # ------------------------------------------------------------------ #
    logger.info("Evaluating on test set...")
    test_loss, test_accuracy = model_wrapper.model.evaluate(X_test, y_test_cat, verbose=0)
    logger.info(f"Test Loss: {test_loss:.4f} | Test Accuracy: {test_accuracy:.4f}")

    y_pred_probs = model_wrapper.model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)

    metrics = calculate_metrics(y_test, y_pred, EMOTIONS)
    print_metrics(metrics)

    plot_confusion_matrix(
        y_test, y_pred, EMOTIONS,
        save_path='docs/facial_confusion_matrix.png'
    )
    # Combine histories for plot
    combined_history = {
        'accuracy': history1.history['accuracy'] + history2.history['accuracy'],
        'val_accuracy': history1.history['val_accuracy'] + history2.history['val_accuracy'],
        'loss': history1.history['loss'] + history2.history['loss'],
        'val_loss': history1.history['val_loss'] + history2.history['val_loss'],
    }
    plot_training_history(combined_history, save_path='docs/facial_training_history.png')
    save_model_info(model_save_path, metrics, FACIAL_CONFIG)

    logger.info("=" * 60)
    logger.info("  FACIAL TRAINING COMPLETED!")
    logger.info(f"  Model saved: {model_save_path}")
    logger.info(f"  Test accuracy: {test_accuracy:.4f}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
