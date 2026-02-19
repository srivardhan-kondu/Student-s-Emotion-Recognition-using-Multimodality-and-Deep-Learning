"""
Training script for speech emotion recognition — Upgraded.
Key improvements:
  - Feeds raw MFCC+delta+delta2 time-series to Attention-BiLSTM
  - On-the-fly audio augmentation (noise, pitch shift, time stretch)
  - Class-weight balancing
  - Label smoothing loss
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import glob
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.utils import to_categorical

from speech_analysis.emotion_model import SpeechEmotionModel
from speech_analysis.audio_features import AudioFeatureExtractor, MAX_TIME_STEPS, FEATURE_DIM
from utils.helpers import (
    plot_confusion_matrix, plot_training_history,
    calculate_metrics, print_metrics, save_model_info
)
from config import SPEECH_CONFIG, MODEL_SAVE_PATHS
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

EMOTIONS = ['happy', 'sad', 'angry', 'neutral', 'fear', 'surprise']


def load_ravdess_sequences(data_dir: str, augment: bool = True):
    """
    Load RAVDESS and return (N, TIME_STEPS, FEATURE_DIM) sequences.
    With augmentation, dataset size is multiplied ~4x.

    RAVDESS filename: 03-01-06-01-02-01-12.wav
    Position 3 (index 2): emotion code
      01=neutral, 02=calm, 03=happy, 04=sad, 05=angry, 06=fearful, 08=surprised
    """
    emotion_map = {
        '01': 3,  # neutral
        '03': 0,  # happy
        '04': 1,  # sad
        '05': 2,  # angry
        '06': 4,  # fear
        '08': 5   # surprise
    }

    extractor = AudioFeatureExtractor(sample_rate=SPEECH_CONFIG.get('sample_rate', 22050))
    sequences, labels = [], []
    augmented_count = 0

    audio_files = glob.glob(os.path.join(data_dir, "**/*.wav"), recursive=True)
    logger.info(f"Found {len(audio_files)} audio files in {data_dir}")

    for audio_file in audio_files:
        fname = os.path.basename(audio_file)
        parts = fname.split('-')
        if len(parts) < 3:
            continue

        code = parts[2]
        if code not in emotion_map:
            continue

        label = emotion_map[code]

        try:
            audio, sr = extractor.load_audio(audio_file)

            # Original + augmented versions
            if augment:
                versions = extractor.augment_audio(audio, sr=sr)
            else:
                versions = [audio]

            for aug_audio in versions:
                seq = extractor.extract_mfcc_sequence(aug_audio, max_steps=MAX_TIME_STEPS)
                sequences.append(seq)
                labels.append(label)
                augmented_count += 1

        except Exception as e:
            logger.warning(f"Error processing {audio_file}: {e}")

    logger.info(f"Total sequences (with augmentation): {len(sequences)}")
    return np.array(sequences, dtype='float32'), np.array(labels)


def main():
    """Main training function — Attention-BiLSTM with raw MFCC sequences."""
    logger.info("=" * 60)
    logger.info("  SPEECH EMOTION RECOGNITION - TRAINING (Attention-BiLSTM)")
    logger.info("=" * 60)

    dataset_path = "data/speech/ravdess"
    model_save_path = str(MODEL_SAVE_PATHS['speech'])

    # ------------------------------------------------------------------ #
    #  Load data
    # ------------------------------------------------------------------ #
    if not os.path.exists(dataset_path):
        logger.warning(f"Dataset not found at {dataset_path}")
        logger.info("Using synthetic sequences for demonstration...")
        X = np.random.randn(800, MAX_TIME_STEPS, FEATURE_DIM).astype('float32')
        y = np.random.randint(0, 6, 800)
    else:
        logger.info("Loading RAVDESS with augmentation...")
        X, y = load_ravdess_sequences(dataset_path, augment=True)

    logger.info(f"Dataset shape: X={X.shape}, y={y.shape}")

    # ------------------------------------------------------------------ #
    #  Split
    # ------------------------------------------------------------------ #
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    logger.info(f"Train={X_train.shape} | Val={X_val.shape} | Test={X_test.shape}")

    # Class weights
    classes = np.unique(y_train)
    weights = compute_class_weight('balanced', classes=classes, y=y_train)
    class_weights = {int(c): float(w) for c, w in zip(classes, weights)}
    logger.info(f"Class weights: {class_weights}")

    # One-hot encode
    y_train_cat = to_categorical(y_train, 6)
    y_val_cat = to_categorical(y_val, 6)
    y_test_cat = to_categorical(y_test, 6)

    # ------------------------------------------------------------------ #
    #  Build Attention-BiLSTM
    # ------------------------------------------------------------------ #
    input_shape = (X_train.shape[1], X_train.shape[2])  # (TIME_STEPS, FEATURE_DIM)
    logger.info(f"Model input shape: {input_shape}")

    model_obj = SpeechEmotionModel(input_shape=input_shape, num_classes=6)
    model = model_obj.build_attention_bilstm()
    model.summary()

    # ------------------------------------------------------------------ #
    #  Train
    # ------------------------------------------------------------------ #
    os.makedirs('saved_models', exist_ok=True)
    os.makedirs('docs', exist_ok=True)

    history = model_obj.train(
        X_train, y_train_cat,
        X_val, y_val_cat,
        epochs=SPEECH_CONFIG.get('epochs', 80),
        batch_size=SPEECH_CONFIG.get('batch_size', 32),
        class_weight=class_weights
    )

    # ------------------------------------------------------------------ #
    #  Evaluate
    # ------------------------------------------------------------------ #
    logger.info("Evaluating on test set...")
    test_loss, test_accuracy = model.evaluate(X_test, y_test_cat, verbose=0)
    logger.info(f"Test Loss: {test_loss:.4f} | Test Accuracy: {test_accuracy:.4f}")

    y_pred_probs = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)

    metrics = calculate_metrics(y_test, y_pred, EMOTIONS)
    print_metrics(metrics)

    plot_confusion_matrix(y_test, y_pred, EMOTIONS,
                          save_path='docs/speech_confusion_matrix.png')
    plot_training_history(history, save_path='docs/speech_training_history.png')

    logger.info(f"Saving model to {model_save_path}...")
    model_obj.save_model(str(model_save_path))
    save_model_info(str(model_save_path), metrics, SPEECH_CONFIG)

    logger.info("=" * 60)
    logger.info("  SPEECH TRAINING COMPLETED!")
    logger.info(f"  Model saved: {model_save_path}")
    logger.info(f"  Test accuracy: {test_accuracy:.4f}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
