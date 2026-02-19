"""
Data preprocessing for facial emotion recognition.
"""

import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from typing import Tuple
import logging

logger = logging.getLogger(__name__)


def load_fer2013(csv_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load FER2013 dataset from CSV file.
    
    Args:
        csv_path: Path to FER2013 CSV file
        
    Returns:
        Tuple of (images, labels)
    """
    import pandas as pd
    
    df = pd.read_csv(csv_path)
    
    # Extract pixels and convert to numpy arrays
    images = []
    labels = []
    
    for idx, row in df.iterrows():
        pixels = np.array(row['pixels'].split(), dtype='float32')
        image = pixels.reshape(48, 48, 1)
        images.append(image)
        labels.append(row['emotion'])
    
    images = np.array(images)
    labels = np.array(labels)
    
    logger.info(f"Loaded {len(images)} images from FER2013")
    return images, labels


def preprocess_images(images: np.ndarray, 
                     target_size: Tuple[int, int] = (48, 48),
                     normalize: bool = True) -> np.ndarray:
    """
    Preprocess images for model input.
    
    Args:
        images: Input images
        target_size: Target size for resizing
        normalize: Normalize pixel values if True
        
    Returns:
        Preprocessed images
    """
    processed = []
    
    for img in images:
        # Resize if needed
        if img.shape[:2] != target_size:
            img = cv2.resize(img, target_size)
        
        # Ensure correct shape
        if len(img.shape) == 2:
            img = np.expand_dims(img, axis=-1)
        
        processed.append(img)
    
    processed = np.array(processed)
    
    # Normalize
    if normalize:
        processed = processed.astype('float32') / 255.0
    
    logger.info(f"Preprocessed {len(processed)} images")
    return processed


def prepare_data(images: np.ndarray, labels: np.ndarray,
                test_size: float = 0.15, val_size: float = 0.15,
                num_classes: int = 6) -> Tuple:
    """
    Prepare data for training.
    
    Args:
        images: Input images
        labels: Labels
        test_size: Test set size
        val_size: Validation set size
        num_classes: Number of emotion classes
        
    Returns:
        Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    # Filter to only include 6 emotions (0-5)
    # FER2013 has 7 emotions, we exclude 'disgust' (emotion 1)
    emotion_mapping = {0: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5}  # Map to 0-5
    
    filtered_images = []
    filtered_labels = []
    
    for img, label in zip(images, labels):
        if label in emotion_mapping:
            filtered_images.append(img)
            filtered_labels.append(emotion_mapping[label])
    
    images = np.array(filtered_images)
    labels = np.array(filtered_labels)
    
    # Split into train and temp (val + test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        images, labels, test_size=(test_size + val_size), 
        random_state=42, stratify=labels
    )
    
    # Split temp into val and test
    val_ratio = val_size / (test_size + val_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=(1 - val_ratio),
        random_state=42, stratify=y_temp
    )
    
    # Convert labels to categorical
    y_train = to_categorical(y_train, num_classes)
    y_val = to_categorical(y_val, num_classes)
    y_test = to_categorical(y_test, num_classes)
    
    logger.info(f"Data split - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def augment_data(images: np.ndarray, labels: np.ndarray,
                augmentation_factor: int = 2) -> Tuple[np.ndarray, np.ndarray]:
    """
    Augment dataset with transformations.
    
    Args:
        images: Input images
        labels: Labels
        augmentation_factor: Number of augmented versions per image
        
    Returns:
        Tuple of (augmented_images, augmented_labels)
    """
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        zoom_range=0.2,
        fill_mode='nearest'
    )
    
    augmented_images = [images]
    augmented_labels = [labels]
    
    for _ in range(augmentation_factor - 1):
        aug_imgs = []
        for img in images:
            img_aug = datagen.random_transform(img)
            aug_imgs.append(img_aug)
        augmented_images.append(np.array(aug_imgs))
        augmented_labels.append(labels)
    
    augmented_images = np.concatenate(augmented_images)
    augmented_labels = np.concatenate(augmented_labels)
    
    logger.info(f"Augmented dataset to {len(augmented_images)} images")
    return augmented_images, augmented_labels
