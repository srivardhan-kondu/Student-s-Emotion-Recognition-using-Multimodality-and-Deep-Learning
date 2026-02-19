"""
Model Architectures for Facial Emotion Recognition.
- MiniXception: Lightweight depthwise-separable CNN (original)  
- EfficientNetV2B0: Transfer learning backbone (upgraded - best accuracy)

Based on:
- "Real-time Convolutional Neural Networks for Emotion and Gender Classification"
- EfficientNetV2: Smaller Models and Faster Training (Tan & Le, 2021)
"""

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv2D, SeparableConv2D, BatchNormalization, Activation,
    MaxPooling2D, GlobalAveragePooling2D, Dense, Dropout, Add
)
from tensorflow.keras.applications import EfficientNetV2B0


def MiniXception(input_shape=(48, 48, 1), num_classes=6):
    """
    Lightweight MiniXception model for low-resource environments.
    Input: 48x48 grayscale
    """
    input_img = Input(shape=input_shape)

    # Entry flow
    x = Conv2D(8, (3, 3), padding='same', use_bias=False)(input_img)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(8, (3, 3), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # Residual depthwise-separable blocks
    for filters in [16, 32, 64, 128]:
        residual = Conv2D(filters, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)
        residual = BatchNormalization()(residual)

        x = SeparableConv2D(filters, (3, 3), padding='same', use_bias=False)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = SeparableConv2D(filters, (3, 3), padding='same', use_bias=False)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
        x = Add()([x, residual])

    # Classification block
    x = Conv2D(num_classes, (3, 3), padding='same')(x)
    x = GlobalAveragePooling2D()(x)
    output = Activation('softmax')(x)

    return Model(inputs=input_img, outputs=output)


def build_efficientnet_model(num_classes=6, input_shape=(96, 96, 3),
                              trainable_base_layers=30):
    """
    EfficientNetV2-B0 with transfer learning for facial emotion recognition.
    
    Architecture:
        EfficientNetV2B0 (ImageNet pretrained, partial fine-tune)
        → GlobalAveragePooling2D
        → Dense(512, relu) + BatchNorm + Dropout(0.4)
        → Dense(256, relu) + BatchNorm + Dropout(0.3)
        → Dense(num_classes, softmax)

    Args:
        num_classes: Number of emotion classes (default 6)
        input_shape: RGB input shape (96x96x3 recommended)
        trainable_base_layers: How many top base layers to fine-tune

    Returns:
        Compiled Keras model
    """
    # Load EfficientNetV2B0 pretrained on ImageNet, no top
    base_model = EfficientNetV2B0(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )

    # Freeze all layers first
    base_model.trainable = False

    # Unfreeze top N layers for fine-tuning
    for layer in base_model.layers[-trainable_base_layers:]:
        layer.trainable = True

    # Build classification head
    inputs = Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D()(x)

    x = Dense(512, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)

    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model


def cutout_layer(image, n_holes=1, length=12):
    """
    Apply cutout augmentation to a single image tensor.
    Randomly masks n_holes square patches of size length x length.
    
    Args:
        image: Image tensor (H, W, C)
        n_holes: Number of holes to cut
        length: Side length of each hole

    Returns:
        Augmented image tensor
    """
    h = tf.shape(image)[0]
    w = tf.shape(image)[1]

    mask = tf.ones((h, w), dtype=tf.float32)

    for _ in range(n_holes):
        y = tf.random.uniform((), 0, h, dtype=tf.int32)
        x = tf.random.uniform((), 0, w, dtype=tf.int32)

        y1 = tf.maximum(0, y - length // 2)
        y2 = tf.minimum(h, y + length // 2)
        x1 = tf.maximum(0, x - length // 2)
        x2 = tf.minimum(w, x + length // 2)

        holes_h = y2 - y1
        holes_w = x2 - x1

        padding = [[y1, h - y2], [x1, w - x2]]
        hole = tf.zeros((holes_h, holes_w), dtype=tf.float32)
        hole_padded = tf.pad(hole, padding, constant_values=1.0)
        mask = mask * hole_padded

    mask = tf.expand_dims(mask, axis=-1)
    mask = tf.tile(mask, [1, 1, tf.shape(image)[2]])
    return image * mask
