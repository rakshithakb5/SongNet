"""SongNet CNN-RNN architecture."""

import tensorflow as tf
from tensorflow.keras import layers, Model
from .config import GENRES

def build_crnn(input_shape, num_classes=len(GENRES)):
    inputs = layers.Input(shape=input_shape, name="mel_spectrogram")
    x = layers.Conv2D(32, (3, 3), padding="same", activation="relu")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)
    x = layers.Conv2D(64, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)
    x = layers.Conv2D(128, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)
    x = layers.Permute((2, 1, 3))(x)
    x = layers.TimeDistributed(layers.Flatten())(x)
    x = layers.GRU(128, return_sequences=True)(x)
    x = layers.Dropout(0.30)(x)
    x = layers.TimeDistributed(layers.Dense(num_classes, activation="softmax"))(x)
    outputs = layers.Lambda(lambda t: tf.reduce_mean(t, axis=1), name="genre_probabilities")(x)
    return Model(inputs, outputs, name="SongNet_CRNN")
