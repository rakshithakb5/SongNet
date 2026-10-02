"""Training utilities for the SongNet CRNN model."""

import tensorflow as tf
from .config import MODEL_DIR
from .model import build_crnn

def compile_model(input_shape):
    model = build_crnn(input_shape)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model

def get_callbacks():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    return [
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=7, restore_best_weights=True, verbose=1),
        tf.keras.callbacks.ModelCheckpoint(MODEL_DIR / "songnet_best.keras", monitor="val_accuracy", save_best_only=True, verbose=1),
    ]

def train(model, x_train, y_train, x_val, y_val, epochs=50, batch_size=32):
    return model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=epochs, batch_size=batch_size, callbacks=get_callbacks(), shuffle=True)
