"""Train SongNet from precomputed mel-spectrogram arrays.

Expected files:
data/processed/X_train.npy, y_train.npy
data/processed/X_validation.npy, y_validation.npy
"""

import numpy as np

from src.train import compile_model, train

def main():
    x_train = np.load("data/processed/X_train.npy")
    y_train = np.load("data/processed/y_train.npy")
    x_val = np.load("data/processed/X_validation.npy")
    y_val = np.load("data/processed/y_validation.npy")

    model = compile_model(x_train.shape[1:])
    model.summary()
    history = train(model, x_train, y_train, x_val, y_val)
    print("Best validation accuracy:", max(history.history["val_accuracy"]))

if __name__ == "__main__":
    main()
