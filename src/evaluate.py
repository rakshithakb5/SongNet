"""Evaluation utilities for SongNet."""

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from .config import GENRES

def evaluate_predictions(y_true, probabilities):
    y_pred = np.argmax(probabilities, axis=1)
    return {"accuracy": accuracy_score(y_true, y_pred), "confusion_matrix": confusion_matrix(y_true, y_pred), "classification_report": classification_report(y_true, y_pred, target_names=GENRES, zero_division=0)}
