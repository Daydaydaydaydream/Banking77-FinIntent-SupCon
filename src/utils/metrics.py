"""Evaluation metrics shared by all baselines.

All baselines report Macro-F1 as the primary metric (per the project plan),
with per-class F1 and a confusion matrix available for analysis.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
)


def macro_f1(y_true, y_pred) -> float:
    """Macro-averaged F1 score across all classes."""
    return float(f1_score(y_true, y_pred, average="macro"))


def per_class_f1(y_true, y_pred, labels: List[str]) -> Dict[str, float]:
    """F1 score computed independently for each class label."""
    scores = f1_score(y_true, y_pred, average=None)
    return {name: float(v) for name, v in zip(labels, scores)}


def report(y_true, y_pred, labels: List[str]) -> Dict[str, float]:
    """Return a small summary: macro-F1 and the per-class F1 dict."""
    return {
        "macro_f1": macro_f1(y_true, y_pred),
        "per_class_f1": per_class_f1(y_true, y_pred, labels),
    }


def confusion_matrix_np(y_true, y_pred) -> np.ndarray:
    """Return the raw confusion matrix as a numpy array."""
    return confusion_matrix(y_true, y_pred)


def full_classification_report(y_true, y_pred, labels: List[str]) -> str:
    """Return the sklearn classification report as text."""
    return classification_report(y_true, y_pred, target_names=labels, digits=4)
