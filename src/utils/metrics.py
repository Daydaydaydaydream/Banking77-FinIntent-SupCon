"""Evaluation metrics shared by all baselines.

Every baseline reports Macro-F1 as the primary metric. Stage A additionally
standardised the secondary metrics (accuracy, micro-F1, per-class precision /
recall / F1) and added a stratified paired-bootstrap test so that a reported
improvement can be accompanied by a confidence interval instead of a bare
point estimate.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)


def macro_f1(y_true, y_pred) -> float:
    """Macro-averaged F1 score across all classes."""
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def micro_f1(y_true, y_pred) -> float:
    """Micro-averaged F1 (equals accuracy for single-label classification)."""
    return float(f1_score(y_true, y_pred, average="micro", zero_division=0))


def accuracy(y_true, y_pred) -> float:
    """Overall accuracy."""
    return float(accuracy_score(y_true, y_pred))


def per_class_f1(y_true, y_pred, labels: Sequence[str]) -> Dict[str, float]:
    """F1 score computed independently for each class label."""
    scores = f1_score(y_true, y_pred, average=None, zero_division=0)
    return {name: float(v) for name, v in zip(labels, scores)}


def per_class_prf(y_true, y_pred, labels: Sequence[str]) -> Dict[str, Dict[str, float]]:
    """Per-class precision, recall, F1 and support.

    Support is reported so that a per-class F1 can always be read together with
    the number of test examples behind it — with 40 examples per class, a single
    flipped prediction moves F1 by roughly 1.2 points.
    """
    precision, recall, fscore, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(len(labels))), zero_division=0
    )
    return {
        name: {
            "precision": float(p),
            "recall": float(r),
            "f1": float(f),
            "support": int(s),
        }
        for name, p, r, f, s in zip(labels, precision, recall, fscore, support)
    }


def summarize(
    y_true,
    y_pred,
    labels: Sequence[str],
    extra: Optional[Dict] = None,
) -> Dict:
    """Return the standard metric block written to ``metrics.json``."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    block = {
        "macro_f1": macro_f1(y_true, y_pred),
        "micro_f1": micro_f1(y_true, y_pred),
        "accuracy": accuracy(y_true, y_pred),
        "n_samples": int(y_true.shape[0]),
        "n_classes": int(len(labels)),
        "per_class": per_class_prf(y_true, y_pred, labels),
    }
    if extra:
        block.update(extra)
    return block


def report(y_true, y_pred, labels: Sequence[str]) -> Dict:
    """Return a small summary: macro-F1 and the per-class F1 dict."""
    return {
        "macro_f1": macro_f1(y_true, y_pred),
        "per_class_f1": per_class_f1(y_true, y_pred, labels),
    }


def confusion_matrix_np(y_true, y_pred) -> np.ndarray:
    """Return the raw confusion matrix as a numpy array."""
    return confusion_matrix(y_true, y_pred)


def full_classification_report(y_true, y_pred, labels: Sequence[str]) -> str:
    """Return the sklearn classification report as text."""
    return classification_report(
        y_true, y_pred, target_names=list(labels), digits=4, zero_division=0
    )


def paired_bootstrap_macro_f1(
    y_true,
    pred_a,
    pred_b,
    n_resamples: int = 1000,
    seed: int = 42,
    confidence: float = 0.95,
) -> Dict[str, float]:
    """Stratified paired bootstrap over per-class Macro-F1 differences.

    Resampling is done **within** each true class so every resample keeps the
    original class balance; an unstratified bootstrap would inject variance that
    has nothing to do with the models being compared.

    Returns the mean delta ``F1(a) - F1(b)``, its percentile confidence
    interval, and a two-sided p-value computed as twice the smaller tail mass
    (i.e. how often the sign of the difference flips under resampling).
    """
    y_true = np.asarray(y_true)
    pred_a = np.asarray(pred_a)
    pred_b = np.asarray(pred_b)

    rng = np.random.default_rng(seed)
    strata = [np.where(y_true == c)[0] for c in np.unique(y_true)]

    deltas = np.empty(n_resamples, dtype=float)
    for i in range(n_resamples):
        idx = np.concatenate(
            [rng.choice(s, size=s.shape[0], replace=True) for s in strata]
        )
        f1_a = f1_score(y_true[idx], pred_a[idx], average="macro", zero_division=0)
        f1_b = f1_score(y_true[idx], pred_b[idx], average="macro", zero_division=0)
        deltas[i] = f1_a - f1_b

    alpha = (1.0 - confidence) / 2.0
    low, high = np.percentile(deltas, [alpha * 100, (1 - alpha) * 100])
    tail = min(float(np.mean(deltas <= 0.0)), float(np.mean(deltas >= 0.0)))

    return {
        "mean_delta": float(np.mean(deltas)),
        "ci_low": float(low),
        "ci_high": float(high),
        "p_value": float(min(1.0, 2.0 * tail)),
        "n_resamples": int(n_resamples),
        "confidence": float(confidence),
    }
