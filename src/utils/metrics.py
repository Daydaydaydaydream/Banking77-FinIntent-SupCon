"""Common classification metrics."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support


def classification_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    categories: list[str],
) -> dict[str, Any]:
    labels = np.arange(len(categories))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(y_true, y_pred, labels=labels, average="micro", zero_division=0)),
        "per_class": {
            category: {
                "precision": float(precision[idx]),
                "recall": float(recall[idx]),
                "f1": float(f1[idx]),
                "support": int(support[idx]),
            }
            for idx, category in enumerate(categories)
        },
    }


def confusion_rows(y_true: Sequence[int], y_pred: Sequence[int], categories: list[str]) -> list[list[int | str]]:
    matrix = confusion_matrix(y_true, y_pred, labels=np.arange(len(categories)))
    return [[category, *map(int, matrix[row_idx]) ] for row_idx, category in enumerate(categories)]
