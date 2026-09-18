"""TF-IDF + Linear SVM baseline."""

from __future__ import annotations

import time
from typing import Any

import joblib
import numpy as np
from scipy.special import softmax
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from utils.artifacts import RunArtifacts
from utils.data import encode_labels, load_splits
from utils.metrics import classification_metrics
from utils.seed import seed_everything


METHOD = "tfidf_svm"


def run(
    setting: str,
    seed: int,
    *,
    c: float = 1.0,
    min_df: int | None = None,
    save_model: bool = True,
) -> dict[str, Any]:
    seed_info = seed_everything(seed)
    train, validation, test, categories = load_splits(setting, seed)
    y_train = encode_labels(train, categories)
    y_val = encode_labels(validation, categories)
    y_test = encode_labels(test, categories)
    actual_min_df = min_df if min_df is not None else (1 if setting != "full" else 2)

    artifacts = RunArtifacts(METHOD, setting, seed)
    config = {
        "method": METHOD,
        "setting": setting,
        "seed": seed,
        "train_rows": len(train),
        "validation_rows": len(validation),
        "test_rows": len(test),
        "tfidf": {
            "ngram_range": [1, 2],
            "sublinear_tf": True,
            "min_df": actual_min_df,
            "max_features": None,
        },
        "svm": {"C": c, "class_weight": "balanced", "max_iter": 5000},
        "score_note": "top-k scores are softmax-normalized SVM decision scores and are not calibrated probabilities",
        "seed_info": seed_info,
    }
    artifacts.write_config(config)

    model = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    sublinear_tf=True,
                    min_df=actual_min_df,
                ),
            ),
            (
                "classifier",
                LinearSVC(C=c, class_weight="balanced", max_iter=5000, random_state=seed),
            ),
        ]
    )
    started = time.perf_counter()
    model.fit(train["text"], y_train)
    train_seconds = time.perf_counter() - started

    val_scores = softmax(model.decision_function(validation["text"]), axis=1)
    val_pred = val_scores.argmax(axis=1)
    validation_metrics = classification_metrics(y_val, val_pred, categories)

    started = time.perf_counter()
    test_scores = softmax(model.decision_function(test["text"]), axis=1)
    inference_seconds = time.perf_counter() - started
    y_pred = artifacts.write_predictions(test["text"].tolist(), y_test, test_scores, categories)
    test_metrics = classification_metrics(y_test, y_pred, categories)
    artifacts.write_confusion(y_test, y_pred, categories)
    artifacts.write_metrics(
        validation_metrics,
        test_metrics,
        {
            "train_seconds": train_seconds,
            "test_inference_seconds": inference_seconds,
            "test_examples_per_second": len(test) / inference_seconds,
        },
    )
    if save_model:
        joblib.dump(model, artifacts.model_dir / "pipeline.joblib")

    return {
        "method": METHOD,
        "setting": setting,
        "seed": seed,
        "validation_macro_f1": validation_metrics["macro_f1"],
        "test_macro_f1": test_metrics["macro_f1"],
        "output_dir": str(artifacts.path),
    }
