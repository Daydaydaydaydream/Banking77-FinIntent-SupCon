"""Baseline 1: TF-IDF + Linear SVM.

A classical bag-of-words baseline: sublinear TF-IDF features (word bigrams)
fed into a linear Support Vector Classifier. Fast to train and a useful lower
bound against which the deep models are compared.

The pipeline writes the standard stage-A artefact bundle. ``LinearSVC`` has no
``predict_proba``, so confidence and top-k labels are derived from a softmax
over the decision function; they are indicative, not calibrated. Test-set
TF-IDF vectors are stored as a sparse matrix rather than a dense array.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from config import SEED
from utils.artifacts import RunArtifacts
from utils.data import load_split, load_label_map
from utils.seed import seed_everything

METHOD = "tfidf_svm"


def build_pipeline(seed: int = SEED) -> Pipeline:
    """Return the TF-IDF + LinearSVC pipeline."""
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=2,
        strip_accents="unicode",
    )
    clf = LinearSVC(C=1.0, class_weight="balanced", max_iter=2000, random_state=seed)
    return Pipeline([("tfidf", vectorizer), ("svm", clf)])


def _softmax(scores: np.ndarray) -> np.ndarray:
    """Row-wise softmax with the max subtracted for numerical stability."""
    scores = np.asarray(scores, dtype=float)
    if scores.ndim == 1:
        return np.ones(1)
    shifted = scores - scores.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def run(
    split_name: str,
    seed: int = SEED,
    verbose: bool = False,
    dedup: bool = False,
    artifacts: Optional[RunArtifacts] = None,
) -> Dict:
    """Train and evaluate on a named split, writing the standard artefacts."""
    seed_state = seed_everything(seed, deterministic=False)
    train, test = load_split(split_name, seed=seed, dedup=dedup)
    id2label, _ = load_label_map(dedup)

    x_train = train["text"]
    y_train = train["label_id"].to_numpy()
    x_test = test["text"]
    y_test = test["label_id"].to_numpy()

    pipe = build_pipeline(seed)
    pipe.fit(x_train, y_train)
    y_pred = pipe.predict(x_test)

    decisions = pipe.decision_function(x_test)
    probabilities = _softmax(decisions)

    if artifacts is None:
        artifacts = RunArtifacts.create(METHOD, split_name, seed)
    artifacts.save_config(
        {
            "vectorizer": {"ngram_range": [1, 2], "sublinear_tf": True, "min_df": 2},
            "classifier": {"C": 1.0, "class_weight": "balanced", "max_iter": 2000},
            "n_train": int(len(train)),
            "n_test": int(len(test)),
            "dedup": dedup,
        }
    )
    artifacts.save_env({"seed_state": seed_state})
    metrics = artifacts.save_metrics(
        y_test,
        y_pred,
        id2label,
        extra={"n_train": int(len(train)), "notes": "confidence is a softmax over decision_function"},
    )
    artifacts.save_predictions(y_test, y_pred, id2label, probabilities=probabilities)
    artifacts.save_embeddings(
        pipe.named_steps["tfidf"].transform(x_test), sparse=True
    )

    if verbose:
        print(artifacts.summary_line(metrics["macro_f1"]))

    return {
        "macro_f1": metrics["macro_f1"],
        "per_class_f1": {k: v["f1"] for k, v in metrics["per_class"].items()},
        "artifacts": str(artifacts.root),
    }


if __name__ == "__main__":
    import sys

    name = sys.argv[1] if len(sys.argv) > 1 else "full"
    result = run(name, verbose=True)
    print({k: v for k, v in result.items() if k != "per_class_f1"})
