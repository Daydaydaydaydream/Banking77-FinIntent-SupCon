"""Baseline 1: TF-IDF + Linear SVM.

A classical bag-of-words baseline: sublinear TF-IDF features (word and
character n-grams) fed into a linear Support Vector Classifier. Fast to train
and a useful lower bound against which the deep models are compared.
"""

from __future__ import annotations

from typing import Dict, Tuple

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from utils.data import load_split
from utils.metrics import report


def build_pipeline() -> Pipeline:
    """Return the TF-IDF + LinearSVC pipeline."""
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=2,
        strip_accents="unicode",
    )
    clf = LinearSVC(C=1.0, class_weight="balanced", max_iter=2000, random_state=42)
    return Pipeline([("tfidf", vectorizer), ("svm", clf)])


def run(split_name: str, verbose: bool = False) -> Dict[str, float]:
    """Train and evaluate on a named split; return {macro_f1, per_class_f1}."""
    train, test = load_split(split_name)
    x_train: pd.Series = train["text"]
    y_train = train["label_id"].to_numpy()
    x_test: pd.Series = test["text"]
    y_test = test["label_id"].to_numpy()

    pipe = build_pipeline()
    pipe.fit(x_train, y_train)
    y_pred = pipe.predict(x_test)

    from utils.data import load_label_map

    id2label, _ = load_label_map()
    result = report(y_test, y_pred, id2label)
    if verbose:
        print(f"[tfidf-svm] {split_name}: Macro-F1 = {result['macro_f1']:.4f}")
    return result


if __name__ == "__main__":
    import sys

    name = sys.argv[1] if len(sys.argv) > 1 else "full"
    run(name, verbose=True)
