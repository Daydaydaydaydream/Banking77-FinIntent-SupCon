"""Baseline 3: native SetFit.

Uses the ``setfit`` library's two-stage training: contrastive fine-tuning of a
sentence-transformer body followed by a logistic-regression head trained on the
frozen embeddings (``use_differentiable_head=False``). The backbone is
``sentence-transformers/all-MiniLM-L6-v2``, a small sentence encoder well suited
to short-utterance classification.

Few-shot settings use the canonical SetFit configuration (many contrastive
iterations); the full-data setting uses fewer iterations to keep runtime
reasonable.

Stage A added the missing exports: probabilities, body embeddings and the
trained model, so SetFit predictions can be compared sample-by-sample with the
other methods.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd
from datasets import Dataset
from setfit import SetFitModel, Trainer, TrainingArguments

from config import SEED
from utils.artifacts import RunArtifacts
from utils.data import load_label_map, load_split
from utils.seed import seed_everything

METHOD = "setfit"
BACKBONE = "sentence-transformers/all-MiniLM-L6-v2"

# num_iterations: number of contrastive pairs generated per training example.
# num_epochs: epochs for the contrastive body fine-tuning.
_CONFIGS: Dict[str, Dict] = {
    "full": {"num_iterations": 5, "num_epochs": 1, "batch": 16},
    "5shot": {"num_iterations": 20, "num_epochs": 1, "batch": 16},
    "10shot": {"num_iterations": 20, "num_epochs": 1, "batch": 16},
    "20shot": {"num_iterations": 20, "num_epochs": 1, "batch": 16},
}


def run(
    split_name: str,
    seed: int = SEED,
    verbose: bool = False,
    dedup: bool = False,
    artifacts: Optional[RunArtifacts] = None,
) -> Dict:
    """Train SetFit and export predictions, probabilities and embeddings."""
    cfg = _CONFIGS[split_name]
    seed_state = seed_everything(seed, deterministic=False)
    train, test = load_split(split_name, seed=seed, dedup=dedup)
    id2label, _ = load_label_map(dedup)

    def to_dataset(df: pd.DataFrame) -> Dataset:
        return Dataset.from_pandas(
            df[["text", "label_id"]].rename(columns={"label_id": "label"})
        )

    train_ds = to_dataset(train)

    if artifacts is None:
        artifacts = RunArtifacts.create(METHOD, split_name, seed)

    model = SetFitModel.from_pretrained(BACKBONE, use_differentiable_head=False)
    args = TrainingArguments(
        output_dir=str(artifacts.model_dir),
        batch_size=cfg["batch"],
        num_epochs=cfg["num_epochs"],
        num_iterations=cfg["num_iterations"],
        body_learning_rate=2e-5,
        head_learning_rate=1e-2,
        seed=seed,
        use_amp=False,
        eval_strategy="no",
        save_strategy="no",
        report_to="none",
        show_progress_bar=verbose,
    )

    trainer = Trainer(model=model, args=args, train_dataset=train_ds)
    trainer.train()

    texts = test["text"].tolist()
    y_true = test["label_id"].to_numpy()
    y_pred = np.asarray(model.predict(texts, as_numpy=True, use_labels=False)).ravel().astype(int)

    probabilities = None
    try:
        probabilities = np.asarray(model.predict_proba(texts))
        if probabilities.ndim == 1:
            probabilities = None
    except Exception as exc:  # head may not expose calibrated probabilities
        if verbose:
            print(f"[setfit] predict_proba unavailable: {exc}")

    embeddings = None
    body = getattr(model, "model_body", None)
    if body is not None:
        try:
            embeddings = body.encode(
                texts, convert_to_numpy=True, batch_size=64, show_progress_bar=False
            )
        except Exception as exc:
            if verbose:
                print(f"[setfit] body embeddings unavailable: {exc}")

    artifacts.save_config(
        {
            "backbone": BACKBONE,
            "num_iterations": cfg["num_iterations"],
            "num_epochs": cfg["num_epochs"],
            "batch_size": cfg["batch"],
            "body_learning_rate": 2e-5,
            "head_learning_rate": 1e-2,
            "use_differentiable_head": False,
            "n_train": int(len(train)),
            "n_test": int(len(test)),
            "dedup": dedup,
        }
    )
    artifacts.save_env({"seed_state": seed_state})
    metrics = artifacts.save_metrics(
        y_true, y_pred, id2label, extra={"n_train": int(len(train))}
    )
    artifacts.save_predictions(y_true, y_pred, id2label, probabilities=probabilities)
    artifacts.save_embeddings(embeddings)

    try:
        model.save_pretrained(str(artifacts.model_dir / "final"))
    except Exception as exc:
        if verbose:
            print(f"[setfit] could not save model: {exc}")

    if verbose:
        print(artifacts.summary_line(metrics["macro_f1"]))

    return {
        "macro_f1": metrics["macro_f1"],
        "per_class_f1": {k: v["f1"] for k, v in metrics["per_class"].items()},
        "artifacts": str(artifacts.root),
    }


if __name__ == "__main__":
    import sys

    name = sys.argv[1] if len(sys.argv) > 1 else "5shot"
    result = run(name, verbose=True)
    print({k: v for k, v in result.items() if k != "per_class_f1"})
