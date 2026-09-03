"""Baseline 3: native SetFit.

Uses the ``setfit`` library's two-stage training: contrastive fine-tuning of a
sentence-transformer body followed by a logistic-regression head trained on the
frozen embeddings (``use_differentiable_head=False``). The backbone is
``sentence-transformers/all-MiniLM-L6-v2``, a small sentence encoder well suited
to short-utterance classification.

Few-shot settings use the canonical SetFit configuration (many contrastive
iterations); the full-data setting uses fewer iterations to keep runtime
reasonable.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
from datasets import Dataset
from setfit import SetFitModel, Trainer, TrainingArguments

from config import MODEL_DIR, SEED
from utils.data import load_label_map, load_split
from utils.metrics import macro_f1

BACKBONE = "sentence-transformers/all-MiniLM-L6-v2"

# num_iterations: number of contrastive pairs generated per training example.
# num_epochs: epochs for the contrastive body fine-tuning.
_CONFIGS: Dict[str, Dict] = {
    "full": {"num_iterations": 5, "num_epochs": 1, "batch": 16},
    "5shot": {"num_iterations": 20, "num_epochs": 1, "batch": 16},
    "10shot": {"num_iterations": 20, "num_epochs": 1, "batch": 16},
    "20shot": {"num_iterations": 20, "num_epochs": 1, "batch": 16},
}


def run(split_name: str, verbose: bool = False) -> Dict[str, float]:
    cfg = _CONFIGS[split_name]
    train, test = load_split(split_name)

    def to_dataset(df: pd.DataFrame) -> Dataset:
        return Dataset.from_pandas(
            df[["text", "label_id"]].rename(columns={"label_id": "label"})
        )

    train_ds = to_dataset(train)

    model = SetFitModel.from_pretrained(BACKBONE, use_differentiable_head=False)
    args = TrainingArguments(
        output_dir=str(MODEL_DIR / f"setfit_{split_name}"),
        batch_size=cfg["batch"],
        num_epochs=cfg["num_epochs"],
        num_iterations=cfg["num_iterations"],
        body_learning_rate=2e-5,
        head_learning_rate=1e-2,
        seed=SEED,
        use_amp=False,
        eval_strategy="no",
        save_strategy="no",
        report_to="none",
        show_progress_bar=verbose,
    )

    trainer = Trainer(model=model, args=args, train_dataset=train_ds)
    trainer.train()

    preds = trainer.model.predict(test["text"].tolist(), as_numpy=True, use_labels=False)
    y_pred = np.asarray(preds).ravel().astype(int)
    y_true = test["label_id"].to_numpy()
    f1 = macro_f1(y_true, y_pred)

    if verbose:
        print(f"[setfit] {split_name}: Macro-F1 = {f1:.4f}")

    return {"macro_f1": f1, "model_dir": str(MODEL_DIR / f"setfit_{split_name}")}


if __name__ == "__main__":
    import sys

    name = sys.argv[1] if len(sys.argv) > 1 else "5shot"
    print(run(name, verbose=True))
