"""Baseline 2: BERT fine-tuning.

Fine-tunes ``bert-base-uncased`` with a linear classification head using the
HuggingFace Trainer. Banking77 utterances are short, so inputs are truncated
at 64 tokens. Runs on MPS when available and falls back to CPU.
"""

from __future__ import annotations

import json
from typing import Dict

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from config import MODEL_DIR, NUM_CLASSES, RESULT_DIR, SEED
from utils.data import load_label_map, load_split
from utils.metrics import macro_f1, per_class_f1

MODEL_NAME = "bert-base-uncased"
MAX_LEN = 64

# Per-setting training hyper-parameters. Few-shot settings use more epochs and
# a slightly higher learning rate to compensate for the small training set.
#
# ``full``: increased to 10 epochs with a 5% stratified validation split so the
# best checkpoint is selected by validation Macro-F1 (load_best_model_at_end)
# instead of taking the last epoch, which overfits; lifts Macro-F1 towards the
# ~0.93 literature level (previous 5-epoch run: 0.8811).
_BERT_CONFIGS: Dict[str, Dict] = {
    "full": {
        "epochs": 10,
        "lr": 3e-5,
        "batch": 32,
        "val_ratio": 0.05,
        "warmup_ratio": 0.1,
    },
    "5shot": {"epochs": 20, "lr": 5e-5, "batch": 16},
    "10shot": {"epochs": 15, "lr": 5e-5, "batch": 16},
    "20shot": {"epochs": 10, "lr": 5e-5, "batch": 16},
}


def _device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _tokenize(tokenizer, examples) -> Dict:
    return tokenizer(
        examples["text"], truncation=True, padding="max_length", max_length=MAX_LEN
    )


def run(split_name: str, verbose: bool = False) -> Dict[str, float]:
    cfg = _BERT_CONFIGS[split_name]
    device = _device()
    if verbose:
        print(f"[bert] {split_name}: device={device}, cfg={cfg}")

    train, test = load_split(split_name)
    id2label, _ = load_label_map()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=NUM_CLASSES, attn_implementation="eager"
    ).to(device)

    def preprocess(df: pd.DataFrame) -> Dataset:
        ds = Dataset.from_pandas(df[["text", "label_id"]])
        ds = ds.map(lambda x: _tokenize(tokenizer, x), batched=True)
        ds = ds.rename_column("label_id", "labels")
        ds.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
        return ds

    # Optional stratified validation split (only "full" configures it) so that
    # the best checkpoint is selected by validation Macro-F1 rather than using
    # the last epoch, which can overfit as training continues.
    test_ds = preprocess(test)
    val_ds = None
    compute_metrics = None
    if cfg.get("val_ratio"):
        from sklearn.model_selection import train_test_split

        train_df, val_df = train_test_split(
            train,
            test_size=cfg["val_ratio"],
            stratify=train["label_id"],
            random_state=SEED,
        )
        train_ds = preprocess(train_df)
        val_ds = preprocess(val_df)

        def compute_metrics(eval_pred) -> Dict[str, float]:
            y_true = np.asarray(eval_pred.label_ids)
            y_pred = np.argmax(eval_pred.predictions, axis=1)
            return {"macro_f1": macro_f1(y_true, y_pred)}
    else:
        train_ds = preprocess(train)

    args = TrainingArguments(
        output_dir=str(MODEL_DIR / f"bert_{split_name}"),
        num_train_epochs=cfg["epochs"],
        per_device_train_batch_size=cfg["batch"],
        per_device_eval_batch_size=cfg["batch"],
        learning_rate=cfg["lr"],
        weight_decay=0.01,
        warmup_ratio=cfg.get("warmup_ratio", 0.06),
        eval_strategy="epoch" if val_ds is not None else "no",
        save_strategy="epoch" if val_ds is not None else "no",
        load_best_model_at_end=val_ds is not None,
        metric_for_best_model="macro_f1",
        save_total_limit=2,
        logging_strategy="no",
        seed=SEED,
        remove_unused_columns=True,
        use_mps_device=(device.type == "mps"),
        disable_tqdm=not verbose,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
    )
    trainer.train()

    preds = trainer.predict(test_ds).predictions
    y_pred = np.argmax(preds, axis=1)
    y_true = test["label_id"].to_numpy()
    f1 = macro_f1(y_true, y_pred)
    pcf1 = per_class_f1(y_true, y_pred, id2label)

    _persist_predictions(split_name, y_true, y_pred)

    if verbose:
        print(f"[bert] {split_name}: Macro-F1 = {f1:.4f}")

    return {
        "macro_f1": f1,
        "per_class_f1": pcf1,
        "model_dir": str(MODEL_DIR / f"bert_{split_name}"),
    }


def _persist_predictions(split_name: str, y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """Persist test-set predictions for the analysis stage."""
    analysis_dir = RESULT_DIR.parent / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    np.save(analysis_dir / f"bert_ft_{split_name}_y_true.npy", y_true)
    np.save(analysis_dir / f"bert_ft_{split_name}_y_pred.npy", y_pred)


if __name__ == "__main__":
    import sys

    name = sys.argv[1] if len(sys.argv) > 1 else "full"
    result = run(name, verbose=True)
    print(json.dumps(result, indent=2))
