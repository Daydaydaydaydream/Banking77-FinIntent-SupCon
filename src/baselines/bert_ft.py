"""Baseline 2: BERT fine-tuning.

Fine-tunes ``bert-base-uncased`` with a linear classification head using the
HuggingFace Trainer. Banking77 utterances are short, so inputs are truncated
at 64 tokens. Runs on MPS when available and falls back to CPU.

Stage A turned this into a validation-selected run: every setting now trains
against the stratified validation split, evaluates once per epoch, and keeps the
checkpoint with the best validation Macro-F1 (``load_best_model_at_end``).
Previously the last epoch was evaluated directly on the test set, which both
overfit and leaked the test split into model selection.
"""

from __future__ import annotations

from typing import Dict, Optional

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

from config import NUM_CLASSES, SEED
from utils.artifacts import RunArtifacts
from utils.data import load_label_map, load_train_val_test
from utils.metrics import macro_f1
from utils.seed import seed_everything

METHOD = "bert_ft"
MODEL_NAME = "bert-base-uncased"
MAX_LEN = 64

# Per-setting training hyper-parameters. Few-shot settings use more epochs and
# a slightly higher learning rate to compensate for the small training set.
_BERT_CONFIGS: Dict[str, Dict] = {
    "full": {"epochs": 10, "lr": 3e-5, "batch": 32, "warmup_ratio": 0.1},
    "5shot": {"epochs": 20, "lr": 5e-5, "batch": 16, "warmup_ratio": 0.1},
    "10shot": {"epochs": 15, "lr": 5e-5, "batch": 16, "warmup_ratio": 0.1},
    "20shot": {"epochs": 10, "lr": 5e-5, "batch": 16, "warmup_ratio": 0.1},
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


def _softmax(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=float)
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


@torch.no_grad()
def _extract_cls_embeddings(
    model, dataset: Dataset, device: torch.device, batch_size: int = 64
) -> Optional[np.ndarray]:
    """Return pooled ``[CLS]`` representations for a tokenised dataset.

    Runs the encoder body only, avoiding the memory blow-up of requesting all
    hidden states from ``Trainer.predict`` (13 layers x 3,080 x 64 x 768 would
    be tens of gigabytes).
    """
    encoder = getattr(model, "bert", None) or getattr(model, "base_model", None)
    if encoder is None:
        return None
    encoder.eval()

    input_ids = dataset["input_ids"]
    attention_mask = dataset["attention_mask"]
    chunks = []
    for start in range(0, len(input_ids), batch_size):
        ids = input_ids[start : start + batch_size].to(device)
        mask = attention_mask[start : start + batch_size].to(device)
        hidden = encoder(input_ids=ids, attention_mask=mask).last_hidden_state
        chunks.append(hidden[:, 0, :].detach().cpu().numpy())
    return np.concatenate(chunks, axis=0)


def run(
    split_name: str,
    seed: int = SEED,
    verbose: bool = False,
    dedup: bool = False,
    artifacts: Optional[RunArtifacts] = None,
    epochs: Optional[int] = None,
    lr: Optional[float] = None,
) -> Dict:
    """Train, select the best checkpoint on validation, evaluate on test."""
    cfg = dict(_BERT_CONFIGS[split_name])
    if epochs is not None:
        cfg["epochs"] = epochs
    if lr is not None:
        cfg["lr"] = lr

    seed_state = seed_everything(seed, deterministic=True)
    device = _device()
    if verbose:
        print(f"[bert] {split_name}: device={device}, seed={seed}, cfg={cfg}")

    train, val, test = load_train_val_test(split_name, seed=seed, dedup=dedup)
    id2label, _ = load_label_map(dedup)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=NUM_CLASSES, attn_implementation="eager"
    ).to(device)

    def preprocess(df: pd.DataFrame) -> Dataset:
        ds = Dataset.from_pandas(df[["text", "label_id"]].reset_index(drop=True))
        ds = ds.map(lambda x: _tokenize(tokenizer, x), batched=True)
        ds = ds.rename_column("label_id", "labels")
        ds.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
        return ds

    train_ds = preprocess(train)
    val_ds = preprocess(val)
    test_ds = preprocess(test)

    def compute_metrics(eval_pred) -> Dict[str, float]:
        y_true = np.asarray(eval_pred.label_ids)
        y_pred = np.argmax(eval_pred.predictions, axis=1)
        return {"macro_f1": macro_f1(y_true, y_pred)}

    if artifacts is None:
        artifacts = RunArtifacts.create(METHOD, split_name, seed)

    args = TrainingArguments(
        output_dir=str(artifacts.model_dir),
        num_train_epochs=cfg["epochs"],
        per_device_train_batch_size=cfg["batch"],
        per_device_eval_batch_size=cfg["batch"],
        learning_rate=cfg["lr"],
        weight_decay=0.01,
        warmup_ratio=cfg.get("warmup_ratio", 0.06),
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        save_total_limit=2,
        logging_strategy="no",
        seed=seed,
        data_seed=seed,
        remove_unused_columns=True,
        use_mps_device=(device.type == "mps"),
        disable_tqdm=not verbose,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        data_collator=None,
    )
    trainer.train()

    test_output = trainer.predict(test_ds)
    logits = test_output.predictions
    y_pred = np.argmax(logits, axis=1)
    y_true = test["label_id"].to_numpy()
    probabilities = _softmax(logits)
    embeddings = _extract_cls_embeddings(trainer.model, test_ds, device)

    best_metric = getattr(trainer.state, "best_metric", None)
    best_ckpt = getattr(trainer.state, "best_model_checkpoint", None)

    artifacts.save_config(
        {
            "model_name": MODEL_NAME,
            "max_len": MAX_LEN,
            "epochs": cfg["epochs"],
            "lr": cfg["lr"],
            "batch_size": cfg["batch"],
            "warmup_ratio": cfg.get("warmup_ratio", 0.06),
            "weight_decay": 0.01,
            "attn_implementation": "eager",
            "n_train": int(len(train)),
            "n_val": int(len(val)),
            "n_test": int(len(test)),
            "dedup": dedup,
            "checkpoint_selection": "best validation Macro-F1",
        }
    )
    artifacts.save_env({"seed_state": seed_state, "torch_generator_seeded": True})
    metrics = artifacts.save_metrics(
        y_true,
        y_pred,
        id2label,
        extra={
            "val_macro_f1": None if best_metric is None else float(best_metric),
            "best_checkpoint": None if best_ckpt is None else str(best_ckpt),
            "n_train": int(len(train)),
        },
    )
    artifacts.save_predictions(y_true, y_pred, id2label, probabilities=probabilities)
    artifacts.save_embeddings(embeddings)

    trainer.save_model(str(artifacts.model_dir))
    tokenizer.save_pretrained(str(artifacts.model_dir))

    if verbose:
        print(artifacts.summary_line(metrics["macro_f1"]))

    return {
        "macro_f1": metrics["macro_f1"],
        "per_class_f1": {k: v["f1"] for k, v in metrics["per_class"].items()},
        "val_macro_f1": None if best_metric is None else float(best_metric),
        "artifacts": str(artifacts.root),
    }


if __name__ == "__main__":
    import sys

    name = sys.argv[1] if len(sys.argv) > 1 else "full"
    result = run(name, verbose=True)
    print({k: v for k, v in result.items() if k != "per_class_f1"})
