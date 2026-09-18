"""SetFit baseline using MiniLM embeddings and a logistic-regression head."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from config import SETFIT_MODEL
from utils.artifacts import RunArtifacts
from utils.data import encode_labels, load_splits
from utils.metrics import classification_metrics
from utils.seed import seed_everything, select_torch_device


METHOD = "setfit"


def run(
    setting: str,
    seed: int,
    *,
    model_name: str = SETFIT_MODEL,
    epochs: int = 1,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    num_iterations: int | None = None,
    max_length: int = 64,
    device: str = "auto",
    local_files_only: bool = False,
    save_model: bool = True,
) -> dict[str, Any]:
    try:
        from datasets import Dataset
        from setfit import SetFitModel, Trainer, TrainingArguments
    except ImportError as exc:
        raise RuntimeError(
            "SetFit baseline dependencies are missing. Run: pip install -r requirements.txt"
        ) from exc

    seed_info = seed_everything(seed)
    selected_device = select_torch_device(device)
    train, validation, test, categories = load_splits(setting, seed)
    y_train = encode_labels(train, categories)
    y_val = encode_labels(validation, categories)
    y_test = encode_labels(test, categories)
    actual_iterations = num_iterations if num_iterations is not None else (1 if setting == "full" else 20)

    train_dataset = Dataset.from_dict({"text": train["text"].tolist(), "label": y_train.tolist()})
    val_dataset = Dataset.from_dict({"text": validation["text"].tolist(), "label": y_val.tolist()})

    artifacts = RunArtifacts(METHOD, setting, seed)
    config = {
        "method": METHOD,
        "setting": setting,
        "seed": seed,
        "model_name": model_name,
        "train_rows": len(train),
        "validation_rows": len(validation),
        "test_rows": len(test),
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "num_iterations": actual_iterations,
        "max_length": max_length,
        "device": selected_device,
        "local_files_only": local_files_only,
        "seed_info": seed_info,
    }
    artifacts.write_config(config)

    model = SetFitModel.from_pretrained(
        model_name,
        labels=categories,
        use_differentiable_head=False,
        local_files_only=local_files_only,
    )
    model.to(selected_device)

    def metric(y_pred: np.ndarray, y_true: np.ndarray) -> dict[str, float]:
        return {
            "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
            "accuracy": float(accuracy_score(y_true, y_pred)),
        }

    training_args = TrainingArguments(
        output_dir=str(artifacts.model_dir / "checkpoints"),
        batch_size=batch_size,
        num_epochs=epochs,
        num_iterations=actual_iterations,
        body_learning_rate=learning_rate,
        max_length=max_length,
        seed=seed,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        save_total_limit=1,
        report_to="none",
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        metric=metric,
    )

    started = time.perf_counter()
    trainer.train()
    train_seconds = time.perf_counter() - started

    val_scores = np.asarray(
        model.predict_proba(validation["text"].tolist(), batch_size=batch_size * 2, as_numpy=True)
    )
    val_pred = val_scores.argmax(axis=1)
    validation_metrics = classification_metrics(y_val, val_pred, categories)

    started = time.perf_counter()
    test_scores = np.asarray(
        model.predict_proba(test["text"].tolist(), batch_size=batch_size * 2, as_numpy=True)
    )
    inference_seconds = time.perf_counter() - started
    y_pred = artifacts.write_predictions(test["text"].tolist(), y_test, test_scores, categories)
    test_metrics = classification_metrics(y_test, y_pred, categories)
    artifacts.write_confusion(y_test, y_pred, categories)
    history = getattr(getattr(trainer, "state", None), "log_history", [])
    artifacts.write_history(history)
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
        model.save_pretrained(artifacts.model_dir / "best")

    return {
        "method": METHOD,
        "setting": setting,
        "seed": seed,
        "validation_macro_f1": validation_metrics["macro_f1"],
        "test_macro_f1": test_metrics["macro_f1"],
        "output_dir": str(artifacts.path),
    }
