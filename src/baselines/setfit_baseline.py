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
from utils.precision import prediction_health
from utils.seed import seed_everything, select_torch_device


METHOD = "setfit"


def training_arguments_kwargs(
    *,
    output_dir: str,
    batch_size: int,
    epochs: int,
    num_iterations: int,
    learning_rate: float,
    max_length: int,
    seed: int,
    save_checkpoints: bool = False,
) -> dict[str, Any]:
    """Build SetFit arguments with a metric available during embedding training."""

    return {
        "output_dir": output_dir,
        "batch_size": batch_size,
        "num_epochs": epochs,
        "num_iterations": num_iterations,
        "body_learning_rate": learning_rate,
        "max_length": max_length,
        "seed": seed,
        "eval_strategy": "epoch",
        "save_strategy": "epoch" if save_checkpoints else "no",
        "logging_strategy": "epoch",
        "load_best_model_at_end": save_checkpoints,
        # SetFit evaluates the sentence-transformer phase before fitting its
        # classifier head. At this point it exposes eval_embedding_loss, not
        # the downstream classification metrics returned by `metric` below.
        "metric_for_best_model": "embedding_loss" if save_checkpoints else None,
        "greater_is_better": False if save_checkpoints else None,
        "save_total_limit": 1,
        "report_to": "none",
    }


def extract_history(trainer: Any) -> list[dict[str, Any]]:
    """Collect the training log from whichever trainer actually recorded it.

    ``setfit.Trainer.train()`` delegates the embedding phase to an internal
    sentence-transformers trainer held on ``st_trainer``; the outer object has
    no usable ``state.log_history``, which is why the archived runs recorded
    ``[]``.  Both phases are checked and the first non-empty log wins.
    """

    candidates: list[Any] = [
        getattr(trainer, "state", None),
        getattr(getattr(trainer, "st_trainer", None), "state", None),
        getattr(getattr(trainer, "trainer", None), "state", None),
        getattr(getattr(getattr(trainer, "model", None), "trainer", None), "state", None),
    ]
    for state in candidates:
        history = getattr(state, "log_history", None)
        if history:
            return list(history)
    return []


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
    keep_checkpoints: bool = False,
    config_provenance: dict[str, Any] | None = None,
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

    artifacts = RunArtifacts(METHOD, setting, seed, keep_checkpoints=keep_checkpoints)
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
    if config_provenance:
        config["config_provenance"] = config_provenance
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
        **training_arguments_kwargs(
            output_dir=str(artifacts.checkpoint_dir),
            batch_size=batch_size,
            epochs=epochs,
            num_iterations=actual_iterations,
            learning_rate=learning_rate,
            max_length=max_length,
            seed=seed,
            save_checkpoints=save_model,
        )
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

    history = extract_history(trainer)
    artifacts.write_history(history)

    validation_health = prediction_health(y_val, val_pred, categories)
    test_health = prediction_health(y_test, y_pred, categories)
    artifacts.write_metrics(
        validation_metrics,
        test_metrics,
        {
            "train_seconds": train_seconds,
            "test_inference_seconds": inference_seconds,
            "test_examples_per_second": len(test) / inference_seconds,
            "training_history_entries": len(history),
            "validation_health": validation_health,
            "test_health": test_health,
            "run_flagged_as_collapsed": test_health["collapsed"],
        },
    )
    artifacts.write_status("completed", {"collapsed": test_health["collapsed"]})
    if save_model:
        model.save_pretrained(artifacts.model_dir / "best")
    artifacts.cleanup_checkpoints()

    return {
        "method": METHOD,
        "setting": setting,
        "seed": seed,
        "validation_macro_f1": validation_metrics["macro_f1"],
        "test_macro_f1": test_metrics["macro_f1"],
        "predicted_class_count": test_health["predicted_class_count"],
        "run_flagged_as_collapsed": test_health["collapsed"],
        "output_dir": str(artifacts.path),
    }
