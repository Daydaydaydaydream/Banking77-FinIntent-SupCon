"""BERT sequence-classification baseline with validation checkpoint selection."""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from config import BERT_MODEL
from utils.artifacts import RunArtifacts
from utils.data import encode_labels, load_splits
from utils.metrics import classification_metrics
from utils.seed import seed_everything, select_torch_device


METHOD = "bert"
DEFAULT_FEW_SHOT_MIN_STEPS = 300
DEFAULT_FEW_SHOT_MAX_STEPS = 1000
DEFAULT_FEW_SHOT_EVAL_STEPS = 50


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def resolve_training_schedule(
    setting: str,
    *,
    min_steps: int | None,
    max_steps: int | None,
    eval_steps: int | None,
) -> dict[str, int | str | None]:
    """Resolve a convergence-oriented schedule without using test results."""

    is_few_shot = setting != "full"
    resolved_min_steps = (
        DEFAULT_FEW_SHOT_MIN_STEPS if min_steps is None and is_few_shot else min_steps
    )
    if resolved_min_steps is None:
        resolved_min_steps = 0
    resolved_max_steps = (
        DEFAULT_FEW_SHOT_MAX_STEPS if max_steps is None and is_few_shot else max_steps
    )
    if resolved_max_steps is None:
        resolved_max_steps = -1
    use_step_strategy = resolved_max_steps > 0
    resolved_eval_steps = (
        DEFAULT_FEW_SHOT_EVAL_STEPS if eval_steps is None and use_step_strategy else eval_steps
    )

    if resolved_min_steps < 0:
        raise ValueError("min_steps must be non-negative")
    if resolved_max_steps == 0 or resolved_max_steps < -1:
        raise ValueError("max_steps must be positive or -1")
    if not use_step_strategy and resolved_min_steps > 0:
        raise ValueError("min_steps requires a positive max_steps")
    if use_step_strategy and resolved_min_steps > resolved_max_steps:
        raise ValueError("min_steps cannot exceed max_steps")
    if use_step_strategy and (resolved_eval_steps is None or resolved_eval_steps <= 0):
        raise ValueError("eval_steps must be positive when max_steps is used")

    return {
        "strategy": "steps" if use_step_strategy else "epoch",
        "min_steps": resolved_min_steps,
        "max_steps": resolved_max_steps,
        "eval_steps": resolved_eval_steps,
    }


def run(
    setting: str,
    seed: int,
    *,
    model_name: str = BERT_MODEL,
    epochs: float = 5.0,
    batch_size: int = 32,
    learning_rate: float = 2e-5,
    weight_decay: float = 0.01,
    warmup_ratio: float = 0.1,
    max_length: int = 64,
    patience: int = 2,
    early_stopping_threshold: float = 1e-4,
    min_steps: int | None = None,
    max_steps: int | None = None,
    eval_steps: int | None = None,
    device: str = "auto",
    local_files_only: bool = False,
    save_model: bool = True,
) -> dict[str, Any]:
    try:
        import torch
        from torch.utils.data import Dataset
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            DataCollatorWithPadding,
            EarlyStoppingCallback,
            Trainer,
            TrainingArguments,
        )
    except ImportError as exc:
        raise RuntimeError(
            "BERT baseline dependencies are missing. Run: pip install -r requirements.txt"
        ) from exc

    seed_info = seed_everything(seed)
    selected_device = select_torch_device(device)
    train, validation, test, categories = load_splits(setting, seed)
    y_train = encode_labels(train, categories)
    y_val = encode_labels(validation, categories)
    y_test = encode_labels(test, categories)
    label_to_id = {label: idx for idx, label in enumerate(categories)}
    schedule = resolve_training_schedule(
        setting,
        min_steps=min_steps,
        max_steps=max_steps,
        eval_steps=eval_steps,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=local_files_only)

    class TextDataset(Dataset):
        def __init__(self, texts: list[str], labels: np.ndarray) -> None:
            self.encodings = tokenizer(texts, truncation=True, max_length=max_length)
            self.labels = labels

        def __len__(self) -> int:
            return len(self.labels)

        def __getitem__(self, index: int) -> dict[str, Any]:
            item = {key: value[index] for key, value in self.encodings.items()}
            item["labels"] = int(self.labels[index])
            return item

    train_dataset = TextDataset(train["text"].tolist(), y_train)
    val_dataset = TextDataset(validation["text"].tolist(), y_val)
    test_dataset = TextDataset(test["text"].tolist(), y_test)

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
        "weight_decay": weight_decay,
        "warmup_ratio": warmup_ratio,
        "max_length": max_length,
        "early_stopping_patience": patience,
        "early_stopping_threshold": early_stopping_threshold,
        "training_schedule": schedule,
        "device": selected_device,
        "local_files_only": local_files_only,
        "seed_info": seed_info,
    }
    artifacts.write_config(config)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(categories),
        id2label={idx: label for idx, label in enumerate(categories)},
        label2id=label_to_id,
        local_files_only=local_files_only,
    )

    def compute_metrics(eval_prediction: Any) -> dict[str, float]:
        logits, labels = eval_prediction
        predictions = np.asarray(logits).argmax(axis=1)
        metrics = classification_metrics(np.asarray(labels), predictions, categories)
        return {
            "macro_f1": metrics["macro_f1"],
            "micro_f1": metrics["micro_f1"],
            "accuracy": metrics["accuracy"],
        }

    strategy = str(schedule["strategy"])
    resolved_eval_steps = schedule["eval_steps"]
    training_args = TrainingArguments(
        output_dir=str(artifacts.model_dir / "checkpoints"),
        num_train_epochs=epochs,
        max_steps=int(schedule["max_steps"]),
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        eval_strategy=strategy,
        save_strategy=strategy,
        logging_strategy=strategy,
        eval_steps=resolved_eval_steps,
        save_steps=resolved_eval_steps,
        logging_steps=resolved_eval_steps,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        save_total_limit=1,
        seed=seed,
        data_seed=seed,
        report_to="none",
        fp16=selected_device == "cuda",
        dataloader_pin_memory=selected_device == "cuda",
    )

    class MinimumStepsEarlyStoppingCallback(EarlyStoppingCallback):
        """Do not allow plateau stopping before the minimum optimization budget."""

        def __init__(self) -> None:
            super().__init__(
                early_stopping_patience=patience,
                early_stopping_threshold=early_stopping_threshold,
            )
            self.stopped_for_plateau = False

        def on_evaluate(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
            if state.global_step < int(schedule["min_steps"]):
                return control
            control = super().on_evaluate(args, state, control, **kwargs)
            if self.early_stopping_patience_counter >= patience:
                self.stopped_for_plateau = True
            return control

    early_stopping = MinimumStepsEarlyStoppingCallback() if patience > 0 else None
    callbacks = [early_stopping] if early_stopping is not None else []
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
        callbacks=callbacks,
    )

    started = time.perf_counter()
    trainer.train()
    train_seconds = time.perf_counter() - started

    val_output = trainer.predict(val_dataset)
    val_scores = _softmax(np.asarray(val_output.predictions))
    val_pred = val_scores.argmax(axis=1)
    validation_metrics = classification_metrics(y_val, val_pred, categories)

    started = time.perf_counter()
    test_output = trainer.predict(test_dataset)
    inference_seconds = time.perf_counter() - started
    test_scores = _softmax(np.asarray(test_output.predictions))
    y_pred = artifacts.write_predictions(test["text"].tolist(), y_test, test_scores, categories)
    test_metrics = classification_metrics(y_test, y_pred, categories)
    artifacts.write_confusion(y_test, y_pred, categories)
    artifacts.write_history(trainer.state.log_history)
    artifacts.write_metrics(
        validation_metrics,
        test_metrics,
        {
            "train_seconds": train_seconds,
            "test_inference_seconds": inference_seconds,
            "test_examples_per_second": len(test) / inference_seconds,
            "best_checkpoint": trainer.state.best_model_checkpoint,
            "best_validation_metric": trainer.state.best_metric,
            "optimizer_steps": trainer.state.global_step,
            "epochs_completed": trainer.state.epoch,
            "stopped_for_validation_plateau": bool(
                early_stopping is not None and early_stopping.stopped_for_plateau
            ),
        },
    )
    if save_model:
        final_dir = artifacts.model_dir / "best"
        trainer.save_model(str(final_dir))
        tokenizer.save_pretrained(final_dir)

    return {
        "method": METHOD,
        "setting": setting,
        "seed": seed,
        "validation_macro_f1": validation_metrics["macro_f1"],
        "test_macro_f1": test_metrics["macro_f1"],
        "output_dir": str(artifacts.path),
    }
