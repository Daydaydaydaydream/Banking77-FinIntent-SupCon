"""BERT sequence-classification baseline with validation checkpoint selection."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np

from config import BERT_MODEL
from utils.artifacts import RunArtifacts
from utils.data import encode_labels, load_splits
from utils.metrics import classification_metrics
from utils.precision import (
    GradientHealthCallback,
    precision_flags,
    prediction_health,
    resolve_precision,
)
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


def _relative_checkpoint(artifacts: RunArtifacts, checkpoint: str | None) -> str | None:
    """Store the best checkpoint as a run-relative path.

    Absolute paths recorded on the training machine stop resolving as soon as
    the run directory is archived, copied to another machine, or committed.
    """

    if not checkpoint:
        return None
    try:
        return str(Path(checkpoint).resolve().relative_to(artifacts.path.resolve()))
    except ValueError:
        return Path(checkpoint).name


def freeze_bert_layers(model: Any, num_layers: int) -> list[str]:
    """Freeze embeddings and the lowest ``num_layers`` encoder blocks.

    Returns the names of every frozen tensor so the choice is auditable from
    ``config.json`` rather than being implicit in the training code.
    """

    if num_layers <= 0:
        return []
    encoder = getattr(model, "bert", model)
    embeddings = getattr(encoder, "embeddings", None)
    if embeddings is not None:
        for parameter in embeddings.parameters():
            parameter.requires_grad = False
    layers = getattr(encoder, "encoder", None)
    layers = getattr(layers, "layer", layers) if layers is not None else None
    if layers is None:
        raise ValueError("Could not locate encoder layers to freeze")
    for layer in list(layers)[:num_layers]:
        for parameter in layer.parameters():
            parameter.requires_grad = False
    return [name for name, parameter in model.named_parameters() if not parameter.requires_grad]


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
    keep_checkpoints: bool = False,
    config_provenance: dict[str, Any] | None = None,
    precision: str = "auto",
    freeze_layers: int = 0,
    attn_implementation: str = "eager",
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
    precision_policy = resolve_precision(precision, selected_device)
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
        "weight_decay": weight_decay,
        "warmup_ratio": warmup_ratio,
        "max_length": max_length,
        "early_stopping_patience": patience,
        "early_stopping_threshold": early_stopping_threshold,
        "training_schedule": schedule,
        "device": selected_device,
        "precision": precision_policy,
        "freeze_layers": freeze_layers,
        "attn_implementation": attn_implementation,
        "local_files_only": local_files_only,
        "seed_info": seed_info,
    }
    if config_provenance:
        # Why these values were chosen, recorded by the validation-only sweep.
        config["config_provenance"] = config_provenance
    artifacts.write_config(config)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(categories),
        id2label={idx: label for idx, label in enumerate(categories)},
        label2id=label_to_id,
        local_files_only=local_files_only,
        attn_implementation=attn_implementation,
    )
    if freeze_layers > 0:
        # Freezing the embedding table and the lowest transformer blocks is the
        # cheapest stabilisation for few-shot fine-tuning: it removes most of
        # the parameters that can drift far from the pretrained optimum when
        # only a handful of examples per class are available.
        frozen = freeze_bert_layers(model, freeze_layers)
        config["frozen_parameter_tensors"] = frozen
        artifacts.write_config(config)

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
        output_dir=str(artifacts.checkpoint_dir),
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
        # float16 silently overflowed on few-shot runs and produced class
        # collapse.  resolve_precision() prefers bfloat16, whose exponent range
        # matches float32.
        **precision_flags(str(precision_policy["resolved"])),
        max_grad_norm=1.0,
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
    gradient_health = GradientHealthCallback()
    callbacks: list[Any] = [gradient_health]
    if early_stopping is not None:
        callbacks.append(early_stopping)
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

    validation_health = prediction_health(y_val, val_pred, categories)
    test_health = prediction_health(y_test, y_pred, categories)
    gradient_report = gradient_health.report()
    collapsed = validation_health["collapsed"] or gradient_report["non_finite_grad_norm_events"] > 0

    artifacts.write_metrics(
        validation_metrics,
        test_metrics,
        {
            "train_seconds": train_seconds,
            "test_inference_seconds": inference_seconds,
            "test_examples_per_second": len(test) / inference_seconds,
            # Stored relative to the run directory: absolute paths from another
            # machine are meaningless once the run is archived or re-cloned.
            "best_checkpoint": _relative_checkpoint(artifacts, trainer.state.best_model_checkpoint),
            "best_validation_metric": trainer.state.best_metric,
            "optimizer_steps": trainer.state.global_step,
            "epochs_completed": trainer.state.epoch,
            "stopped_for_validation_plateau": bool(
                early_stopping is not None and early_stopping.stopped_for_plateau
            ),
            "gradient_health": gradient_report,
            "validation_health": validation_health,
            "test_health": test_health,
            "run_flagged_as_collapsed": collapsed,
        },
    )
    artifacts.write_status("completed", {"collapsed": collapsed})
    if save_model:
        final_dir = artifacts.model_dir / "best"
        trainer.save_model(str(final_dir))
        tokenizer.save_pretrained(final_dir)
    artifacts.cleanup_checkpoints()

    return {
        "method": METHOD,
        "setting": setting,
        "seed": seed,
        "validation_macro_f1": validation_metrics["macro_f1"],
        "test_macro_f1": test_metrics["macro_f1"],
        "predicted_class_count": test_health["predicted_class_count"],
        "run_flagged_as_collapsed": collapsed,
        "output_dir": str(artifacts.path),
    }
