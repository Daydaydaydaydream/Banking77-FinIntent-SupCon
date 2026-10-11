"""Two-stage training: contrastive pretraining, then a linear classification head.

The classification head never sees the test split.  Model selection between
the ``baseline_bert`` / ``supcon`` / ``supcon_hn`` arms uses validation Macro-F1
only; the test split is touched once at the end of a frozen configuration.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from config import BERT_MODEL
from supcon.losses import hard_negative_weight_matrix, supervised_contrastive_loss
from supcon.model import (
    BalancedBatchSampler,
    SupConModel,
    TextDataset,
    build_loader,
    collate_with_padding,
)
from utils.artifacts import RunArtifacts
from utils.data import encode_labels, load_splits
from utils.metrics import classification_metrics
from utils.precision import GradientHealthCallback, precision_flags, prediction_health, resolve_precision
from utils.seed import seed_everything, select_torch_device


def _relative_checkpoint(artifacts: RunArtifacts, checkpoint: str | None) -> str | None:
    if not checkpoint:
        return None
    from pathlib import Path

    try:
        return str(Path(checkpoint).resolve().relative_to(artifacts.path.resolve()))
    except ValueError:
        return Path(checkpoint).name


def _prepare(model: SupConModel, batch: dict[str, torch.Tensor], device: str) -> tuple[torch.Tensor, torch.Tensor]:
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)
    pooled = model.embed(input_ids, attention_mask)
    return pooled, labels


def run(
    setting: str,
    seed: int,
    *,
    method: str = "supcon",
    model_name: str = BERT_MODEL,
    supcon_temperature: float = 0.1,
    supcon_epochs: int = 20,
    classifier_epochs: int = 20,
    classifier_lr: float = 1e-3,
    weight_decay: float = 0.01,
    batch_size: int = 77,
    eval_batch_size: int = 128,
    max_length: int = 64,
    hn_temperature: float | None = None,
    device: str = "auto",
    local_files_only: bool = False,
    save_model: bool = True,
    keep_checkpoints: bool = False,
    config_provenance: dict[str, Any] | None = None,
    precision: str = "auto",
    freeze_layers: int = 0,
    attn_implementation: str = "eager",
) -> dict[str, Any]:
    """Train with supervised contrastive learning, optionally with hard negatives.

    ``method="supcon"`` is standard SupCon.  ``method="supcon_hn"`` adds
    similarity-weighted negatives in the denominator.  Everything else -
    backbone, split, tokenizer, batch construction, evaluation code - is
    identical so that the difference between the two arms is attributable to
    the hard-negative weighting alone.
    """

    if method not in ("supcon", "supcon_hn"):
        raise ValueError(f"method must be supcon or supcon_hn, got {method!r}")
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "SupCon dependencies are missing. Run: pip install -r requirements.txt"
        ) from exc

    seed_info = seed_everything(seed)
    selected_device = select_torch_device(device)
    precision_policy = resolve_precision(precision, selected_device)
    train, validation, test, categories = load_splits(setting, seed)
    y_train = encode_labels(train, categories)
    y_val = encode_labels(validation, categories)
    y_test = encode_labels(test, categories)

    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=local_files_only)
    train_encodings = tokenizer(train["text"].tolist(), truncation=True, max_length=max_length)
    val_encodings = tokenizer(validation["text"].tolist(), truncation=True, max_length=max_length)
    test_encodings = tokenizer(test["text"].tolist(), truncation=True, max_length=max_length)

    train_dataset = TextDataset(train_encodings, y_train)
    val_dataset = TextDataset(val_encodings, y_val)
    test_dataset = TextDataset(test_encodings, y_test)

    effective_hn_temperature = (
        hn_temperature if method == "supcon_hn" and hn_temperature is not None else None
    )

    artifacts = RunArtifacts(method, setting, seed, keep_checkpoints=keep_checkpoints)
    config = {
        "method": method,
        "setting": setting,
        "seed": seed,
        "model_name": model_name,
        "train_rows": len(train),
        "validation_rows": len(validation),
        "test_rows": len(test),
        "supcon_temperature": supcon_temperature,
        "supcon_epochs": supcon_epochs,
        "classifier_epochs": classifier_epochs,
        "classifier_lr": classifier_lr,
        "weight_decay": weight_decay,
        "batch_size": batch_size,
        "max_length": max_length,
        "hard_negative_temperature": effective_hn_temperature,
        "device": selected_device,
        "precision": precision_policy,
        "freeze_layers": freeze_layers,
        "attn_implementation": attn_implementation,
        "local_files_only": local_files_only,
        "seed_info": seed_info,
    }
    if config_provenance:
        config["config_provenance"] = config_provenance
    artifacts.write_config(config)

    model = SupConModel(
        model_name,
        num_labels=len(categories),
        local_files_only=local_files_only,
        attn_implementation=attn_implementation,
    ).to(selected_device)

    if freeze_layers > 0:
        from baselines.bert_ft import freeze_bert_layers

        model.encoder = model.encoder  # keep the reference explicit for the reader
        config["frozen_parameter_tensors"] = freeze_bert_layers(model.encoder, freeze_layers)
        artifacts.write_config(config)

    contrastive_loader = build_loader(
        train_dataset,
        tokenizer,
        batch_size=batch_size,
        num_classes=len(categories),
        seed=seed,
        shuffle=True,
    )
    effective_batch = getattr(contrastive_loader.batch_sampler, "effective_batch_size", batch_size)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=classifier_lr,
        weight_decay=weight_decay,
    )
    amp_dtype = torch.bfloat16 if str(precision_policy["resolved"]) == "bf16" else None
    scaler = torch.amp.GradScaler("cuda", enabled=str(precision_policy["resolved"]) == "fp16")

    gradient_health = GradientHealthCallback()
    contrastive_history: list[dict[str, Any]] = []
    started = time.perf_counter()
    for epoch in range(supcon_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_anchors = 0
        for batch in contrastive_loader:
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type="cuda",
                dtype=amp_dtype or torch.float32,
                enabled=str(precision_policy["resolved"]) != "fp32",
            ):
                pooled, labels = _prepare(model, batch, selected_device)
                projected = model.project(pooled)
            features = F.normalize(projected.float(), dim=1)
            similarity = features @ features.t()
            weights = None
            if effective_hn_temperature is not None:
                weights = hard_negative_weight_matrix(similarity, labels, effective_hn_temperature)
            loss = supervised_contrastive_loss(
                features,
                labels,
                temperature=supcon_temperature,
                hard_negative_weights=weights,
            )
            if not torch.isfinite(loss):
                gradient_health.violations.append(
                    {"epoch": epoch, "step": epoch_anchors, "reason": "non_finite_loss"}
                )
                continue
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not torch.isfinite(grad_norm):
                gradient_health.violations.append(
                    {"epoch": epoch, "step": epoch_anchors, "reason": "non_finite_grad_norm"}
                )
                scaler.step(optimizer)
                scaler.update()
                continue
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += float(loss.detach())
            epoch_anchors += 1
        contrastive_history.append(
            {"epoch": epoch, "supcon_loss": epoch_loss / max(1, epoch_anchors), "steps": epoch_anchors}
        )
    contrastive_seconds = time.perf_counter() - started

    # Stage 2: freeze the contrastive encoder and fit a linear head.
    for parameter in model.encoder.parameters():
        parameter.requires_grad = False
    model.eval()

    classifier_optimizer = torch.optim.AdamW(
        model.classifier.parameters(), lr=classifier_lr, weight_decay=weight_decay
    )
    classifier_loader = build_loader(
        train_dataset,
        tokenizer,
        batch_size=eval_batch_size,
        num_classes=len(categories),
        seed=seed,
        shuffle=False,
    )
    val_eval_loader = build_loader(
        val_dataset, tokenizer, batch_size=eval_batch_size, num_classes=len(categories), seed=seed, shuffle=False
    )

    def predict_logits(dataset: TextDataset) -> np.ndarray:
        model.eval()
        outputs: list[np.ndarray] = []
        loader = build_loader(
            dataset,
            tokenizer,
            batch_size=eval_batch_size,
            num_classes=len(categories),
            seed=seed,
            shuffle=False,
        )
        with torch.no_grad():
            for batch in loader:
                input_ids = batch["input_ids"].to(selected_device)
                attention_mask = batch["attention_mask"].to(selected_device)
                pooled = model.embed(input_ids, attention_mask)
                logits = model.classify(pooled)
                outputs.append(logits.float().cpu().numpy())
        return np.concatenate(outputs, axis=0)

    def _softmax(logits: np.ndarray) -> np.ndarray:
        shifted = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(shifted)
        return exp / exp.sum(axis=1, keepdims=True)

    classifier_history: list[dict[str, Any]] = []
    best_val_f1 = -1.0
    best_state = {key: value.detach().clone() for key, value in model.classifier.state_dict().items()}
    started = time.perf_counter()
    for epoch in range(classifier_epochs):
        model.classifier.train()
        for batch in classifier_loader:
            classifier_optimizer.zero_grad(set_to_none=True)
            input_ids = batch["input_ids"].to(selected_device)
            attention_mask = batch["attention_mask"].to(selected_device)
            labels = batch["labels"].to(selected_device)
            pooled = model.embed(input_ids, attention_mask)
            loss = F.cross_entropy(model.classify(pooled), labels)
            loss.backward()
            classifier_optimizer.step()
        val_pred = predict_logits(val_dataset).argmax(axis=1)
        val_macro_f1 = classification_metrics(y_val, val_pred, categories)["macro_f1"]
        classifier_history.append({"epoch": epoch, "validation_macro_f1": val_macro_f1})
        if val_macro_f1 > best_val_f1:
            best_val_f1 = val_macro_f1
            best_state = {key: value.detach().clone() for key, value in model.classifier.state_dict().items()}
    classifier_seconds = time.perf_counter() - started
    model.classifier.load_state_dict(best_state)

    val_scores = _softmax(predict_logits(val_dataset))
    val_pred = val_scores.argmax(axis=1)
    validation_metrics = classification_metrics(y_val, val_pred, categories)

    started = time.perf_counter()
    test_scores = _softmax(predict_logits(test_dataset))
    inference_seconds = time.perf_counter() - started
    y_pred = artifacts.write_predictions(test["text"].tolist(), y_test, test_scores, categories)
    test_metrics = classification_metrics(y_test, y_pred, categories)
    artifacts.write_confusion(y_test, y_pred, categories)
    artifacts.write_history(
        {"contrastive": contrastive_history, "classifier": classifier_history}
    )

    validation_health = prediction_health(y_val, val_pred, categories)
    test_health = prediction_health(y_test, y_pred, categories)
    gradient_report = gradient_health.report()
    collapsed = validation_health["collapsed"] or gradient_report["non_finite_grad_norm_events"] > 0

    artifacts.write_metrics(
        validation_metrics,
        test_metrics,
        {
            "contrastive_seconds": contrastive_seconds,
            "classifier_seconds": classifier_seconds,
            "train_seconds": contrastive_seconds + classifier_seconds,
            "test_inference_seconds": inference_seconds,
            "test_examples_per_second": len(test) / inference_seconds,
            "effective_contrastive_batch": effective_batch,
            "best_validation_macro_f1": best_val_f1,
            "gradient_health": gradient_report,
            "validation_health": validation_health,
            "test_health": test_health,
            "run_flagged_as_collapsed": collapsed,
        },
    )
    artifacts.write_status("completed", {"collapsed": collapsed})
    if save_model:
        model.encoder.save_pretrained(artifacts.model_dir / "best")
        torch.save(model.classifier.state_dict(), artifacts.model_dir / "best" / "classifier.pt")
        tokenizer.save_pretrained(artifacts.model_dir / "best")
    artifacts.cleanup_checkpoints()

    return {
        "method": method,
        "setting": setting,
        "seed": seed,
        "validation_macro_f1": validation_metrics["macro_f1"],
        "test_macro_f1": test_metrics["macro_f1"],
        "predicted_class_count": test_health["predicted_class_count"],
        "run_flagged_as_collapsed": collapsed,
        "output_dir": str(artifacts.path),
    }


def run_hard_negative(setting: str, seed: int, **kwargs: Any) -> dict[str, Any]:
    """SupCon plus similarity-weighted hard negatives."""

    kwargs.setdefault("method", "supcon_hn")
    return run(setting, seed, **kwargs)