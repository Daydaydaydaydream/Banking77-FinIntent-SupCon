"""Mixed-precision policy and gradient health checks.

Few-shot full fine-tuning of BERT overflowed under float16: the training log
contained repeated ``grad_norm: inf`` followed by ``nan``.  float16 has the same
exponent range as float32 (max ~65504) but a much smaller one, so a single large
loss or gradient silently poisons the run and the model collapses to a handful
of classes.

Two independent guards are used here:

1. bfloat16 is the default on CUDA.  It keeps float32's exponent range, so the
   overflow cannot happen at the same magnitude.
2. A callback inspects ``grad_norm`` on every logging step and records a
   violation instead of letting the run continue silently.  The count ends up
   in ``metrics.json`` so a collapsed run is identifiable after the fact.
"""

from __future__ import annotations

from typing import Any

import torch

try:  # transformers is a project dependency, but keep this module importable
    from transformers import TrainerCallback
except ImportError:  # pragma: no cover - only hit outside a full install
    TrainerCallback = object  # type: ignore[assignment,misc]


PRECISIONS = ("fp32", "bf16", "fp16")
DEFAULT_PRECISION = "auto"


def resolve_precision(requested: str, device: str) -> dict[str, Any]:
    """Choose a training precision and report whether the device supports it."""

    if requested not in PRECISIONS + (DEFAULT_PRECISION,):
        raise ValueError(f"precision must be one of {PRECISIONS + (DEFAULT_PRECISION,)}")
    if device != "cuda":
        # bf16/fp16 autocast only applies to CUDA kernels here.
        return {"requested": requested, "resolved": "fp32", "reason": f"device={device}"}

    supports_bf16 = torch.cuda.is_bf16_supported()
    if requested == "auto":
        # bfloat16 first: same exponent range as fp32, so gradient overflow is
        # not expected even in the few-shot regime.
        resolved = "bf16" if supports_bf16 else "fp32"
        reason = "bf16 supported" if supports_bf16 else "device lacks bf16, falling back to fp32"
    else:
        resolved = requested
        if requested == "bf16" and not supports_bf16:
            raise RuntimeError("bf16 was requested but this CUDA device does not support it")
        reason = "explicitly requested"

    return {"requested": requested, "resolved": resolved, "reason": reason}


def precision_flags(precision: str) -> dict[str, bool]:
    """Translate a resolved precision into ``TrainingArguments`` flags."""

    return {
        "fp16": precision == "fp16",
        "bf16": precision == "bf16",
        "fp16_full_eval": precision == "fp16",
        "half_precision_backend": precision == "fp16",
    }


class GradientHealthCallback(TrainerCallback):
    """Record non-finite gradient norms instead of training through them.

    HF Trainer already skips the optimizer step when ``grad_norm`` is not
    finite, so this does not change the trajectory of a healthy run.  It exists
    so that a run which silently lost most of its updates is distinguishable
    from a genuinely bad configuration.
    """

    def __init__(self) -> None:
        super().__init__()
        self.violations: list[dict[str, Any]] = []
        self.checked_steps = 0

    def on_log(self, args: Any, state: Any, control: Any, logs: dict | None = None, **kwargs: Any) -> Any:
        if not logs or "grad_norm" not in logs:
            return control
        grad_norm = logs["grad_norm"]
        self.checked_steps += 1
        try:
            numeric = float(grad_norm)
        except (TypeError, ValueError):
            return control
        if numeric != numeric or numeric in (float("inf"), float("-inf")):
            self.violations.append({"step": int(state.global_step), "grad_norm": str(grad_norm)})
        return control

    def report(self) -> dict[str, Any]:
        return {
            "gradient_checks": self.checked_steps,
            "non_finite_grad_norm_events": len(self.violations),
            "non_finite_grad_norm_steps": self.violations[:20],
        }


def prediction_health(y_true: Any, y_pred: Any, categories: list[str]) -> dict[str, Any]:
    """Coverage statistics that expose class-collapse runs.

    A model that predicts only a subset of the label space scores a plausible
    accuracy on the classes it does emit, so Macro-F1 alone is not enough to
    distinguish "hard task" from "broken run".
    """

    import numpy as np

    y_pred_arr = np.asarray(y_pred)
    predicted = np.unique(y_pred_arr)
    per_class_f1 = []
    for label in range(len(categories)):
        tp = int(((y_pred_arr == label) & (np.asarray(y_true) == label)).sum())
        fp = int(((y_pred_arr == label) & (np.asarray(y_true) != label)).sum())
        fn = int(((y_pred_arr != label) & (np.asarray(y_true) == label)).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        per_class_f1.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)

    counts = np.bincount(y_pred_arr, minlength=len(categories)).astype(float)
    probs = counts / counts.sum()
    nonzero = probs[probs > 0]
    entropy = float(-(nonzero * np.log(nonzero)).sum() / np.log(len(categories)))
    return {
        "predicted_class_count": int(predicted.size),
        "zero_f1_class_count": int(sum(1 for f in per_class_f1 if f == 0.0)),
        "prediction_distribution_entropy": entropy,
        "collapsed": bool(predicted.size < 0.5 * len(categories)),
    }