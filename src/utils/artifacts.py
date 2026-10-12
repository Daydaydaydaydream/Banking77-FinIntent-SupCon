"""Uniform output writer for all baseline methods."""

from __future__ import annotations

import csv
import importlib.metadata
import json
import platform
import shutil
import sys
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from config import run_dir
from utils.metrics import confusion_rows


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Cannot serialize {type(value)!r}")


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )


def environment_info() -> dict[str, Any]:
    packages = {}
    for name in (
        "numpy",
        "pandas",
        "scikit-learn",
        "torch",
        "transformers",
        "datasets",
        "sentence-transformers",
        "setfit",
    ):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": packages,
    }


@dataclass
class RunArtifacts:
    method: str
    setting: str
    seed: int
    keep_checkpoints: bool = field(default=False)
    root: Path | None = None

    def __post_init__(self) -> None:
        self.path = (self.root / self.method / self.setting / f"seed{self.seed}") if self.root is not None else run_dir(self.method, self.setting, self.seed)
        self.path.mkdir(parents=True, exist_ok=True)

    @property
    def complete(self) -> bool:
        return (self.path / "metrics.json").exists()

    @property
    def model_dir(self) -> Path:
        path = self.path / "model"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def checkpoint_dir(self) -> Path:
        """Where the HF Trainer writes intermediate checkpoints."""

        path = self.path / "model" / "checkpoints"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def cleanup_checkpoints(self) -> None:
        """Delete intermediate trainer checkpoints.

        With ``save_total_limit=1`` the trainer still leaves one full
        checkpoint on disk (weights, optimizer and scheduler state).  For BERT
        that is roughly 1.3 GB per run and it is fully reproducible from
        ``config.json``, so it is removed after the best model is saved.
        """

        if self.keep_checkpoints:
            return
        checkpoint_root = self.path / "model" / "checkpoints"
        if checkpoint_root.exists():
            shutil.rmtree(checkpoint_root, ignore_errors=True)

    def write_config(self, config: dict[str, Any]) -> None:
        _write_json(self.path / "config.json", config)
        _write_json(self.path / "env.json", environment_info())

    def write_status(self, status: str, detail: dict[str, Any] | None = None) -> None:
        """Record run lifecycle state so failures leave evidence behind.

        Previously a run that raised before ``write_metrics`` left nothing but
        a stray model directory, so silent failures were indistinguishable from
        runs that were never started.
        """

        payload: dict[str, Any] = {"status": status}
        if detail:
            payload.update(detail)
        _write_json(self.path / "status.json", payload)

    def write_failure(self, exc: BaseException, stage: str) -> None:
        """Persist a traceback for an aborted run."""

        _write_json(
            self.path / "failure.json",
            {
                "stage": stage,
                "exception_type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            },
        )
        self.write_status("failed", {"stage": stage, "exception_type": type(exc).__name__})

    def write_metrics(self, validation: dict[str, Any], test: dict[str, Any], extra: dict[str, Any]) -> None:
        _write_json(
            self.path / "metrics.json",
            {"validation": validation, "test": test, **extra},
        )

    def write_predictions(
        self,
        texts: list[str],
        y_true: np.ndarray,
        scores: np.ndarray,
        categories: list[str],
        filename: str = "predictions.csv",
    ) -> np.ndarray:
        if scores.shape != (len(texts), len(categories)):
            raise ValueError(f"Unexpected score shape {scores.shape}")
        ranking = np.argsort(-scores, axis=1)
        y_pred = ranking[:, 0]
        if filename not in ("predictions.csv", "validation_predictions.csv"):
            raise ValueError("Unsupported predictions filename")
        output = self.path / filename
        with output.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(
                [
                    "text",
                    "y_true_id",
                    "y_true",
                    "y_pred_id",
                    "y_pred",
                    "top1_score",
                    "top2",
                    "top2_score",
                    "top3",
                    "top3_score",
                ]
            )
            for idx, text in enumerate(texts):
                top = ranking[idx, :3]
                writer.writerow(
                    [
                        text,
                        int(y_true[idx]),
                        categories[int(y_true[idx])],
                        int(top[0]),
                        categories[int(top[0])],
                        float(scores[idx, top[0]]),
                        categories[int(top[1])],
                        float(scores[idx, top[1]]),
                        categories[int(top[2])],
                        float(scores[idx, top[2]]),
                    ]
                )
        return y_pred

    def write_confusion(self, y_true: np.ndarray, y_pred: np.ndarray, categories: list[str]) -> None:
        with (self.path / "confusion_matrix.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(["true\\pred", *categories])
            writer.writerows(confusion_rows(y_true, y_pred, categories))

    def write_history(self, history: Any) -> None:
        _write_json(self.path / "training_history.json", history)

    def relative(self, path: Path | str) -> str:
        """Express ``path`` relative to the run directory when possible."""

        candidate = Path(path)
        try:
            return str(candidate.resolve().relative_to(self.path.resolve()))
        except ValueError:
            return candidate.name
