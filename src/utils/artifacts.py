"""Uniform output writer for all baseline methods."""

from __future__ import annotations

import csv
import importlib.metadata
import json
import platform
import sys
from dataclasses import dataclass
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

    def __post_init__(self) -> None:
        self.path = run_dir(self.method, self.setting, self.seed)
        self.path.mkdir(parents=True, exist_ok=True)

    @property
    def complete(self) -> bool:
        return (self.path / "metrics.json").exists()

    @property
    def model_dir(self) -> Path:
        path = self.path / "model"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def write_config(self, config: dict[str, Any]) -> None:
        _write_json(self.path / "config.json", config)
        _write_json(self.path / "env.json", environment_info())

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
    ) -> np.ndarray:
        if scores.shape != (len(texts), len(categories)):
            raise ValueError(f"Unexpected score shape {scores.shape}")
        ranking = np.argsort(-scores, axis=1)
        y_pred = ranking[:, 0]
        output = self.path / "predictions.csv"
        with output.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
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
            writer = csv.writer(handle)
            writer.writerow(["true\\pred", *categories])
            writer.writerows(confusion_rows(y_true, y_pred, categories))

    def write_history(self, history: Any) -> None:
        _write_json(self.path / "training_history.json", history)
