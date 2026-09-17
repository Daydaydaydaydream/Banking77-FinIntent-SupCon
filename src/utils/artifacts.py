"""Uniform run artefacts (stage A.3).

Every method writes the same directory tree so that downstream analysis,
reporting and significance testing can be written once instead of per method:

    outputs/runs/<method>/<setting>/seed<k>/
    ├── metrics.json      macro/micro-F1, accuracy, per-class P/R/F1
    ├── predictions.csv   per-example truth, prediction, top-3, confidence
    ├── embeddings.npy    test-set encoder representations (optional)
    ├── config.json       every hyper-parameter plus method/setting/seed
    ├── env.json          package versions, device, determinism state
    └── model/            weights, tokenizer, label map (git-ignored)

Methods that cannot produce embeddings (or probabilities) simply omit the
corresponding field instead of failing, and the analysis layer degrades
gracefully.
"""

from __future__ import annotations

import json
import platform
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from config import run_dir
from utils.metrics import summarize

_TRACKED_PACKAGES = (
    "torch",
    "transformers",
    "setfit",
    "sentence-transformers",
    "datasets",
    "scikit-learn",
    "numpy",
    "pandas",
    "scipy",
)


def _package_version(name: str) -> Optional[str]:
    try:
        from importlib.metadata import PackageNotFoundError, version

        return version(name)
    except Exception:
        return None


def _jsonable(value):
    """Convert numpy / Path values so ``json.dump`` never raises."""
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


class RunArtifacts:
    """Writer for one ``method / setting / seed`` run directory."""

    def __init__(self, root: Path, method: str, setting: str, seed: int) -> None:
        self.root = Path(root)
        self.method = method
        self.setting = setting
        self.seed = int(seed)
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "model").mkdir(parents=True, exist_ok=True)

    @classmethod
    def create(cls, method: str, setting: str, seed: int) -> "RunArtifacts":
        return cls(run_dir(method, setting, seed), method, setting, seed)

    @property
    def model_dir(self) -> Path:
        return self.root / "model"

    # ------------------------------------------------------------------ #
    # Writers
    # ------------------------------------------------------------------ #
    def save_config(self, config: Dict) -> Path:
        payload = {
            "method": self.method,
            "setting": self.setting,
            "seed": self.seed,
            **_jsonable(config),
        }
        return self._write("config.json", payload)

    def save_env(self, extra: Optional[Dict] = None) -> Path:
        device = "cpu"
        try:
            import torch

            if torch.backends.mps.is_available():
                device = "mps"
            elif torch.cuda.is_available():
                device = "cuda"
        except ImportError:
            pass

        payload: Dict = {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "machine": platform.machine(),
            "device": device,
            "packages": {name: _package_version(name) for name in _TRACKED_PACKAGES},
        }
        if extra:
            payload.update(_jsonable(extra))
        return self._write("env.json", payload)

    def save_metrics(
        self,
        y_true,
        y_pred,
        id2label: Sequence[str],
        extra: Optional[Dict] = None,
    ) -> Dict:
        block = summarize(y_true, y_pred, id2label, extra=extra)
        self._write("metrics.json", block)
        return block

    def save_predictions(
        self,
        y_true,
        y_pred,
        id2label: Sequence[str],
        probabilities=None,
    ) -> Path:
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        frame: Dict[str, object] = {
            "y_true": y_true,
            "y_true_intent": [id2label[int(i)] for i in y_true],
            "y_pred": y_pred,
            "y_pred_intent": [id2label[int(i)] for i in y_pred],
            "correct": (y_true == y_pred).astype(int),
        }

        if probabilities is not None:
            probs = np.asarray(probabilities)
            top = np.argsort(-probs, axis=1)[:, :3]
            frame["confidence"] = probs[np.arange(probs.shape[0]), y_pred]
            for rank in range(3):
                frame[f"top{rank + 1}_intent"] = [id2label[int(i)] for i in top[:, rank]]
        else:
            frame["confidence"] = np.full(y_true.shape[0], np.nan)
            frame["top1_intent"] = frame["y_pred_intent"]
            frame["top2_intent"] = ""
            frame["top3_intent"] = ""

        path = self.root / "predictions.csv"
        pd.DataFrame(frame).to_csv(path, index=False)
        return path

    def save_embeddings(self, embeddings, sparse: bool = False) -> Optional[Path]:
        """Persist test-set embeddings. Returns ``None`` when unavailable."""
        if embeddings is None:
            return None
        if sparse:
            from scipy import sparse as sp

            path = self.root / "embeddings.npz"
            sp.save_npz(path, embeddings)
            return path
        path = self.root / "embeddings.npy"
        np.save(path, np.asarray(embeddings))
        return path

    def summary_line(self, macro_f1: Optional[float] = None) -> str:
        base = f"{self.method}/{self.setting}/seed{self.seed}"
        score = "n/a" if macro_f1 is None else f"{macro_f1:.4f}"
        return f"[run] {base}: Macro-F1 = {score} -> {self.root}"

    # ------------------------------------------------------------------ #
    # Internal
    # ------------------------------------------------------------------ #
    def _write(self, filename: str, payload: Dict) -> Path:
        path = self.root / filename
        with open(path, "w", encoding="utf-8") as f:
            json.dump(_jsonable(payload), f, indent=2, ensure_ascii=False)
        return path


def load_run_metrics(method: str, setting: str, seed: int) -> Optional[Dict]:
    """Read ``metrics.json`` for a run, or ``None`` if it does not exist."""
    path = run_dir(method, setting, seed) / "metrics.json"
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def iter_runs() -> List[Dict]:
    """List every completed run as ``{method, setting, seed, path}``."""
    from config import RUNS_DIR

    runs: List[Dict] = []
    if not RUNS_DIR.exists():
        return runs
    for metric_path in sorted(RUNS_DIR.glob("*/*/seed*/metrics.json")):
        parts = metric_path.relative_to(RUNS_DIR).parts
        if len(parts) != 3:
            continue
        method, setting, seed_dir = parts
        runs.append(
            {
                "method": method,
                "setting": setting,
                "seed": int(seed_dir.replace("seed", "")),
                "path": metric_path.parent,
            }
        )
    return runs
