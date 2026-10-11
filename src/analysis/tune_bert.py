"""Validation-only hyperparameter search for the BERT baseline.

This script exists to keep the protocol honest: it never selects on the test
split.  For every candidate it trains on the few-shot training pool, selects
the checkpoint by validation Macro-F1, and reports validation Macro-F1 only.
The frozen result is written to ``configs/bert_frozen.json`` and consumed by
``run_baselines --config``.

Every candidate writes into a temporary directory, so a tuning run can never
be mistaken for a recorded experiment in ``outputs/runs``.

Usage::

    PYTHONPATH=src python -m analysis.tune_bert --setting 5shot --freeze
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any

from config import DEFAULT_SEED, LOCAL_BERT_MODEL, ROOT, SETTINGS


FROZEN_PATH = ROOT / "configs" / "bert_frozen.json"


# Deliberately small grid: each point costs a full training run, and the goal
# is to find a configuration that does not collapse rather than to squeeze the
# last decimal out of the validation split.
CANDIDATES: list[dict[str, Any]] = [
    {"name": "baseline_fp32", "precision": "fp32", "learning_rate": 2e-5, "freeze_layers": 0},
    {"name": "bf16", "precision": "bf16", "learning_rate": 2e-5, "freeze_layers": 0},
    {"name": "bf16_lr3e5", "precision": "bf16", "learning_rate": 3e-5, "freeze_layers": 0},
    {"name": "bf16_freeze6", "precision": "bf16", "learning_rate": 3e-5, "freeze_layers": 6},
    {"name": "bf16_lr5e5", "precision": "bf16", "learning_rate": 5e-5, "freeze_layers": 0},
]


def run_candidate(setting: str, seed: int, candidate: dict[str, Any]) -> dict[str, Any]:
    """Train one candidate into a scratch directory and report validation only."""

    from baselines import bert_ft

    with tempfile.TemporaryDirectory(prefix="bert_tune_") as scratch:
        scratch_root = Path(scratch)
        base_artifacts = bert_ft.RunArtifacts

        class _ScratchArtifacts(base_artifacts):  # type: ignore[misc, valid-type]
            def __post_init__(self) -> None:
                self.path = scratch_root / self.method / self.setting / f"seed{self.seed}"
                self.path.mkdir(parents=True, exist_ok=True)

        bert_ft.RunArtifacts = _ScratchArtifacts  # type: ignore[assignment]
        try:
            bert_ft.run(
                setting,
                seed,
                model_name=LOCAL_BERT_MODEL,
                local_files_only=True,
                save_model=False,
                precision=str(candidate["precision"]),
                learning_rate=float(candidate["learning_rate"]),
                freeze_layers=int(candidate["freeze_layers"]),
                epochs=10.0 if setting == "full" else 5.0,
                batch_size=32,
                warmup_ratio=0.1,
            )
        finally:
            bert_ft.RunArtifacts = base_artifacts  # type: ignore[assignment]

        metrics_path = scratch_root / "bert" / setting / f"seed{seed}" / "metrics.json"
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        return {
            "candidate": candidate["name"],
            "precision": candidate["precision"],
            "learning_rate": candidate["learning_rate"],
            "freeze_layers": candidate["freeze_layers"],
            "validation_macro_f1": metrics["validation"]["macro_f1"],
            "validation_health": metrics.get("validation_health", {}),
            "gradient_health": metrics.get("gradient_health", {}),
            "optimizer_steps": metrics.get("optimizer_steps"),
            "best_checkpoint": metrics.get("best_checkpoint"),
            # Recorded for completeness but never used for selection.
            "test_macro_f1_not_used_for_selection": metrics["test"]["macro_f1"],
        }


def sweep(setting: str, seed: int, candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        print(f"[tune] {setting} seed={seed} candidate={candidate['name']}", flush=True)
        row = run_candidate(setting, seed, candidate)
        rows.append(row)
        health = row["validation_health"]
        print(
            f"    val_macro_f1={row['validation_macro_f1']:.4f} "
            f"pred_classes={health.get('predicted_class_count')} "
            f"zero_f1={health.get('zero_f1_class_count')} "
            f"nonfinite_grad={row['gradient_health'].get('non_finite_grad_norm_events')}",
            flush=True,
        )
    return rows


def select(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Pick the best candidate on validation Macro-F1.

    Collapsed candidates are excluded rather than merely losing on score: a run
    that emits 26 of 77 classes has not converged, whatever its Macro-F1.
    """

    healthy = [row for row in rows if not row["validation_health"].get("collapsed")]
    pool = healthy or rows
    return max(pool, key=lambda row: row["validation_macro_f1"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--setting", choices=[*SETTINGS, "all"], default="5shot")
    parser.add_argument("--seed", type=int, nargs="+", default=[DEFAULT_SEED])
    parser.add_argument("--freeze", action="store_true", help="write configs/bert_frozen.json")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "tuning")
    return parser.parse_args()


def main() -> None:
    from baselines import bert_ft  # noqa: F401 - validate imports before a long sweep

    args = parse_args()
    settings = list(SETTINGS) if args.setting == "all" else [args.setting]
    args.output.mkdir(parents=True, exist_ok=True)

    frozen: dict[str, Any] = {}
    for setting in settings:
        all_rows: list[dict[str, Any]] = []
        for seed in args.seed:
            all_rows.extend(sweep(setting, seed, CANDIDATES))

        # Aggregate per candidate across seeds before selecting, so one lucky
        # split cannot decide the configuration.
        by_name: dict[str, list[dict[str, Any]]] = {}
        for row in all_rows:
            by_name.setdefault(row["candidate"], []).append(row)
        summary = {
            name: {
                "validation_macro_f1_mean": sum(r["validation_macro_f1"] for r in rows) / len(rows),
                "any_collapsed": any(r["validation_health"].get("collapsed") for r in rows),
                "precision": rows[0]["precision"],
                "learning_rate": rows[0]["learning_rate"],
                "freeze_layers": rows[0]["freeze_layers"],
            }
            for name, rows in by_name.items()
        }
        healthy_names = [n for n, s in summary.items() if not s["any_collapsed"]] or list(summary)
        best_name = max(healthy_names, key=lambda n: summary[n]["validation_macro_f1_mean"])
        best = summary[best_name]
        print(f"[tune] {setting}: selected {best_name} val={best['validation_macro_f1_mean']:.4f}")

        frozen[setting] = {
            "precision": best["precision"],
            "learning_rate": best["learning_rate"],
            "freeze_layers": best["freeze_layers"],
            "epochs": 10.0 if setting == "full" else 5.0,
            "batch_size": 32,
            "warmup_ratio": 0.1,
            "selected_on": f"validation Macro-F1 over seeds {args.seed}",
            "selected_candidate": best_name,
        }
        (args.output / f"bert_{setting}_sweep.json").write_text(
            json.dumps({"rows": all_rows, "summary": summary, "selected": best_name}, indent=2),
            encoding="utf-8",
        )

    if args.freeze:
        existing: dict[str, Any] = {}
        if FROZEN_PATH.exists():
            existing = json.loads(FROZEN_PATH.read_text(encoding="utf-8"))
        existing.setdefault("bert", {}).update(frozen)
        FROZEN_PATH.parent.mkdir(parents=True, exist_ok=True)
        FROZEN_PATH.write_text(json.dumps(existing, indent=2), encoding="utf-8")
        print(f"wrote {FROZEN_PATH}")


if __name__ == "__main__":
    main()