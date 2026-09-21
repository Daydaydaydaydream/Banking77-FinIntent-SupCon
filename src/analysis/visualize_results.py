"""Visualize saved BANKING77 experiment results.

The module discovers completed runs under ``outputs/runs`` and produces both
cross-run comparisons and per-run diagnostic figures. It only reads existing
artifacts and never modifies experiment results.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import ROOT, RUNS_DIR, SETTINGS


DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "figures" / "model_results"
METRIC_NAMES = ("macro_f1", "accuracy", "micro_f1")


@dataclass(frozen=True)
class RunResult:
    """A completed experiment run and its metrics."""

    method: str
    setting: str
    seed: int
    path: Path
    metrics: dict[str, Any]


def _safe_name(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "-", value).strip("-").lower()


def _read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def discover_runs(runs_dir: Path = RUNS_DIR) -> list[RunResult]:
    """Return completed runs found under ``method/setting/seedN``."""

    if not runs_dir.exists():
        return []

    results: list[RunResult] = []
    for metrics_path in sorted(runs_dir.glob("*/*/seed*/metrics.json")):
        run_path = metrics_path.parent
        seed_match = re.fullmatch(r"seed(-?\d+)", run_path.name)
        if seed_match is None:
            continue
        try:
            metrics = _read_json(metrics_path)
        except (json.JSONDecodeError, OSError) as exc:
            raise ValueError(f"Cannot read metrics file: {metrics_path}") from exc
        results.append(
            RunResult(
                method=run_path.parent.parent.name,
                setting=run_path.parent.name,
                seed=int(seed_match.group(1)),
                path=run_path,
                metrics=metrics,
            )
        )
    return results


def select_runs(
    runs: Sequence[RunResult],
    methods: Sequence[str] | None = None,
    settings: Sequence[str] | None = None,
    seeds: Sequence[int] | None = None,
) -> list[RunResult]:
    """Filter runs while preserving discovery order."""

    method_set = set(methods) if methods else None
    setting_set = set(settings) if settings else None
    seed_set = set(seeds) if seeds else None
    return [
        run
        for run in runs
        if (method_set is None or run.method in method_set)
        and (setting_set is None or run.setting in setting_set)
        and (seed_set is None or run.seed in seed_set)
    ]


def metrics_frame(runs: Sequence[RunResult], split: str) -> pd.DataFrame:
    """Create a tidy frame containing the scalar metrics for each run."""

    rows: list[dict[str, Any]] = []
    for run in runs:
        split_metrics = run.metrics.get(split)
        if not isinstance(split_metrics, dict):
            continue
        row: dict[str, Any] = {
            "method": run.method,
            "setting": run.setting,
            "seed": run.seed,
        }
        row.update({name: split_metrics.get(name, np.nan) for name in METRIC_NAMES})
        row["train_seconds"] = run.metrics.get("train_seconds", np.nan)
        rows.append(row)
    return pd.DataFrame(rows)


def _ordered_settings(values: Iterable[str]) -> list[str]:
    observed = list(dict.fromkeys(values))
    known = [setting for setting in SETTINGS if setting in observed]
    return known + sorted(set(observed) - set(known))


def _save_figure(fig: plt.Figure, path: Path, dpi: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_metric_comparison(
    frame: pd.DataFrame,
    output_path: Path,
    metric: str = "macro_f1",
    dpi: int = 180,
) -> Path:
    """Plot mean metric by method and setting, with seed standard deviation."""

    if frame.empty:
        raise ValueError("No scalar metrics are available for comparison")
    if metric not in frame.columns:
        raise ValueError(f"Unknown metric: {metric}")

    methods = sorted(frame["method"].unique())
    settings = _ordered_settings(frame["setting"])
    grouped = frame.groupby(["method", "setting"])[metric].agg(["mean", "std"])

    fig, ax = plt.subplots(figsize=(max(8, 1.7 * len(settings)), 5.5))
    x = np.arange(len(settings), dtype=float)
    total_width = 0.82
    width = total_width / max(len(methods), 1)
    colors = plt.get_cmap("tab10").colors

    for method_idx, method in enumerate(methods):
        means = []
        errors = []
        for setting in settings:
            if (method, setting) in grouped.index:
                row = grouped.loc[(method, setting)]
                means.append(float(row["mean"]))
                errors.append(0.0 if pd.isna(row["std"]) else float(row["std"]))
            else:
                means.append(np.nan)
                errors.append(0.0)
        positions = x - total_width / 2 + width / 2 + method_idx * width
        bars = ax.bar(
            positions,
            means,
            width=width * 0.9,
            yerr=errors,
            capsize=3,
            label=method,
            color=colors[method_idx % len(colors)],
        )
        ax.bar_label(bars, fmt="%.3f", padding=2, fontsize=8, rotation=90)

    ax.set_title(f"{metric.replace('_', ' ').title()} by method and label budget")
    ax.set_xlabel("Label budget")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_xticks(x, settings)
    ax.set_ylim(0, 1.08)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(title="Method", frameon=False)
    fig.tight_layout()
    return _save_figure(fig, output_path, dpi)


def _per_class_frame(runs: Sequence[RunResult], split: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for run in runs:
        per_class = run.metrics.get(split, {}).get("per_class", {})
        for category, values in per_class.items():
            rows.append(
                {
                    "method": run.method,
                    "setting": run.setting,
                    "seed": run.seed,
                    "category": category,
                    "f1": values.get("f1", np.nan),
                }
            )
    return pd.DataFrame(rows)


def plot_per_class_f1(
    runs: Sequence[RunResult],
    split: str,
    output_path: Path,
    dpi: int = 180,
) -> Path | None:
    """Compare per-class F1 for runs sharing one setting and seed."""

    frame = _per_class_frame(runs, split)
    if frame.empty:
        return None

    pivot = frame.pivot_table(index="category", columns="method", values="f1", aggfunc="mean")
    pivot = pivot.assign(_mean=pivot.mean(axis=1)).sort_values("_mean").drop(columns="_mean")
    methods = list(pivot.columns)
    y = np.arange(len(pivot), dtype=float)
    offsets = np.linspace(-0.25, 0.25, len(methods)) if len(methods) > 1 else np.array([0.0])
    colors = plt.get_cmap("tab10").colors

    fig_height = max(8, 0.25 * len(pivot) + 2)
    fig, ax = plt.subplots(figsize=(11, fig_height))
    for method_idx, method in enumerate(methods):
        ax.scatter(
            pivot[method],
            y + offsets[method_idx],
            s=25,
            label=method,
            color=colors[method_idx % len(colors)],
            alpha=0.9,
        )
    ax.set_title(f"Per-class F1 ({split})")
    ax.set_xlabel("F1")
    ax.set_ylabel("Intent")
    ax.set_xlim(-0.02, 1.02)
    ax.set_yticks(y, pivot.index, fontsize=7)
    ax.grid(axis="x", alpha=0.25)
    ax.legend(title="Method", frameon=False, loc="lower right")
    fig.tight_layout()
    return _save_figure(fig, output_path, dpi)


def load_confusion_matrix(path: Path) -> pd.DataFrame:
    """Load and validate a saved confusion matrix."""

    frame = pd.read_csv(path, index_col=0)
    if frame.empty or frame.shape[0] != frame.shape[1]:
        raise ValueError(f"Confusion matrix must be non-empty and square: {path}")
    if list(frame.index) != list(frame.columns):
        raise ValueError(f"Confusion matrix row and column labels differ: {path}")
    return frame.apply(pd.to_numeric, errors="raise")


def top_confusion_pairs(matrix: pd.DataFrame, top_k: int = 15) -> pd.DataFrame:
    """Return highest bidirectional off-diagonal confusion counts."""

    values = matrix.to_numpy(dtype=float)
    labels = list(matrix.index)
    columns = ["intent_a", "intent_b", "a_to_b", "b_to_a", "total"]
    rows: list[dict[str, Any]] = []
    for left in range(len(labels)):
        for right in range(left + 1, len(labels)):
            left_to_right = int(values[left, right])
            right_to_left = int(values[right, left])
            total = left_to_right + right_to_left
            if total:
                rows.append(
                    {
                        "intent_a": labels[left],
                        "intent_b": labels[right],
                        "a_to_b": left_to_right,
                        "b_to_a": right_to_left,
                        "total": total,
                    }
                )
    if not rows:
        return pd.DataFrame(columns=columns)
    return (
        pd.DataFrame(rows, columns=columns)
        .sort_values("total", ascending=False)
        .head(top_k)
        .reset_index(drop=True)
    )


def plot_confusion_matrix(
    matrix: pd.DataFrame,
    output_path: Path,
    title: str,
    dpi: int = 180,
) -> Path:
    """Plot a row-normalized confusion matrix."""

    values = matrix.to_numpy(dtype=float)
    row_totals = values.sum(axis=1, keepdims=True)
    normalized = np.divide(values, row_totals, out=np.zeros_like(values), where=row_totals != 0)

    fig, ax = plt.subplots(figsize=(18, 16))
    image = ax.imshow(normalized, cmap="Blues", vmin=0, vmax=1, interpolation="nearest", aspect="auto")
    ax.set_title(f"{title} — row-normalized confusion matrix")
    ax.set_xlabel("Predicted intent")
    ax.set_ylabel("True intent")
    labels = list(matrix.index)
    ticks = np.arange(len(labels))
    ax.set_xticks(ticks, labels, rotation=90, fontsize=5)
    ax.set_yticks(ticks, labels, fontsize=5)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02)
    colorbar.set_label("Share within true class")
    fig.tight_layout()
    return _save_figure(fig, output_path, dpi)


def plot_top_confusions(
    pairs: pd.DataFrame,
    output_path: Path,
    title: str,
    dpi: int = 180,
) -> Path | None:
    """Plot the most frequent bidirectional confusion pairs."""

    if pairs.empty:
        return None
    shown = pairs.sort_values("total")
    labels = shown["intent_a"] + "  ↔  " + shown["intent_b"]
    fig, ax = plt.subplots(figsize=(12, max(5, 0.52 * len(shown) + 1.5)))
    ax.barh(labels, shown["total"], color="#D97706")
    for y_idx, (_, row) in enumerate(shown.iterrows()):
        ax.text(
            row["total"] + 0.2,
            y_idx,
            f"{int(row['a_to_b'])} + {int(row['b_to_a'])}",
            va="center",
            fontsize=8,
        )
    ax.set_title(f"{title} — top bidirectional confusions")
    ax.set_xlabel("Misclassified examples (A→B + B→A)")
    ax.set_ylabel("")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    return _save_figure(fig, output_path, dpi)


def plot_prediction_distribution(
    predictions_path: Path,
    categories: Sequence[str],
    output_path: Path,
    title: str,
    dpi: int = 180,
) -> Path | None:
    """Plot predicted counts for every class when predictions are available."""

    if not predictions_path.exists():
        return None
    predictions = pd.read_csv(predictions_path, usecols=["y_pred"])
    counts = predictions["y_pred"].value_counts().reindex(categories, fill_value=0).sort_values()
    colors = np.where(counts.to_numpy() == 0, "#B91C1C", "#2563EB")

    fig, ax = plt.subplots(figsize=(11, max(8, 0.24 * len(counts) + 2)))
    ax.barh(counts.index, counts.values, color=colors)
    ax.set_title(f"{title} — predicted class distribution")
    ax.set_xlabel("Predicted examples")
    ax.set_ylabel("Intent")
    ax.tick_params(axis="y", labelsize=7)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    return _save_figure(fig, output_path, dpi)


def plot_training_history(
    history_path: Path,
    output_path: Path,
    title: str,
    dpi: int = 180,
) -> Path | None:
    """Plot validation Macro-F1 and loss from a deep-model history file."""

    if not history_path.exists():
        return None
    history = _read_json(history_path)
    evaluations = [row for row in history if "eval_macro_f1" in row or "eval_loss" in row]
    if not evaluations:
        return None

    epochs = [row.get("epoch", row.get("step")) for row in evaluations]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].plot(
        epochs,
        [row.get("eval_macro_f1", np.nan) for row in evaluations],
        marker="o",
        color="#2563EB",
    )
    axes[0].set_title("Validation Macro-F1")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Macro-F1")
    axes[0].set_ylim(0, 1)
    axes[0].grid(alpha=0.25)

    axes[1].plot(
        epochs,
        [row.get("eval_loss", np.nan) for row in evaluations],
        marker="o",
        color="#D97706",
    )
    axes[1].set_title("Validation loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Cross-entropy loss")
    axes[1].grid(alpha=0.25)
    fig.suptitle(f"{title} — training history")
    fig.tight_layout()
    return _save_figure(fig, output_path, dpi)


def visualize_results(
    runs_dir: Path = RUNS_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    methods: Sequence[str] | None = None,
    settings: Sequence[str] | None = None,
    seeds: Sequence[int] | None = None,
    split: str = "test",
    metric: str = "macro_f1",
    top_k_confusions: int = 15,
    dpi: int = 180,
) -> list[Path]:
    """Generate comparison and diagnostic figures for selected completed runs."""

    if split not in {"validation", "test"}:
        raise ValueError("split must be 'validation' or 'test'")
    if metric not in METRIC_NAMES:
        raise ValueError(f"metric must be one of {METRIC_NAMES}")
    if top_k_confusions < 1:
        raise ValueError("top_k_confusions must be positive")

    selected = select_runs(discover_runs(runs_dir), methods, settings, seeds)
    if not selected:
        raise ValueError("No completed runs match the requested filters")

    output_dir.mkdir(parents=True, exist_ok=True)
    generated: list[Path] = []
    scalar_metrics = metrics_frame(selected, split)
    scalar_metrics.to_csv(output_dir / f"{split}_metrics_summary.csv", index=False)
    generated.append(output_dir / f"{split}_metrics_summary.csv")
    generated.append(
        plot_metric_comparison(
            scalar_metrics,
            output_dir / f"{split}_{metric}_comparison.png",
            metric=metric,
            dpi=dpi,
        )
    )

    grouped: dict[tuple[str, int], list[RunResult]] = {}
    for run in selected:
        grouped.setdefault((run.setting, run.seed), []).append(run)
    for (setting, seed), group in grouped.items():
        output = plot_per_class_f1(
            group,
            split,
            output_dir / f"{split}_per_class_f1_{_safe_name(setting)}_seed{seed}.png",
            dpi=dpi,
        )
        if output is not None:
            generated.append(output)

    for run in selected:
        run_name = f"{_safe_name(run.method)}_{_safe_name(run.setting)}_seed{run.seed}"
        title = f"{run.method} / {run.setting} / seed {run.seed}"
        categories = list(run.metrics.get(split, {}).get("per_class", {}))

        confusion_path = run.path / "confusion_matrix.csv"
        if split == "test" and confusion_path.exists():
            matrix = load_confusion_matrix(confusion_path)
            generated.append(
                plot_confusion_matrix(
                    matrix,
                    output_dir / f"confusion_matrix_{run_name}.png",
                    title,
                    dpi=dpi,
                )
            )
            pairs = top_confusion_pairs(matrix, top_k=top_k_confusions)
            pairs_path = output_dir / f"top_confusions_{run_name}.csv"
            pairs.to_csv(pairs_path, index=False)
            generated.append(pairs_path)
            output = plot_top_confusions(
                pairs,
                output_dir / f"top_confusions_{run_name}.png",
                title,
                dpi=dpi,
            )
            if output is not None:
                generated.append(output)

        if split == "test" and categories:
            output = plot_prediction_distribution(
                run.path / "predictions.csv",
                categories,
                output_dir / f"prediction_distribution_{run_name}.png",
                title,
                dpi=dpi,
            )
            if output is not None:
                generated.append(output)

        output = plot_training_history(
            run.path / "training_history.json",
            output_dir / f"training_history_{run_name}.png",
            title,
            dpi=dpi,
        )
        if output is not None:
            generated.append(output)

    manifest = {
        "runs_dir": str(runs_dir.resolve()),
        "output_dir": str(output_dir.resolve()),
        "split": split,
        "metric": metric,
        "selected_runs": [
            {"method": run.method, "setting": run.setting, "seed": run.seed} for run in selected
        ],
        "generated_files": [path.name for path in generated],
    }
    manifest_path = output_dir / "visualization_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    generated.append(manifest_path)
    return generated


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--method", action="append", dest="methods", help="method filter; repeatable")
    parser.add_argument("--setting", action="append", dest="settings", help="setting filter; repeatable")
    parser.add_argument("--seed", action="append", dest="seeds", type=int, help="seed filter; repeatable")
    parser.add_argument("--split", choices=["validation", "test"], default="test")
    parser.add_argument("--metric", choices=METRIC_NAMES, default="macro_f1")
    parser.add_argument("--top-k-confusions", type=int, default=15)
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generated = visualize_results(
        runs_dir=args.runs_dir,
        output_dir=args.output_dir,
        methods=args.methods,
        settings=args.settings,
        seeds=args.seeds,
        split=args.split,
        metric=args.metric,
        top_k_confusions=args.top_k_confusions,
        dpi=args.dpi,
    )
    print(f"Generated {len(generated)} files in {args.output_dir}")
    for path in generated:
        print(path)


if __name__ == "__main__":
    main()
