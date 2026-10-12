"""Aggregate single-seed runs into mean +/- std result tables.

A single seed cannot separate a method effect from split or initialisation
noise.  Every table produced here therefore reports the mean and standard
deviation across seeds, the per-seed values, and a paired comparison against a
baseline method on the same seeds.

Usage::

    PYTHONPATH=src python -m analysis.aggregate --setting all --seeds 42 43 44
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from config import ROOT, RUNS_DIR, SETTINGS


MAIN_METHODS = ("tfidf_svm", "bert", "setfit", "supcon", "supcon_hn")
SETTING_ORDER = {setting: idx for idx, setting in enumerate(("5shot", "10shot", "20shot", "full"))}


def discover(runs_dir: Path = RUNS_DIR) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for metrics_path in sorted(runs_dir.glob("*/*/seed*/metrics.json")):
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        run_path = metrics_path.parent
        seed_text = run_path.name.replace("seed", "")
        try:
            seed = int(seed_text)
        except ValueError:
            continue
        health = payload.get("test_health", {})
        gradient = payload.get("gradient_health", {})
        rows.append(
            {
                "method": run_path.parent.parent.name,
                "setting": run_path.parent.name,
                "seed": seed,
                "validation_macro_f1": payload["validation"]["macro_f1"],
                "test_macro_f1": payload["test"]["macro_f1"],
                "test_accuracy": payload["test"]["accuracy"],
                "predicted_class_count": health.get("predicted_class_count"),
                "zero_f1_class_count": health.get("zero_f1_class_count"),
                "collapsed": payload.get("run_flagged_as_collapsed"),
                "non_finite_grad_events": gradient.get("non_finite_grad_norm_events"),
                "train_seconds": payload.get("train_seconds"),
                "path": str(metrics_path.parent.relative_to(runs_dir)),
            }
        )
    return rows


def summarise(rows: list[dict[str, Any]]) -> pd.DataFrame:
    """Mean +/- std of test Macro-F1 per (method, setting)."""

    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows)
    grouped = frame.groupby(["method", "setting"])
    summary = grouped.agg(
        seeds=("seed", "nunique"),
        test_macro_f1_mean=("test_macro_f1", "mean"),
        test_macro_f1_std=("test_macro_f1", "std"),
        validation_macro_f1_mean=("validation_macro_f1", "mean"),
        validation_macro_f1_std=("validation_macro_f1", "std"),
        test_accuracy_mean=("test_accuracy", "mean"),
        any_collapsed=("collapsed", "any"),
        train_seconds_mean=("train_seconds", "mean"),
    ).reset_index()
    # A single seed has no standard deviation; report it as missing rather
    # than 0.0, which would read as "perfectly stable".
    summary["test_macro_f1_std"] = summary["test_macro_f1_std"].round(4)
    summary["validation_macro_f1_std"] = summary["validation_macro_f1_std"].round(4)
    summary["test_macro_f1_mean"] = summary["test_macro_f1_mean"].round(4)
    summary["validation_macro_f1_mean"] = summary["validation_macro_f1_mean"].round(4)
    summary["test_accuracy_mean"] = summary["test_accuracy_mean"].round(4)
    summary["setting_order"] = summary["setting"].map(SETTING_ORDER).fillna(99)
    summary = summary.sort_values(["method", "setting_order"]).drop(columns="setting_order")
    return summary.reset_index(drop=True)


def pivot_table(summary: pd.DataFrame) -> pd.DataFrame:
    """method x setting matrix of ``mean +/- std`` strings."""

    if summary.empty:
        return summary
    cells: dict[str, dict[str, str]] = {}
    for row in summary.itertuples(index=False):
        mean = getattr(row, "test_macro_f1_mean")
        std = getattr(row, "test_macro_f1_std")
        n = getattr(row, "seeds")
        label = f"{mean:.4f}" + (f" ± {std:.4f}" if n > 1 else " (1 seed)")
        if getattr(row, "any_collapsed"):
            label += " ⚠"
        cells.setdefault(getattr(row, "method"), {})[getattr(row, "setting")] = label
    table = pd.DataFrame(cells).T
    ordered_settings = [s for s in ("5shot", "10shot", "20shot", "full") if s in table.columns]
    return table[ordered_settings]


def paired_comparison(
    rows: list[dict[str, Any]],
    baseline: str,
    treatment: str,
) -> list[dict[str, Any]]:
    """Per-seed paired deltas between two methods on the same split."""

    frame = pd.DataFrame(rows)
    if frame.empty:
        return []
    subset = frame[frame["method"].isin([baseline, treatment])]
    results: list[dict[str, Any]] = []
    for (setting, seed), group in subset.groupby(["setting", "seed"]):
        by_method = group.set_index("method")
        if baseline not in by_method.index or treatment not in by_method.index:
            continue
        base = by_method.loc[baseline]
        treat = by_method.loc[treatment]
        results.append(
            {
                "setting": setting,
                "seed": seed,
                "baseline_test_macro_f1": base["test_macro_f1"],
                "treatment_test_macro_f1": treat["test_macro_f1"],
                "delta": treat["test_macro_f1"] - base["test_macro_f1"],
            }
        )
    if not results:
        return []
    detail = pd.DataFrame(results)
    summary = (
        detail.groupby("setting")["delta"]
        .agg(["count", "mean", "std", "min", "max"])
        .reset_index()
        .rename(
            columns={
                "count": "paired_seeds",
                "mean": "delta_mean",
                "std": "delta_std",
                "min": "delta_min",
                "max": "delta_max",
            }
        )
    )
    summary["baseline"] = baseline
    summary["treatment"] = treatment
    summary["consistent_gain"] = summary["delta_min"] > 0
    summary["delta_mean"] = summary["delta_mean"].round(4)
    summary["delta_std"] = summary["delta_std"].round(4)
    return summary.astype(object).where(pd.notna(summary), None).to_dict("records")


def seed_consistency(rows: list[dict[str, Any]]) -> pd.DataFrame:
    """Spread of test Macro-F1 across seeds for each (method, setting)."""

    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows)
    out = (
        frame.groupby(["method", "setting"])["test_macro_f1"]
        .agg(["count", "min", "max", "std"])
        .reset_index()
        .rename(columns={"count": "seeds", "min": "min_f1", "max": "max_f1", "std": "std_f1"})
    )
    out["range"] = (out["max_f1"] - out["min_f1"]).round(4)
    out["std_f1"] = out["std_f1"].round(4)
    out["min_f1"] = out["min_f1"].round(4)
    out["max_f1"] = out["max_f1"].round(4)
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--setting", choices=[*SETTINGS, "all"], default="all")
    parser.add_argument("--method", nargs="+", default=list(MAIN_METHODS))
    parser.add_argument("--baseline", default="bert")
    parser.add_argument("--treatment", default="supcon")
    parser.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "results")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    rows = discover(args.runs_dir)
    if args.setting != "all":
        rows = [r for r in rows if r["setting"] == args.setting]
    if args.method:
        rows = [r for r in rows if r["method"] in args.method]

    if not rows:
        print(f"No completed runs under {args.runs_dir}")
        return

    summary = summarise(rows)
    table = pivot_table(summary)
    consistency = seed_consistency(rows)

    # Record every ordered pair among the present methods, not just the
    # requested one. The question the ablation has to answer is where the gain
    # comes from, so supcon_hn - supcon matters as much as supcon_hn - bert.
    present_methods = [m for m in MAIN_METHODS if (summary.method == m).any()]
    pairs: list[dict[str, Any]] = []
    for treatment in present_methods:
        for baseline in present_methods:
            if treatment == baseline:
                continue
            pairs.extend(paired_comparison(rows, baseline, treatment))
    requested = [p for p in pairs if {p["baseline"], p["treatment"]} == {args.baseline, args.treatment}]
    comparisons = requested or pairs

    summary.to_csv(args.output / "macro_f1_summary.csv", index=False)
    consistency.to_csv(args.output / "seed_consistency.csv", index=False)
    (args.output / "runs_index.json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8"
    )
    (args.output / "paired_comparisons.json").write_text(
        json.dumps(pairs, indent=2), encoding="utf-8"
    )

    print("\n=== Test Macro-F1 (mean over seeds) ===")
    print(table.to_string())
    print("\n=== Seed spread ===")
    print(consistency.to_string(index=False))
    if comparisons:
        label = f"{args.treatment} - {args.baseline}" if requested else "all pairs"
        print(f"\n=== Paired deltas ({label}) ===")
        print(pd.DataFrame(comparisons).to_string(index=False))
    collapsed = [r for r in rows if r.get("collapsed")]
    if collapsed:
        print(f"\n! {len(collapsed)} run(s) flagged as collapsed and are not trustworthy:")
        for row in collapsed:
            print(f"    {row['method']}/{row['setting']}/seed{row['seed']}")


if __name__ == "__main__":
    main()