"""Predefined analysis of the semantically closest intent pairs.

The category pairs are declared here as research hypotheses, before any model
is trained, and are derived from the label names and the business grouping of
BANKING77 - never from a test confusion matrix.  Using test-set confusions to
*choose* the pairs would leak the evaluation set into the analysis, and using
them to *tune* anything would invalidate the ablation.

The module reports, for each declared pair:
  * both directions of confusion (a->b and b->a), counted separately;
  * per-class F1 for both members;
  * whether the pair improved between two methods.

Usage::

    PYTHONPATH=src python -m analysis.confusion_pairs --methods bert supcon
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from config import ROOT, RUNS_DIR, SETTINGS


# Declared up front from label semantics.  Each group is a set of intents that
# share a business object and differ only in state or user goal.
CONFUSION_GROUPS: dict[str, tuple[str, ...]] = {
    "identity_verification": (
        "verify_my_identity",
        "why_verify_identity",
        "verify_source_of_funds",
        "unable_to_verify_identity",
    ),
    "top_up_state": (
        "pending_top_up",
        "top_up_failed",
        "top_up_reverted",
        "top_up_by_bank_transfer_charge",
    ),
    "card_delivery": (
        "card_arrival",
        "order_physical_card",
        "card_delivery_estimate",
        "card_not_working",
    ),
    "declined_transaction": (
        "declined_card_payment",
        "declined_transfer",
        "declined_cash_withdrawal",
    ),
    "balance_not_updated": (
        "balance_not_updated_after_bank_transfer",
        "balance_not_updated_after_cheque_or_cash_deposit",
    ),
    "exchange_rate": (
        "card_payment_wrong_exchange_rate",
        "wrong_exchange_rate_for_cash_withdrawal",
        "exchange_rate",
    ),
    "refund": (
        "request_refund",
        "Refund_not_showing_up",
    ),
    "virtual_card": (
        "get_disposable_virtual_card",
        "disposable_card_limits",
        "virtual_card_not_working",
    ),
}


def declared_pairs() -> list[tuple[str, str, str]]:
    """Expand the groups into ``(group, label_a, label_b)`` triples."""

    pairs: list[tuple[str, str, str]] = []
    for group, members in CONFUSION_GROUPS.items():
        ordered = sorted(members)
        for i in range(len(ordered)):
            for j in range(i + 1, len(ordered)):
                pairs.append((group, ordered[i], ordered[j]))
    return pairs


def load_confusion(method: str, setting: str, seed: int, runs_dir: Path = RUNS_DIR) -> tuple[pd.DataFrame, list[str]] | None:
    path = runs_dir / method / setting / f"seed{seed}" / "confusion_matrix.csv"
    if not path.exists():
        return None
    frame = pd.read_csv(path, index_col=0)
    return frame, [str(c) for c in frame.columns]


def load_per_class(method: str, setting: str, seed: int, runs_dir: Path = RUNS_DIR) -> dict[str, Any] | None:
    path = runs_dir / method / setting / f"seed{seed}" / "metrics.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def pair_confusions(
    frame: pd.DataFrame,
    pairs: Sequence[tuple[str, str, str]],
) -> list[dict[str, Any]]:
    """Count both directions of every declared pair."""

    labels = [str(c) for c in frame.columns]
    index = {label: pos for pos, label in enumerate(frame.index.astype(str))}
    rows: list[dict[str, Any]] = []
    for group, label_a, label_b in pairs:
        if label_a not in index or label_b not in index or label_a not in labels or label_b not in labels:
            continue
        i, j = index[label_a], labels.index(label_b)
        i2, j2 = index[label_b], labels.index(label_a)
        a_to_b = int(frame.iloc[i, j])
        b_to_a = int(frame.iloc[i2, j2])
        rows.append(
            {
                "group": group,
                "label_a": label_a,
                "label_b": label_b,
                f"{label_a}->{label_b}": a_to_b,
                f"{label_b}->{label_a}": b_to_a,
                "bidirectional_total": a_to_b + b_to_a,
            }
        )
    return rows


def group_summary(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Roll pair counts up to one row per declared group."""

    buckets: dict[str, dict[str, Any]] = {}
    for row in rows:
        entry = buckets.setdefault(
            row["group"], {"group": row["group"], "pairs": 0, "bidirectional_total": 0}
        )
        entry["pairs"] += 1
        entry["bidirectional_total"] += row["bidirectional_total"]
    return sorted(buckets.values(), key=lambda item: -item["bidirectional_total"])


def compare(
    method_a: str,
    method_b: str,
    setting: str,
    seed: int,
    runs_dir: Path = RUNS_DIR,
) -> dict[str, Any]:
    """Compare bidirectional confusion between two methods on the same split."""

    conf_a = load_confusion(method_a, setting, seed, runs_dir)
    conf_b = load_confusion(method_b, setting, seed, runs_dir)
    if conf_a is None or conf_b is None:
        return {"error": f"missing confusion matrix for {method_a}/{setting}/seed{seed}"}
    pairs = declared_pairs()
    rows_a = {(r["label_a"], r["label_b"]): r for r in pair_confusions(conf_a[0], pairs)}
    rows_b = {(r["label_a"], r["label_b"]): r for r in pair_confusions(conf_b[0], pairs)}

    comparisons: list[dict[str, Any]] = []
    for key, row_a in rows_a.items():
        row_b = rows_b.get(key)
        if row_b is None:
            continue
        total_a = row_a["bidirectional_total"]
        total_b = row_b["bidirectional_total"]
        comparisons.append(
            {
                "group": row_a["group"],
                "label_a": key[0],
                "label_b": key[1],
                f"{method_a}_total": total_a,
                f"{method_b}_total": total_b,
                "delta": total_b - total_a,
                "improved": total_b < total_a,
            }
        )
    comparisons.sort(key=lambda item: -abs(item["delta"]))

    per_class_a = load_per_class(method_a, setting, seed, runs_dir)
    per_class_b = load_per_class(method_b, setting, seed, runs_dir)
    f1_deltas: list[dict[str, Any]] = []
    if per_class_a and per_class_b:
        classes = per_class_a["test"]["per_class"]
        classes_b = per_class_b["test"]["per_class"]
        for label in sorted(set(classes) & set(classes_b)):
            f1_a = classes[label]["f1"]
            f1_b = classes_b[label]["f1"]
            f1_deltas.append({"label": label, "f1_a": f1_a, "f1_b": f1_b, "delta": f1_b - f1_a})
        f1_deltas.sort(key=lambda item: item["delta"])

    return {
        "method_a": method_a,
        "method_b": method_b,
        "setting": setting,
        "seed": seed,
        "pairs": comparisons,
        "groups": group_summary(
            [
                {
                    "group": c["group"],
                    "pairs": 1,
                    "bidirectional_total": c[f"{method_a}_total"],
                }
                for c in comparisons
            ]
        ),
        "per_class_worst": f1_deltas[:10],
        "per_class_best": f1_deltas[-10:][::-1],
        "improved_pair_count": sum(1 for c in comparisons if c["improved"]),
        "total_pair_count": len(comparisons),
        "macro_f1_a": per_class_a["test"]["macro_f1"] if per_class_a else None,
        "macro_f1_b": per_class_b["test"]["macro_f1"] if per_class_b else None,
    }


def group_aggregate_over_seeds(
    baseline: str,
    treatment: str,
    setting: str,
    seeds: Sequence[int],
    runs_dir: Path = RUNS_DIR,
) -> dict[str, Any]:
    """Aggregate pair-level changes across seeds.

    A single seed can make one pair look fixed while the error simply moves to
    a pair that was not inspected.  Aggregating per group over every seed, and
    counting how many declared pairs improve in *every* seed, separates a real
    effect from a reshuffle.
    """

    pair_flags: dict[tuple[str, str], list[bool]] = {}
    group_baseline: dict[str, list[int]] = {}
    group_treatment: dict[str, list[int]] = {}
    compared_seeds: list[int] = []

    for seed in seeds:
        conf_base = load_confusion(baseline, setting, seed, runs_dir)
        conf_treat = load_confusion(treatment, setting, seed, runs_dir)
        if conf_base is None or conf_treat is None:
            continue
        compared_seeds.append(seed)

        pairs = declared_pairs()
        rows_base = pair_confusions(conf_base[0], pairs)
        rows_treat = pair_confusions(conf_treat[0], pairs)

        # Sum per group within this seed first, so each group contributes one
        # number per seed rather than one per declared pair.
        base_by_group: dict[str, int] = {}
        treat_by_group: dict[str, int] = {}
        for row_base, row_treat in zip(rows_base, rows_treat):
            pair_flags.setdefault((row_base["label_a"], row_base["label_b"]), []).append(
                row_treat["bidirectional_total"] < row_base["bidirectional_total"]
            )
            base_by_group[row_base["group"]] = (
                base_by_group.get(row_base["group"], 0) + row_base["bidirectional_total"]
            )
            treat_by_group[row_base["group"]] = (
                treat_by_group.get(row_base["group"], 0) + row_treat["bidirectional_total"]
            )
        for group, total in base_by_group.items():
            group_baseline.setdefault(group, []).append(total)
            group_treatment.setdefault(group, []).append(treat_by_group.get(group, 0))

    if not compared_seeds:
        return {"error": f"no comparable runs for {baseline}/{treatment}/{setting}"}

    group_table = []
    for group in CONFUSION_GROUPS:
        base_totals = group_baseline.get(group)
        treat_totals = group_treatment.get(group)
        if not base_totals or not treat_totals:
            continue
        base_mean = sum(base_totals) / len(base_totals)
        treat_mean = sum(treat_totals) / len(treat_totals)
        group_table.append(
            {
                "group": group,
                "baseline_mean": base_mean,
                "treatment_mean": treat_mean,
                "delta": treat_mean - base_mean,
                "improved": treat_mean < base_mean,
                "per_seed_delta": [t - b for b, t in zip(base_totals, treat_totals)],
            }
        )
    group_table.sort(key=lambda item: item["delta"])

    consistently_improved = [k for k, flags in pair_flags.items() if all(flags)]
    consistently_worsened = [k for k, flags in pair_flags.items() if not any(flags)]
    improved = sum(1 for item in group_table if item["improved"])

    return {
        "baseline": baseline,
        "treatment": treatment,
        "setting": setting,
        "seeds": compared_seeds,
        "groups": group_table,
        "groups_improved": improved,
        "groups_total": len(group_table),
        "pairs_compared": len(pair_flags),
        "pairs_consistently_improved": len(consistently_improved),
        "pairs_consistently_worsened": len(consistently_worsened),
        "consistently_improved_labels": [f"{a} <-> {b}" for a, b in consistently_improved],
        "consistently_worsened_labels": [f"{a} <-> {b}" for a, b in consistently_worsened],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--methods", nargs="+", default=["bert", "supcon", "supcon_hn"])
    parser.add_argument("--setting", default="5shot")
    parser.add_argument("--seed", type=int, nargs="+", default=[42])
    parser.add_argument(
        "--compare",
        nargs=2,
        metavar=("BASELINE", "TREATMENT"),
        default=["bert", "supcon"],
        help="baseline and treatment method to compare",
    )
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "analysis")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    for method in args.methods:
        for seed in args.seed:
            conf = load_confusion(method, args.setting, seed)
            if conf is None:
                print(f"[skip] {method}/{args.setting}/seed{seed}: no confusion matrix")
                continue
            rows = pair_confusions(conf[0], declared_pairs())
            payload = {
                "method": method,
                "setting": args.setting,
                "seed": seed,
                "selection_note": (
                    "Pairs are declared from label semantics before training; no test "
                    "confusion matrix was used to choose them."
                ),
                "pairs": rows,
                "groups": group_summary(rows),
            }
            path = args.output / f"confusion_pairs_{method}_{args.setting}_seed{seed}.json"
            path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            print(f"\n=== {method} / {args.setting} / seed{seed} ===")
            for entry in payload["groups"]:
                print(f"  {entry['group']:28s} bidirectional errors {entry['bidirectional_total']:4d}")

    baseline, treatment = args.compare
    aggregate = group_aggregate_over_seeds(baseline, treatment, args.setting, args.seed)
    if "error" not in aggregate:
        name = f"confusion_groups_{baseline}_vs_{treatment}_{args.setting}.json"
        (args.output / name).write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
        print(f"\n=== {baseline} -> {treatment} | {args.setting} | seeds {aggregate['seeds']} ===")
        print(f"  {'group':28s} {'baseline':>9s} {'treatment':>10s} {'delta':>7s}")
        for entry in aggregate["groups"]:
            mark = "improved" if entry["improved"] else "worsened"
            print(
                f"  {entry['group']:28s} {entry['baseline_mean']:9.1f} "
                f"{entry['treatment_mean']:10.1f} {entry['delta']:+7.1f}  {mark}"
            )
        print(
            f"  groups improved: {aggregate['groups_improved']}/{aggregate['groups_total']}\n"
            f"  pairs improved in every seed: {aggregate['pairs_consistently_improved']}"
            f"/{aggregate['pairs_compared']}\n"
            f"  pairs worsened in every seed: {aggregate['pairs_consistently_worsened']}"
            f"/{aggregate['pairs_compared']}"
        )
        if aggregate["consistently_improved_labels"]:
            print("  consistently improved:")
            for label in aggregate["consistently_improved_labels"]:
                print(f"    {label}")
        if aggregate["consistently_worsened_labels"]:
            print("  consistently worsened:")
            for label in aggregate["consistently_worsened_labels"]:
                print(f"    {label}")


if __name__ == "__main__":
    main()