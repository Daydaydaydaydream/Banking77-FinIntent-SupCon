"""Ablation experiment orchestration.

Runs the three ablation arms across the full-data and few-shot settings and
collects Macro-F1 plus per-class F1:

1. ``baseline_bert`` — plain BERT fine-tuning (Baseline).
2. ``supcon``        — BERT + supervised contrastive pre-training (Baseline + SupCon).
3. ``supcon_hn``     — BERT + SupCon with hard-negative weighting (Baseline + SupCon + HN).

Results are persisted incrementally to ``outputs/results/ablation_results.json``
so partial runs can be resumed.
"""

from __future__ import annotations

import argparse
import json

import pandas as pd

from config import FEW_SHOT_K, RESULT_DIR, ensure_dirs
from baselines import bert_ft
from supcon import train as supcon_train

SETTINGS = ["full"] + [f"{k}shot" for k in FEW_SHOT_K]

# method name -> (display label, callable run(split_name, verbose=...))
METHODS = {
    "baseline_bert": ("BERT 微调", lambda s, **kw: bert_ft.run(s, **kw)),
    "supcon": (
        "SupCon",
        lambda s, **kw: supcon_train.run(s, hard_negative_weighting=False, **kw),
    ),
    "supcon_hn": (
        "SupCon+难负例",
        lambda s, **kw: supcon_train.run(s, hard_negative_weighting=True, **kw),
    ),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run Banking77 ablation experiments")
    p.add_argument("--method", choices=list(METHODS) + ["all"], default="all")
    p.add_argument("--setting", choices=SETTINGS + ["all"], default="all")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dirs()

    method_names = list(METHODS) if args.method == "all" else [args.method]
    settings = SETTINGS if args.setting == "all" else [args.setting]

    results_path = RESULT_DIR / "ablation_results.json"
    results = {}
    if results_path.exists():
        with open(results_path, "r", encoding="utf-8") as f:
            results = json.load(f)

    for setting in settings:
        results.setdefault(setting, {})
        for mname in method_names:
            if mname in results[setting]:
                print(
                    f"[skip] {setting} / {mname}: already done "
                    f"(Macro-F1 = {results[setting][mname]['macro_f1']:.4f})",
                    flush=True,
                )
                continue
            print(f"\n=== {setting} / {mname} ===", flush=True)
            outcome = METHODS[mname][1](setting, verbose=False)
            results[setting][mname] = {
                "macro_f1": outcome["macro_f1"],
                "per_class_f1": outcome.get("per_class_f1", {}),
            }
            print(f"    Macro-F1 = {outcome['macro_f1']:.4f}", flush=True)
            with open(results_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2)

    _print_table(results)
    print(f"\n[run] results saved to {results_path}")


def _print_table(results: dict) -> None:
    rows = []
    for mname, (label, _) in METHODS.items():
        row = {"method": label}
        for s in SETTINGS:
            row[s] = results.get(s, {}).get(mname, {}).get("macro_f1")
        rows.append(row)
    df = pd.DataFrame(rows).set_index("method")
    df = df[SETTINGS]
    print("\n" + "=" * 60)
    print("Ablation Macro-F1 (test set)")
    print("=" * 60)
    print(df.to_string(float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
