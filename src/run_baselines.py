"""Baseline orchestration entry point.

Runs the three baselines (TF-IDF+SVM, BERT fine-tuning, SetFit) across the
full-data and few-shot (5/10/20-shot) settings, collects Macro-F1, writes a
JSON summary, and prints a results table.
"""

from __future__ import annotations

import argparse
import json

import pandas as pd

from config import FEW_SHOT_K, RESULT_DIR, ensure_dirs
from baselines import bert_ft, setfit_baseline, tfidf_svm

SETTINGS = ["full"] + [f"{k}shot" for k in FEW_SHOT_K]
BASELINES = {
    "tfidf_svm": tfidf_svm,
    "bert_ft": bert_ft,
    "setfit": setfit_baseline,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run Banking77 baselines")
    p.add_argument(
        "--baseline",
        choices=list(BASELINES) + ["all"],
        default="all",
        help="which baseline to run (default: all)",
    )
    p.add_argument(
        "--setting",
        choices=SETTINGS + ["all"],
        default="all",
        help="which data setting to run (default: all)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dirs()

    baseline_names = list(BASELINES) if args.baseline == "all" else [args.baseline]
    settings = SETTINGS if args.setting == "all" else [args.setting]

    # Load any previously saved results so partial runs can be resumed.
    results_path = RESULT_DIR / "results.json"
    results = {}
    if results_path.exists():
        with open(results_path, "r", encoding="utf-8") as f:
            results = json.load(f)

    for bname in baseline_names:
        module = BASELINES[bname]
        results.setdefault(bname, {})
        for setting in settings:
            if setting in results[bname]:
                print(f"[skip] {bname} / {setting}: already done "
                      f"(Macro-F1 = {results[bname][setting]:.4f})", flush=True)
                continue
            print(f"\n=== {bname} / {setting} ===", flush=True)
            metric = module.run(setting, verbose=False)["macro_f1"]
            results[bname][setting] = metric
            print(f"    Macro-F1 = {metric:.4f}", flush=True)
            # Persist incrementally so partial progress survives a crash.
            with open(results_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2)

    _print_table(results)
    print(f"\n[run] results saved to {results_path}")


def _print_table(results: dict) -> None:
    rows = []
    for bname, by_setting in results.items():
        row = {"baseline": bname}
        for s in SETTINGS:
            row[s] = by_setting.get(s)
        rows.append(row)
    df = pd.DataFrame(rows).set_index("baseline")
    df = df[SETTINGS]  # enforce column order
    print("\n" + "=" * 60)
    print("Macro-F1 (test set)")
    print("=" * 60)
    print(df.to_string(float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
