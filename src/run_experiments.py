"""Experiment matrix orchestration (methods x settings x seeds).

Runs every arm through the shared :class:`RunArtifacts` contract:

1. ``tfidf_svm``    — TF-IDF + LinearSVC (classical lower bound).
2. ``bert_ft``      — BERT fine-tuned end to end, checkpoint picked on validation.
3. ``setfit``       — native SetFit (contrastive body + logistic-regression head).
4. ``supcon``       — BERT + supervised contrastive pre-training, frozen head.
5. ``supcon_hn``    — as above with similarity-weighted hard negatives.

Each (method, setting, seed) writes ``outputs/runs/...``. Completed runs are
skipped unless ``--overwrite`` is passed, so a long sweep can be resumed.
The summary table aggregates Macro-F1 across seeds as ``mean ± std``.
"""

from __future__ import annotations

import argparse
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

from baselines import bert_ft, setfit_baseline, tfidf_svm
from config import RESULT_DIR, SEEDS, SETTINGS, ensure_dirs, run_dir
from supcon import train as supcon_train

# method name -> (display label, callable run(setting, seed=..., verbose=..., dedup=...))
METHODS: Dict[str, tuple] = {
    "tfidf_svm": ("TF-IDF + LinearSVC", lambda s, **kw: tfidf_svm.run(s, **kw)),
    "bert_ft": ("BERT 微调", lambda s, **kw: bert_ft.run(s, **kw)),
    "setfit": ("SetFit", lambda s, **kw: setfit_baseline.run(s, **kw)),
    "supcon": (
        "SupCon",
        lambda s, **kw: supcon_train.run(s, hard_negative_weighting=False, **kw),
    ),
    "supcon_hn": (
        "SupCon + 难负例",
        lambda s, **kw: supcon_train.run(s, hard_negative_weighting=True, **kw),
    ),
}

BASELINE_METHODS = ["tfidf_svm", "bert_ft", "setfit"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run the Banking77 experiment matrix")
    p.add_argument("--method", choices=list(METHODS) + ["all", "baselines"], default="all")
    p.add_argument("--setting", choices=SETTINGS + ["all"], default="all")
    p.add_argument(
        "--seed",
        type=int,
        nargs="+",
        default=None,
        help=f"seeds to run (default: {SEEDS})",
    )
    p.add_argument(
        "--dedup",
        action="store_true",
        help="use the de-duplicated dataset variant",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="re-run experiments even if metrics.json already exists",
    )
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def resolve(
    method_arg: str, setting_arg: str, seed_arg: Sequence[int] | None
) -> tuple[List[str], List[str], List[int]]:
    if method_arg == "all":
        methods = list(METHODS)
    elif method_arg == "baselines":
        methods = list(BASELINE_METHODS)
    else:
        methods = [method_arg]

    settings = SETTINGS if setting_arg == "all" else [setting_arg]
    seeds = list(seed_arg) if seed_arg else list(SEEDS)
    return methods, settings, seeds


def run_matrix(
    methods: Sequence[str],
    settings: Sequence[str],
    seeds: Sequence[int],
    verbose: bool = False,
    dedup: bool = False,
    overwrite: bool = False,
) -> None:
    """Run the cartesian product, skipping completed cells by default."""
    ensure_dirs()
    for method in methods:
        label, fn = METHODS[method]
        for setting in settings:
            for seed in seeds:
                target = run_dir(method, setting, seed)
                if not overwrite and (target / "metrics.json").exists():
                    print(
                        f"[skip] {method}/{setting}/seed{seed}: already done",
                        flush=True,
                    )
                    continue
                print(
                    f"\n=== {label} | {setting} | seed{seed} ===",
                    flush=True,
                )
                outcome = fn(setting, seed=seed, verbose=verbose, dedup=dedup)
                print(
                    f"    Macro-F1 = {outcome['macro_f1']:.4f}"
                    + (
                        f" (val {outcome['val_macro_f1']:.4f})"
                        if outcome.get("val_macro_f1") is not None
                        else ""
                    ),
                    flush=True,
                )


def _cell(method: str, setting: str, seeds: Sequence[int]) -> str:
    scores, vals = [], []
    for seed in seeds:
        path = run_dir(method, setting, seed) / "metrics.json"
        if not path.exists():
            continue
        import json

        with open(path, "r", encoding="utf-8") as f:
            block = json.load(f)
        scores.append(block["macro_f1"])
        if block.get("val_macro_f1") is not None:
            vals.append(block["val_macro_f1"])
    if not scores:
        return "—"
    mean = float(np.mean(scores))
    text = f"{mean:.4f}" if len(scores) == 1 else f"{mean:.4f} ± {np.std(scores):.4f}"
    if vals:
        text += f" (val {np.mean(vals):.4f})"
    return text


def print_summary(
    methods: Sequence[str], settings: Sequence[str], seeds: Sequence[int]
) -> None:
    rows = []
    for method in methods:
        label = METHODS[method][0]
        rows.append(
            {"method": label, **{s: _cell(method, s, seeds) for s in settings}}
        )
    df = pd.DataFrame(rows).set_index("method")
    df = df[list(settings)]
    print("\n" + "=" * 72)
    print(f"Test Macro-F1 (mean over seeds {list(seeds)}; val = validation Macro-F1)")
    print("=" * 72)
    print(df.to_string())


def main() -> None:
    args = parse_args()
    methods, settings, seeds = resolve(args.method, args.setting, args.seed)
    run_matrix(
        methods,
        settings,
        seeds,
        verbose=args.verbose,
        dedup=args.dedup,
        overwrite=args.overwrite,
    )
    print_summary(methods, settings, seeds)
    print(f"\n[run] artefacts under {RESULT_DIR.parent / 'runs'}")


if __name__ == "__main__":
    main()
