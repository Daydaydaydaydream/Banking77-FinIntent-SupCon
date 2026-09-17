"""Convenience entry point for the three external baselines.

Kept as a thin wrapper so the original command still works::

    python -m run_baselines --baseline all --setting all

It restricts the full matrix in :mod:`run_experiments` to TF-IDF+SVM,
BERT fine-tuning and SetFit. SupCon and its hard-negative variant live in
``run_experiments``, which additionally supports the seed dimension.
"""

from __future__ import annotations

import argparse

from config import SETTINGS
from run_experiments import (
    BASELINE_METHODS,
    print_summary,
    resolve,
    run_matrix,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run Banking77 baselines")
    p.add_argument(
        "--baseline",
        choices=BASELINE_METHODS + ["all"],
        default="all",
        help="which baseline to run (default: all)",
    )
    p.add_argument(
        "--setting",
        choices=SETTINGS + ["all"],
        default="all",
        help="which data setting to run (default: all)",
    )
    p.add_argument("--seed", type=int, nargs="+", default=None)
    p.add_argument("--dedup", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    methods, settings, seeds = resolve(args.baseline, args.setting, args.seed)
    methods = [m for m in methods if m in BASELINE_METHODS]

    run_matrix(
        methods,
        settings,
        seeds,
        verbose=args.verbose,
        dedup=args.dedup,
        overwrite=args.overwrite,
    )
    print_summary(methods, settings, seeds)


if __name__ == "__main__":
    main()
