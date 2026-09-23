"""Run BANKING77 baseline experiments with a shared artifact format."""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from typing import Any

from baselines import bert_ft, setfit_baseline, tfidf_svm
from config import DEFAULT_SEED, SETTINGS, run_dir


RUNNERS: dict[str, Callable[..., dict[str, Any]]] = {
    "tfidf_svm": tfidf_svm.run,
    "bert": bert_ft.run,
    "setfit": setfit_baseline.run,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=[*RUNNERS, "all"], default="all")
    parser.add_argument("--setting", choices=[*SETTINGS, "all"], default="full")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--overwrite", action="store_true", help="rerun even if metrics.json exists")
    parser.add_argument("--no-save-model", action="store_true", help="do not retain fitted model files")
    parser.add_argument("--local-files-only", action="store_true", help="never download Hugging Face models")
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--epochs", type=float, default=None, help="override deep-model epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="override deep-model batch size")
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--min-steps", type=int, default=None, help="BERT minimum optimizer steps")
    parser.add_argument("--max-steps", type=int, default=None, help="BERT maximum optimizer steps")
    parser.add_argument("--eval-steps", type=int, default=None, help="BERT validation interval in steps")
    parser.add_argument("--early-stopping-patience", type=int, default=None)
    parser.add_argument("--early-stopping-threshold", type=float, default=None)
    parser.add_argument("--num-iterations", type=int, default=None, help="SetFit contrastive iterations")
    parser.add_argument("--svm-c", type=float, default=1.0)
    return parser.parse_args()


def kwargs_for(method: str, args: argparse.Namespace) -> dict[str, Any]:
    common = {"save_model": not args.no_save_model}
    if method == "tfidf_svm":
        return {**common, "c": args.svm_c}
    if method == "bert":
        values: dict[str, Any] = {
            **common,
            "device": args.device,
            "local_files_only": args.local_files_only,
            "max_length": args.max_length,
        }
        if args.epochs is not None:
            values["epochs"] = args.epochs
        if args.batch_size is not None:
            values["batch_size"] = args.batch_size
        if args.learning_rate is not None:
            values["learning_rate"] = args.learning_rate
        if args.min_steps is not None:
            values["min_steps"] = args.min_steps
        if args.max_steps is not None:
            values["max_steps"] = args.max_steps
        if args.eval_steps is not None:
            values["eval_steps"] = args.eval_steps
        if args.early_stopping_patience is not None:
            values["patience"] = args.early_stopping_patience
        if args.early_stopping_threshold is not None:
            values["early_stopping_threshold"] = args.early_stopping_threshold
        return values
    values = {
        **common,
        "device": args.device,
        "local_files_only": args.local_files_only,
        "max_length": args.max_length,
        "num_iterations": args.num_iterations,
    }
    if args.epochs is not None:
        values["epochs"] = int(args.epochs)
    if args.batch_size is not None:
        values["batch_size"] = args.batch_size
    if args.learning_rate is not None:
        values["learning_rate"] = args.learning_rate
    return values


def main() -> None:
    args = parse_args()
    methods = list(RUNNERS) if args.method == "all" else [args.method]
    settings = list(SETTINGS) if args.setting == "all" else [args.setting]
    results = []
    for method in methods:
        for setting in settings:
            metrics_path = run_dir(method, setting, args.seed) / "metrics.json"
            if metrics_path.exists() and not args.overwrite:
                print(f"skip {method}/{setting}/seed{args.seed}: {metrics_path} exists")
                continue
            print(f"run {method}/{setting}/seed{args.seed}")
            result = RUNNERS[method](setting, args.seed, **kwargs_for(method, args))
            results.append(result)
            print(json.dumps(result, ensure_ascii=False, indent=2))
    if not results:
        print("No experiments ran. Use --overwrite to replace completed runs.")


if __name__ == "__main__":
    main()
