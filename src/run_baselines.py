"""Uniform experiment entry point with failure capture and multi-seed support."""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from baselines import bert_ft, setfit_baseline, tfidf_svm
from supcon import train as supcon
from config import DEFAULT_SEED, SETTINGS, run_dir
from utils.artifacts import RunArtifacts


RUNNERS: dict[str, Callable[..., dict[str, Any]]] = {
    "tfidf_svm": tfidf_svm.run,
    "bert": bert_ft.run,
    "setfit": setfit_baseline.run,
    "supcon": supcon.run,
    "supcon_hn": supcon.run_hard_negative,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=[*RUNNERS, "all"], default="all")
    parser.add_argument("--setting", choices=[*SETTINGS, "all"], default="full")
    parser.add_argument(
        "--seed",
        type=int,
        nargs="+",
        default=[DEFAULT_SEED],
        help="one or more seeds; results are written per (method, setting, seed)",
    )
    parser.add_argument("--overwrite", action="store_true", help="rerun even if metrics.json exists")
    parser.add_argument("--no-save-model", action="store_true", help="do not retain fitted model files")
    parser.add_argument("--keep-checkpoints", action="store_true", help="retain intermediate trainer checkpoints")
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
    parser.add_argument("--precision", choices=["auto", "fp32", "bf16", "fp16"], default="auto")
    parser.add_argument("--freeze-layers", type=int, default=0, help="freeze embeddings and N lowest BERT layers")
    parser.add_argument("--supcon-temperature", type=float, default=None)
    parser.add_argument("--supcon-epochs", type=int, default=None)
    parser.add_argument("--hn-temperature", type=float, default=None)
    parser.add_argument("--config", type=str, default=None, help="frozen JSON config file (overrides defaults)")
    return parser.parse_args()


def kwargs_for(method: str, args: argparse.Namespace) -> dict[str, Any]:
    common = {
        "save_model": not args.no_save_model,
        "keep_checkpoints": args.keep_checkpoints,
    }
    if method == "tfidf_svm":
        return {**common, "c": args.svm_c}
    if method in ("bert", "supcon", "supcon_hn"):
        values: dict[str, Any] = {
            **common,
            "device": args.device,
            "local_files_only": args.local_files_only,
            "max_length": args.max_length,
            "precision": args.precision,
            "freeze_layers": args.freeze_layers,
        }
        if method != "bert":
            values["supcon_temperature"] = args.supcon_temperature
            values["supcon_epochs"] = args.supcon_epochs
            if method == "supcon_hn":
                values["hn_temperature"] = args.hn_temperature
        else:
            for key, value in (
                ("epochs", args.epochs),
                ("batch_size", args.batch_size),
                ("learning_rate", args.learning_rate),
                ("min_steps", args.min_steps),
                ("max_steps", args.max_steps),
                ("eval_steps", args.eval_steps),
                ("patience", args.early_stopping_patience),
                ("early_stopping_threshold", args.early_stopping_threshold),
            ):
                if value is not None:
                    values[key] = value
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


# Fields the sweep records for provenance. They document *why* a value was
# chosen and must not be forwarded to a runner.
PROVENANCE_KEYS = frozenset({"selected_on", "selected_candidate"})


def _read_frozen(path: str | None) -> dict[str, Any]:
    """Load and validate the frozen config file."""

    if path is None:
        return {}
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(
            f"Frozen config not found: {file_path}. "
            "Run: PYTHONPATH=src python -m analysis.tune_bert --setting all --freeze"
        )
    payload = json.loads(file_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Frozen config must be a JSON object: {file_path}")
    return payload


def _frozen_config(path: str | None, method: str, setting: str) -> dict[str, Any]:
    """Load the frozen hyperparameters for one (method, setting).

    The file produced by ``analysis.tune_bert`` is keyed by setting, because
    the best learning rate for 5-shot is not necessarily the best for full.
    Applying one set of values everywhere would silently ignore that.
    Provenance fields are dropped here so the run records them separately.
    """

    method_section = _read_frozen(path).get(method, {})
    if not isinstance(method_section, dict):
        return {}
    per_setting = method_section.get(setting)
    section = per_setting if isinstance(per_setting, dict) else {
        key: value for key, value in method_section.items() if not isinstance(value, dict)
    }
    return {key: value for key, value in section.items() if key not in PROVENANCE_KEYS}


def _provenance(path: str | None, method: str, setting: str) -> dict[str, Any]:
    """Read only the audit fields from the frozen config, for run records."""

    if path is None:
        return {}
    method_section = _read_frozen(path).get(method, {})
    if not isinstance(method_section, dict):
        return {}
    per_setting = method_section.get(setting)
    section = per_setting if isinstance(per_setting, dict) else method_section
    return {key: section[key] for key in PROVENANCE_KEYS if key in section}


def main() -> None:
    args = parse_args()
    methods = list(RUNNERS) if args.method == "all" else [args.method]
    settings = list(SETTINGS) if args.setting == "all" else [args.setting]

    results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    for method in methods:
        for setting in settings:
            overrides = _frozen_config(args.config, method, setting)
            provenance = _provenance(args.config, method, setting)
            if overrides:
                print(f"frozen config for {method}/{setting}: {overrides}")
            for seed in args.seed:
                metrics_path = run_dir(method, setting, seed) / "metrics.json"
                if metrics_path.exists() and not args.overwrite:
                    print(f"skip {method}/{setting}/seed{seed}: {metrics_path} exists")
                    continue
                # A stale failure record must not survive a successful rerun.
                artifacts = RunArtifacts(method, setting, seed, keep_checkpoints=args.keep_checkpoints)
                for stale in ("failure.json", "status.json"):
                    (artifacts.path / stale).unlink(missing_ok=True)
                print(f"run {method}/{setting}/seed{seed}")
                call_kwargs = {**kwargs_for(method, args), **overrides}
                # Recorded in config.json, never forwarded as a hyperparameter.
                if provenance:
                    call_kwargs["config_provenance"] = provenance
                try:
                    result = RUNNERS[method](setting, seed, **call_kwargs)
                except Exception as exc:  # noqa: BLE001 - recorded, then reported below
                    artifacts.write_failure(exc, stage=f"{method}/{setting}/seed{seed}")
                    failures.append(
                        {"method": method, "setting": setting, "seed": seed, "error": repr(exc)}
                    )
                    print(f"FAILED {method}/{setting}/seed{seed}: {exc}")
                    continue
                results.append(result)
                print(json.dumps(result, ensure_ascii=False, indent=2))

    if failures:
        print(f"\n{len(failures)} run(s) failed; see failure.json in each run directory")
    if not results:
        print("No experiments ran. Use --overwrite to replace completed runs.")


if __name__ == "__main__":
    main()