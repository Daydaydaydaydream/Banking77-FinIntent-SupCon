"""Create a reproducible BANKING77 train/validation split.

The official test set is never sampled or modified.  The initial split is
stratified by intent.  A deterministic repair step then keeps normalized
duplicate utterances on the same side without changing split sizes or the
per-class counts produced by the stratified split.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "banking_data"
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "datasets"


def normalize_text(text: str) -> str:
    """Normalize text for duplicate detection, not for model input."""
    normalized = unicodedata.normalize("NFKC", str(text)).lower()
    return " ".join(re.findall(r"[a-z0-9']+", normalized))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _repair_normalized_overlap(
    data: pd.DataFrame,
    train_indices: set[int],
    val_indices: set[int],
) -> tuple[set[int], set[int], dict[str, object]]:
    """Consolidate normalized duplicate groups into validation.

    For every training row moved into validation, a singleton validation row
    of the same class is moved back.  This preserves total and per-class split
    counts while eliminating normalized train/validation overlap.
    """
    normalized = data["text"].map(normalize_text)
    groups: dict[str, list[int]] = {}
    for idx, value in normalized.items():
        groups.setdefault(value, []).append(int(idx))

    overlaps = sorted(
        value
        for value, members in groups.items()
        if train_indices.intersection(members) and val_indices.intersection(members)
    )
    moved_to_val: list[int] = []
    for value in overlaps:
        members = set(groups[value])
        crossing = sorted(train_indices.intersection(members))
        train_indices.difference_update(crossing)
        val_indices.update(crossing)
        moved_to_val.extend(crossing)

    required_swaps = Counter(data.loc[moved_to_val, "category"])
    singleton_indices = {members[0] for members in groups.values() if len(members) == 1}
    moved_to_train: list[int] = []
    for category in sorted(required_swaps):
        candidates = sorted(
            idx
            for idx in val_indices.intersection(singleton_indices)
            if data.at[idx, "category"] == category
        )
        count = required_swaps[category]
        if len(candidates) < count:
            raise RuntimeError(f"Not enough singleton validation rows for {category!r}")
        chosen = candidates[:count]
        val_indices.difference_update(chosen)
        train_indices.update(chosen)
        moved_to_train.extend(chosen)

    remaining_overlap = {
        normalized.at[idx] for idx in train_indices
    }.intersection(normalized.at[idx] for idx in val_indices)
    if remaining_overlap:
        raise RuntimeError("Normalized duplicate leakage remains after repair")

    return train_indices, val_indices, {
        "overlap_groups_repaired": len(overlaps),
        "rows_moved_train_to_val": moved_to_val,
        "rows_moved_val_to_train": moved_to_train,
    }


def create_split(
    raw_dir: Path = RAW_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    seed: int = 42,
    val_size: int = 1003,
) -> dict[str, object]:
    train_source = raw_dir / "train.csv"
    test_source = raw_dir / "test.csv"
    categories_source = raw_dir / "categories.json"

    data = pd.read_csv(train_source)
    test = pd.read_csv(test_source)
    categories = json.loads(categories_source.read_text(encoding="utf-8"))

    expected_columns = ["text", "category"]
    if data.columns.tolist() != expected_columns or test.columns.tolist() != expected_columns:
        raise ValueError(f"Expected columns {expected_columns}")
    if data.isna().any().any() or test.isna().any().any():
        raise ValueError("Missing values found in source data")
    if set(data["category"]) != set(categories) or set(test["category"]) != set(categories):
        raise ValueError("categories.json does not match CSV labels")
    if not 0 < val_size < len(data):
        raise ValueError("val_size must be between 1 and len(train)-1")

    all_indices = data.index.to_numpy()
    train_idx, val_idx = train_test_split(
        all_indices,
        test_size=val_size,
        random_state=seed,
        shuffle=True,
        stratify=data["category"],
    )
    train_set, val_set, repair = _repair_normalized_overlap(
        data,
        set(map(int, train_idx)),
        set(map(int, val_idx)),
    )

    train_counts = data.loc[sorted(train_set), "category"].value_counts().sort_index()
    val_counts = data.loc[sorted(val_set), "category"].value_counts().sort_index()
    if len(train_set) != len(data) - val_size or len(val_set) != val_size:
        raise RuntimeError("Split size changed during duplicate repair")
    if len(train_set.intersection(val_set)) != 0:
        raise RuntimeError("Row indices overlap between train and validation")
    if len(train_counts) != len(categories) or len(val_counts) != len(categories):
        raise RuntimeError("Every class must appear in both train and validation")

    train_out = (
        data.loc[sorted(train_set), expected_columns]
        .sample(frac=1, random_state=seed)
        .reset_index(drop=True)
    )
    val_out = (
        data.loc[sorted(val_set), expected_columns]
        .sample(frac=1, random_state=seed)
        .reset_index(drop=True)
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "train.csv"
    val_path = output_dir / "val.csv"
    test_path = output_dir / "test.csv"
    label_path = output_dir / "categories.json"
    train_out.to_csv(train_path, index=False)
    val_out.to_csv(val_path, index=False)
    test.to_csv(test_path, index=False)
    label_path.write_text(json.dumps(categories, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    manifest: dict[str, object] = {
        "dataset": "BANKING77",
        "seed": seed,
        "strategy": "stratified split with normalized-duplicate group repair",
        "normalization": "NFKC + lowercase + ASCII alphanumeric/apostrophe tokens",
        "source": {
            "train_path": str(train_source.relative_to(ROOT)),
            "test_path": str(test_source.relative_to(ROOT)),
            "train_sha256": sha256_file(train_source),
            "test_sha256": sha256_file(test_source),
        },
        "sizes": {"train": len(train_out), "validation": len(val_out), "test": len(test)},
        "class_count": len(categories),
        "train_class_min_max": [int(train_counts.min()), int(train_counts.max())],
        "validation_class_min_max": [int(val_counts.min()), int(val_counts.max())],
        "duplicate_repair": repair,
        "source_indices": {
            "train": sorted(train_set),
            "validation": sorted(val_set),
        },
    }
    manifest_path = output_dir / "split_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-size", type=int, default=1003)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = create_split(output_dir=args.output_dir, seed=args.seed, val_size=args.val_size)
    print(json.dumps({k: manifest[k] for k in ("seed", "strategy", "sizes", "duplicate_repair")}, indent=2))


if __name__ == "__main__":
    main()
