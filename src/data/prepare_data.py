"""Dataset preparation for Banking77 (stage A.2).

Builds, under ``outputs/datasets`` (or ``outputs/datasets_dedup``):

* a **frozen test split** copied verbatim from the official BANKING77 test set;
* a **stratified train/validation split** carved out of the official 10,003
  training examples (9,000 / 1,003 by default) so that checkpoints can be
  selected without ever touching the test set;
* **few-shot training subsets** with 5/10/20 labelled examples per class, drawn
  from the training pool only — the validation split never contributes to the
  annotation budget;
* a shared label<->id mapping.

Few-shot draws vary with the seed, so they are written under ``seed<k>/``.
The train/validation split is fixed across seeds by construction (it is derived
from the split seed only), which is what stage D.1 requires.

Running with ``--dedup`` additionally removes training examples whose
normalised text also occurs in the test split, and writes the result to a
separate directory so the official-split results stay the primary numbers.
"""

from __future__ import annotations

import argparse
import json
import re
from typing import Dict, List, Sequence, Set, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from config import (
    CATEGORIES_JSON,
    FEW_SHOT_K,
    LABEL_COL,
    SEED,
    SEEDS,
    TEST_CSV,
    TRAIN_CSV,
    VAL_RATIO,
    VAL_SIZE,
    dataset_root,
    label_map_json,
    test_csv_out,
    train_csv_out,
    train_pool_csv,
    val_csv_out,
)

Id2Label = List[str]
Label2Id = Dict[str, int]

_WHITESPACE = re.compile(r"\s+")


def load_raw() -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """Load the original train/test CSVs and the canonical category list."""
    train = pd.read_csv(TRAIN_CSV)
    test = pd.read_csv(TEST_CSV)
    with open(CATEGORIES_JSON, "r", encoding="utf-8") as f:
        categories = json.load(f)
    assert len(categories) == 77, f"expected 77 categories, got {len(categories)}"
    return train, test, categories


def build_label_map(categories: List[str]) -> Tuple[Label2Id, Id2Label]:
    """Map category names to consecutive integer ids in canonical order."""
    label2id = {cat: i for i, cat in enumerate(categories)}
    id2label = list(categories)
    return label2id, id2label


def encode(df: pd.DataFrame, label2id: Label2Id) -> pd.DataFrame:
    """Add a ``label_id`` column to a dataframe with a ``category`` column."""
    out = df.copy()
    out["label_id"] = out[LABEL_COL].map(label2id)
    assert out["label_id"].notna().all(), "some categories are missing from the map"
    out["label_id"] = out["label_id"].astype(int)
    return out[["text", LABEL_COL, "label_id"]]


def normalize_text(series: pd.Series) -> pd.Series:
    """Lower-case, strip and collapse whitespace so duplicates can be matched."""
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(_WHITESPACE, " ", regex=True)
    )


def duplicated_train_rows(train: pd.DataFrame, test: pd.DataFrame) -> pd.Series:
    """Boolean mask of training rows whose normalised text appears in the test set."""
    test_keys: Set[str] = set(normalize_text(test["text"]))
    return normalize_text(train["text"]).isin(test_keys)


def split_train_val(
    train: pd.DataFrame, val_size: int, seed: int
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified split of the official training pool into train and validation."""
    train_part, val_part = train_test_split(
        train,
        test_size=val_size,
        stratify=train[LABEL_COL],
        random_state=seed,
    )
    return train_part.reset_index(drop=True), val_part.reset_index(drop=True)


def sample_few_shot(train: pd.DataFrame, k: int, seed: int) -> pd.DataFrame:
    """Sample exactly ``k`` examples per class (stratified).

    The dataframe is shuffled once with the given seed, then the first ``k``
    rows of every class are taken. This keeps the draw reproducible and avoids
    ``groupby.apply`` semantics that changed across pandas versions.
    """
    shuffled = train.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return shuffled.groupby(LABEL_COL, group_keys=False).head(k).reset_index(drop=True)


def build(dedup: bool = False, seeds: Sequence[int] = tuple(SEEDS)) -> Dict[str, object]:
    """Build every split for one dataset variant and return a summary."""
    train, test, categories = load_raw()
    label2id, id2label = build_label_map(categories)

    removed = 0
    if dedup:
        dup_mask = duplicated_train_rows(train, test)
        removed = int(dup_mask.sum())
        train = train.loc[~dup_mask].reset_index(drop=True)

    # The official test split is frozen: de-duplication only ever removes
    # training rows, so the test set stays comparable across variants.
    val_size = VAL_SIZE if not dedup else int(round(len(train) * VAL_RATIO))
    train_pool, val = split_train_val(train, val_size, seed=SEED)

    root = dataset_root(dedup)
    root.mkdir(parents=True, exist_ok=True)

    with open(label_map_json(dedup), "w", encoding="utf-8") as f:
        json.dump({"label2id": label2id, "id2label": id2label}, f, indent=2)

    encode(test, label2id).to_csv(test_csv_out(dedup), index=False)
    encode(train_pool, label2id).to_csv(train_pool_csv(dedup), index=False)
    encode(val, label2id).to_csv(val_csv_out(dedup), index=False)

    few_shot_counts: Dict[str, int] = {}
    for seed in seeds:
        for k in FEW_SHOT_K:
            few = sample_few_shot(train_pool, k, seed=seed)
            path = train_csv_out(f"{k}shot", seed=seed, dedup=dedup)
            path.parent.mkdir(parents=True, exist_ok=True)
            encode(few, label2id).to_csv(path, index=False)
            few_shot_counts[f"seed{seed}/{k}shot"] = len(few)

    return {
        "dedup": dedup,
        "root": str(root),
        "test": len(test),
        "train_pool": len(train_pool),
        "val": len(val),
        "removed_duplicates": removed,
        "seeds": list(seeds),
        "few_shot": few_shot_counts,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build Banking77 splits (stage A.2)")
    p.add_argument(
        "--dedup",
        action="store_true",
        help="drop training rows whose text also appears in the test split; "
        "writes to outputs/datasets_dedup and keeps the official variant intact",
    )
    p.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help=f"seeds for the few-shot draws (default: {SEEDS})",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    seeds = args.seeds if args.seeds is not None else SEEDS
    summary = build(dedup=args.dedup, seeds=seeds)

    variant = "de-duplicated" if summary["dedup"] else "official"
    print(f"[data] built {variant} splits under {summary['root']}")
    print(f"  test       : {summary['test']} (frozen official test set)")
    print(f"  train pool : {summary['train_pool']}")
    print(f"  validation : {summary['val']}")
    if summary["dedup"]:
        print(f"  removed    : {summary['removed_duplicates']} duplicate training rows")
    print(f"  seeds      : {summary['seeds']}")
    for name, n in summary["few_shot"].items():
        print(f"  - {name:>16}: {n} samples")


if __name__ == "__main__":
    main()
