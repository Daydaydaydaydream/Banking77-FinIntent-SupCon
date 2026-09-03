"""Dataset preparation for Banking77.

Builds:
- a full-data train/test split (identical to the original BANKING77 split),
- few-shot training subsets with 5/10/20 labelled examples per class,
- a shared label<->id mapping,

and writes them under ``outputs/datasets`` for consumption by all baselines.
Few-shot sampling is seeded for reproducibility.
"""

from __future__ import annotations

import json
from typing import Dict, List, Tuple

import pandas as pd

from config import (
    CATEGORIES_JSON,
    FEW_SHOT_K,
    LABEL_COL,
    LABEL_MAP_JSON,
    SEED,
    TEST_CSV,
    TEST_CSV_OUT,
    TEXT_COL,
    TRAIN_CSV,
    train_csv_out,
)

Id2Label = List[str]
Label2Id = Dict[str, int]


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
    return out[[TEXT_COL, LABEL_COL, "label_id"]]


def sample_few_shot(train: pd.DataFrame, k: int, seed: int) -> pd.DataFrame:
    """Sample exactly ``k`` examples per class (stratified).

    The dataframe is shuffled once with a fixed seed, then the first ``k``
    rows of every class are taken. This keeps the draw reproducible and avoids
    ``groupby.apply`` semantics that changed across pandas versions.
    """
    shuffled = train.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return shuffled.groupby(LABEL_COL, group_keys=False).head(k).reset_index(drop=True)


def main() -> None:
    train, test, categories = load_raw()
    label2id, id2label = build_label_map(categories)

    # Shared label map and test set (test split is identical across settings).
    LABEL_MAP_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(LABEL_MAP_JSON, "w", encoding="utf-8") as f:
        json.dump({"label2id": label2id, "id2label": id2label}, f, indent=2)

    encode(test, label2id).to_csv(TEST_CSV_OUT, index=False)

    # Full-data training split.
    encode(train, label2id).to_csv(train_csv_out("full"), index=False)

    # Few-shot splits.
    for k in FEW_SHOT_K:
        few = sample_few_shot(train, k, seed=SEED)
        encode(few, label2id).to_csv(train_csv_out(f"{k}shot"), index=False)

    print("[data] wrote label map, test set, and training splits:")
    for k in ["full"] + [f"{k}shot" for k in FEW_SHOT_K]:
        n = len(pd.read_csv(train_csv_out(k)))
        print(f"  - {k:>8}: {n} samples")


if __name__ == "__main__":
    main()
