"""Dataset loading and deterministic few-shot sampling."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from config import CATEGORIES_JSON, SETTINGS, SHOT_COUNTS, TEST_CSV, TRAIN_CSV, VAL_CSV


def load_categories(path: Path = CATEGORIES_JSON) -> list[str]:
    categories = json.loads(path.read_text(encoding="utf-8"))
    if len(categories) != 77 or len(categories) != len(set(categories)):
        raise ValueError("Expected 77 unique BANKING77 categories")
    return categories


def _class_seed(seed: int, category: str) -> int:
    digest = hashlib.sha256(f"{seed}:{category}".encode()).digest()
    return int.from_bytes(digest[:4], "little")


def sample_few_shot(train: pd.DataFrame, shots: int, seed: int) -> pd.DataFrame:
    sampled: list[pd.DataFrame] = []
    for category in sorted(train["category"].unique()):
        group = train.loc[train["category"] == category]
        if len(group) < shots:
            raise ValueError(f"Category {category!r} has only {len(group)} rows, fewer than {shots}")
        rng = np.random.default_rng(_class_seed(seed, category))
        chosen = rng.choice(group.index.to_numpy(), size=shots, replace=False)
        sampled.append(train.loc[chosen])
    result = pd.concat(sampled, ignore_index=True)
    return result.sample(frac=1, random_state=seed).reset_index(drop=True)


def load_splits(setting: str, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    if setting not in SETTINGS:
        raise ValueError(f"Unknown setting {setting!r}; choose from {SETTINGS}")
    for path in (TRAIN_CSV, VAL_CSV, TEST_CSV, CATEGORIES_JSON):
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}. Run: python -m data.prepare_data")

    train = pd.read_csv(TRAIN_CSV)
    validation = pd.read_csv(VAL_CSV)
    test = pd.read_csv(TEST_CSV)
    categories = load_categories()
    for name, frame in (("train", train), ("validation", validation), ("test", test)):
        if frame.columns.tolist() != ["text", "category"]:
            raise ValueError(f"{name} must contain exactly text,category columns")
        unknown = set(frame["category"]) - set(categories)
        if unknown:
            raise ValueError(f"{name} contains unknown categories: {sorted(unknown)}")

    if setting != "full":
        train = sample_few_shot(train, SHOT_COUNTS[setting], seed)
    return train, validation, test, categories


def encode_labels(frame: pd.DataFrame, categories: list[str]) -> np.ndarray:
    label_to_id = {label: idx for idx, label in enumerate(categories)}
    encoded = frame["category"].map(label_to_id)
    if encoded.isna().any():
        raise ValueError("Found category missing from label map")
    return encoded.to_numpy(dtype=np.int64)
