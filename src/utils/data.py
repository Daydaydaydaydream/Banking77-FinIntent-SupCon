"""Shared data-loading helpers for baselines.

Stage A split the official training set into a training pool and a stratified
validation set, and made the few-shot draws seed-dependent. Loaders therefore
take a ``seed`` and an optional ``dedup`` flag, and expose the validation split
so training code can select checkpoints without touching the frozen test set.
"""

from __future__ import annotations

import json
from typing import List, Tuple

import pandas as pd

from config import (
    SEED,
    label_map_json,
    test_csv_out,
    train_csv_out,
    val_csv_out,
)


def load_label_map(dedup: bool = False) -> Tuple[List[str], dict]:
    """Return (id2label, label2id) from the generated label map."""
    with open(label_map_json(dedup), "r", encoding="utf-8") as f:
        m = json.load(f)
    return m["id2label"], m["label2id"]


def load_split(
    name: str, seed: int = SEED, dedup: bool = False
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load a training split and the (shared) frozen test split.

    ``name`` is one of ``full``, ``5shot``, ``10shot``, ``20shot``.
    """
    train = pd.read_csv(train_csv_out(name, seed=seed, dedup=dedup))
    test = pd.read_csv(test_csv_out(dedup))
    return train, test


def load_val(dedup: bool = False) -> pd.DataFrame:
    """Load the stratified validation split (seed-independent)."""
    return pd.read_csv(val_csv_out(dedup))


def load_train_val_test(
    name: str, seed: int = SEED, dedup: bool = False
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (train, validation, test) for a setting and seed."""
    train, test = load_split(name, seed=seed, dedup=dedup)
    return train, load_val(dedup), test
