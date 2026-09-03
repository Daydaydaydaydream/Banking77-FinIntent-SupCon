"""Shared data-loading helpers for baselines."""

from __future__ import annotations

import json
from typing import List, Tuple

import pandas as pd

from config import LABEL_MAP_JSON, TEST_CSV_OUT, train_csv_out


def load_label_map() -> Tuple[List[str], dict]:
    """Return (id2label, label2id) from the generated label map."""
    with open(LABEL_MAP_JSON, "r", encoding="utf-8") as f:
        m = json.load(f)
    return m["id2label"], m["label2id"]


def load_split(name: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load a training split and the (shared) test split.

    ``name`` is one of ``full``, ``5shot``, ``10shot``, ``20shot``.
    """
    train = pd.read_csv(train_csv_out(name))
    test = pd.read_csv(TEST_CSV_OUT)
    return train, test
