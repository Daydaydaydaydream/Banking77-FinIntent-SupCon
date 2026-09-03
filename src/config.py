"""Global configuration for the Banking77 intent-classification project.

Centralizes paths, the random seed, and dataset constants so that every
baseline shares a single source of truth.
"""

from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "banking_data"
SRC_DIR = PROJECT_ROOT / "src"

OUTPUT_DIR = PROJECT_ROOT / "outputs"
DATASET_DIR = OUTPUT_DIR / "datasets"
MODEL_DIR = OUTPUT_DIR / "models"
RESULT_DIR = OUTPUT_DIR / "results"

TRAIN_CSV = DATA_DIR / "train.csv"
TEST_CSV = DATA_DIR / "test.csv"
CATEGORIES_JSON = DATA_DIR / "categories.json"

# --------------------------------------------------------------------------- #
# Dataset constants
# --------------------------------------------------------------------------- #
SEED = 42
NUM_CLASSES = 77
TEXT_COL = "text"
LABEL_COL = "category"

# Few-shot label budgets (number of labelled examples per class).
FEW_SHOT_K = [5, 10, 20]

# --------------------------------------------------------------------------- #
# Shared output files
# --------------------------------------------------------------------------- #
LABEL_MAP_JSON = DATASET_DIR / "label_map.json"
TEST_CSV_OUT = DATASET_DIR / "test.csv"


def train_csv_out(name: str) -> Path:
    """Return the path of a generated training split."""
    return DATASET_DIR / f"train_{name}.csv"


def ensure_dirs() -> None:
    """Create output directories if they do not exist."""
    for d in (DATASET_DIR, MODEL_DIR, RESULT_DIR):
        d.mkdir(parents=True, exist_ok=True)
