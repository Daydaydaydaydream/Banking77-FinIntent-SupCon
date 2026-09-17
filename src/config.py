"""Global configuration for the Banking77 intent-classification project.

Centralizes paths, random seeds, dataset constants and the stage-A split
budgets so that every baseline shares a single source of truth.

Stage A (see ``docs/experiment-plan.md``) introduced two structural changes:

* a stratified validation split carved out of the official 10,003 training
  examples, so checkpoints can be selected without touching the frozen test set;
* a per-seed dataset namespace, because few-shot draws vary with the seed while
  the train/validation split itself stays fixed across seeds.
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
DATASET_DEDUP_DIR = OUTPUT_DIR / "datasets_dedup"
MODEL_DIR = OUTPUT_DIR / "models"
RESULT_DIR = OUTPUT_DIR / "results"
RUNS_DIR = OUTPUT_DIR / "runs"
FIG_DIR = OUTPUT_DIR / "figures"
ANALYSIS_DIR = OUTPUT_DIR / "analysis"

TRAIN_CSV = DATA_DIR / "train.csv"
TEST_CSV = DATA_DIR / "test.csv"
CATEGORIES_JSON = DATA_DIR / "categories.json"

# --------------------------------------------------------------------------- #
# Dataset constants
# --------------------------------------------------------------------------- #
SEED = 42
# Seeds used for the main results. ``SEED`` remains the split/derivation seed.
SEEDS = [42, 1, 2]

NUM_CLASSES = 77
TEXT_COL = "text"
LABEL_COL = "category"
LABEL_ID_COL = "label_id"

# Few-shot label budgets (number of labelled examples per class).
FEW_SHOT_K = [5, 10, 20]

# Settings: the full training pool plus every few-shot budget.
SETTINGS = ["full"] + [f"{k}shot" for k in FEW_SHOT_K]

# --------------------------------------------------------------------------- #
# Stage A.2: stratified train / validation split
# --------------------------------------------------------------------------- #
# Carved out of the 10,003 official training examples only. The 3,080-example
# official test split is frozen and never used for model selection.
VAL_SIZE = 1003
TRAIN_POOL_SIZE = 9000
VAL_RATIO = VAL_SIZE / (VAL_SIZE + TRAIN_POOL_SIZE)

# --------------------------------------------------------------------------- #
# Shared output files (primary / official split)
# --------------------------------------------------------------------------- #
LABEL_MAP_JSON = DATASET_DIR / "label_map.json"
TEST_CSV_OUT = DATASET_DIR / "test.csv"
TRAIN_POOL_CSV = DATASET_DIR / "train_pool.csv"
VAL_CSV_OUT = DATASET_DIR / "val.csv"


def dataset_root(dedup: bool = False) -> Path:
    """Return the dataset directory for the official or de-duplicated variant."""
    return DATASET_DEDUP_DIR if dedup else DATASET_DIR


def label_map_json(dedup: bool = False) -> Path:
    return dataset_root(dedup) / "label_map.json"


def test_csv_out(dedup: bool = False) -> Path:
    return dataset_root(dedup) / "test.csv"


def val_csv_out(dedup: bool = False) -> Path:
    """Validation split path. Fixed across seeds by construction."""
    return dataset_root(dedup) / "val.csv"


def train_pool_csv(dedup: bool = False) -> Path:
    """Full training pool (9,000 examples). Fixed across seeds."""
    return dataset_root(dedup) / "train_pool.csv"


def train_csv_out(name: str, seed: int = SEED, dedup: bool = False) -> Path:
    """Return the path of a generated training split.

    ``full`` reuses the shared training pool because it does not depend on the
    seed; few-shot draws do, so they live under ``seed<k>/``.
    """
    if name == "full":
        return train_pool_csv(dedup)
    return dataset_root(dedup) / f"seed{seed}" / f"train_{name}.csv"


def run_dir(method: str, setting: str, seed: int) -> Path:
    """Return ``outputs/runs/<method>/<setting>/seed<k>`` for a single run."""
    return RUNS_DIR / method / setting / f"seed{seed}"


def ensure_dirs() -> None:
    """Create output directories if they do not exist."""
    for d in (DATASET_DIR, DATASET_DEDUP_DIR, MODEL_DIR, RESULT_DIR, RUNS_DIR, FIG_DIR):
        d.mkdir(parents=True, exist_ok=True)
