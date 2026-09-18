"""Shared configuration for BANKING77 baseline experiments."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT / "outputs" / "datasets"
RUNS_DIR = ROOT / "outputs" / "runs"

TRAIN_CSV = DATASET_DIR / "train.csv"
VAL_CSV = DATASET_DIR / "val.csv"
TEST_CSV = DATASET_DIR / "test.csv"
CATEGORIES_JSON = DATASET_DIR / "categories.json"

SETTINGS = ("full", "5shot", "10shot", "20shot")
SHOT_COUNTS = {"5shot": 5, "10shot": 10, "20shot": 20}
DEFAULT_SEED = 42

BERT_MODEL = "bert-base-uncased"
SETFIT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def run_dir(method: str, setting: str, seed: int) -> Path:
    return RUNS_DIR / method / setting / f"seed{seed}"
