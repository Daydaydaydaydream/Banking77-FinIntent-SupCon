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

# Checked-out snapshots.  Using the local paths keeps every run offline and
# removes the dependency on hub availability during an experiment sweep.
LOCAL_BERT_MODEL = str(ROOT / "models" / "bert-base-uncased")
LOCAL_SETFIT_MODEL = str(ROOT / "models" / "all-MiniLM-L6-v2")

# Default to the local snapshot when it is present.  ``--local-files-only``
# still requires a resolvable path: passing the hub name with that flag makes
# transformers raise "couldn't connect to huggingface.co" because there is no
# cache entry for it under that name.
BERT_MODEL = LOCAL_BERT_MODEL if Path(LOCAL_BERT_MODEL).is_dir() else "bert-base-uncased"
SETFIT_MODEL = (
    LOCAL_SETFIT_MODEL if Path(LOCAL_SETFIT_MODEL).is_dir() else "sentence-transformers/all-MiniLM-L6-v2"
)


def run_dir(method: str, setting: str, seed: int) -> Path:
    return RUNS_DIR / method / setting / f"seed{seed}"
