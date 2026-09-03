"""Two-stage SupCon training: contrastive representation learning followed by
linear classification.

Stage 1 trains the encoder together with the projection head using the
supervised contrastive objective (optionally with hard-negative weighting).
Stage 2 freezes the encoder and trains only the classification head with
cross-entropy.  Both stages use balanced batches (each batch samples several
classes with several examples per class) so that the contrastive objective
always observes same-class positives.

The ``run`` entry point mirrors the baseline interface and returns Macro-F1
together with per-class F1; predictions and test-set embeddings are persisted
under ``outputs/analysis`` for the visualisation stage.
"""

from __future__ import annotations

import json
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from config import MODEL_DIR, NUM_CLASSES, SEED, RESULT_DIR
from utils.data import load_label_map, load_split
from utils.metrics import macro_f1, per_class_f1
from supcon.losses import SupConLoss
from supcon.model import SupConModel

MODEL_NAME = "bert-base-uncased"
MAX_LEN = 64
TEMPERATURE = 0.07
PROJ_DIM = 128

# Per-setting training hyper-parameters for the two stages.
_SUPCON_CONFIGS: Dict[str, Dict] = {
    "full": {
        "stage1_epochs": 5,
        "stage1_lr": 2e-5,
        "stage2_epochs": 10,
        "stage2_lr": 1e-3,
        "batch_classes": 16,
        "batch_per_class": 8,
        "warmup_ratio": 0.06,
    },
    "5shot": {
        "stage1_epochs": 20,
        "stage1_lr": 5e-5,
        "stage2_epochs": 40,
        "stage2_lr": 1e-3,
        "batch_classes": 16,
        "batch_per_class": 5,
        "warmup_ratio": 0.1,
    },
    "10shot": {
        "stage1_epochs": 15,
        "stage1_lr": 5e-5,
        "stage2_epochs": 30,
        "stage2_lr": 1e-3,
        "batch_classes": 16,
        "batch_per_class": 8,
        "warmup_ratio": 0.1,
    },
    "20shot": {
        "stage1_epochs": 10,
        "stage1_lr": 5e-5,
        "stage2_epochs": 25,
        "stage2_lr": 1e-3,
        "batch_classes": 16,
        "batch_per_class": 8,
        "warmup_ratio": 0.1,
    },
}


def _device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class _TokenizedDataset(Dataset):
    """Tokenised utterances with their integer labels."""

    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_len: int) -> None:
        enc = tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        )
        self.input_ids = enc["input_ids"]
        self.attention_mask = enc["attention_mask"]
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.input_ids[idx], self.attention_mask[idx], self.labels[idx]


def _indices_by_class(labels: np.ndarray) -> Dict[int, np.ndarray]:
    """Group dataset row indices by class label."""
    classes = np.unique(labels)
    return {int(c): np.where(labels == c)[0] for c in classes}


def _balanced_batches(
    indices_by_class: Dict[int, np.ndarray],
    num_classes: int,
    num_per_class: int,
    num_batches: int,
    seed: int,
):
    """Yield lists of row indices forming class-balanced batches."""
    rng = np.random.default_rng(seed)
    class_ids = list(indices_by_class.keys())
    for _ in range(num_batches):
        chosen = rng.choice(class_ids, size=num_classes, replace=False)
        batch: List[int] = []
        for c in chosen:
            idxs = indices_by_class[int(c)]
            replace = len(idxs) < num_per_class
            batch.extend(rng.choice(idxs, size=num_per_class, replace=replace).tolist())
        yield batch


def _num_batches_per_epoch(num_samples: int, num_classes: int, num_per_class: int) -> int:
    return max(1, num_samples // (num_classes * num_per_class))


def _stage1_epoch(
    model: SupConModel,
    dataset: _TokenizedDataset,
    indices_by_class: Dict[int, np.ndarray],
    criterion: SupConLoss,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
    cfg: Dict,
    seed: int,
) -> float:
    """Run one SupCon epoch and return the average loss."""
    model.train()
    total_loss = 0.0
    num_batches = _num_batches_per_epoch(
        len(dataset), cfg["batch_classes"], cfg["batch_per_class"]
    )
    batches = list(
        _balanced_batches(
            indices_by_class, cfg["batch_classes"], cfg["batch_per_class"], num_batches, seed
        )
    )
    for batch_idx in batches:
        input_ids, attention_mask, labels = dataset[batch_idx]
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)

        _, z, _ = model(input_ids, attention_mask)
        loss = criterion(z, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

    return total_loss / max(1, len(batches))


def _stage2_epoch(
    model: SupConModel,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
) -> float:
    """Run one classification-head epoch and return the average loss."""
    model.train()
    total_loss = 0.0
    for input_ids, attention_mask, labels in loader:
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)

        _, _, logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

    return total_loss / max(1, len(loader))


@torch.no_grad()
def _predict(
    model: SupConModel, loader: DataLoader, device: torch.device
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (y_true, y_pred, embeddings) over a dataset."""
    model.eval()
    all_true, all_pred, all_emb = [], [], []
    for input_ids, attention_mask, labels in loader:
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        h, _, logits = model(input_ids, attention_mask)
        preds = logits.argmax(dim=1)
        all_true.append(labels.numpy())
        all_pred.append(preds.cpu().numpy())
        all_emb.append(h.cpu().numpy())
    return (
        np.concatenate(all_true),
        np.concatenate(all_pred),
        np.concatenate(all_emb),
    )


def run(
    split_name: str,
    hard_negative_weighting: bool = False,
    verbose: bool = False,
) -> Dict:
    """Train and evaluate the SupCon model on a given split.

    Parameters
    ----------
    split_name:
        One of ``full``, ``5shot``, ``10shot``, ``20shot``.
    hard_negative_weighting:
        Enable similarity-weighted hard negatives in the SupCon objective.
    verbose:
        Print stage-by-stage progress.

    Returns
    -------
    dict
        Contains ``macro_f1`` and ``per_class_f1``; ``y_true``, ``y_pred`` and
        ``embeddings`` are persisted under ``outputs/analysis``.
    """
    cfg = _SUPCON_CONFIGS[split_name]
    device = _device()
    if verbose:
        print(
            f"[supcon] {split_name}: device={device}, hard_negative={hard_negative_weighting}, "
            f"cfg={cfg}"
        )

    train, test = load_split(split_name)
    id2label, _ = load_label_map()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_ds = _TokenizedDataset(train["text"].tolist(), train["label_id"].tolist(), tokenizer, MAX_LEN)
    test_ds = _TokenizedDataset(test["text"].tolist(), test["label_id"].tolist(), tokenizer, MAX_LEN)

    model = SupConModel(MODEL_NAME, NUM_CLASSES, proj_dim=PROJ_DIM).to(device)
    model.freeze_encoder(freeze=False)

    train_labels = train["label_id"].to_numpy()
    idx_by_class = _indices_by_class(train_labels)

    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

    # --- Stage 1: supervised contrastive representation learning --- #
    criterion = SupConLoss(
        temperature=TEMPERATURE,
        hard_negative_weighting=hard_negative_weighting,
    )
    stage1_params = list(model.encoder.parameters()) + list(model.projection_head.parameters())
    opt1 = torch.optim.AdamW(stage1_params, lr=cfg["stage1_lr"], weight_decay=0.01)
    total_steps1 = cfg["stage1_epochs"] * _num_batches_per_epoch(
        len(train_ds), cfg["batch_classes"], cfg["batch_per_class"]
    )
    sched1 = get_linear_schedule_with_warmup(
        opt1, int(total_steps1 * cfg["warmup_ratio"]), total_steps1
    )

    for epoch in range(1, cfg["stage1_epochs"] + 1):
        loss = _stage1_epoch(
            model, train_ds, idx_by_class, criterion, opt1, sched1, device, cfg, SEED + epoch
        )
        if verbose:
            print(f"    stage1 epoch {epoch}/{cfg['stage1_epochs']}: loss={loss:.4f}")

    # --- Stage 2: freeze encoder, train classification head --- #
    model.freeze_encoder(freeze=True)
    model.reset_classification_head()
    criterion2 = nn.CrossEntropyLoss()
    opt2 = torch.optim.AdamW(model.classification_head.parameters(), lr=cfg["stage2_lr"], weight_decay=0.01)
    stage2_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    total_steps2 = cfg["stage2_epochs"] * len(stage2_loader)
    sched2 = get_linear_schedule_with_warmup(
        opt2, int(total_steps2 * cfg["warmup_ratio"]), total_steps2
    )

    for epoch in range(1, cfg["stage2_epochs"] + 1):
        loss = _stage2_epoch(model, stage2_loader, criterion2, opt2, sched2, device)
        if verbose:
            print(f"    stage2 epoch {epoch}/{cfg['stage2_epochs']}: loss={loss:.4f}")

    # --- Evaluate --- #
    y_true, y_pred, embeddings = _predict(model, test_loader, device)
    f1 = macro_f1(y_true, y_pred)
    pcf1 = per_class_f1(y_true, y_pred, id2label)

    _persist_analysis(split_name, hard_negative_weighting, y_true, y_pred, embeddings)

    if verbose:
        print(f"[supcon] {split_name}: Macro-F1 = {f1:.4f}")

    return {"macro_f1": f1, "per_class_f1": pcf1}


def _persist_analysis(
    split_name: str,
    hard_negative_weighting: bool,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    embeddings: np.ndarray,
) -> None:
    """Persist predictions and embeddings for the visualisation stage."""
    analysis_dir = RESULT_DIR.parent / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    tag = "supcon_hn" if hard_negative_weighting else "supcon"
    stem = f"{tag}_{split_name}"
    np.save(analysis_dir / f"{stem}_y_true.npy", y_true)
    np.save(analysis_dir / f"{stem}_y_pred.npy", y_pred)
    np.save(analysis_dir / f"{stem}_emb.npy", embeddings)


if __name__ == "__main__":
    import sys

    name = sys.argv[1] if len(sys.argv) > 1 else "full"
    hn = "--hn" in sys.argv
    result = run(name, hard_negative_weighting=hn, verbose=True)
    print(json.dumps({"macro_f1": result["macro_f1"]}, indent=2))
