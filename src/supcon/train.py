"""Two-stage SupCon training: contrastive representation learning followed by
linear classification.

Stage 1 trains the encoder together with the projection head using the
supervised contrastive objective (optionally with hard-negative weighting).
Stage 2 freezes the encoder and trains only the classification head with
cross-entropy.  Both stages use balanced batches (each batch samples several
classes with several examples per class) so that the contrastive objective
always observes same-class positives.

Stage A changes:

* every randomness source is seeded from one run seed (including the DataLoader
  generator and the per-epoch balanced sampler, which used ``SEED + epoch``);
* stage 2 selects its classification head by **validation** Macro-F1 instead of
  taking the last epoch, matching how the BERT baseline is now selected;
* encoder, projection head, classification head and label map are persisted;
* results go through the shared :class:`RunArtifacts` contract.
"""

from __future__ import annotations

import json
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from config import NUM_CLASSES, SEED
from utils.artifacts import RunArtifacts
from utils.data import load_label_map, load_train_val_test
from utils.metrics import macro_f1, per_class_f1
from utils.seed import derive_seed, numpy_rng, seed_everything, torch_generator
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

    def __getitem__(self, idx):
        if isinstance(idx, list):
            return (
                self.input_ids[idx],
                self.attention_mask[idx],
                self.labels[idx],
            )
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
    rng: np.random.Generator,
):
    """Yield lists of row indices forming class-balanced batches."""
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


def _loader(
    dataset: Dataset, batch_size: int, seed: int, shuffle: bool
) -> DataLoader:
    """Build a DataLoader whose shuffling is reproducible."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=torch_generator(seed) if shuffle else None,
    )


def _stage1_epoch(
    model: SupConModel,
    dataset: _TokenizedDataset,
    indices_by_class: Dict[int, np.ndarray],
    criterion: SupConLoss,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
    cfg: Dict,
    rng: np.random.Generator,
) -> float:
    """Run one SupCon epoch and return the average loss."""
    model.train()
    total_loss = 0.0
    num_batches = _num_batches_per_epoch(
        len(dataset), cfg["batch_classes"], cfg["batch_per_class"]
    )
    batches = list(
        _balanced_batches(
            indices_by_class, cfg["batch_classes"], cfg["batch_per_class"], num_batches, rng
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


def _softmax(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=float)
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def run(
    split_name: str,
    hard_negative_weighting: bool = False,
    seed: int = SEED,
    verbose: bool = False,
    dedup: bool = False,
    artifacts: Optional[RunArtifacts] = None,
) -> Dict:
    """Train and evaluate the SupCon model on a given split.

    Returns ``macro_f1`` (test), the best validation Macro-F1 seen in stage 2,
    per-class F1 and the artefact directory.
    """
    cfg = _SUPCON_CONFIGS[split_name]
    method = "supcon_hn" if hard_negative_weighting else "supcon"
    seed_state = seed_everything(seed, deterministic=True)
    device = _device()
    if verbose:
        print(
            f"[{method}] {split_name}: device={device}, seed={seed}, "
            f"hard_negative={hard_negative_weighting}"
        )

    train, val, test = load_train_val_test(split_name, seed=seed, dedup=dedup)
    id2label, label2id = load_label_map(dedup)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_ds = _TokenizedDataset(train["text"].tolist(), train["label_id"].tolist(), tokenizer, MAX_LEN)
    val_ds = _TokenizedDataset(val["text"].tolist(), val["label_id"].tolist(), tokenizer, MAX_LEN)
    test_ds = _TokenizedDataset(test["text"].tolist(), test["label_id"].tolist(), tokenizer, MAX_LEN)

    model = SupConModel(MODEL_NAME, NUM_CLASSES, proj_dim=PROJ_DIM).to(device)
    model.freeze_encoder(freeze=False)

    train_labels = train["label_id"].to_numpy()
    idx_by_class = _indices_by_class(train_labels)

    test_loader = _loader(test_ds, 32, derive_seed(seed, 99), shuffle=False)
    val_loader = _loader(val_ds, 32, derive_seed(seed, 98), shuffle=False)

    if artifacts is None:
        artifacts = RunArtifacts.create(method, split_name, seed)

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
        rng = numpy_rng(derive_seed(seed, 1, epoch))
        loss = _stage1_epoch(
            model, train_ds, idx_by_class, criterion, opt1, sched1, device, cfg, rng
        )
        if verbose:
            print(f"    stage1 epoch {epoch}/{cfg['stage1_epochs']}: loss={loss:.4f}")

    # --- Stage 2: freeze encoder, train head, select on validation --- #
    model.freeze_encoder(freeze=True)
    model.reset_classification_head()
    criterion2 = nn.CrossEntropyLoss()
    opt2 = torch.optim.AdamW(
        model.classification_head.parameters(), lr=cfg["stage2_lr"], weight_decay=0.01
    )
    stage2_loader = _loader(train_ds, 32, derive_seed(seed, 2), shuffle=True)
    total_steps2 = cfg["stage2_epochs"] * len(stage2_loader)
    sched2 = get_linear_schedule_with_warmup(
        opt2, int(total_steps2 * cfg["warmup_ratio"]), total_steps2
    )

    best_val_f1 = -1.0
    best_epoch = 0
    best_state: Optional[Dict[str, torch.Tensor]] = None

    for epoch in range(1, cfg["stage2_epochs"] + 1):
        loss = _stage2_epoch(model, stage2_loader, criterion2, opt2, sched2, device)
        val_true, val_pred, _ = _predict(model, val_loader, device)
        val_f1 = macro_f1(val_true, val_pred)
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.classification_head.state_dict().items()
            }
        if verbose:
            print(
                f"    stage2 epoch {epoch}/{cfg['stage2_epochs']}: "
                f"loss={loss:.4f} val_f1={val_f1:.4f}"
            )

    if best_state is not None:
        model.classification_head.load_state_dict(best_state)

    # --- Evaluate on the frozen test set --- #
    y_true, y_pred, embeddings = _predict(model, test_loader, device)
    f1 = macro_f1(y_true, y_pred)
    pcf1 = per_class_f1(y_true, y_pred, id2label)

    h_test = embeddings
    logits_for_probs = None
    model.eval()
    with torch.no_grad():
        chunks = []
        for start in range(0, len(test), 32):
            enc = tokenizer(
                test["text"].tolist()[start : start + 32],
                padding="max_length",
                truncation=True,
                max_length=MAX_LEN,
                return_tensors="pt",
            )
            _, _, logits = model(enc["input_ids"].to(device), enc["attention_mask"].to(device))
            chunks.append(logits.cpu().numpy())
        logits_for_probs = np.concatenate(chunks, axis=0)
    probabilities = _softmax(logits_for_probs)

    artifacts.save_config(
        {
            "model_name": MODEL_NAME,
            "max_len": MAX_LEN,
            "temperature": TEMPERATURE,
            "proj_dim": PROJ_DIM,
            "hard_negative_weighting": hard_negative_weighting,
            "stage1": {
                "epochs": cfg["stage1_epochs"],
                "lr": cfg["stage1_lr"],
                "batch_classes": cfg["batch_classes"],
                "batch_per_class": cfg["batch_per_class"],
            },
            "stage2": {
                "epochs": cfg["stage2_epochs"],
                "lr": cfg["stage2_lr"],
                "batch_size": 32,
            },
            "warmup_ratio": cfg["warmup_ratio"],
            "n_train": int(len(train)),
            "n_val": int(len(val)),
            "n_test": int(len(test)),
            "dedup": dedup,
            "checkpoint_selection": "best validation Macro-F1 (stage 2 head)",
        }
    )
    artifacts.save_env({"seed_state": seed_state})
    metrics = artifacts.save_metrics(
        y_true,
        y_pred,
        id2label,
        extra={
            "val_macro_f1": float(best_val_f1),
            "best_stage2_epoch": int(best_epoch),
            "n_train": int(len(train)),
        },
    )
    artifacts.save_predictions(y_true, y_pred, id2label, probabilities=probabilities)
    artifacts.save_embeddings(h_test)

    torch.save(
        {
            "encoder": model.encoder.state_dict(),
            "projection_head": model.projection_head.state_dict(),
            "classification_head": model.classification_head.state_dict(),
        },
        artifacts.model_dir / "supcon.pt",
    )
    with open(artifacts.model_dir / "label_map.json", "w", encoding="utf-8") as f:
        json.dump({"id2label": id2label, "label2id": label2id}, f, indent=2)

    if verbose:
        print(artifacts.summary_line(metrics["macro_f1"]))

    return {
        "macro_f1": f1,
        "per_class_f1": pcf1,
        "val_macro_f1": float(best_val_f1),
        "artifacts": str(artifacts.root),
    }


if __name__ == "__main__":
    import sys

    name = sys.argv[1] if len(sys.argv) > 1 else "full"
    hn = "--hn" in sys.argv
    result = run(name, hard_negative_weighting=hn, verbose=True)
    print({k: v for k, v in result.items() if k != "per_class_f1"})
