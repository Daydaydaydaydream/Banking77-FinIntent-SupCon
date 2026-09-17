"""Visualisation for the Banking77 experiments.

Reads the stage-A artefact tree (``outputs/runs/<method>/<setting>/seed<k>``)
and writes figures under ``outputs/figures``:

1. Confusion matrices — full 77-class heatmaps plus focused views over the
   confusable intent pairs identified in the project plan.
2. Per-class F1 bars — comparison across methods for one setting.
3. Confidence distributions — correct vs incorrect predictions.
4. t-SNE projections — 2-D embeddings restricted to the confusable pairs.

Embeddings are **optional**: methods or runs that do not persist them still get
every other figure, and only the t-SNE plot is skipped. Previously a missing
embedding file raised ``FileNotFoundError`` and dropped the method entirely.

All plots use a non-interactive ``Agg`` backend and write PNG files.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.manifold import TSNE  # noqa: E402
from sklearn.metrics import confusion_matrix  # noqa: E402

from config import FIG_DIR, RUNS_DIR, SEED  # noqa: E402
from utils.data import load_label_map  # noqa: E402

# Confusable intent pairs identified in the project plan.  Each pair groups
# semantically overlapping intents whose misclassification is the focus of the
# SupCon + hard-negative experiments.
CONFUSABLE_PAIRS: List[Tuple[str, str]] = [
    ("declined_card_payment", "declined_transfer"),
    ("card_arrival", "order_physical_card"),
    ("request_refund", "Refund_not_showing_up"),
    ("top_up_failed", "topping_up_by_card"),
]


def load_run(
    method: str, setting: str, seed: int = SEED
) -> Optional[Dict[str, Optional[np.ndarray]]]:
    """Load predictions (required) and embeddings (optional) for one run.

    Returns ``None`` when the run has no ``predictions.csv``.
    """
    root = RUNS_DIR / method / setting / f"seed{seed}"
    pred_path = root / "predictions.csv"
    if not pred_path.exists():
        return None

    frame = pd.read_csv(pred_path)
    data: Dict[str, Optional[np.ndarray]] = {
        "y_true": frame["y_true"].to_numpy(),
        "y_pred": frame["y_pred"].to_numpy(),
        "confidence": (
            frame["confidence"].to_numpy() if "confidence" in frame else None
        ),
        "emb": None,
    }

    dense = root / "embeddings.npy"
    sparse = root / "embeddings.npz"
    if dense.exists():
        data["emb"] = np.load(dense)
    elif sparse.exists():
        from scipy import sparse as sp

        matrix = sp.load_npz(sparse)
        data["emb"] = matrix.toarray() if matrix.shape[1] <= 512 else None

    return data


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Sequence[str],
    title: str,
    save_path: Path,
    normalize: bool = True,
) -> None:
    """Plot a (normalised) confusion matrix and save it to ``save_path``."""
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(labels))))
    if normalize:
        cm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-9)

    size = max(10, len(labels) * 0.12)
    fig, ax = plt.subplots(figsize=(size, size))
    im = ax.imshow(cm, cmap="Blues", aspect="auto")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=4)
    ax.set_yticklabels(labels, fontsize=4)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_confusable_pairs(
    data: Dict[str, Dict[str, np.ndarray]],
    labels: Sequence[str],
    label2id: Dict[str, int],
    save_path: Path,
) -> None:
    """Plot a focused confusion sub-matrix for the confusable intent pairs."""
    pair_ids = sorted({label2id[a] for a, _ in CONFUSABLE_PAIRS} | {label2id[b] for _, b in CONFUSABLE_PAIRS})
    pair_labels = [labels[i] for i in pair_ids]

    n_methods = len(data)
    fig, axes = plt.subplots(1, n_methods, figsize=(6 * n_methods, 6), squeeze=False)

    for ax, (method, d) in zip(axes[0], data.items()):
        sub = np.isin(d["y_true"], pair_ids) | np.isin(d["y_pred"], pair_ids)
        cm = confusion_matrix(d["y_true"][sub], d["y_pred"][sub], labels=pair_ids)
        cm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-9)

        im = ax.imshow(cm, cmap="Reds", aspect="auto")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_xticks(np.arange(len(pair_ids)))
        ax.set_yticks(np.arange(len(pair_ids)))
        ax.set_xticklabels(pair_labels, rotation=90, fontsize=6)
        ax.set_yticklabels(pair_labels, fontsize=6)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(method)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_per_class_f1(
    per_class_by_method: Dict[str, Dict[str, float]],
    save_path: Path,
    title: str = "Per-class F1 by method",
) -> None:
    """Grouped bar chart of per-class F1 across methods."""
    methods = list(per_class_by_method)
    intents = list(per_class_by_method[methods[0]])
    x = np.arange(len(intents))
    width = 0.8 / max(1, len(methods))

    fig, ax = plt.subplots(figsize=(max(12, len(intents) * 0.22), 6))
    for i, method in enumerate(methods):
        values = [per_class_by_method[method].get(intent, 0.0) for intent in intents]
        ax.bar(x + i * width, values, width=width, label=method)

    ax.set_xticks(x + width * (len(methods) - 1) / 2)
    ax.set_xticklabels(intents, rotation=90, fontsize=6)
    ax.set_ylabel("F1")
    ax.set_ylim(0, 1.05)
    ax.set_title(title)
    ax.legend(fontsize=8)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_confidence_distribution(
    data: Dict[str, Dict[str, np.ndarray]],
    save_path: Path,
) -> None:
    """Histogram of prediction confidence split by correctness."""
    available = {m: d for m, d in data.items() if d.get("confidence") is not None}
    if not available:
        return

    n = len(available)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), squeeze=False)
    for ax, (method, d) in zip(axes[0], available.items()):
        conf = np.asarray(d["confidence"], dtype=float)
        correct = np.asarray(d["y_true"]) == np.asarray(d["y_pred"])
        conf = conf[~np.isnan(conf)]
        correct = correct[~np.isnan(np.asarray(d["confidence"], dtype=float))]
        ax.hist(conf[correct], bins=20, alpha=0.7, label="correct", color="#378ADD")
        ax.hist(conf[~correct], bins=20, alpha=0.7, label="incorrect", color="#E24B4A")
        ax.set_title(method)
        ax.set_xlabel("confidence")
        ax.set_ylabel("count")
        ax.legend(fontsize=8)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_tsne(
    embeddings: np.ndarray,
    labels_arr: np.ndarray,
    id2label: Sequence[str],
    label2id: Dict[str, int],
    save_path: Path,
    n_samples_per_class: int = 60,
) -> None:
    """Project test embeddings of the confusable pairs to 2-D via t-SNE."""
    pair_ids = sorted({label2id[a] for a, _ in CONFUSABLE_PAIRS} | {label2id[b] for _, b in CONFUSABLE_PAIRS})

    keep_idx = []
    rng = np.random.default_rng(42)
    for cid in pair_ids:
        idx = np.where(labels_arr == cid)[0]
        if len(idx) > n_samples_per_class:
            idx = rng.choice(idx, size=n_samples_per_class, replace=False)
        keep_idx.extend(idx.tolist())

    keep_idx = np.array(keep_idx)
    sub_emb = embeddings[keep_idx]
    sub_lab = labels_arr[keep_idx]

    tsne = TSNE(n_components=2, random_state=42, perplexity=30, init="pca")
    coords = tsne.fit_transform(sub_emb)

    fig, ax = plt.subplots(figsize=(8, 6))
    for cid in pair_ids:
        m = sub_lab == cid
        ax.scatter(coords[m, 0], coords[m, 1], s=12, alpha=0.7, label=id2label[cid])

    ax.set_title("t-SNE of confusable intent pairs (test set)")
    ax.set_xlabel("t-SNE dim 1")
    ax.set_ylabel("t-SNE dim 2")
    ax.legend(fontsize=6, loc="best", ncol=2)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def main(
    methods: Sequence[str] = ("bert_ft", "supcon", "supcon_hn"),
    settings: Sequence[str] = ("full",),
    seed: int = SEED,
) -> None:
    """Generate every figure for the given methods, settings and seed."""
    id2label, label2id = load_label_map()
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    for setting in settings:
        data: Dict[str, Dict[str, Optional[np.ndarray]]] = {}
        for method in methods:
            run = load_run(method, setting, seed)
            if run is None:
                print(f"[skip] {method}/{setting}: no predictions found")
                continue
            data[method] = run

        if not data:
            print(f"[skip] {setting}: no runs available")
            continue

        for method, d in data.items():
            plot_confusion_matrix(
                d["y_true"],
                d["y_pred"],
                id2label,
                f"{method} ({setting}, seed{seed})",
                FIG_DIR / f"confusion_{method}_{setting}_seed{seed}.png",
            )

        plot_confusable_pairs(
            data, id2label, label2id, FIG_DIR / f"confusable_pairs_{setting}_seed{seed}.png"
        )
        plot_confidence_distribution(
            data, FIG_DIR / f"confidence_{setting}_seed{seed}.png"
        )

        skipped_tsne = []
        for method, d in data.items():
            if d.get("emb") is None:
                skipped_tsne.append(method)
                continue
            plot_tsne(
                d["emb"],
                d["y_true"],
                id2label,
                label2id,
                FIG_DIR / f"tsne_{method}_{setting}_seed{seed}.png",
            )
        if skipped_tsne:
            print(f"[info] {setting}: no embeddings for {skipped_tsne}; t-SNE skipped")

    print(f"[analysis] figures written to {FIG_DIR}")


if __name__ == "__main__":
    main()
