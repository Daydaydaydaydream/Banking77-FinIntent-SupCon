"""Visualisation for the Banking77 experiments.

Generates two artefact families under ``outputs/figures``:

1. Confusion matrices — full 77-class heatmaps plus focused views over the
   confusable intent pairs identified in the project plan.
2. t-SNE projections — 2-D embeddings of test examples restricted to the
   confusable pairs, coloured by intent, to show whether SupCon separates
   the semantically overlapping classes.

All plots use a non-interactive ``Agg`` backend and write PNG files, so they
are suitable for inclusion in the final report.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix

from config import RESULT_DIR
from utils.data import load_label_map

FIG_DIR = RESULT_DIR.parent / "figures"

# Confusable intent pairs identified in the project plan.  Each pair groups
# semantically overlapping intents whose misclassification is the focus of
# the SupCon + hard-negative experiments.
CONFUSABLE_PAIRS: List[Tuple[str, str]] = [
    ("declined_card_payment", "declined_transfer"),
    ("card_arrival", "order_physical_card"),
    ("request_refund", "Refund_not_showing_up"),
    ("top_up_failed", "topping_up_by_card"),
]


def load_analysis(tag: str, split_name: str) -> Dict[str, np.ndarray]:
    """Load persisted predictions/embeddings for a method/split.

    Parameters
    ----------
    tag:
        One of ``bert_ft``, ``supcon``, ``supcon_hn``.
    split_name:
        One of ``full``, ``5shot``, ``10shot``, ``20shot``.
    """
    analysis_dir = RESULT_DIR.parent / "analysis"
    stem = f"{tag}_{split_name}"

    def _load(kind: str, required: bool = True) -> Optional[np.ndarray]:
        p = analysis_dir / f"{stem}_{kind}.npy"
        if p.exists():
            return np.load(p)
        if required:
            raise FileNotFoundError(f"Missing analysis artefact: {p}")
        return None

    return {
        "y_true": _load("y_true"),
        "y_pred": _load("y_pred"),
        "emb": _load("emb"),
    }


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: List[str],
    title: str,
    save_path: Path,
    normalize: bool = True,
) -> None:
    """Plot a (normalised) confusion matrix and save it to ``save_path``."""
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(labels))))
    if normalize:
        cm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-9)

    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 0.12), max(10, len(labels) * 0.12)))
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
    labels: List[str],
    label2id: Dict[str, int],
    save_path: Path,
) -> None:
    """Plot a focused confusion sub-matrix for the confusable intent pairs.

    Parameters
    ----------
    data:
        Mapping ``method -> {"y_true", "y_pred"}`` for the methods to compare.
    labels:
        ``id2label`` list.
    label2id:
        ``label2id`` mapping.
    save_path:
        Output PNG path.
    """
    pair_ids = sorted({label2id[a] for a, b in CONFUSABLE_PAIRS for _ in (0,)} |
                      {label2id[b] for a, b in CONFUSABLE_PAIRS})
    pair_labels = [labels[i] for i in pair_ids]

    n_methods = len(data)
    fig, axes = plt.subplots(
        1, n_methods, figsize=(6 * n_methods, 6), squeeze=False
    )

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


def plot_tsne(
    embeddings: np.ndarray,
    labels_arr: np.ndarray,
    id2label: List[str],
    label2id: Dict[str, int],
    save_path: Path,
    n_samples_per_class: int = 60,
) -> None:
    """Project test embeddings of the confusable pairs to 2-D via t-SNE.

    Only examples belonging to the confusable pairs are kept, so the plot
    shows whether semantically overlapping intents are separated in the
    representation space.
    """
    pair_ids = sorted({label2id[a] for a, _ in CONFUSABLE_PAIRS} |
                      {label2id[b] for _, b in CONFUSABLE_PAIRS})

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


if __name__ == "__main__":
    # Generate visualisations for the full-data setting across all methods.
    id2label, label2id = load_label_map()
    tags = ["bert_ft", "supcon", "supcon_hn"]

    data = {}
    for tag in tags:
        try:
            data[tag] = load_analysis(tag, "full")
        except FileNotFoundError:
            print(f"[skip] {tag}: analysis artefacts not found")
    if not data:
        raise SystemExit("No analysis artefacts found; run experiments first.")

    # Full confusion matrix for each method.
    for tag, d in data.items():
        plot_confusion_matrix(
            d["y_true"], d["y_pred"], id2label, f"{tag} (full)",
            FIG_DIR / f"confusion_{tag}_full.png",
        )

    # Focused confusable-pair view.
    plot_confusable_pairs(data, id2label, label2id, FIG_DIR / "confusable_pairs_full.png")

    # t-SNE for methods that persisted embeddings.
    for tag, d in data.items():
        if d.get("emb") is None:
            continue
        plot_tsne(d["emb"], d["y_true"], id2label, label2id, FIG_DIR / f"tsne_{tag}_full.png")

    print(f"[analysis] figures written to {FIG_DIR}")
