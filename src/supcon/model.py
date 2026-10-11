"""Encoder, projection head and class-balanced batching for SupCon."""

from __future__ import annotations

from typing import Any, Iterator

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


class SupConModel(nn.Module):
    """BERT encoder with a projection head and a linear classifier.

    The projection head exists only during the contrastive stage and is
    discarded afterwards; the classifier head is what produces predictions.
    Keeping both in one module lets the contrastive phase export an encoder
    that the classification phase can reuse without re-downloading anything.
    """

    def __init__(
        self,
        model_name: str,
        num_labels: int,
        *,
        projection_dim: int = 128,
        local_files_only: bool = False,
        attn_implementation: str = "eager",
    ) -> None:
        super().__init__()
        from transformers import AutoModel, AutoModelForSequenceClassification

        self.encoder = AutoModel.from_pretrained(
            model_name,
            local_files_only=local_files_only,
            attn_implementation=attn_implementation,
        )
        hidden_size = self.encoder.config.hidden_size
        self.projection = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, projection_dim),
        )
        self.classifier = nn.Linear(hidden_size, num_labels)

    def embed(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        output = self.encoder(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        # Mean pooling with the attention mask: [CLS] is not trained for this
        # task and mean pooling is more stable in the few-shot regime.
        hidden = output.last_hidden_state
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        summed = (hidden * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-6)
        return summed / counts

    def project(self, pooled: torch.Tensor) -> torch.Tensor:
        return self.projection(pooled)

    def classify(self, pooled: torch.Tensor) -> torch.Tensor:
        return self.classifier(pooled)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, **kwargs: Any):
        pooled = self.embed(input_ids, attention_mask, **kwargs)
        return self.project(pooled), self.classify(pooled)


class TextDataset(Dataset):
    def __init__(self, encodings: dict[str, Any], labels: np.ndarray) -> None:
        self.encodings = encodings
        self.labels = np.asarray(labels, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = {key: value[index] for key, value in self.encodings.items()}
        item["labels"] = int(self.labels[index])
        return item


class BalancedBatchSampler(torch.utils.data.Sampler[list[int]]):
    """Yield batches in which every class is represented equally.

    Supervised contrastive learning needs each anchor to have at least one
    positive in its batch.  Uniform shuffling makes that unlikely for 77
    classes, so each batch is built from whole per-class groups.

    ``shots_per_class`` must be at least 2.  With ``batch_size == num_classes``
    the natural reading is one sample per class, which leaves *no* anchor with
    a positive and makes the loss identically zero - the model trains on
    nothing.  The sampler therefore uses two samples per class by default, so
    every anchor in a 77-class batch has exactly one positive.
    """

    def __init__(
        self,
        labels: np.ndarray,
        batch_size: int,
        num_classes: int,
        shots_per_class: int = 2,
        generator: torch.Generator | None = None,
    ) -> None:
        self.labels = np.asarray(labels)
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.generator = generator
        if shots_per_class < 2:
            raise ValueError(
                "shots_per_class must be >= 2; with one sample per class no anchor "
                "has a positive and the supervised contrastive loss is zero"
            )
        self.per_class = max(2, shots_per_class)
        self.effective_batch_size = self.per_class * num_classes
        self.class_indices: list[np.ndarray] = [
            np.flatnonzero(self.labels == cls) for cls in range(num_classes)
        ]
        too_small = [
            int(cls) for cls, idx in enumerate(self.class_indices) if len(idx) < self.per_class
        ]
        if too_small:
            raise ValueError(
                f"classes with fewer than {self.per_class} training examples: {too_small}"
            )
        # Enough steps for every example to appear at least once per epoch.
        largest = max(len(idx) for idx in self.class_indices)
        self.epoch_size = largest
        self.steps_per_epoch = max(1, largest // self.per_class)

    def __len__(self) -> int:
        return self.steps_per_epoch

    def __iter__(self) -> Iterator[list[int]]:
        seed = int(self.generator.initial_seed()) if self.generator is not None else 0
        rng = np.random.default_rng(seed)
        offsets = [rng.permutation(idx) for idx in self.class_indices]
        cursors = [0] * self.num_classes
        for _ in range(self.steps_per_epoch):
            batch: list[int] = []
            for cls in range(self.num_classes):
                pool = offsets[cls]
                if cursors[cls] + self.per_class > len(pool):
                    offsets[cls] = rng.permutation(self.class_indices[cls])
                    pool = offsets[cls]
                    cursors[cls] = 0
                batch.extend(int(pool[pos]) for pos in range(cursors[cls], cursors[cls] + self.per_class))
                cursors[cls] += self.per_class
            yield batch


def collate_with_padding(tokenizer: Any):
    def collate(features: list[dict[str, Any]]) -> dict[str, Any]:
        labels = [int(feature.pop("labels")) for feature in features]
        batch = tokenizer.pad(features, padding=True, return_tensors="pt")
        batch["labels"] = torch.tensor(labels, dtype=torch.long)
        return batch

    return collate


def build_loader(
    dataset: Dataset,
    tokenizer: Any,
    *,
    batch_size: int,
    num_classes: int,
    seed: int,
    shuffle: bool,
    shots_per_class: int = 2,
) -> DataLoader:
    collate = collate_with_padding(tokenizer)
    generator = torch.Generator()
    generator.manual_seed(seed)
    if shuffle:
        sampler: Any = BalancedBatchSampler(
            np.asarray([dataset[i]["labels"] for i in range(len(dataset))]),
            batch_size=batch_size,
            num_classes=num_classes,
            shots_per_class=shots_per_class,
            generator=generator,
        )
        return DataLoader(
            dataset,
            batch_sampler=sampler,
            collate_fn=collate,
            num_workers=0,
            generator=generator,
        )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate,
        num_workers=0,
    )