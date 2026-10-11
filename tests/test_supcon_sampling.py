"""Tests for the class-balanced sampler used by supervised contrastive learning."""

from __future__ import annotations

import collections
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from supcon.model import BalancedBatchSampler


def _few_shot_labels(shots: int, num_classes: int = 77) -> np.ndarray:
    return np.repeat(np.arange(num_classes), shots)


class BalancedBatchSamplerTest(unittest.TestCase):
    def test_every_anchor_has_a_same_class_positive(self) -> None:
        """The regression this guards: one sample per class gives loss 0.

        With 77 classes and a batch sized to the class count, the naive
        ``batch_size // num_classes`` rule yields exactly one sample per class,
        so no anchor has a positive and the supervised contrastive loss is
        identically zero while the encoder receives no gradient.
        """

        labels = _few_shot_labels(5)
        sampler = BalancedBatchSampler(
            labels, batch_size=77, num_classes=77, shots_per_class=2,
            generator=torch.Generator().manual_seed(42),
        )
        batch = next(iter(sampler))
        counts = collections.Counter(labels[batch])
        self.assertEqual(set(counts.values()), {2})
        # Each of the 77 classes contributes 2 samples; one acts as the anchor
        # and the other is its single positive.
        anchors_with_positive = sum(count - 1 for count in counts.values())
        self.assertEqual(anchors_with_positive, 77)

    def test_single_sample_per_class_is_rejected(self) -> None:
        labels = _few_shot_labels(5)
        with self.assertRaises(ValueError) as ctx:
            BalancedBatchSampler(labels, batch_size=77, num_classes=77, shots_per_class=1)
        self.assertIn("no anchor", str(ctx.exception))

    def test_class_with_too_few_examples_is_rejected(self) -> None:
        labels = _few_shot_labels(5)
        labels = np.where(labels == 3, 0, labels)
        with self.assertRaises(ValueError):
            BalancedBatchSampler(labels, batch_size=77, num_classes=77, shots_per_class=2)

    def test_batches_stay_balanced_and_within_range(self) -> None:
        labels = _few_shot_labels(20)
        sampler = BalancedBatchSampler(
            labels, batch_size=77, num_classes=77, shots_per_class=2,
            generator=torch.Generator().manual_seed(7),
        )
        for batch in sampler:
            counts = collections.Counter(labels[batch])
            self.assertEqual(len(counts), 77)
            self.assertEqual(set(counts.values()), {2})
            self.assertEqual(len(batch), sampler.effective_batch_size)

    def test_sampler_is_deterministic_for_a_fixed_seed(self) -> None:
        labels = _few_shot_labels(10)
        kwargs = dict(batch_size=77, num_classes=77, shots_per_class=2)
        first = list(BalancedBatchSampler(labels, generator=torch.Generator().manual_seed(3), **kwargs))
        second = list(BalancedBatchSampler(labels, generator=torch.Generator().manual_seed(3), **kwargs))
        self.assertEqual(first, second)

    def test_different_seeds_give_different_orders(self) -> None:
        labels = _few_shot_labels(10)
        kwargs = dict(batch_size=77, num_classes=77, shots_per_class=2)
        first = list(BalancedBatchSampler(labels, generator=torch.Generator().manual_seed(1), **kwargs))
        second = list(BalancedBatchSampler(labels, generator=torch.Generator().manual_seed(2), **kwargs))
        self.assertNotEqual(first, second)

    def test_epoch_length_reflects_the_largest_class(self) -> None:
        labels = _few_shot_labels(5)
        sampler = BalancedBatchSampler(
            labels, batch_size=77, num_classes=77, shots_per_class=2,
            generator=torch.Generator().manual_seed(42),
        )
        self.assertEqual(len(sampler), 2)
        self.assertGreater(len(sampler), 0)

    def test_indices_stay_in_range(self) -> None:
        labels = _few_shot_labels(5)
        sampler = BalancedBatchSampler(
            labels, batch_size=77, num_classes=77, shots_per_class=2,
            generator=torch.Generator().manual_seed(42),
        )
        for batch in sampler:
            for index in batch:
                self.assertGreaterEqual(index, 0)
                self.assertLess(index, len(labels))


if __name__ == "__main__":
    unittest.main()