"""Unit tests for the supervised contrastive loss and hard-negative weighting."""

from __future__ import annotations

import unittest

import torch
import torch.nn.functional as F

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from supcon.losses import hard_negative_weight_matrix, supervised_contrastive_loss


def _unit(vectors: torch.Tensor) -> torch.Tensor:
    return F.normalize(vectors, dim=1)


class SupervisedContrastiveLossTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)
        self.labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
        self.random = _unit(torch.randn(8, 16))

    def test_loss_is_finite_with_balanced_batch(self) -> None:
        loss = supervised_contrastive_loss(self.random, self.labels, temperature=0.1)
        self.assertTrue(bool(torch.isfinite(loss)))

    def test_perfectly_clustered_embeddings_get_lower_loss(self) -> None:
        centres = torch.zeros(8, 16)
        for row, label in enumerate(self.labels):
            centres[row, int(label)] = 10.0
        clustered = _unit(centres + torch.randn(8, 16) * 0.01)
        clustered_loss = supervised_contrastive_loss(clustered, self.labels, temperature=0.1)
        random_loss = supervised_contrastive_loss(self.random, self.labels, temperature=0.1)
        self.assertLess(float(clustered_loss), float(random_loss))

    def test_anchors_without_positives_do_not_produce_nan(self) -> None:
        """Unbalanced batches are common; the loss must stay finite, not nan.

        A zero-mask multiplied by the -inf diagonal yields nan, which is the
        failure this test guards against.
        """

        all_distinct = torch.arange(8)
        loss = supervised_contrastive_loss(self.random, all_distinct, temperature=0.1)
        self.assertTrue(bool(torch.isfinite(loss)))
        self.assertEqual(float(loss), 0.0)

    def test_low_temperature_does_not_overflow(self) -> None:
        loss = supervised_contrastive_loss(self.random, self.labels, temperature=0.01)
        self.assertTrue(bool(torch.isfinite(loss)))

    def test_backward_produces_finite_gradients(self) -> None:
        # ``_unit`` returns a new tensor, so gradients land on the leaf only if
        # normalisation happens after the leaf is created.
        raw = torch.randn(8, 16, requires_grad=True)
        embeddings = F.normalize(raw, dim=1)
        supervised_contrastive_loss(embeddings, self.labels, temperature=0.1).backward()
        self.assertIsNotNone(raw.grad)
        self.assertTrue(bool(torch.isfinite(raw.grad).all()))

    def test_rejects_non_positive_temperature(self) -> None:
        with self.assertRaises(ValueError):
            supervised_contrastive_loss(self.random, self.labels, temperature=0.0)

    def test_rejects_mismatched_batch_dimensions(self) -> None:
        with self.assertRaises(ValueError):
            supervised_contrastive_loss(self.random, torch.tensor([0, 1]), temperature=0.1)

    def test_is_minimised_by_pulling_towards_the_positive(self) -> None:
        """A third, different-label sample must exist for this to be meaningful.

        With only two same-label samples there are no negatives at all, so the
        denominator is a single term and the loss collapses to zero no matter
        how the positive is placed.
        """

        anchor = torch.tensor([[1.0, 0.0]])
        negative = torch.tensor([[0.0, 1.0]])
        labels = torch.tensor([0, 0, 1])

        near = torch.cat([anchor, anchor * 0.9 + negative * 0.1, negative])
        far = torch.cat([anchor, anchor * 0.5 + negative * 0.5, negative])
        near_loss = supervised_contrastive_loss(_unit(near), labels, temperature=0.1)
        far_loss = supervised_contrastive_loss(_unit(far), labels, temperature=0.1)
        self.assertLess(float(near_loss), float(far_loss))


class HardNegativeWeightTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)
        self.labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
        self.embeddings = _unit(torch.randn(8, 16))
        self.similarity = self.embeddings @ self.embeddings.t()

    def test_diagonal_and_positives_are_excluded(self) -> None:
        weights = hard_negative_weight_matrix(self.similarity, self.labels, temperature=1.0)
        self.assertEqual(float(torch.diagonal(weights).sum()), 0.0)
        # Pairs with the same label must receive zero weight.
        same_label = self.labels[:, None] == self.labels[None, :]
        self.assertEqual(float(weights[same_label].sum()), 0.0)

    def test_high_temperature_approaches_uniform_weighting(self) -> None:
        """The ablation depends on a uniform limit recovering standard SupCon."""

        cold = hard_negative_weight_matrix(self.similarity, self.labels, temperature=0.1)
        hot = hard_negative_weight_matrix(self.similarity, self.labels, temperature=50.0)
        num_negatives = float((self.labels[:, None] != self.labels[None, :]).sum()) / self.labels.numel()
        self.assertAlmostEqual(
            float(hot.sum()) / self.labels.numel(), num_negatives, delta=0.05
        )
        self.assertGreater(float(cold.max()), float(hot.max()) * 5)

    def test_weights_are_non_negative_and_finite(self) -> None:
        weights = hard_negative_weight_matrix(self.similarity, self.labels, temperature=0.5)
        self.assertTrue(bool((weights >= 0).all()))
        self.assertTrue(bool(torch.isfinite(weights).all()))

    def test_rejects_non_positive_temperature(self) -> None:
        with self.assertRaises(ValueError):
            hard_negative_weight_matrix(self.similarity, self.labels, temperature=0.0)


if __name__ == "__main__":
    unittest.main()