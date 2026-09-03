"""Supervised contrastive loss (Khosla et al., NeurIPS 2020).

The supervised contrastive objective pulls representations of the same class
together and pushes representations of different classes apart within a
batch.  Given a batch of normalised embeddings ``z`` and their labels, the
loss for anchor ``i`` is

    L_i = - (1 / |P(i)|) * sum_{p in P(i)}
              log( exp(z_i . z_p / tau) / sum_{a in A(i)} exp(z_i . z_a / tau) )

where ``P(i)`` is the set of same-class positives and ``A(i)`` is the set of
all other samples in the batch.

The ``hard_negative_weighting`` variant replaces the uniform denominator over
negatives with a similarity-weighted one, so that confusable cross-class
pairs (e.g. ``declined_card_payment`` vs ``declined_transfer``) receive a
larger penalty.  Concretely, each negative ``n`` of anchor ``i`` is weighted
by ``w_{i,n} = softmax_n(z_i . z_n / tau_hn) * |N(i)|``; as ``tau_hn -> inf``
the weights become uniform and the objective reduces to standard SupCon.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SupConLoss(nn.Module):
    """Supervised contrastive loss with optional hard-negative weighting.

    Parameters
    ----------
    temperature:
        Scalar temperature ``tau`` applied to the similarity logits.
    hard_negative_weighting:
        When ``True``, negatives are re-weighted by their similarity to the
        anchor so that hard negatives dominate the denominator.
    hard_negative_temperature:
        Temperature ``tau_hn`` controlling how sharply the negative weights
        concentrate on the hardest negatives.  Smaller values focus more
        strongly on the hardest negatives; larger values approach uniform
        weighting (i.e. standard SupCon).
    """

    def __init__(
        self,
        temperature: float = 0.07,
        hard_negative_weighting: bool = False,
        hard_negative_temperature: float = 0.5,
    ) -> None:
        super().__init__()
        self.temperature = temperature
        self.hard_negative_weighting = hard_negative_weighting
        self.hard_negative_temperature = hard_negative_temperature

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute the supervised contrastive loss.

        Parameters
        ----------
        features:
            Embedding matrix of shape ``(B, D)``.
        labels:
            Integer labels of shape ``(B,)``.

        Returns
        -------
        torch.Tensor
            A scalar loss averaged over anchors.
        """
        device = features.device
        batch_size = features.shape[0]
        labels = labels.contiguous().view(-1, 1)

        if labels.shape[0] != batch_size:
            raise ValueError("Number of labels does not match number of features")

        # L2-normalise embeddings so the dot product is a cosine similarity.
        features = F.normalize(features, dim=1)

        # Similarity logits: logits[i, j] = z_i . z_j / tau.
        logits = torch.matmul(features, features.T) / self.temperature

        # Masks over the (B, B) pair matrix.
        same_class = torch.eq(labels, labels.T).float().to(device)
        self_mask = torch.eye(batch_size, device=device)
        pos_mask = same_class * (1.0 - self_mask)          # positives, excl. self
        neg_mask = 1.0 - same_class                        # negatives (self is same class)

        exp_logits = torch.exp(logits)

        if self.hard_negative_weighting:
            # Similarity-based weight for every negative pair.
            hn_scores = torch.matmul(features, features.T) / self.hard_negative_temperature
            hn_scores = hn_scores.masked_fill(neg_mask == 0, float("-inf"))
            neg_weights = torch.softmax(hn_scores, dim=1)  # sums to 1 over negatives

            # Rescale so the total negative mass equals |N(i)| (uniform case).
            num_neg = neg_mask.sum(dim=1, keepdim=True).clamp(min=1.0)
            neg_weights = neg_weights * num_neg

            denom = (exp_logits * pos_mask).sum(dim=1, keepdim=True) + (
                exp_logits * neg_weights * neg_mask
            ).sum(dim=1, keepdim=True)
        else:
            denom = (exp_logits * (1.0 - self_mask)).sum(dim=1, keepdim=True)

        log_prob = logits - torch.log(denom + 1e-8)

        # Mean log-likelihood over positives.
        pos_count = pos_mask.sum(dim=1).clamp(min=1.0)
        mean_log_prob_pos = (pos_mask * log_prob).sum(dim=1) / pos_count

        return -mean_log_prob_pos.mean()
