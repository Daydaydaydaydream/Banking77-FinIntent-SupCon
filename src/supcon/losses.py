"""Numerically stable losses for supervised contrastive learning."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def supervised_contrastive_loss(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 0.1,
    hard_negative_weights: torch.Tensor | None = None,
    hard_negative_temperature: float = 1.0,
) -> torch.Tensor:
    """Supervised contrastive loss over a batch.

    ``embeddings`` is expected to be L2-normalised.  For every anchor ``i`` the
    loss pulls every same-label sample ``p in P(i)`` closer and pushes every
    different-label sample away:

    .. math::

        L_i = -\\frac{1}{|P(i)|} \\sum_{p \\in P(i)}
        \\log \\frac{\\exp(z_i \\cdot z_p / \\tau)}
        {\\sum_{a \\ne i} \\exp(z_i \\cdot z_a / \\tau)}

    Implementation notes
    --------------------
    * The similarity matrix is computed in float32 even when the encoder runs in
      bfloat16.  ``exp`` of a low-temperature logit can exceed the float16
      range, which is the same overflow that collapsed the BERT baseline.
    * The log-sum-exp denominator is taken over the full matrix minus the
      self-similarity, so the result never forms ``log(0)``.
    * Anchors without a positive in the batch are masked out and contribute
      nothing; with a class-balanced batch sampler this should not happen, but
      masking keeps a malformed batch from producing ``nan`` silently.

    ``hard_negative_weights`` optionally re-weights the denominator per anchor
    column.  Passing all-ones (or ``None``) recovers standard SupCon exactly,
    which is what the ablation relies on.
    """

    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if embeddings.shape[0] != labels.shape[0]:
        raise ValueError("embeddings and labels must have the same batch dimension")

    device = embeddings.device
    features = F.normalize(embeddings.float(), dim=1)
    labels = labels.to(device)

    similarity = features @ features.t() / temperature
    batch_size = labels.shape[0]

    # Mask the diagonal: an anchor is never its own positive or negative.
    self_mask = torch.eye(batch_size, dtype=torch.bool, device=device)
    similarity = similarity.masked_fill(self_mask, float("-inf"))

    positives = (labels[:, None] == labels[None, :]) & ~self_mask
    has_positive = positives.any(dim=1)

    denominator_logits = similarity
    if hard_negative_weights is not None:
        weights = hard_negative_weights.detach().to(device=device, dtype=torch.float32)
        if weights.shape != similarity.shape or not torch.isfinite(weights).all() or (weights < 0).any():
            raise ValueError("hard-negative weights must be finite, nonnegative and batch-square")
        negatives = labels[:, None] != labels[None, :]
        # Only negative terms are weighted; positive terms retain weight one.
        effective = torch.where(negatives, weights, torch.ones_like(weights))
        denominator_logits = similarity + effective.clamp_min(1e-30).log()
        denominator_logits = denominator_logits.masked_fill(negatives & (weights == 0), float("-inf"))
    log_prob = similarity - torch.logsumexp(denominator_logits, dim=1, keepdim=True)
    positive_log_prob = torch.where(positives, log_prob, torch.zeros_like(log_prob)).sum(dim=1)
    positive_count = positives.sum(dim=1).clamp(min=1)

    anchor_loss = -(positive_log_prob / positive_count)
    anchor_loss = torch.where(has_positive, anchor_loss, torch.zeros_like(anchor_loss))
    return anchor_loss.sum() / batch_size


def hard_negative_weight_matrix(
    similarity: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Weight each negative in proportion to its similarity to the anchor.

    ``similarity`` is a cosine-similarity matrix (before temperature scaling).
    The returned matrix has zeros on diagonal and positive pairs. The loss
    applies weights only to negatives, preserving positive terms. Weights
    are detached and normalised to mean one over negatives in each row.  With a large ``temperature`` the weights flatten towards
    uniform, recovering standard SupCon.
    """

    if temperature <= 0:
        raise ValueError("temperature must be positive")
    device = similarity.device
    labels = labels.to(device)
    batch_size = labels.shape[0]

    cosine = similarity.detach().float().clamp(min=-1.0, max=1.0)
    is_negative = labels[:, None] != labels[None, :]
    counts = is_negative.sum(dim=1, keepdim=True)
    logits = (cosine / temperature).masked_fill(~is_negative, float("-inf"))
    logits = torch.where(counts > 0, logits, torch.zeros_like(logits))
    # Mean negative weight is one, preserving the total negative mass.
    weights = torch.softmax(logits, dim=1) * counts
    return weights.masked_fill(~is_negative, 0.0)
