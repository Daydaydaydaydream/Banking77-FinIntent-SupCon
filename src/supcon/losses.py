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

    log_prob = similarity - torch.logsumexp(similarity, dim=1, keepdim=True)

    # ``where`` rather than a multiplication by a 0/1 mask: the diagonal of
    # ``log_prob`` is -inf, and (-inf) * 0 is nan rather than 0.
    positive_log_prob = torch.where(positives, log_prob, torch.zeros_like(log_prob)).sum(dim=1)
    positive_count = positives.sum(dim=1).clamp(min=1)

    if hard_negative_weights is not None:
        weights = hard_negative_weights.to(device=device, dtype=torch.float32)
        weights = weights / weights.sum(dim=1, keepdim=True).clamp(min=1e-12)
        # Re-weighting a log-sum-exp denominator: subtract the log of the
        # weighted partition so the result stays a proper log-probability.
        weighted = torch.log((torch.exp(similarity.masked_fill(self_mask, 0.0)) * weights).sum(dim=1, keepdim=True))
        log_prob = similarity - weighted

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
    The returned matrix has ones on the diagonal and on positive pairs, so it
    leaves the numerator of the loss untouched and only reshapes the
    denominator.  With a large ``temperature`` the weights flatten towards
    uniform, recovering standard SupCon.
    """

    if temperature <= 0:
        raise ValueError("temperature must be positive")
    device = similarity.device
    labels = labels.to(device)
    batch_size = labels.shape[0]

    cosine = similarity.float().clamp(min=-1.0, max=1.0)
    is_negative = labels[:, None] != labels[None, :]
    identity = torch.eye(batch_size, dtype=torch.bool, device=device)

    weights = torch.exp(cosine / temperature) * is_negative
    weights = weights.masked_fill(identity, 0.0)
    return weights