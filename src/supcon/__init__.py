"""Supervised contrastive learning (SupCon) package.

Implements the supervised contrastive objective of Khosla et al. (2020)
together with a hard-negative weighting variant, a BERT-based model with a
projection head, and a two-stage training routine.
"""

from supcon.losses import SupConLoss
from supcon.model import SupConModel

__all__ = ["SupConLoss", "SupConModel"]
