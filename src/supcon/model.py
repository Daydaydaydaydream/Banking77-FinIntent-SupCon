"""SupCon model: a BERT encoder with a projection head and a classification
head.

The encoder produces a pooled ``[CLS]`` representation ``h``.  The projection
head maps ``h`` to a lower-dimensional, L2-normalised contrastive embedding
``z`` used during the SupCon stage.  The classification head maps the frozen
encoder representation to intent logits during the linear-probing stage.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel


class ProjectionHead(nn.Module):
    """Two-layer MLP producing the contrastive embedding ``z``."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ClassificationHead(nn.Module):
    """Classification head mapping the pooled representation to logits."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if hidden_dim is not None:
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes),
            )
        else:
            self.net = nn.Linear(input_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SupConModel(nn.Module):
    """BERT encoder plus projection and classification heads.

    Parameters
    ----------
    model_name:
        HuggingFace model identifier for the encoder backbone.
    num_classes:
        Number of intent classes.
    proj_dim:
        Dimensionality of the contrastive embedding ``z``.
    proj_hidden:
        Hidden size of the projection MLP (defaults to the encoder size).
    classifier_hidden:
        Optional hidden size for a two-layer classification head; ``None``
        yields a linear head.
    """

    def __init__(
        self,
        model_name: str,
        num_classes: int,
        proj_dim: int = 128,
        proj_hidden: Optional[int] = None,
        classifier_hidden: Optional[int] = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        config = AutoConfig.from_pretrained(model_name)
        hidden_size = config.hidden_size
        proj_hidden = proj_hidden or hidden_size

        self.encoder = AutoModel.from_pretrained(model_name)
        self.projection_head = ProjectionHead(hidden_size, proj_hidden, proj_dim, dropout)
        self.classification_head = ClassificationHead(
            hidden_size, num_classes, classifier_hidden, dropout
        )
        self.hidden_size = hidden_size

    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Return the pooled ``[CLS]`` representation ``h`` (``B, H``)."""
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state[:, 0, :]

    def project(self, h: torch.Tensor) -> torch.Tensor:
        """Map ``h`` to the L2-normalised contrastive embedding ``z``."""
        return torch.nn.functional.normalize(self.projection_head(h), dim=1)

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(h, z, logits)`` for a batch of tokenised inputs."""
        h = self.encode(input_ids, attention_mask)
        z = self.project(h)
        logits = self.classification_head(h)
        return h, z, logits

    def freeze_encoder(self, freeze: bool = True) -> None:
        """Freeze or unfreeze the encoder backbone."""
        for param in self.encoder.parameters():
            param.requires_grad = not freeze

    def reset_classification_head(self) -> None:
        """Reinitialise the classification head weights."""
        for layer in self.classification_head.modules():
            if isinstance(layer, nn.Linear):
                layer.reset_parameters()
