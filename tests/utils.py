"""Shared test helpers: a tiny CNN backbone so unit tests avoid timm overhead."""

from __future__ import annotations

import torch.nn as nn


class TinyBackbone(nn.Module):
    """Minimal conv feature extractor with the same contract as timm backbones."""

    num_features = 16

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 8, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, self.num_features, 3, stride=2, padding=1),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


def tiny_embedding_model(embed_dim: int = 8, num_classes: int | None = 4, pooling: str = "gem"):
    from embedkd.models import EmbedHead, EmbeddingModel

    head = EmbedHead(TinyBackbone.num_features, embed_dim, pooling=pooling)
    return EmbeddingModel(TinyBackbone(), head, num_classes)
