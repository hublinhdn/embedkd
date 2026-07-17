"""Embedding head: pooling + projection + L2 normalisation.

GeM pooling deliberately uses ``adaptive_avg_pool2d((1, 1))`` rather than
``avg_pool2d`` with a dynamic kernel: the latter breaks ONNX export for
variable input sizes.
"""

from __future__ import annotations

import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F


class GeM(nn.Module):
    """Generalised-mean pooling (Radenovic et al., TPAMI 2019). ONNX-safe."""

    def __init__(self, p: float = 3.0, trainable: bool = False, eps: float = 1e-6) -> None:
        super().__init__()
        if trainable:
            self.p = nn.Parameter(torch.tensor(float(p)))
        else:
            self.register_buffer("p", torch.tensor(float(p)))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.clamp(min=self.eps).pow(self.p)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        return x.pow(1.0 / self.p).flatten(1)


class EmbedHead(nn.Module):
    """Pool backbone features to a fixed-size, optionally L2-normalised embedding.

    A BatchNorm1d "neck" sits between projection and normalisation (the BNNeck
    of Luo et al., 2019): it decouples the classification logits from the
    metric-learning embedding and measurably stabilises joint training.
    """

    def __init__(
        self,
        in_dim: int,
        embed_dim: int,
        pooling: str = "gem",
        gem_p: float = 3.0,
        gem_p_trainable: bool = False,
        normalize: bool = True,
    ) -> None:
        super().__init__()
        if pooling not in ("gem", "gap", "token_mean"):
            raise ValueError(f"Unknown pooling '{pooling}' (use gem | gap | token_mean)")
        self.in_dim = in_dim
        self.pooling = pooling
        self.normalize = normalize
        self.gem = GeM(gem_p, gem_p_trainable) if pooling == "gem" else None
        self.proj = nn.Linear(in_dim, embed_dim, bias=False)
        self.bn_neck = nn.BatchNorm1d(embed_dim)
        self._warned_token_fallback = False

    def _pool(self, feats: torch.Tensor) -> torch.Tensor:
        if feats.dim() == 4:
            # Some transformer families (e.g. swin) emit BHWC; detect and fix.
            if feats.shape[1] != self.in_dim and feats.shape[-1] == self.in_dim:
                feats = feats.permute(0, 3, 1, 2).contiguous()
            if self.pooling == "gem":
                return self.gem(feats)
            return F.adaptive_avg_pool2d(feats, (1, 1)).flatten(1)
        if feats.dim() == 3:  # transformer tokens (B, N, C)
            if self.pooling == "gem" and not self._warned_token_fallback:
                warnings.warn(
                    "GeM pooling is undefined on token sequences; falling back to "
                    "token mean pooling for this backbone.",
                    stacklevel=2,
                )
                self._warned_token_fallback = True
            return feats.mean(dim=1)
        if feats.dim() == 2:  # backbone already pooled
            return feats
        raise ValueError(f"Unsupported feature shape {tuple(feats.shape)}")

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        emb = self.bn_neck(self.proj(self._pool(feats)))
        if self.normalize:
            emb = F.normalize(emb, dim=-1)
        return emb


class CosineClassifier(nn.Module):
    """Normalised-softmax classifier: cos(embedding, class weight) * scale.

    A plain Linear on an L2-normalised embedding produces bounded, weakly
    separated logits; scaling the cosine (s ~ 64) restores usable softmax
    dynamics. Also the logits source for KL distillation.
    """

    def __init__(self, embed_dim: int, num_classes: int, scale: float = 64.0) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_classes, embed_dim))
        nn.init.xavier_uniform_(self.weight)
        self.scale = float(scale)

    def forward(self, emb: torch.Tensor) -> torch.Tensor:
        cos = F.normalize(emb, dim=-1) @ F.normalize(self.weight, dim=-1).t()
        return cos * self.scale


class EmbeddingModel(nn.Module):
    """Backbone + EmbedHead (+ optional cosine classifier for sce / kl)."""

    def __init__(self, backbone: nn.Module, head: EmbedHead, num_classes: int | None = None,
                 logit_scale: float = 64.0) -> None:
        super().__init__()
        self.backbone = backbone
        self.head = head
        self.classifier = (
            CosineClassifier(head.proj.out_features, num_classes, logit_scale)
            if num_classes else None
        )

    @property
    def embed_dim(self) -> int:
        return self.head.proj.out_features

    def forward(self, x: torch.Tensor, return_logits: bool = False):
        emb = self.head(self.backbone(x))
        if return_logits:
            logits = self.classifier(emb) if self.classifier is not None else None
            return emb, logits
        return emb
