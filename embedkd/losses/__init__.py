"""Metric-learning task losses applied to the student's embedding (and logits).

These train the retrieval task itself; distillation objectives live in
:mod:`embedkd.objectives`.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..registry import registry


class TaskLoss(nn.Module):
    """Base class. forward(emb, logits, labels) -> scalar."""

    def forward(self, emb: torch.Tensor, logits: torch.Tensor | None, labels: torch.Tensor):
        raise NotImplementedError


@registry.task_loss("sce")
class SoftmaxCrossEntropy(TaskLoss):
    """Cross-entropy on the model's classifier logits.

    Label smoothing tempers class-specific overfitting, which directly hurts
    open-set retrieval generalisation.
    """

    def __init__(self, label_smoothing: float = 0.0):
        super().__init__()
        self.label_smoothing = float(label_smoothing)

    def forward(self, emb, logits, labels):
        if logits is None:
            raise ValueError(
                "'sce' needs classifier logits; the model was built without a "
                "classifier head (num_classes unknown)."
            )
        return F.cross_entropy(logits, labels, label_smoothing=self.label_smoothing)


@registry.task_loss("arcface")
class ArcFace(TaskLoss):
    """Additive angular margin loss (Deng et al., CVPR 2019).

    Numerically stable formulation (cos_m / sin_m expansion with the
    theta + m > pi fallback) instead of acos, as proven in the authors' prior
    training code. Defaults follow that code (margin 0.35, scale 64); for
    low-capacity students a lighter margin (0.20-0.25) avoids underfitting.
    """

    def __init__(self, embed_dim: int, num_classes: int, margin: float = 0.35, scale: float = 64.0):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_classes, embed_dim))
        nn.init.xavier_uniform_(self.weight)
        self.margin = float(margin)
        self.scale = float(scale)
        self.cos_m = math.cos(self.margin)
        self.sin_m = math.sin(self.margin)
        self.th = math.cos(math.pi - self.margin)
        self.mm = math.sin(math.pi - self.margin) * self.margin

    def forward(self, emb, logits, labels):
        cos = F.normalize(emb, dim=-1) @ F.normalize(self.weight, dim=-1).t()
        cos = cos.clamp(-1.0 + 1e-7, 1.0 - 1e-7)
        sine = torch.sqrt((1.0 - cos.pow(2)).clamp(min=1e-12))
        phi = cos * self.cos_m - sine * self.sin_m
        phi = torch.where(cos > self.th, phi, cos - self.mm)
        one_hot = F.one_hot(labels, cos.shape[1]).to(cos.dtype)
        logits_margin = (one_hot * phi + (1.0 - one_hot) * cos) * self.scale
        return F.cross_entropy(logits_margin, labels)


def _pairwise_distances(e: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    # NaN-safe: clamp before sqrt (infinite derivative at 0) and zero the
    # diagonal by mask multiplication; see embedkd.objectives._pairwise_distances.
    prod = e @ e.t()
    sq = prod.diagonal()
    d2 = (sq.unsqueeze(0) + sq.unsqueeze(1) - 2.0 * prod).clamp(min=eps)
    off_diag = 1.0 - torch.eye(e.shape[0], device=e.device, dtype=e.dtype)
    return d2.sqrt() * off_diag


@registry.task_loss("triplet")
class BatchHardTriplet(TaskLoss):
    """Triplet loss with batch-hard mining (Hermans et al., 2017).

    Requires the PK sampler so every anchor has at least one in-batch positive.
    """

    def __init__(self, embed_dim: int = 0, num_classes: int = 0, margin: float = 0.2,
                 mining: str = "batch_hard"):
        super().__init__()
        if mining != "batch_hard":
            raise ValueError("Only 'batch_hard' mining is implemented in v0.1")
        self.margin = float(margin)

    def forward(self, emb, logits, labels):
        dist = _pairwise_distances(emb)
        same = labels.unsqueeze(0) == labels.unsqueeze(1)
        eye = torch.eye(len(labels), dtype=torch.bool, device=labels.device)
        pos_mask = same & ~eye
        neg_mask = ~same
        valid = pos_mask.any(dim=1) & neg_mask.any(dim=1)
        if not valid.any():
            return emb.new_zeros(())
        hardest_pos = torch.where(pos_mask, dist, dist.new_full(dist.shape, -torch.inf)).max(dim=1).values
        hardest_neg = torch.where(neg_mask, dist, dist.new_full(dist.shape, torch.inf)).min(dim=1).values
        loss = F.relu(hardest_pos - hardest_neg + self.margin)
        return loss[valid].mean()


@registry.task_loss("contrastive")
class Contrastive(TaskLoss):
    """Pairwise contrastive loss (Hadsell et al., 2006) over all in-batch pairs.

    Squared form; ``margin`` is the negative-pair margin (positive margin 0),
    matching the configuration proven in the authors' prior experiments.
    """

    def __init__(self, embed_dim: int = 0, num_classes: int = 0, margin: float = 1.0):
        super().__init__()
        self.margin = float(margin)

    def forward(self, emb, logits, labels):
        dist = _pairwise_distances(emb)
        same = labels.unsqueeze(0) == labels.unsqueeze(1)
        eye = torch.eye(len(labels), dtype=torch.bool, device=labels.device)
        triu = torch.triu(torch.ones_like(same), diagonal=1).bool()
        pos = (same & triu & ~eye)
        neg = (~same & triu)
        terms = []
        if pos.any():
            terms.append(dist[pos].pow(2).mean())
        if neg.any():
            terms.append(F.relu(self.margin - dist[neg]).pow(2).mean())
        if not terms:
            return emb.new_zeros(())
        return sum(terms) / len(terms)


class CombinedTaskLoss(nn.Module):
    """Weighted sum of named task losses; reports per-component values."""

    def __init__(self, parts: dict[str, tuple[TaskLoss, float]]) -> None:
        super().__init__()
        self.weights = {name: float(w) for name, (_, w) in parts.items()}
        self.parts = nn.ModuleDict({name: loss for name, (loss, _) in parts.items()})

    def forward(self, emb, logits, labels):
        total = emb.new_zeros(())
        components: dict[str, float] = {}
        for name, loss_mod in self.parts.items():
            value = loss_mod(emb, logits, labels)
            components[name] = float(value.detach())
            total = total + self.weights[name] * value
        self.last_components = components
        return total


_NEEDS_DIMS = ("arcface",)


def build_task_loss(head_cfg: dict, embed_dim: int, num_classes: int) -> CombinedTaskLoss:
    parts: dict[str, tuple[TaskLoss, float]] = {}
    for name, weight in (head_cfg.get("losses") or {}).items():
        cls = registry.get("task_loss", name)
        params = dict(head_cfg.get(name, {}) or {})
        if name in _NEEDS_DIMS:
            params.update(embed_dim=embed_dim, num_classes=num_classes)
        parts[name] = (cls(**params), float(weight))
    return CombinedTaskLoss(parts)
