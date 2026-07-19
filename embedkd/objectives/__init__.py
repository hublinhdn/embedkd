"""Distillation objectives: align the student's embedding space with the teacher's.

Terminology (see docs glossary): an *objective* aligns teacher and student,
a *task loss* (:mod:`embedkd.losses`) trains the metric-learning task itself.

Numerical-safety contract: objectives that are unstable in fp16 declare
``needs_fp32 = True``. The trainer always evaluates losses in fp32 outside
the autocast region; the flag additionally documents and enforces the
requirement for user code that calls objectives directly.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..registry import registry


class DistillObjective(nn.Module):
    """Base class. Subclasses implement forward(s_emb, t_emb, **extras)."""

    needs_fp32: bool = False
    # Relational objectives (RKD-style) spike on an immature student embedding
    # and can collapse training; the trainer ramps them in after warmup.
    relational: bool = False

    def forward(
        self,
        s_emb: torch.Tensor,
        t_emb: torch.Tensor,
        s_logits: torch.Tensor | None = None,
        t_logits: torch.Tensor | None = None,
    ) -> torch.Tensor:
        raise NotImplementedError


@registry.distill_objective("cosine")
class CosineObjective(DistillObjective):
    """1 - cos(student, teacher), averaged over the batch."""

    def forward(self, s_emb, t_emb, **_):
        s = F.normalize(s_emb, dim=-1)
        t = F.normalize(t_emb, dim=-1)
        return (1.0 - (s * t).sum(dim=-1)).mean()


@registry.distill_objective("mse")
class MSEObjective(DistillObjective):
    """Mean squared error between L2-normalised embeddings."""

    def forward(self, s_emb, t_emb, **_):
        return F.mse_loss(F.normalize(s_emb, dim=-1), F.normalize(t_emb, dim=-1))


@registry.distill_objective("kl")
class KLObjective(DistillObjective):
    """Hinton et al. (2015) logit distillation: KL(softmax(t/T) || softmax(s/T)) * T^2.

    Requires classifier logits from both models (student: 'sce' task loss or a
    classifier head; teacher: a checkpoint that includes a classifier).
    """

    def __init__(self, temperature: float = 4.0) -> None:
        super().__init__()
        self.temperature = float(temperature)

    def forward(self, s_emb, t_emb, s_logits=None, t_logits=None):
        if s_logits is None or t_logits is None:
            raise ValueError(
                "The 'kl' objective needs classifier logits from both teacher and "
                "student. Add 'sce' to head.losses and use a teacher checkpoint "
                "that includes a classifier."
            )
        temp = self.temperature
        log_p_s = F.log_softmax(s_logits / temp, dim=-1)
        p_t = F.softmax(t_logits / temp, dim=-1)
        return F.kl_div(log_p_s, p_t, reduction="batchmean") * temp * temp


def _pairwise_distances(e: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Euclidean distance matrix with a NaN-safe backward pass.

    sqrt has an infinite derivative at 0; the diagonal (and duplicate points)
    would turn masked-out zero gradients into 0 * inf = NaN. Clamping the
    squared distances to eps before sqrt and zeroing the diagonal via a mask
    multiplication keeps every gradient finite.
    """
    prod = e @ e.t()
    sq = prod.diagonal()
    d2 = (sq.unsqueeze(0) + sq.unsqueeze(1) - 2.0 * prod).clamp(min=eps)
    dist = d2.sqrt()
    off_diag = 1.0 - torch.eye(e.shape[0], device=e.device, dtype=e.dtype)
    return dist * off_diag


@registry.distill_objective("rkd")
class RKDObjective(DistillObjective):
    """Relational KD (Park et al., CVPR 2019): distance + angle relations.

    fp16-unsafe: tiny pairwise distances underflow in half precision, which
    silently produces infinite gradients and frozen training. Always computed
    in fp32 (``needs_fp32 = True``); inputs are upcast defensively.
    """

    needs_fp32 = True
    relational = True

    def __init__(self, distance_weight: float = 25.0, angle_weight: float = 50.0) -> None:
        # Defaults follow Park et al. (2019): 25 / 50. With these built in,
        # distill.alpha stays ~1.0 for RKD configs.
        super().__init__()
        self.distance_weight = float(distance_weight)
        self.angle_weight = float(angle_weight)

    @staticmethod
    def _distance_loss(s: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            t_d = _pairwise_distances(t)
            mask = ~torch.eye(t_d.shape[0], dtype=torch.bool, device=t_d.device)
            t_mean = t_d[mask].mean().clamp(min=1e-12)
            t_d = t_d / t_mean
        s_d = _pairwise_distances(s)
        s_mean = s_d[mask].mean().clamp(min=1e-12)
        s_d = s_d / s_mean
        return F.smooth_l1_loss(s_d[mask], t_d[mask])

    @staticmethod
    def _angle_loss(s: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            t_vec = F.normalize(t.unsqueeze(0) - t.unsqueeze(1), dim=2, eps=1e-12)
            t_angle = torch.bmm(t_vec, t_vec.transpose(1, 2)).flatten()
        s_vec = F.normalize(s.unsqueeze(0) - s.unsqueeze(1), dim=2, eps=1e-12)
        s_angle = torch.bmm(s_vec, s_vec.transpose(1, 2)).flatten()
        return F.smooth_l1_loss(s_angle, t_angle)

    def forward(self, s_emb, t_emb, **_):
        s = s_emb.float()
        t = t_emb.float()
        loss = s.new_zeros(())
        if self.distance_weight:
            loss = loss + self.distance_weight * self._distance_loss(s, t)
        if self.angle_weight:
            loss = loss + self.angle_weight * self._angle_loss(s, t)
        return loss


class CombinedObjective(DistillObjective):
    """Weighted sum of named objectives; reports per-component values.

    Inherited stability mechanics (from the authors' prior training code):
    relational parts are scaled by ``relational_scale`` (the trainer ramps it
    0 -> 1 after warmup), and any non-finite component is skipped and counted
    rather than poisoning the whole step.
    """

    def __init__(self, parts: dict[str, tuple[DistillObjective, float]]) -> None:
        super().__init__()
        self.weights = {name: float(w) for name, (_, w) in parts.items()}
        self.parts = nn.ModuleDict({name: obj for name, (obj, _) in parts.items()})
        self.needs_fp32 = any(obj.needs_fp32 for obj, _ in parts.values())
        self.relational = any(obj.relational for obj, _ in parts.values())
        self.nonfinite_count = 0

    def forward(self, s_emb, t_emb, s_logits=None, t_logits=None,
                relational_scale: float = 1.0):
        total = s_emb.new_zeros(())
        components: dict[str, float] = {}
        for name, obj in self.parts.items():
            weight = self.weights[name]
            if obj.relational:
                weight = weight * float(relational_scale)
                if weight == 0.0:
                    components[name] = 0.0
                    continue
            value = obj(s_emb, t_emb, s_logits=s_logits, t_logits=t_logits)
            if not torch.isfinite(value):
                self.nonfinite_count += 1
                components[name] = float("nan")
                continue
            components[name] = float(value.detach())
            total = total + weight * value
        self.last_components = components
        return total


def build_objective(distill_cfg: dict) -> CombinedObjective:
    """Build the (possibly combined) objective declared in the config."""
    spec = distill_cfg["objective"]
    weights = {spec: 1.0} if isinstance(spec, str) else {k: float(v) for k, v in spec.items()}
    parts: dict[str, tuple[DistillObjective, float]] = {}
    for name, weight in weights.items():
        cls = registry.get("distill_objective", name)
        params = distill_cfg.get(name, {}) or {}
        parts[name] = (cls(**params), weight)
    return CombinedObjective(parts)
