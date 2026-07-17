"""Teacher-student compatibility diagnostics.

Answers the question no generic KD framework asks: is this teacher-student
pair worth distilling at all? Run :func:`compatibility_report` BEFORE burning
GPU-days, and :func:`distill_report` after training to classify the outcome.

The risk thresholds below are heuristics; their empirical origin is
documented in docs/concepts/diagnostics.md and in the accompanying paper.
"""

from __future__ import annotations

import torch
from torch.utils.data import Dataset

from ..evaluation import extract_embeddings

# Pre-distillation embedding CKA below these values indicates elevated risk
# that distillation aligns representations without improving retrieval
# ("aligned-but-worse") or actively hurts the student ("negative transfer").
RISK_THRESHOLDS = {"high": 0.35, "moderate": 0.60}


def linear_cka(x: torch.Tensor, y: torch.Tensor) -> float:
    """Linear Centered Kernel Alignment (Kornblith et al., ICML 2019).

    x: [N, D1], y: [N, D2]; features are centered per dimension.
    """
    if x.shape[0] != y.shape[0]:
        raise ValueError(f"Need matching sample counts, got {x.shape[0]} vs {y.shape[0]}")
    x = x.float() - x.float().mean(dim=0, keepdim=True)
    y = y.float() - y.float().mean(dim=0, keepdim=True)
    cross = (x.t() @ y).norm() ** 2
    self_x = (x.t() @ x).norm()
    self_y = (y.t() @ y).norm()
    denom = self_x * self_y
    return float(cross / denom) if denom > 0 else 0.0


def _num_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def _risk_level(cka: float) -> str:
    if cka < RISK_THRESHOLDS["high"]:
        return "HIGH"
    if cka < RISK_THRESHOLDS["moderate"]:
        return "MODERATE"
    return "LOW"


@torch.no_grad()
def compatibility_report(
    teacher: torch.nn.Module,
    student: torch.nn.Module,
    probe: Dataset,
    batch_size: int = 128,
    device: str | torch.device = "cpu",
) -> dict:
    """Pre-distillation report: run this before training."""
    t_emb, _ = extract_embeddings(teacher, probe, batch_size, device)
    s_emb, _ = extract_embeddings(student, probe, batch_size, device)
    cka = linear_cka(t_emb, s_emb)
    t_params = _num_params(teacher)
    s_params = _num_params(student)
    risk = _risk_level(cka)
    return {
        "cka_pre": round(cka, 4),
        "teacher_params": t_params,
        "student_params": s_params,
        "capacity_ratio": round(t_params / max(1, s_params), 2),
        "probe_size": len(probe),
        "risk": risk,
        "note": (
            "Low pre-distillation CKA signals dissimilar representation geometry; "
            "see docs/concepts/diagnostics.md#thresholds for the empirical basis."
        ),
    }


def distill_report(
    pre: dict,
    cka_post: float,
    student_metrics_before: dict,
    student_metrics_after: dict,
    metric: str = "map",
) -> dict:
    """Post-distillation report: classify what distillation actually did."""
    delta = student_metrics_after.get(metric, 0.0) - student_metrics_before.get(metric, 0.0)
    cka_delta = cka_post - pre["cka_pre"]
    if delta > 0:
        pattern = "improved"
    elif cka_delta > 0:
        pattern = "aligned_but_worse"
    else:
        pattern = "diverged"
    return {
        "cka_pre": pre["cka_pre"],
        "cka_post": round(cka_post, 4),
        "cka_delta": round(cka_delta, 4),
        f"{metric}_before": round(student_metrics_before.get(metric, 0.0), 4),
        f"{metric}_after": round(student_metrics_after.get(metric, 0.0), 4),
        f"{metric}_delta": round(delta, 4),
        "pattern": pattern,
    }


def format_report(report: dict) -> str:
    """Human-readable console rendering of a report dict."""
    lines = [f"{key:>18}: {value}" for key, value in report.items()]
    return "\n".join(lines)
