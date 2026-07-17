"""Retrieval evaluation: gallery-query mAP and Recall@k on cosine similarity."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


@torch.no_grad()
def extract_embeddings(
    model: torch.nn.Module,
    dataset: Dataset,
    batch_size: int = 256,
    device: str | torch.device = "cpu",
    num_workers: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (embeddings [N, D] fp32 L2-normalised, labels [N])."""
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    embs, labels = [], []
    for images, batch_labels in loader:
        emb = model(images.to(device))
        embs.append(F.normalize(emb.float(), dim=-1).cpu())
        labels.append(torch.as_tensor(batch_labels))
    return torch.cat(embs), torch.cat(labels)


def retrieval_metrics(
    gallery_emb: torch.Tensor,
    gallery_labels: torch.Tensor,
    query_emb: torch.Tensor,
    query_labels: torch.Tensor,
    ks: tuple[int, ...] = (1, 5),
) -> dict[str, float]:
    """mAP and Recall@k. Queries with no relevant gallery item are skipped."""
    sim = query_emb @ gallery_emb.t()
    order = sim.argsort(dim=1, descending=True)
    ranked_labels = gallery_labels[order]
    relevant = ranked_labels == query_labels.unsqueeze(1)

    has_match = relevant.any(dim=1)
    metrics: dict[str, float] = {}

    rel = relevant[has_match].float()
    if rel.numel() == 0:
        return {"map": 0.0, "mrr": 0.0, **{f"r{k}": 0.0 for k in ks}, "num_queries": 0}

    # Average precision per query: mean over relevant positions of
    # (number of relevant items up to that rank) / rank.
    cumulative = rel.cumsum(dim=1)
    ranks = torch.arange(1, rel.shape[1] + 1, dtype=torch.float32)
    precision_at = cumulative / ranks
    ap = (precision_at * rel).sum(dim=1) / rel.sum(dim=1)
    metrics["map"] = float(ap.mean())

    # Mean reciprocal rank of the FIRST relevant item. Reported alongside mAP
    # because the authors' prior work scores class-level reciprocal rank;
    # mrr is the closest image-level bridge to those historical numbers.
    first_rank = rel.float().argmax(dim=1) + 1
    metrics["mrr"] = float((1.0 / first_rank.float()).mean())

    for k in ks:
        metrics[f"r{k}"] = float(rel[:, :k].any(dim=1).float().mean())
    metrics["num_queries"] = int(has_match.sum())
    return metrics


def evaluate_model(
    model: torch.nn.Module,
    gallery: Dataset,
    query: Dataset,
    batch_size: int = 256,
    device: str | torch.device = "cpu",
) -> dict[str, float]:
    g_emb, g_labels = extract_embeddings(model, gallery, batch_size, device)
    q_emb, q_labels = extract_embeddings(model, query, batch_size, device)
    return retrieval_metrics(g_emb, g_labels, q_emb, q_labels)
