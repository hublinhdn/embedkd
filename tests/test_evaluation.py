import pytest
import torch

from embedkd.evaluation import retrieval_metrics


def test_retrieval_metrics_hand_computed():
    # Gallery: 4 unit vectors in 2D; labels 0,0,1,1.
    gallery = torch.tensor([
        [1.0, 0.0],
        [0.9, 0.1],
        [0.0, 1.0],
        [0.1, 0.9],
    ])
    gallery = torch.nn.functional.normalize(gallery, dim=-1)
    g_labels = torch.tensor([0, 0, 1, 1])

    # Query 1 (label 0) points at class-0 items: ranking puts both relevant first.
    # Query 2 (label 1) points between: crafted so one relevant is at rank 1, other rank 3.
    q1 = torch.nn.functional.normalize(torch.tensor([[1.0, 0.05]]), dim=-1)
    q2 = torch.nn.functional.normalize(torch.tensor([[0.5, 0.6]]), dim=-1)
    query = torch.cat([q1, q2])
    q_labels = torch.tensor([0, 1])

    metrics = retrieval_metrics(gallery, g_labels, query, q_labels, ks=(1, 2))

    # Query 1: relevant at ranks 1,2 -> AP = (1/1 + 2/2) / 2 = 1.0
    # Query 2 similarities: g0=0.64, g1=0.65, g2=0.77, g3=0.75 (approx)
    #   ranking: g2(rel), g3(rel), g1, g0 -> AP = (1/1 + 2/2)/2 = 1.0? verify below.
    sim2 = (query[1:2] @ gallery.t()).squeeze()
    order = sim2.argsort(descending=True)
    ranked = g_labels[order]
    # Compute AP for query 2 independently of the implementation:
    hits, precisions = 0, []
    for rank, label in enumerate(ranked.tolist(), start=1):
        if label == 1:
            hits += 1
            precisions.append(hits / rank)
    expected_ap2 = sum(precisions) / hits
    expected_map = (1.0 + expected_ap2) / 2
    assert metrics["map"] == pytest.approx(expected_map, abs=1e-6)
    assert metrics["num_queries"] == 2
    assert 0.0 <= metrics["r1"] <= 1.0


def test_r1_counts_top_hit():
    gallery = torch.eye(3)
    g_labels = torch.tensor([0, 1, 2])
    query = torch.eye(3)
    q_labels = torch.tensor([0, 1, 2])
    metrics = retrieval_metrics(gallery, g_labels, query, q_labels, ks=(1,))
    assert metrics["r1"] == pytest.approx(1.0)
    assert metrics["map"] == pytest.approx(1.0)


def test_mrr_hand_computed():
    # Query 1: first relevant at rank 1 -> 1.0 ; Query 2: first relevant at rank 2 -> 0.5
    gallery = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.7, 0.7]])
    gallery = torch.nn.functional.normalize(gallery, dim=-1)
    g_labels = torch.tensor([0, 1, 1])
    query = torch.nn.functional.normalize(torch.tensor([[1.0, 0.1], [0.9, 0.5]]), dim=-1)
    q_labels = torch.tensor([0, 1])
    metrics = retrieval_metrics(gallery, g_labels, query, q_labels, ks=(1,))
    # q2 similarities: g0=0.874, g1=0.485, g2=0.961 -> ranking g2(rel,1st) => rr=1.0
    # q1: g0=0.995(rel first) => rr=1.0 ; adjust q2 to hit rank 2 instead:
    q2 = torch.nn.functional.normalize(torch.tensor([[1.0, 0.4]]), dim=-1)
    metrics2 = retrieval_metrics(gallery, g_labels, q2, torch.tensor([1]), ks=(1,))
    # similarities: g0=0.928, g2=0.919, g1=0.371 -> first relevant (label 1) at rank 2
    assert metrics2["mrr"] == pytest.approx(0.5, abs=1e-6)
    assert metrics["mrr"] == pytest.approx(1.0, abs=1e-6)


def test_queries_without_relevant_are_skipped():
    gallery = torch.eye(2)
    g_labels = torch.tensor([0, 0])
    query = torch.eye(2)
    q_labels = torch.tensor([0, 9])  # second query has no match in gallery
    metrics = retrieval_metrics(gallery, g_labels, query, q_labels)
    assert metrics["num_queries"] == 1
