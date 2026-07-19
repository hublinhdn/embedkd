import pytest
import torch

from embedkd.losses import (
    ArcFace,
    BatchHardTriplet,
    Contrastive,
    SoftmaxCrossEntropy,
    build_task_loss,
)


def test_sce_matches_cross_entropy_and_requires_logits():
    logits = torch.randn(4, 6)
    labels = torch.tensor([0, 1, 2, 3])
    loss = SoftmaxCrossEntropy()(None, logits, labels)
    assert float(loss) == pytest.approx(float(torch.nn.functional.cross_entropy(logits, labels)))
    with pytest.raises(ValueError, match="classifier"):
        SoftmaxCrossEntropy()(torch.randn(4, 8), None, labels)


def test_triplet_batch_hard_reference_value():
    # 2 classes x 2 samples on a line: positions 0, 1 (class 0) and 10, 12 (class 1).
    emb = torch.tensor([[0.0], [1.0], [10.0], [12.0]])
    labels = torch.tensor([0, 0, 1, 1])
    # anchors: hardest pos / hardest neg distances:
    #  a0: dp=1,  dn=10 -> relu(1-10+0.2)=0
    #  a1: dp=1,  dn=9  -> 0
    #  a2: dp=2,  dn=9  -> 0
    #  a3: dp=2,  dn=11 -> 0
    loss = BatchHardTriplet(margin=0.2)(emb, None, labels)
    assert float(loss) == pytest.approx(0.0, abs=1e-6)
    # Bring classes together so hinge activates: positions 0,1 vs 1.5, 3.
    emb2 = torch.tensor([[0.0], [1.0], [1.5], [3.0]])
    # a0: dp=1, dn=1.5 -> 0 ; a1: dp=1, dn=0.5 -> 0.7 ; a2: dp=1.5, dn=0.5 -> 1.2 ;
    # a3: dp=1.5, dn=2.0 -> 0. mean = (0 + 0.7 + 1.2 + 0) / 4
    loss2 = BatchHardTriplet(margin=0.2)(emb2, None, labels)
    assert float(loss2) == pytest.approx((0.7 + 1.2) / 4, abs=1e-5)


def test_contrastive_reference_value():
    emb = torch.tensor([[0.0], [1.0]])
    labels_same = torch.tensor([0, 0])
    labels_diff = torch.tensor([0, 1])
    # same class: d^2 = 1
    assert float(Contrastive(margin=2.0)(emb, None, labels_same)) == pytest.approx(1.0)
    # different class: relu(2 - 1)^2 = 1
    assert float(Contrastive(margin=2.0)(emb, None, labels_diff)) == pytest.approx(1.0)


def test_arcface_reduces_to_ce_at_zero_margin_scale_one():
    torch.manual_seed(0)
    arc = ArcFace(embed_dim=8, num_classes=5, margin=0.0, scale=1.0)
    emb = torch.randn(4, 8)
    labels = torch.tensor([0, 1, 2, 3])
    cos = torch.nn.functional.normalize(emb, dim=-1) @ \
        torch.nn.functional.normalize(arc.weight, dim=-1).t()
    expected = torch.nn.functional.cross_entropy(cos.clamp(-1 + 1e-7, 1 - 1e-7), labels)
    assert float(arc(emb, None, labels)) == pytest.approx(float(expected), abs=1e-5)


def test_arcface_margin_increases_loss():
    torch.manual_seed(0)
    emb = torch.randn(8, 16)
    labels = torch.randint(0, 4, (8,))
    base = ArcFace(16, 4, margin=0.0, scale=30.0)
    with torch.no_grad():
        margined = ArcFace(16, 4, margin=0.5, scale=30.0)
        margined.weight.copy_(base.weight)
    assert float(margined(emb, None, labels)) > float(base(emb, None, labels))


def test_build_task_loss_components():
    head_cfg = {
        "losses": {"sce": 1.0, "triplet": 0.5},
        "triplet": {"margin": 0.3, "mining": "batch_hard"},
    }
    combined = build_task_loss(head_cfg, embed_dim=8, num_classes=4)
    emb = torch.randn(8, 8)
    logits = torch.randn(8, 4)
    labels = torch.randint(0, 4, (8,))
    total = combined(emb, logits, labels)
    assert torch.isfinite(total)
    assert set(combined.last_components) == {"sce", "triplet"}
    assert combined.parts["triplet"].margin == pytest.approx(0.3)
