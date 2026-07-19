import math

import pytest
import torch
import torch.nn.functional as F

from embedkd.objectives import (
    CombinedObjective,
    CosineObjective,
    KLObjective,
    MSEObjective,
    RKDObjective,
    build_objective,
)


def test_cosine_identical_embeddings_is_zero():
    emb = F.normalize(torch.randn(8, 16), dim=-1)
    assert float(CosineObjective()(emb, emb)) == pytest.approx(0.0, abs=1e-6)


def test_cosine_reference_value():
    # Orthogonal unit vectors: cos = 0, so loss = 1 for both pairs.
    s = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    t = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    assert float(CosineObjective()(s, t)) == pytest.approx(1.0, abs=1e-6)
    # Opposite vectors: cos = -1, loss = 2.
    assert float(CosineObjective()(s, -s)) == pytest.approx(2.0, abs=1e-6)


def test_mse_reference_value():
    s = torch.tensor([[1.0, 0.0]])
    t = torch.tensor([[0.0, 1.0]])
    # Normalised difference: (1,-1); mean of squares = 1.
    assert float(MSEObjective()(s, t)) == pytest.approx(1.0, abs=1e-6)


def test_kl_identical_logits_is_zero_and_requires_logits():
    logits = torch.randn(4, 10)
    emb = torch.randn(4, 8)
    obj = KLObjective(temperature=4.0)
    value = obj(emb, emb, s_logits=logits, t_logits=logits)
    assert float(value) == pytest.approx(0.0, abs=1e-6)
    with pytest.raises(ValueError, match="logits"):
        obj(emb, emb)


def test_kl_reference_value():
    # Two classes, T=1: KL(t||s) with hand-computed softmax values.
    s_logits = torch.tensor([[0.0, math.log(3.0)]])  # softmax = [0.25, 0.75]
    t_logits = torch.tensor([[0.0, 0.0]])  # softmax = [0.5, 0.5]
    expected = 0.5 * math.log(0.5 / 0.25) + 0.5 * math.log(0.5 / 0.75)
    obj = KLObjective(temperature=1.0)
    value = obj(None, None, s_logits=s_logits, t_logits=t_logits)
    assert float(value) == pytest.approx(expected, abs=1e-6)


def test_rkd_declares_fp32_and_identical_is_zero():
    obj = RKDObjective()
    assert obj.needs_fp32 is True
    emb = torch.randn(8, 16)
    assert float(obj(emb, emb)) == pytest.approx(0.0, abs=1e-6)


def test_rkd_survives_fp16_inputs():
    # Regression guard for the original bug: RKD computed inside fp16 autocast
    # underflows tiny pairwise distances and freezes training. The design fix
    # is fp32 master parameters + fp32 loss computation on upcast embeddings;
    # this simulates exactly that: fp32 leaf, half-precision embeddings from a
    # tightly clustered batch (small distances), gradient must stay finite.
    leaf = torch.randn(6, 8, requires_grad=True)
    s_half = (F.normalize(leaf, dim=-1) * 1e-3).half()
    t_half = torch.randn(6, 8).half()
    loss = RKDObjective()(s_half, t_half)
    loss.backward()
    assert torch.isfinite(loss)
    assert leaf.grad is not None
    assert torch.isfinite(leaf.grad).all()


def test_combined_objective_components_and_weights():
    obj = build_objective({"objective": {"cosine": 2.0, "mse": 1.0}, "alpha": 1.0})
    assert isinstance(obj, CombinedObjective)
    s = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    t = -s
    total = obj(s, t)
    # cosine part = 2.0 (opposite vectors), mse part = mean((s - t)^2) = 2.0
    assert float(total) == pytest.approx(2.0 * 2.0 + 1.0 * 2.0, abs=1e-5)
    assert set(obj.last_components) == {"cosine", "mse"}


def test_build_objective_passes_params():
    obj = build_objective({"objective": "kl", "kl": {"temperature": 2.5}})
    assert obj.parts["kl"].temperature == 2.5


def test_relational_ramp_gates_rkd_only():
    obj = build_objective({"objective": {"cosine": 1.0, "rkd": 1.0}})
    assert obj.relational is True
    s = torch.randn(6, 8)
    t = torch.randn(6, 8)
    gated = obj(s, t, relational_scale=0.0)
    assert obj.last_components["rkd"] == 0.0  # gated off, not even computed
    assert obj.last_components["cosine"] > 0.0  # pointwise unaffected
    full = obj(s, t, relational_scale=1.0)
    assert float(full) > float(gated)
    assert obj.last_components["rkd"] > 0.0


def test_nonfinite_component_is_skipped_and_counted():
    from embedkd.objectives import CombinedObjective

    class ExplodingObjective(CosineObjective):
        def forward(self, s_emb, t_emb, **kw):
            return s_emb.new_tensor(float("nan"))

    obj = CombinedObjective({"boom": (ExplodingObjective(), 1.0),
                             "cosine": (CosineObjective(), 1.0)})
    s, t = torch.randn(4, 8), torch.randn(4, 8)
    total = obj(s, t)
    assert torch.isfinite(total)  # nan part skipped, cosine survives
    assert obj.nonfinite_count == 1


def test_trainer_relational_scale_schedule():
    from embedkd.engine.trainer import Trainer

    trainer = Trainer.__new__(Trainer)  # schedule logic only, no model build
    trainer.cfg = {"distill": {"relational_ramp": {"start_epoch": None, "epochs": 5}},
                   "train": {"warmup_epochs": 5}}
    assert trainer._relational_scale(0) == 0.0
    assert trainer._relational_scale(4) == 0.0
    assert trainer._relational_scale(5) == pytest.approx(0.2)
    assert trainer._relational_scale(9) == pytest.approx(1.0)
    assert trainer._relational_scale(30) == 1.0
