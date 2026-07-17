import pytest
import torch

from embedkd.data import SyntheticDataset
from embedkd.diagnostics import compatibility_report, distill_report, linear_cka

from .utils import tiny_embedding_model


def test_cka_self_similarity_is_one():
    x = torch.randn(50, 16)
    assert linear_cka(x, x) == pytest.approx(1.0, abs=1e-5)


def test_cka_invariant_to_orthogonal_rotation():
    x = torch.randn(50, 8)
    q, _ = torch.linalg.qr(torch.randn(8, 8))
    assert linear_cka(x, x @ q) == pytest.approx(1.0, abs=1e-4)


def test_cka_low_for_independent_features():
    torch.manual_seed(0)
    x = torch.randn(200, 16)
    y = torch.randn(200, 16)
    assert linear_cka(x, y) < 0.3


def test_cka_shape_mismatch_raises():
    with pytest.raises(ValueError, match="sample counts"):
        linear_cka(torch.randn(10, 4), torch.randn(9, 4))


def test_linear_cka_matches_gram_matrix_form():
    # The prior analysis code computes CKA on double-centered Gram matrices;
    # our feature-space form must agree to numerical precision.
    torch.manual_seed(0)
    x = torch.randn(40, 12)
    y = torch.randn(40, 7)

    def gram_form(a, b):
        def center(k):
            return k - k.mean(0, keepdim=True) - k.mean(1, keepdim=True) + k.mean()

        k_a, k_b = center(a @ a.t()), center(b @ b.t())
        return float((k_a * k_b).sum() / (k_a.norm() * k_b.norm()))

    assert linear_cka(x, y) == pytest.approx(gram_form(x, y), abs=1e-5)


def test_rbf_cka_self_similarity_and_range():
    from embedkd.diagnostics import rbf_cka

    torch.manual_seed(0)
    x = torch.randn(30, 8)
    assert rbf_cka(x, x) == pytest.approx(1.0, abs=1e-5)
    other = rbf_cka(x, torch.randn(30, 8))
    assert 0.0 <= other < 0.6


def test_compatibility_report_keys_and_risk():
    teacher = tiny_embedding_model(embed_dim=8)
    student = tiny_embedding_model(embed_dim=8)
    probe = SyntheticDataset(num_classes=3, per_class=6, size=32)
    report = compatibility_report(teacher, student, probe, batch_size=16)
    assert {"cka_pre", "capacity_ratio", "risk", "probe_size"} <= set(report)
    assert report["risk"] in ("LOW", "MODERATE", "HIGH")
    assert report["probe_size"] == 18


def test_distill_report_patterns():
    pre = {"cka_pre": 0.5}
    improved = distill_report(pre, 0.7, {"map": 0.4}, {"map": 0.5})
    assert improved["pattern"] == "improved"
    aligned_worse = distill_report(pre, 0.8, {"map": 0.5}, {"map": 0.4})
    assert aligned_worse["pattern"] == "aligned_but_worse"
    diverged = distill_report(pre, 0.3, {"map": 0.5}, {"map": 0.4})
    assert diverged["pattern"] == "diverged"
