import pytest
import torch

from embedkd.models import EmbedHead, GeM

from .utils import tiny_embedding_model


def test_gem_shape_and_gap_equivalence_at_p1():
    x = torch.rand(2, 16, 7, 7) + 0.1
    gem = GeM(p=1.0)
    expected = x.mean(dim=(2, 3))
    assert torch.allclose(gem(x), expected, atol=1e-5)
    assert GeM(p=3.0)(x).shape == (2, 16)


def test_head_normalizes_output():
    head = EmbedHead(16, 8, pooling="gem")
    emb = head(torch.rand(4, 16, 5, 5))
    assert emb.shape == (4, 8)
    assert torch.allclose(emb.norm(dim=-1), torch.ones(4), atol=1e-5)


def test_head_token_input_mean_pool_with_warning():
    head = EmbedHead(16, 8, pooling="gem")
    tokens = torch.rand(3, 10, 16)
    with pytest.warns(UserWarning, match="token mean"):
        emb = head(tokens)
    assert emb.shape == (3, 8)


def test_head_channels_last_features_are_fixed():
    head = EmbedHead(16, 8, pooling="gap")
    bhwc = torch.rand(2, 7, 7, 16)  # swin-style output
    assert head(bhwc).shape == (2, 8)


def test_embedding_model_logits_are_scaled_cosines():
    model = tiny_embedding_model(embed_dim=8, num_classes=5)
    emb, logits = model(torch.rand(2, 3, 32, 32), return_logits=True)
    assert emb.shape == (2, 8)
    assert logits.shape == (2, 5)
    # Cosine classifier: |logit| <= scale by construction.
    assert float(logits.abs().max()) <= model.classifier.scale + 1e-4


def test_head_has_bn_neck():
    head = EmbedHead(16, 8)
    assert hasattr(head, "bn_neck")  # BNNeck (Luo et al. 2019), inherited design


def test_gem_head_onnx_export_is_safe(tmp_path):
    onnx = pytest.importorskip("onnx")  # noqa: F841
    model = tiny_embedding_model(embed_dim=8, num_classes=None, pooling="gem")
    path = tmp_path / "head.onnx"
    torch.onnx.export(
        model.eval(), (torch.rand(1, 3, 64, 64),), str(path),
        input_names=["images"], output_names=["embedding"],
        dynamic_axes={"images": {0: "batch"}}, opset_version=17, dynamo=False,
    )
    assert path.exists()
