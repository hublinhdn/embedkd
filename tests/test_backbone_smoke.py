"""Smoke test for EVERY listed backbone: the promise CONTRIBUTING.md makes.

Each entry in SUPPORTED_BACKBONES and EXPERIMENTAL_VERIFIED must instantiate
(without pretrained weights), run a forward pass through the embedding head,
and produce a finite L2-normalised embedding. Being on the list means exactly
this much; published numbers exist only for Tier 1.
"""

import pytest
import torch

from embedkd.models import (
    EXPERIMENTAL_VERIFIED,
    SUPPORTED_BACKBONES,
    EmbedHead,
    EmbeddingModel,
    create_backbone,
)

ALL_LISTED = sorted(SUPPORTED_BACKBONES) + sorted(EXPERIMENTAL_VERIFIED)


def test_legacy_suffixes_rejected_with_hint():
    from embedkd.models import BackboneNotValidatedError

    with pytest.raises(BackboneNotValidatedError, match="Try 'resnet50'"):
        create_backbone("resnet50_tv", policy="experimental")
    with pytest.raises(BackboneNotValidatedError, match="via timm"):
        create_backbone("mobilenetv2_100_timm", policy="experimental")


def test_torchvision_only_names_get_precise_guidance():
    from embedkd.models import BackboneNotValidatedError

    # Has a timm equivalent: point to it (with or without the _tv suffix).
    with pytest.raises(BackboneNotValidatedError, match="mobilenetv3_large_100"):
        create_backbone("mobilenet_v3_large", policy="experimental")
    with pytest.raises(BackboneNotValidatedError, match="mobilenetv2_100"):
        create_backbone("mobilenet_v2_tv", policy="experimental")
    # No timm port: say so explicitly instead of a cryptic unknown-model error.
    with pytest.raises(BackboneNotValidatedError, match="no timm port"):
        create_backbone("shufflenet_v2_x1_0", policy="experimental")


def test_output_stride_doubles_feature_resolution():
    # Last-stride retrieval trick, inherited: os=16 keeps a 14x14 map at 224px.
    backbone, dim = create_backbone("resnet18", policy="supported", output_stride=16)
    with torch.no_grad():
        out = backbone.eval()(torch.zeros(1, 3, 224, 224))
    assert out.shape[-2:] == (14, 14)
    assert dim == 512


@pytest.mark.slow
@pytest.mark.parametrize("name", ALL_LISTED)
def test_listed_backbone_forward(name):
    backbone, dim = create_backbone(name, pretrained=False, policy="experimental")
    model = EmbeddingModel(backbone, EmbedHead(dim, 32, pooling="gem")).eval()
    with torch.no_grad():
        emb = model(torch.randn(2, 3, 224, 224))
    assert emb.shape == (2, 32)
    assert torch.isfinite(emb).all()
    assert torch.allclose(emb.norm(dim=-1), torch.ones(2), atol=1e-4)
