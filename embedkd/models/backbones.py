"""Backbone factory with an explicit, frozen support policy.

Tier 1 (SUPPORTED_BACKBONES) is fully tested in CI and reported in the paper.
Tier 2 (EXPERIMENTAL_VERIFIED) passes smoke tests only. Any other timm model
runs only under ``backbone_policy: experimental`` and emits a warning.

The docs table and the paper table are generated from these dicts by script;
do not maintain the lists anywhere else.
"""

from __future__ import annotations

import warnings

import timm
import torch.nn as nn


class BackboneNotValidatedError(ValueError):
    pass


SUPPORTED_BACKBONES: dict[str, dict] = {
    "resnet50": {"tier": 1, "family": "resnet", "role": "teacher", "params_m": 25.6},
    "convnext_tiny": {"tier": 1, "family": "convnext", "role": "teacher", "params_m": 28.6},
    "resnet18": {"tier": 1, "family": "resnet", "role": "student", "params_m": 11.7},
    "mobilenetv3_large_100": {"tier": 1, "family": "mobilenet", "role": "student", "params_m": 5.5},
    "efficientnet_b0": {"tier": 1, "family": "efficientnet", "role": "student", "params_m": 5.3},
}

EXPERIMENTAL_VERIFIED: dict[str, dict] = {
    "vit_small_patch16_224": {"tier": 2, "family": "vit", "note": "token mean pooling"},
    "swin_tiny_patch4_window7_224": {"tier": 2, "family": "swin", "note": "token mean pooling"},
    "regnety_016": {"tier": 2, "family": "regnet", "note": ""},
}


def create_backbone(
    name: str, pretrained: bool = False, policy: str = "supported"
) -> tuple[nn.Module, int]:
    """Return ``(feature_extractor, num_features)`` for a timm model name.

    The extractor keeps spatial feature maps (CNN: BCHW, transformer: tokens);
    pooling and projection belong to :class:`embedkd.models.head.EmbedHead`.
    """
    if name not in SUPPORTED_BACKBONES:
        if policy != "experimental":
            raise BackboneNotValidatedError(
                f"'{name}' is not a validated backbone. "
                f"Supported: {sorted(SUPPORTED_BACKBONES)}. "
                "To proceed anyway, set 'backbone_policy: experimental'."
            )
        level = "experimental tier" if name in EXPERIMENTAL_VERIFIED else "untested"
        warnings.warn(
            f"Backbone '{name}' is {level}: it runs, but published numbers are "
            "not guaranteed for it.",
            stacklevel=2,
        )
    kwargs: dict = {"pretrained": pretrained, "num_classes": 0, "global_pool": ""}
    # Stochastic depth 0.2 is the proven default for CNN fine-tuning in the
    # authors' prior experiments; edge families predate drop_path support.
    edge_families = ("ghostnet", "mobilenet", "shufflenet", "squeezenet", "repvgg")
    if not any(family in name for family in edge_families):
        kwargs["drop_path_rate"] = 0.2
    try:
        model = timm.create_model(name, **kwargs)
    except TypeError:  # model family without drop_path support
        kwargs.pop("drop_path_rate", None)
        model = timm.create_model(name, **kwargs)
    return model, model.num_features


def backbone_table() -> list[dict]:
    """Rows for ``embedkd backbones`` and for the auto-generated docs table."""
    rows = []
    for name, meta in SUPPORTED_BACKBONES.items():
        rows.append({"name": name, "status": "supported", **meta})
    for name, meta in EXPERIMENTAL_VERIFIED.items():
        rows.append({"name": name, "status": "experimental", **meta})
    return rows
