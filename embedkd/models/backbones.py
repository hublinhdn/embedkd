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

# Tier 2: smoke-tested in CI (creation + forward through the head), no
# published numbers. The selection mirrors the backbone families validated
# in the authors' prior experiments; any other timm model still runs under
# backbone_policy: experimental with a warning.
EXPERIMENTAL_VERIFIED: dict[str, dict] = {
    # resnet family
    "resnet34": {"tier": 2, "family": "resnet", "note": ""},
    "resnet101": {"tier": 2, "family": "resnet", "note": ""},
    "resnest101e": {"tier": 2, "family": "resnest", "note": "teacher in the authors' prior work"},
    "seresnext101_32x4d": {"tier": 2, "family": "resnext", "note": ""},
    # convnext family
    "convnext_small": {"tier": 2, "family": "convnext", "note": ""},
    "convnext_base": {"tier": 2, "family": "convnext", "note": ""},
    # efficientnet family
    "efficientnet_b1": {"tier": 2, "family": "efficientnet", "note": ""},
    "efficientnet_b3": {"tier": 2, "family": "efficientnet", "note": ""},
    # edge CNNs
    "mobilenetv2_100": {"tier": 2, "family": "mobilenet", "note": ""},
    "ghostnet_100": {"tier": 2, "family": "ghostnet", "note": ""},
    "repvgg_a0": {"tier": 2, "family": "repvgg", "note": ""},
    "regnety_016": {"tier": 2, "family": "regnet", "note": ""},
    # transformers (token pooling via the head)
    "vit_small_patch16_224": {"tier": 2, "family": "vit", "note": "token mean pooling"},
    "deit_small_patch16_224": {"tier": 2, "family": "vit", "note": "token mean pooling"},
    "vit_small_patch14_dinov2.lvd142m": {"tier": 2, "family": "dinov2",
                                         "note": "input size must be divisible by 14"},
    "swin_tiny_patch4_window7_224": {"tier": 2, "family": "swin", "note": "BHWC features"},
    "swin_small_patch4_window7_224": {"tier": 2, "family": "swin", "note": "BHWC features"},
    # lightweight hybrids
    "mobilevit_s": {"tier": 2, "family": "mobilevit", "note": ""},
    "tiny_vit_11m_224": {"tier": 2, "family": "tinyvit", "note": ""},
}


def create_backbone(
    name: str, pretrained: bool = False, policy: str = "supported",
    output_stride: int | None = None,
) -> tuple[nn.Module, int]:
    """Return ``(feature_extractor, num_features)`` for a timm model name.

    The extractor keeps spatial feature maps (CNN: BCHW, transformer: tokens);
    pooling and projection belong to :class:`embedkd.models.head.EmbedHead`.

    ``output_stride=16`` on resnet-family models reproduces the last-stride
    retrieval trick from the authors' prior work (higher-resolution final
    feature map); leave None for the architecture default.

    EmbedKD loads every backbone through timm; the legacy ``_tv`` /
    ``_timm`` name suffixes from the authors' prior code are rejected with
    an explanation rather than silently misinterpreted.
    """
    # Known torchvision-only names (the reason prior tooling had a second,
    # '_tv' loader). Values: the timm equivalent, or None when there is none.
    tv_only = {
        "mobilenet_v2": "mobilenetv2_100",
        "mobilenet_v3_large": "mobilenetv3_large_100",
        "shufflenet_v2_x1_0": None,
        "squeezenet1_0": None,
    }
    base = name.rsplit("_", 1)[0] if name.endswith(("_tv", "_timm")) else name
    if name.endswith(("_tv", "_timm")) or base in tv_only:
        equivalent = tv_only.get(base)
        hint = (
            f"Use the timm equivalent '{equivalent}'." if equivalent
            else f"Try '{base}'." if base != name and base not in tv_only
            else "This model has no timm port; torchvision-only models are outside "
                 "EmbedKD's scope."
        )
        raise BackboneNotValidatedError(
            f"'{name}': EmbedKD loads all backbones via timm (the '_tv'/'_timm' "
            f"suffixes from prior tooling are not used). {hint}"
        )
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
    if output_stride is not None:
        kwargs["output_stride"] = int(output_stride)
    # Stochastic depth 0.2 is the proven default for CNN fine-tuning in the
    # authors' prior experiments; edge families predate drop_path support.
    edge_families = ("ghostnet", "mobilenet", "shufflenet", "squeezenet", "repvgg")
    if not any(family in name for family in edge_families):
        kwargs["drop_path_rate"] = 0.2
    # ViT-family models default to their pretraining resolution (e.g. DINOv2
    # at 518); dynamic image size lets them run at the configured input size.
    if any(family in name for family in ("vit_", "deit", "eva")):
        kwargs["dynamic_img_size"] = True
    # Convenience kwargs may be unsupported by a given family and are dropped
    # on TypeError; output_stride is an explicit user request and is NEVER
    # dropped silently (an unsupported model must fail loudly instead).
    droppable = ("drop_path_rate", "dynamic_img_size")
    model = None
    for attempt in (dict(kwargs), *(
        {k: v for k, v in kwargs.items() if k != drop} for drop in droppable)):
        try:
            model = timm.create_model(name, **attempt)
            break
        except TypeError:
            continue
    if model is None:
        minimal = {"pretrained": pretrained, "num_classes": 0, "global_pool": ""}
        if output_stride is not None:
            minimal["output_stride"] = int(output_stride)
        model = timm.create_model(name, **minimal)  # real errors surface here
    return model, _detect_feature_dim(model)


def _detect_feature_dim(model: nn.Module) -> int:
    """Channel count of the actual feature output, found by a dummy forward.

    ``model.num_features`` lies for families whose head convolution sits
    AFTER global pooling (mobilenetv3: 1280 advertised vs 960 on the spatial
    map); probing the real output is the only reliable contract. Inherited
    from the authors' prior model code ("dynamic shape detection").
    """
    import torch

    was_training = model.training
    model.eval()
    with torch.no_grad():
        out = model(torch.zeros(1, 3, 224, 224))
    if was_training:
        model.train()
    if isinstance(out, (tuple, list)):
        out = out[0]
    if out.dim() == 4:  # BCHW (CNN) or BHWC (swin): channels dwarf spatial dims
        if getattr(model, "num_features", None) in (out.shape[1], out.shape[-1]):
            return int(model.num_features)
        return int(max(out.shape[1], out.shape[-1]))
    return int(out.shape[-1])  # tokens (B, N, C) or pooled (B, C)


def backbone_table() -> list[dict]:
    """Rows for ``embedkd backbones`` and for the auto-generated docs table."""
    rows = []
    for name, meta in SUPPORTED_BACKBONES.items():
        rows.append({"name": name, "status": "supported", **meta})
    for name, meta in EXPERIMENTAL_VERIFIED.items():
        rows.append({"name": name, "status": "experimental", **meta})
    return rows
