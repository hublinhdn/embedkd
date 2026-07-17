import pytest

from embedkd.config import ConfigError, DEFAULTS, deep_update, parse_set_override, resolve


def _base(**over):
    cfg = deep_update(DEFAULTS, {"data": {"adapter": "synthetic"}})
    return deep_update(cfg, over)


def test_defaults_are_valid():
    cfg = resolve(overrides=["data.adapter=synthetic"])
    assert cfg["train"]["seed"] == 42


def test_set_override_types():
    nested = parse_set_override("train.lr=0.01")
    assert nested == {"train": {"lr": 0.01}}
    assert parse_set_override("train.amp=false") == {"train": {"amp": False}}


def test_rule_embed_dim_mismatch_for_cosine():
    with pytest.raises(ConfigError, match="embed_dim"):
        resolve(overrides=["data.adapter=synthetic", "teacher.embed_dim=512",
                           "student.embed_dim=256"])


def test_rule_kl_allows_dim_mismatch():
    cfg = resolve(overrides=["data.adapter=synthetic", "distill.objective=kl",
                             "teacher.embed_dim=512", "student.embed_dim=256"])
    assert cfg["student"]["embed_dim"] == 256


def test_rule_triplet_requires_pk_sampler():
    with pytest.raises(ConfigError, match="pk"):
        resolve(overrides=["data.adapter=synthetic", "data.sampler=random",
                           "head.losses={triplet: 1.0}"])


def test_random_sampler_fine_without_pair_losses():
    cfg = resolve(overrides=["data.adapter=synthetic", "data.sampler=random",
                             "head.losses={sce: 1.0}"])
    assert cfg["data"]["sampler"] == "random"


def test_rule_backbone_whitelist():
    with pytest.raises(ConfigError, match="not a validated backbone"):
        resolve(overrides=["data.adapter=synthetic", "student.backbone=resnest269e"])


def test_rule_backbone_experimental_optin():
    cfg = resolve(overrides=["data.adapter=synthetic", "student.backbone=resnest269e",
                             "backbone_policy=experimental"])
    assert cfg["student"]["backbone"] == "resnest269e"


def test_rule_cross_domain_needs_target():
    with pytest.raises(ConfigError, match="data.target"):
        resolve(overrides=["data.adapter=synthetic", "data.protocol=cross_domain"])


def test_unknown_key_with_suggestion():
    with pytest.raises(ConfigError, match="Did you mean 'epochs'"):
        resolve(overrides=["data.adapter=synthetic", "train.epoch=3"])
