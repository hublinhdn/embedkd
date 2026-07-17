"""Configuration loading, defaults, overrides and fail-fast validation.

The resolved configuration (defaults + YAML file + --set overrides) is the
single source of truth for a run and is dumped verbatim into the run
fingerprint, so what you configure is exactly what executes.
"""

from __future__ import annotations

import copy
import difflib
from pathlib import Path
from typing import Any

import yaml

from .registry import registry


class ConfigError(ValueError):
    pass


DEFAULTS: dict[str, Any] = {
    "run": {"tag": None, "output_dir": "runs"},
    "backbone_policy": "supported",
    "teacher": {"backbone": "resnet50", "weights": "random", "embed_dim": 512,
                "output_stride": None},
    "student": {"backbone": "resnet18", "pretrained": True, "embed_dim": 512,
                "output_stride": None},
    "head": {
        "pooling": "gem",
        "gem_p": 3.0,
        "gem_p_trainable": False,
        "normalize": True,
        "logit_scale": 64.0,  # cosine-classifier scale for sce / kl logits
        "losses": {"sce": 1.0},
        "arcface": {"margin": 0.35, "scale": 64.0},
        "triplet": {"margin": 0.2, "mining": "batch_hard"},
        "contrastive": {"margin": 1.0},
    },
    "distill": {
        "objective": "cosine",
        "alpha": 10.0,
        "kl": {"temperature": 4.0},
        "rkd": {"distance_weight": 25.0, "angle_weight": 50.0},
        # Relational objectives stay OFF until start_epoch (None = after LR
        # warmup), then ramp 0 -> 1 over `epochs`. Pointwise objectives are
        # unaffected. Inherited anti-collapse schedule.
        "relational_ramp": {"start_epoch": None, "epochs": 5},
    },
    "data": {
        "adapter": "image_folder",
        "root": None,
        "manifest": None,
        "input_size": 224,
        "protocol": "gallery_query",
        "split": {"mode": "auto", "gallery_ratio": 0.5, "min_per_class": 2},
        "sampler": "pk",
        "p_classes": 16,
        "k_samples": 4,
        "num_workers": 4,
        "synthetic": {"num_classes": 4, "per_class": 32},
        "target": None,
    },
    "train": {
        "epochs": 60,
        "batch_size": 64,
        "optimizer": "adamw",
        "lr": 1.0e-3,
        "lr_backbone": None,  # None = lr / 10 (two-tier fine-tuning)
        "weight_decay": 1.0e-4,
        "scheduler": "cosine",
        "warmup_epochs": 1,
        "amp": True,
        "grad_clip": 5.0,  # global-norm clipping after unscale (0 disables)
        "seed": 42,
        "early_stopping": None,
        "save_every": 0,
        "eval_every": 1,
    },
    "eval": {"batch_size": 256, "metrics": ["map", "r1", "r5"], "report_retention": True},
}

# Sections where unknown keys are rejected outright. "head" and "distill" also
# accept any key that matches a registered component name (its parameter block).
_STRICT_SECTIONS = ("run", "teacher", "student", "data", "train", "eval")
_EMBEDDING_SPACE_OBJECTIVES = ("cosine", "mse")


def deep_update(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_update(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def parse_set_override(expr: str) -> dict:
    """Parse one ``--set a.b.c=value`` expression into a nested dict."""
    if "=" not in expr:
        raise ConfigError(f"--set expects key=value, got '{expr}'")
    dotted, raw = expr.split("=", 1)
    value = yaml.safe_load(raw)
    node: Any = value
    for part in reversed(dotted.strip().split(".")):
        node = {part: node}
    return node


def _unknown_key_error(section: str, key: str, known: list[str]) -> ConfigError:
    close = difflib.get_close_matches(key, known, n=1)
    hint = f" Did you mean '{close[0]}'?" if close else ""
    return ConfigError(f"Unknown config key '{section}.{key}'. Known keys: {sorted(known)}.{hint}")


def _check_unknown_keys(cfg: dict) -> None:
    top_known = list(DEFAULTS)
    for key in cfg:
        if key not in top_known:
            raise _unknown_key_error("<root>", key, top_known)
    for section in _STRICT_SECTIONS:
        known = list(DEFAULTS[section])
        for key in cfg.get(section) or {}:
            if key not in known:
                raise _unknown_key_error(section, key, known)
    for section, namespace in (("head", "task_loss"), ("distill", "distill_objective")):
        known = list(DEFAULTS[section])
        for key in cfg.get(section) or {}:
            if key not in known and not registry.contains(namespace, key):
                raise _unknown_key_error(section, key, known + registry.available(namespace))


def _objective_names(distill_cfg: dict) -> list[str]:
    objective = distill_cfg.get("objective")
    if isinstance(objective, str):
        return [objective]
    if isinstance(objective, dict):
        return list(objective)
    raise ConfigError("distill.objective must be a name or a {name: weight} mapping")


def validate(cfg: dict) -> None:
    """Fail fast with actionable messages. Called on every resolved config."""
    from .models.backbones import EXPERIMENTAL_VERIFIED, SUPPORTED_BACKBONES

    _check_unknown_keys(cfg)

    policy = cfg["backbone_policy"]
    if policy not in ("supported", "experimental"):
        raise ConfigError("backbone_policy must be 'supported' or 'experimental'")
    for role in ("teacher", "student"):
        name = cfg[role]["backbone"]
        if policy == "supported" and name not in SUPPORTED_BACKBONES:
            extra = " (experimental tier)" if name in EXPERIMENTAL_VERIFIED else ""
            raise ConfigError(
                f"'{name}'{extra} is not a validated backbone. "
                f"Supported: {sorted(SUPPORTED_BACKBONES)}. "
                "To proceed anyway, set 'backbone_policy: experimental'."
            )

    objectives = _objective_names(cfg["distill"])
    if cfg["teacher"]["embed_dim"] != cfg["student"]["embed_dim"]:
        clash = [o for o in objectives if o in _EMBEDDING_SPACE_OBJECTIVES]
        if clash:
            raise ConfigError(
                f"Objective(s) {clash} compare embeddings directly, so "
                f"teacher.embed_dim ({cfg['teacher']['embed_dim']}) must equal "
                f"student.embed_dim ({cfg['student']['embed_dim']})."
            )

    losses = cfg["head"]["losses"]
    pair_based = [name for name in ("triplet", "contrastive") if name in losses]
    if pair_based and cfg["data"]["sampler"] != "pk":
        raise ConfigError(
            f"Task loss(es) {pair_based} need in-batch positives. "
            "Set 'data.sampler: pk' (P classes x K samples per batch)."
        )

    if cfg["data"]["protocol"] == "cross_domain" and not cfg["data"].get("target"):
        raise ConfigError("protocol 'cross_domain' requires a 'data.target' block")
    if cfg["data"]["protocol"] not in ("gallery_query", "cross_domain"):
        raise ConfigError("data.protocol must be 'gallery_query' or 'cross_domain'")


def resolve(config_path: str | Path | None = None, overrides: list[str] | dict | None = None) -> dict:
    """defaults <- YAML file <- --set overrides, then validate."""
    cfg = copy.deepcopy(DEFAULTS)
    if config_path is not None:
        with open(config_path, encoding="utf-8") as fh:
            file_cfg = yaml.safe_load(fh) or {}
        if not isinstance(file_cfg, dict):
            raise ConfigError(f"{config_path} must contain a YAML mapping")
        cfg = deep_update(cfg, file_cfg)
        if cfg["run"]["tag"] is None:
            cfg["run"]["tag"] = Path(config_path).stem
    if overrides:
        if isinstance(overrides, dict):
            nested: dict = {}
            for dotted, value in overrides.items():
                nested = deep_update(nested, parse_set_override(f"{dotted}={yaml.safe_dump(value).strip()}"))
            cfg = deep_update(cfg, nested)
        else:
            for expr in overrides:
                cfg = deep_update(cfg, parse_set_override(expr))
    if cfg["run"]["tag"] is None:
        cfg["run"]["tag"] = "run"
    validate(cfg)
    return cfg
