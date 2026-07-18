"""Trainer features added for the GPU campaign: alpha=0 mode and eval_every."""

import copy

import pytest

from embedkd.config import DEFAULTS, deep_update
from embedkd.run import DistillationRun

BASE = {
    "run": {"tag": "engine"},
    "teacher": {"backbone": "resnet18", "weights": "random", "embed_dim": 16},
    "student": {"backbone": "resnet18", "pretrained": False, "embed_dim": 16},
    "head": {"pooling": "gap", "losses": {"sce": 1.0}},
    "distill": {"objective": "cosine", "alpha": 1.0},
    "data": {
        "adapter": "synthetic", "input_size": 32, "sampler": "pk",
        "p_classes": 2, "k_samples": 2, "num_workers": 0,
        "synthetic": {"num_classes": 2, "per_class": 4},
    },
    "train": {"epochs": 1, "amp": False, "seed": 42, "warmup_epochs": 0},
    "eval": {"batch_size": 16, "report_retention": False},
}


def _cfg(tmp_path, **extra):
    cfg = deep_update(DEFAULTS, BASE)
    cfg = deep_update(cfg, {"run": {"output_dir": str(tmp_path)}})
    return deep_update(cfg, extra)


@pytest.mark.slow
def test_alpha_zero_ignores_missing_teacher_checkpoint(tmp_path):
    # Standalone runs launched from a distill config (--set distill.alpha=0)
    # must not trip over the config's placeholder teacher weights path.
    cfg = _cfg(tmp_path,
               teacher={"backbone": "resnet18", "weights": "runs/does_not_exist.pth",
                        "embed_dim": 16},
               distill={"objective": "cosine", "alpha": 0.0})
    run = DistillationRun(copy.deepcopy(cfg), device="cpu")
    assert run.student is not None


@pytest.mark.slow
def test_alpha_zero_trains_standalone_and_skips_teacher(tmp_path):
    cfg = _cfg(tmp_path, distill={"objective": "cosine", "alpha": 0.0})
    run = DistillationRun(copy.deepcopy(cfg), device="cpu")
    result = run.fit()
    record = result["history"][0]
    assert record["distill"] == 0.0
    assert record["total"] == pytest.approx(record["task"])
    # Teacher must stay on CPU untouched; its checkpoint becomes a teacher later.
    assert (run.out_dir / "best.pth").exists() or (run.out_dir / "last.pth").exists()


@pytest.mark.slow
def test_teacher_checkpoint_roundtrip(tmp_path):
    # Train "teacher" standalone (resnet18 with sce), then load its best.pth
    # as the teacher of a distillation run with the cosine objective (which
    # does not need the saved classifier: unexpected classifier keys are ok).
    teacher_cfg = _cfg(tmp_path, distill={"objective": "cosine", "alpha": 0.0})
    teacher_run = DistillationRun(copy.deepcopy(teacher_cfg), device="cpu")
    ckpt = teacher_run.fit()["checkpoints"]["last"]

    distill_cfg = _cfg(
        tmp_path,
        run={"tag": "kd", "output_dir": str(tmp_path / "kd")},
        teacher={"backbone": "resnet18", "weights": ckpt, "embed_dim": 16},
        head={"pooling": "gap", "losses": {"triplet": 1.0}},  # no classifier needed
        distill={"objective": "cosine", "alpha": 1.0},
    )
    run = DistillationRun(copy.deepcopy(distill_cfg), device="cpu")
    result = run.fit()
    assert result["history"][0]["distill"] > 0.0


def test_two_tier_learning_rate_param_groups():
    # Inherited anti-collapse recipe: pretrained backbone at ~1/10 head LR.
    import torch

    from embedkd.engine.trainer import _build_param_groups
    from embedkd.losses import build_task_loss

    from .utils import tiny_embedding_model

    student = tiny_embedding_model(embed_dim=8, num_classes=4)
    task_loss = build_task_loss({"losses": {"sce": 1.0}}, embed_dim=8, num_classes=4)

    groups = _build_param_groups(student, task_loss, {"lr": 3e-4, "lr_backbone": None})
    assert groups[0]["lr"] == pytest.approx(3e-5)  # backbone default = lr / 10
    assert groups[1]["lr"] == pytest.approx(3e-4)

    explicit = _build_param_groups(student, task_loss, {"lr": 3e-4, "lr_backbone": 1e-5})
    assert explicit[0]["lr"] == pytest.approx(1e-5)

    # Every trainable parameter is in exactly one group.
    ids_backbone = {id(p) for p in groups[0]["params"]}
    ids_head = {id(p) for p in groups[1]["params"]}
    assert not ids_backbone & ids_head
    total = len(list(student.parameters())) + len(list(task_loss.parameters()))
    assert len(ids_backbone) + len(ids_head) == total
    assert torch.optim.AdamW(groups, weight_decay=1e-4)  # constructible


@pytest.mark.slow
def test_eval_every_skips_intermediate_epochs(tmp_path):
    cfg = _cfg(tmp_path, train={"epochs": 3, "eval_every": 3, "amp": False,
                                "seed": 42, "warmup_epochs": 0})
    run = DistillationRun(copy.deepcopy(cfg), device="cpu")
    history = run.fit()["history"]
    assert "val_map" not in history[0]
    assert "val_map" not in history[1]
    assert "val_map" in history[2]  # last epoch always evaluated
