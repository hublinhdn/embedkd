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


@pytest.mark.slow
def test_eval_every_skips_intermediate_epochs(tmp_path):
    cfg = _cfg(tmp_path, train={"epochs": 3, "eval_every": 3, "amp": False,
                                "seed": 42, "warmup_epochs": 0})
    run = DistillationRun(copy.deepcopy(cfg), device="cpu")
    history = run.fit()["history"]
    assert "val_map" not in history[0]
    assert "val_map" not in history[1]
    assert "val_map" in history[2]  # last epoch always evaluated
