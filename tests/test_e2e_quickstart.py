"""End-to-end and determinism tests on the synthetic quickstart pipeline."""

import copy
import json

import pytest

from embedkd.config import DEFAULTS, deep_update
from embedkd.run import DistillationRun

QUICK_CFG = {
    "run": {"tag": "e2e"},
    "teacher": {"backbone": "resnet18", "weights": "random", "embed_dim": 32},
    "student": {"backbone": "resnet18", "pretrained": False, "embed_dim": 32},
    "head": {"pooling": "gem", "losses": {"sce": 1.0, "triplet": 1.0}},
    "distill": {"objective": "cosine", "alpha": 1.0},
    "data": {
        "adapter": "synthetic", "input_size": 32, "sampler": "pk",
        "p_classes": 3, "k_samples": 3, "num_workers": 0,
        "synthetic": {"num_classes": 3, "per_class": 9},
    },
    "train": {"epochs": 1, "amp": False, "seed": 42, "warmup_epochs": 0},
    "eval": {"batch_size": 32},
}


def _cfg(tmp_path, **extra):
    cfg = deep_update(DEFAULTS, QUICK_CFG)
    cfg = deep_update(cfg, {"run": {"output_dir": str(tmp_path)}})
    return deep_update(cfg, extra)


@pytest.mark.slow
def test_fit_eval_diagnose_pipeline(tmp_path):
    run = DistillationRun(_cfg(tmp_path), device="cpu")
    result = run.fit()
    assert len(result["history"]) == 1
    record = result["history"][0]
    assert all(k in record for k in ("total", "task", "distill", "val_map"))
    assert (run.out_dir / "fingerprint.yaml").exists()
    assert (run.out_dir / "log.jsonl").exists()
    with open(run.out_dir / "log.jsonl", encoding="utf-8") as fh:
        assert json.loads(fh.readline())["epoch"] == 0

    metrics = run.evaluate(checkpoint=result["checkpoints"]["best"])
    assert 0.0 <= metrics["map"] <= 1.0
    assert "retention" in metrics or "teacher_map" not in metrics

    report = run.diagnose()
    assert 0.0 <= report["cka_pre"] <= 1.0
    assert report["capacity_ratio"] == pytest.approx(1.0, abs=0.05)


@pytest.mark.slow
def test_same_seed_same_results(tmp_path):
    finals = []
    for tag in ("a", "b"):
        cfg = _cfg(tmp_path, run={"tag": tag, "output_dir": str(tmp_path / tag)})
        run = DistillationRun(copy.deepcopy(cfg), device="cpu")
        result = run.fit()
        finals.append(result["history"][-1])
    for key in ("total", "task", "distill", "val_map"):
        assert finals[0][key] == pytest.approx(finals[1][key], abs=1e-6), key


@pytest.mark.slow
def test_different_seed_different_training(tmp_path):
    finals = []
    for seed in (42, 43):
        cfg = _cfg(tmp_path, run={"tag": f"s{seed}", "output_dir": str(tmp_path / str(seed))},
                   train={"seed": seed})
        run = DistillationRun(copy.deepcopy(cfg), device="cpu")
        finals.append(run.fit()["history"][-1]["total"])
    assert finals[0] != finals[1]
