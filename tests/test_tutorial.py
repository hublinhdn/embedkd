"""Execute the tutorial's exact flow so the document cannot drift from the
code: generate the toy dataset, validate, fit, evaluate, and iterate on the
loss mix, all through the same entry points the tutorial shows."""

import json

import pytest
import yaml

from embedkd.cli import main

TUTORIAL_CONFIG = {
    "run": {"tag": "tutorial"},
    "teacher": {"backbone": "resnet50", "weights": "random", "embed_dim": 128},
    "student": {"backbone": "resnet18", "pretrained": False, "embed_dim": 128},
    "head": {"losses": {"sce": 1.0, "triplet": 1.0}, "sce": {"label_smoothing": 0.1}},
    "distill": {"objective": "cosine", "alpha": 10.0},
    "data": {"adapter": "image_folder", "root": None, "input_size": 64,
             "sampler": "pk", "p_classes": 4, "k_samples": 4, "num_workers": 0},
    "train": {"epochs": 1, "amp": False, "seed": 42, "warmup_epochs": 0},
    "eval": {"batch_size": 64},
}


def _toy_dataset(root):
    # Verbatim from docs/getting-started/tutorial.md, step 1.
    from pathlib import Path

    from PIL import Image

    for c in range(6):
        d = Path(root) / f"class_{c:02d}"
        d.mkdir(parents=True, exist_ok=True)
        for i in range(12):
            Image.new("RGB", (80, 80),
                      (40 * c % 255, 30 * i % 255, (60 + 25 * c) % 255)
                      ).save(d / f"img_{i:03d}.jpg")


@pytest.mark.slow
def test_tutorial_flow(tmp_path, capsys):
    _toy_dataset(tmp_path / "data" / "tutorial")

    # Step 2: health check.
    assert main(["datasets", "validate",
                 f"image_folder:{tmp_path / 'data' / 'tutorial'}"]) == 0
    assert "Result: OK" in capsys.readouterr().out
    assert (tmp_path / "data" / "tutorial" / "embedkd_generated_split.csv").exists()

    # Step 3: the tutorial config.
    cfg = dict(TUTORIAL_CONFIG)
    cfg["run"] = {"tag": "tutorial", "output_dir": str(tmp_path / "runs")}
    cfg["data"] = {**cfg["data"], "root": str(tmp_path / "data" / "tutorial")}
    config_path = tmp_path / "tutorial.yaml"
    config_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    # Step 4: diagnose runs and prints a risk level.
    assert main(["diagnose", "--config", str(config_path)]) == 0
    assert "risk" in capsys.readouterr().out

    # Step 5: fit produces the documented artifacts.
    assert main(["fit", "--config", str(config_path)]) == 0
    capsys.readouterr()
    run_dir = next((tmp_path / "runs").iterdir())
    for artifact in ("fingerprint.yaml", "log.jsonl", "best.pth", "last.pth"):
        assert (run_dir / artifact).exists(), artifact

    # Step 6: eval reports the documented metrics.
    assert main(["eval", "--config", str(config_path),
                 "--checkpoint", str(run_dir / "best.pth")]) == 0
    metrics = json.loads(capsys.readouterr().out)
    assert {"map", "mrr", "r1", "r5"} <= set(metrics)

    # Step 7: loss mix via --set; per-component logging appears.
    assert main(["fit", "--config", str(config_path),
                 "--set", "run.tag=tutorial_mix",
                 "--set", "head.losses={sce: 1.0, contrastive: 0.5}",
                 "--set", "distill.objective={cosine: 1.0, rkd: 0.5}"]) == 0
    capsys.readouterr()
    mix_dir = next(d for d in (tmp_path / "runs").iterdir()
                   if d.name.endswith("tutorial_mix"))
    record = json.loads((mix_dir / "log.jsonl").read_text().splitlines()[0])
    assert {"task_sce", "task_contrastive", "distill_cosine", "distill_rkd"} <= set(record)
