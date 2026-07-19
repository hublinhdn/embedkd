"""CLI plumbing tests (no training)."""

import csv

from PIL import Image

from embedkd.cli import main


def _folder(root, classes=2, per_class=4):
    for c in range(classes):
        d = root / f"class_{c}"
        d.mkdir(parents=True)
        for i in range(per_class):
            Image.new("RGB", (16, 16), color=(c * 90, i * 30, 10)).save(d / f"i{i}.jpg")


def test_diagnose_parser_accepts_checkpoint():
    from embedkd.cli import build_parser

    args = build_parser().parse_args(
        ["diagnose", "--config", "c.yaml", "--checkpoint", "s.pth", "--out", "r.json"])
    assert args.checkpoint == "s.pth"


def test_backbones_lists(capsys):
    assert main(["backbones"]) == 0
    out = capsys.readouterr().out
    assert "resnet50" in out and "experimental" in out


def test_datasets_validate_shorthand_with_set_override(tmp_path, capsys):
    # Regression: --set data.manifest=... was ignored in the adapter:root form.
    _folder(tmp_path / "imgs")
    manifest = tmp_path / "manifest.csv"
    rows = []
    for c in range(2):
        for i in range(4):
            split = "train" if i < 2 else ("gallery" if i == 2 else "query")
            rows.append({"path": f"class_{c}/i{i}.jpg", "label": str(c), "split": split})
    with open(manifest, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["path", "label", "split"])
        writer.writeheader()
        writer.writerows(rows)

    code = main(["datasets", "validate", f"csv_manifest:{tmp_path / 'imgs'}",
                 "--set", f"data.manifest={manifest}"])
    out = capsys.readouterr().out
    assert code == 0, out
    assert "Result: OK" in out


def test_datasets_validate_shorthand_image_folder(tmp_path, capsys):
    _folder(tmp_path)
    assert main(["datasets", "validate", f"image_folder:{tmp_path}"]) == 0
    assert "num_classes: 2" in capsys.readouterr().out
