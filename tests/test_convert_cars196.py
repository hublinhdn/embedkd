"""Cars196 converter: verified against fakes of both known distributions."""

import csv
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

scipy = pytest.importorskip("scipy")
from scipy.io import savemat  # noqa: E402

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "convert_cars196.py"


def _run(base: Path):
    return subprocess.run([sys.executable, str(SCRIPT), str(base)],
                          capture_output=True, text=True)


def _annos_struct(entries, fields):
    arr = np.zeros((len(entries),), dtype=[(f, object) for f in fields])
    for i, entry in enumerate(entries):
        for field in fields:
            arr[i][field] = entry[field]
    return arr


def _read_csv(base: Path) -> list[dict]:
    with open(base / "cars_annos.csv", newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def test_original_layout(tmp_path):
    entries = []
    for i, cls in enumerate([1, 1, 2, 196], start=1):
        rel = f"car_ims/{i:06d}.jpg"
        (tmp_path / "car_ims").mkdir(exist_ok=True)
        Image.new("RGB", (8, 8)).save(tmp_path / rel)
        entries.append({"relative_im_path": rel, "class": cls, "test": 0})
    savemat(tmp_path / "cars_annos.mat",
            {"annotations": _annos_struct(entries, ["relative_im_path", "class", "test"])})
    out = _run(tmp_path)
    assert out.returncode == 0, out.stderr
    rows = _read_csv(tmp_path)
    assert len(rows) == 4
    assert rows[0]["path"] == "car_ims/000001.jpg"
    assert {r["label"] for r in rows} == {"1", "2", "196"}


def test_torchvision_layout_nested(tmp_path):
    base = tmp_path / "stanford_cars"
    (base / "devkit").mkdir(parents=True)
    train, test = [], []
    for i, cls in enumerate([1, 2], start=1):
        fname = f"{i:05d}.jpg"
        (base / "cars_train").mkdir(exist_ok=True)
        Image.new("RGB", (8, 8)).save(base / "cars_train" / fname)
        train.append({"fname": fname, "class": cls})
    for i, cls in enumerate([3, 4], start=1):
        fname = f"{i:05d}.jpg"
        (base / "cars_test").mkdir(exist_ok=True)
        Image.new("RGB", (8, 8)).save(base / "cars_test" / fname)
        test.append({"fname": fname, "class": cls})
    savemat(base / "devkit" / "cars_train_annos.mat",
            {"annotations": _annos_struct(train, ["fname", "class"])})
    savemat(base / "cars_test_annos_withlabels.mat",
            {"annotations": _annos_struct(test, ["fname", "class"])})
    out = _run(tmp_path)  # converter must find the nested stanford_cars/ dir
    assert out.returncode == 0, out.stderr
    rows = _read_csv(tmp_path)
    assert len(rows) == 4
    assert all(r["path"].startswith("stanford_cars/cars_") for r in rows)
    assert {r["label"] for r in rows} == {"1", "2", "3", "4"}


def test_missing_withlabels_is_fatal(tmp_path):
    (tmp_path / "cars_train").mkdir()
    savemat(tmp_path / "cars_train_annos.mat",
            {"annotations": _annos_struct([{"fname": "x.jpg", "class": 1}],
                                          ["fname", "class"])})
    out = _run(tmp_path)
    assert out.returncode != 0
    assert "withlabels" in out.stderr.lower() or "withlabels" in out.stdout.lower()
