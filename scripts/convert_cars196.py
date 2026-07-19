#!/usr/bin/env python3
"""Convert Cars196 annotations to the CSV layout expected by the EmbedKD
cars196 adapter. Auto-detects both common distributions:

  A) original archive:     car_ims/ + cars_annos.mat
  B) torchvision/Kaggle:   cars_train/ + cars_test/ + cars_train_annos.mat
                           + cars_test_annos_withlabels.mat (in ./ or devkit/)

Usage:
  python scripts/convert_cars196.py data/cars196
Writes cars_annos.csv into that directory with image paths relative to it.
Nested one-level folders (e.g. stanford_cars/) are searched automatically.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path


def _loadmat(path: Path):
    try:
        from scipy.io import loadmat
    except ImportError:
        raise SystemExit("This converter needs scipy: pip install scipy") from None
    return loadmat(str(path), squeeze_me=True)


def _rows_original(base: Path, prefix: Path) -> list[dict]:
    mat = _loadmat(base / prefix / "cars_annos.mat")
    rows = []
    for entry in mat["annotations"]:
        rel = str(entry["relative_im_path"])
        rows.append({"path": str(prefix / rel), "label": str(int(entry["class"]))})
    return rows


def _find_annos(base: Path, prefix: Path, name: str) -> Path | None:
    for cand in (base / prefix / name, base / prefix / "devkit" / name):
        if cand.exists():
            return cand
    return None


def _rows_torchvision(base: Path, prefix: Path) -> list[dict]:
    rows = []
    specs = [
        ("cars_train_annos.mat", "cars_train"),
        ("cars_test_annos_withlabels.mat", "cars_test"),
        ("cars_test_annos_withlabels (1).mat", "cars_test"),  # common Kaggle name
    ]
    seen_dirs = set()
    for mat_name, img_dir in specs:
        if img_dir in seen_dirs:
            continue
        mat_path = _find_annos(base, prefix, mat_name)
        if mat_path is None:
            continue
        seen_dirs.add(img_dir)
        for entry in _loadmat(mat_path)["annotations"]:
            fname = str(entry["fname"])
            rows.append({"path": str(prefix / img_dir / fname),
                         "label": str(int(entry["class"]))})
    if "cars_test" not in seen_dirs:
        raise SystemExit(
            "Found cars_train annotations but no cars_test_annos_withlabels.mat "
            "(the WITH-LABELS test file is required; the plain test annos have no classes)."
        )
    return rows


def detect_and_convert(base: Path) -> list[dict]:
    for prefix in (Path("."), *[p.relative_to(base) for p in sorted(base.iterdir())
                                if p.is_dir()]):
        if (base / prefix / "cars_annos.mat").exists():
            print(f"Detected ORIGINAL layout under {base / prefix}")
            return _rows_original(base, prefix)
        if (base / prefix / "cars_train").is_dir() and \
                _find_annos(base, prefix, "cars_train_annos.mat"):
            print(f"Detected TORCHVISION layout under {base / prefix}")
            return _rows_torchvision(base, prefix)
    raise SystemExit(
        f"No known Cars196 layout under '{base}'. Expected either "
        "cars_annos.mat + car_ims/, or cars_train/ + cars_test/ + annotation .mat files. "
        f"Top-level entries found: {[p.name for p in sorted(base.iterdir())][:10]}"
    )


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    base = Path(sys.argv[1])
    if not base.is_dir():
        raise SystemExit(f"'{base}' is not a directory")
    rows = detect_and_convert(base)
    if not rows:
        raise SystemExit("No annotations parsed.")

    missing = [r["path"] for r in rows if not (base / r["path"]).exists()]
    if len(missing) > len(rows) * 0.01:
        raise SystemExit(
            f"{len(missing)}/{len(rows)} referenced images are missing; first: {missing[:3]}. "
            "The archive is probably incomplete or the layout differs."
        )

    out = base / "cars_annos.csv"
    with open(out, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["path", "label"])
        writer.writeheader()
        writer.writerows(rows)
    classes = {r["label"] for r in rows}
    print(f"Wrote {out}: {len(rows)} images, {len(classes)} classes"
          + (f", {len(missing)} missing files ignored" if missing else ""))
    if len(classes) != 196:
        print(f"WARNING: expected 196 classes, found {len(classes)}; check the source archive.")


if __name__ == "__main__":
    main()
