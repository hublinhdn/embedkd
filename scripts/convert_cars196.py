#!/usr/bin/env python3
"""Convert the official Cars196 annotations (cars_annos.mat) to the CSV layout
expected by the EmbedKD cars196 adapter.

Usage:
  python scripts/convert_cars196.py /path/to/cars196
where the directory contains car_ims/ and cars_annos.mat (from the official
archive or the Kaggle 'cars196' mirror). Writes cars_annos.csv next to it.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    base = Path(sys.argv[1])
    mat_path = base / "cars_annos.mat"
    if not mat_path.exists():
        raise SystemExit(f"{mat_path} not found. Expected the official cars_annos.mat.")
    try:
        from scipy.io import loadmat
    except ImportError:
        raise SystemExit("This converter needs scipy: pip install scipy") from None

    mat = loadmat(str(mat_path), squeeze_me=True)
    annotations = mat["annotations"]
    rows = []
    for entry in annotations:
        # Fields per the official file: relative_im_path, bbox_*, class, test
        rel_path = str(entry["relative_im_path"])
        label = int(entry["class"])
        rows.append({"path": rel_path, "label": str(label)})
    if not rows:
        raise SystemExit("No annotations parsed; is this the official cars_annos.mat?")

    out = base / "cars_annos.csv"
    with open(out, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["path", "label"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {out} ({len(rows)} images, {len({r['label'] for r in rows})} classes)")
    missing = [r["path"] for r in rows[:2000] if not (base / r["path"]).exists()]
    if missing:
        print(f"WARNING: {len(missing)} referenced images missing (checked first 2000); "
              f"first: {missing[:3]}")


if __name__ == "__main__":
    main()
