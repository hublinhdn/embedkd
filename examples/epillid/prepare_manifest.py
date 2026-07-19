#!/usr/bin/env python3
"""Build an EmbedKD csv_manifest from the ePillID benchmark label file.

Usage:
  python prepare_manifest.py <all_labels.csv> <images_root> [--out manifest.csv]
      [--holdout-fraction 0.25]

Mapping: a deterministic fraction of pill classes is held out; their reference
images become the gallery and their consumer images the queries. All images of
the remaining classes are the training split. See README.md in this directory.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path

REQUIRED = {"image_path", "label", "is_ref"}


def _is_holdout(label: str, fraction: float) -> bool:
    digest = int(hashlib.md5(label.encode("utf-8")).hexdigest(), 16)
    return (digest % 10_000) / 10_000.0 < fraction


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("labels_csv")
    parser.add_argument("images_root")
    parser.add_argument("--out", default="manifest.csv")
    parser.add_argument("--holdout-fraction", type=float, default=0.25)
    args = parser.parse_args()

    root = Path(args.images_root)
    with open(args.labels_csv, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None or not REQUIRED <= set(reader.fieldnames):
            raise SystemExit(
                f"{args.labels_csv} must have columns {sorted(REQUIRED)}; "
                f"found {reader.fieldnames}. Is this the ePillID all_labels.csv?"
            )
        records = list(reader)

    rows, skipped = [], 0
    for record in records:
        rel, label = record["image_path"], record["label"]
        is_ref = str(record["is_ref"]).strip().lower() in ("true", "1", "yes")
        if not (root / rel).exists():
            skipped += 1
            continue
        if _is_holdout(label, args.holdout_fraction):
            split = "gallery" if is_ref else "query"
        else:
            split = "train"
        rows.append({"path": rel, "label": label, "split": split})

    if not rows:
        raise SystemExit(f"No images found under {root}; check images_root.")
    counts = {s: sum(r["split"] == s for r in rows) for s in ("train", "gallery", "query")}
    if not counts["gallery"] or not counts["query"]:
        raise SystemExit(f"Degenerate holdout ({counts}); adjust --holdout-fraction.")

    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["path", "label", "split"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {args.out}: {counts}, skipped {skipped} missing files")


if __name__ == "__main__":
    main()
