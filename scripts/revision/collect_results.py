#!/usr/bin/env python3
"""Collect the best retrieval metrics of every revision run into one table.

Reads runs/<id>/log.jsonl, takes the epoch with the highest val_map (the same
criterion best.pth is written on) and prints a table plus a JSON summary, so
the numbers that go into the response letter come from a committed script
rather than from reading logs by hand.

Usage:
  python scripts/revision/collect_results.py
  python scripts/revision/collect_results.py --out runs/revision/collected.json
"""

from __future__ import annotations

import argparse
import glob
import json
import os

# Reference points the revision compares against, all from the published D1 demo.
BASELINES = {
    "runs/20260717_164702_d1_cosine_s42": "D1 cosine s42 (published)",
    "runs/20260718_015853_d1_student_baseline": "D1 no_kd (published)",
    "runs/20260717_165705_d1_mse_s42": "D1 mse alpha 10 (published)",
    "runs/20260717_170709_d1_kl_s42": "D1 kl (published)",
    "runs/20260717_172344_d1_rkd_s42": "D1 rkd (published)",
    "runs/20260717_163324_d1_teacher_resnet50": "D1 teacher (published)",
    "runs/20260717_174346_d1_cosine_s43": "D1 cosine s43 (published)",
    "runs/20260717_180255_d1_cosine_s44": "D1 cosine s44 (published)",
}


def best_of(path: str) -> dict | None:
    best = None
    epochs = 0
    for line in open(path, encoding="utf-8"):
        row = json.loads(line)
        epochs += 1
        if "val_map" in row and (best is None or row["val_map"] > best["val_map"]):
            best = row
    if best is None:
        return None
    return {
        "epochs_logged": epochs,
        "best_epoch": best.get("epoch"),
        "map": round(best["val_map"], 6),
        "r1": round(best.get("val_r1", 0.0), 6),
        "r5": round(best.get("val_r5", 0.0), 6),
        "mrr": round(best.get("val_mrr", 0.0), 6),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="runs/revision/collected.json")
    args = parser.parse_args()

    dirs = sorted(glob.glob("runs/2026*_e[0-9]*/")) + [d + "/" for d in BASELINES]
    summary = {}
    print(f"{'RUN':<48} {'EP':>4} {'mAP':>9} {'R@1':>9}  NOTE")
    for d in dirs:
        log = os.path.join(d, "log.jsonl")
        if not os.path.exists(log):
            continue
        name = os.path.basename(d.rstrip("/"))
        rec = best_of(log)
        if rec is None:
            continue
        summary[name] = rec
        note = BASELINES.get(d.rstrip("/"), "")
        print(f"{name:<48} {rec['epochs_logged']:>4} {rec['map']:>9.5f} {rec['r1']:>9.5f}  {note}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\nwritten to {args.out}")


if __name__ == "__main__":
    main()
