#!/usr/bin/env python3
"""Aggregate the D1 objective comparison into mean and standard deviation.

Reviewers 1, 2 and 3 all ask for repeated runs on the objectives the paper
compares. Seeds 42, 43 and 44 now exist for every one of them, so this script
turns runs/revision/collected.json into the table that replaces the
single-run rows of the submitted Table 3.

Usage:
  python scripts/revision/objective_summary.py runs/revision/collected.json
"""

from __future__ import annotations

import json
import statistics
import sys

NO_KD = {"map": 0.284426, "r1": 0.513889}

GROUPS = {
    "cosine (alpha 10)": [
        "20260717_164702_d1_cosine_s42",
        "20260717_174346_d1_cosine_s43",
        "20260717_180255_d1_cosine_s44",
    ],
    "rkd (alpha 1)": [
        "20260717_172344_d1_rkd_s42",
        "20260822_054559_e8_d1_rkd_s43",
        "20260822_055605_e8_d1_rkd_s44",
    ],
    "mse (alpha 10, as submitted)": [
        "20260717_165705_d1_mse_s42",
        "20260822_060617_e8_d1_mse_s43",
        "20260822_061621_e8_d1_mse_s44",
    ],
    "mse (alpha 2560, scale matched)": [
        "20260822_051538_e11_d1_mse_matched_s42",
        "20260822_052547_e11_d1_mse_matched_s43",
        "20260822_053552_e11_d1_mse_matched_s44",
    ],
    "kl (temperature 4)": [
        "20260717_170709_d1_kl_s42",
        "20260822_062623_e8_d1_kl_s43",
        "20260822_063633_e8_d1_kl_s44",
    ],
}


def main() -> None:
    path = sys.argv[1] if len(sys.argv) > 1 else "runs/revision/collected.json"
    data = json.load(open(path, encoding="utf-8"))

    print(f"no-KD baseline (single run): map {NO_KD['map']:.5f}  r1 {NO_KD['r1']:.5f}\n")
    header = f"{'objective':<34}{'mAP mean':>10}{'sd':>9}{'gain':>10}{'R@1 mean':>10}{'sd':>9}{'gain':>10}"
    print(header)
    for label, runs in GROUPS.items():
        missing = [r for r in runs if r not in data]
        if missing:
            print(f"{label:<34}  missing: {missing}")
            continue
        maps = [data[r]["map"] for r in runs]
        r1s = [data[r]["r1"] for r in runs]
        print(f"{label:<34}"
              f"{statistics.mean(maps):>10.5f}{statistics.stdev(maps):>9.5f}"
              f"{statistics.mean(maps) - NO_KD['map']:>+10.5f}"
              f"{statistics.mean(r1s):>10.5f}{statistics.stdev(r1s):>9.5f}"
              f"{statistics.mean(r1s) - NO_KD['r1']:>+10.5f}")
    print("\nsd is the sample standard deviation over seeds 42, 43 and 44.")


if __name__ == "__main__":
    main()
