#!/usr/bin/env python3
"""Revision experiment E16: run the diagnostic module across the E6 ladder.

E6 produced a harmful, a negligible and a beneficial outcome on one controlled
axis. E13 showed the module labelling the two cross-domain failures. This
script completes the picture by classifying every rung of the ladder, so the
paper can show the categories firing on outcomes that were produced on
purpose rather than found after the fact.

Probe is the CUB-200 query split, the same one the demos evaluate on.

Usage:
  python scripts/revision/e16_diagnose_ladder.py --out runs/revision/e16_ladder_diagnosis.json
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

CONFIG = "configs/d1_cub200_cosine.yaml"
NO_KD = {"map": 0.284426, "r1": 0.513889}   # published D1 baseline, constant here

LADDER = [
    {"name": "teacher_5_epochs", "teacher_glob": "runs/*_e6_teacher_e5",
     "student_glob": "runs/*_e6_student_from_e5"},
    {"name": "teacher_15_epochs", "teacher_glob": "runs/*_e6_teacher_e15",
     "student_glob": "runs/*_e6_student_from_e15"},
    {"name": "teacher_30_epochs", "teacher_glob": "runs/*_e6_teacher_e30",
     "student_glob": "runs/*_e6_student_from_e30"},
    {"name": "teacher_60_epochs_published", "teacher_glob": "runs/d1_cub200_teacher",
     "student_glob": "runs/20260717_164702_d1_cosine_s42"},
]


def newest(pattern: str) -> str:
    hits = sorted(glob.glob(pattern))
    if not hits:
        raise SystemExit(f"no run matches {pattern}")
    return hits[-1]


def best_metrics(run_dir: str) -> dict:
    best = None
    for line in open(Path(run_dir) / "log.jsonl", encoding="utf-8"):
        row = json.loads(line)
        if "val_map" in row and (best is None or row["val_map"] > best["val_map"]):
            best = row
    return {"map": best["val_map"], "r1": best.get("val_r1", 0.0)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="runs/revision/e16_ladder_diagnosis.json")
    args = parser.parse_args()

    import torch

    from embedkd.diagnostics import compatibility_report, distill_report
    from embedkd.diagnostics.cka import format_report
    from embedkd.run import DistillationRun

    reports = {}
    for rung in LADDER:
        teacher_dir = newest(rung["teacher_glob"])
        student_dir = newest(rung["student_glob"])
        print(f"\n=== {rung['name']}: {teacher_dir} -> {student_dir} ===")

        run = DistillationRun.from_config(CONFIG, [
            f"teacher.weights={teacher_dir}/best.pth",
        ])
        probe = run.bundle.query
        batch_size = run.cfg["eval"]["batch_size"]
        teacher = run.teacher.to(run.device)
        student = run.student.to(run.device)

        pre = compatibility_report(teacher, student, probe,
                                   batch_size=batch_size, device=run.device)
        state = torch.load(f"{student_dir}/best.pth", map_location="cpu", weights_only=True)
        student.load_state_dict(state["state_dict"])
        student = student.to(run.device)
        post = compatibility_report(teacher, student, probe,
                                    batch_size=batch_size, device=run.device)

        after = best_metrics(student_dir)
        report = distill_report(pre, post["cka_pre"], NO_KD, after)
        report["teacher_map"] = round(best_metrics(teacher_dir)["map"], 6)
        report["gap"] = round(report["teacher_map"] - NO_KD["map"], 6)
        report["risk_pre"] = pre["risk"]
        report["capacity_ratio"] = pre["capacity_ratio"]
        print(format_report(report))
        reports[rung["name"]] = report

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(reports, indent=2), encoding="utf-8")
    print(f"\nwritten to {out}")


if __name__ == "__main__":
    main()
