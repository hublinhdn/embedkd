#!/usr/bin/env python3
"""Revision experiment E13: what does the diagnostic module say about a real failure?

Reviewers 1, 2 and 3 all note the same gap: the toolkit defines the outcome
categories improved, aligned_but_worse and diverged, but no reported
demonstration ever produced anything except improved, so the categories were
never shown to fire on a genuine failure.

E1 produced two real failures: a distilled student transfers worse than its
own no-KD baseline in both cross-domain directions. This script runs the
diagnostic module on those exact cases and prints the label it assigns, using
the target domain query set as the probe:

  cka_pre   teacher against a freshly initialised student, on the target probe
  cka_post  teacher against the distilled student, on the same probe
  metrics   the cross-domain numbers measured by E1

Usage:
  python scripts/revision/e13_diagnose_failure_case.py \
      --results-dir runs/revision/e1_cross_domain \
      --out runs/revision/e13_failure_diagnosis.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

CASES = [
    {
        "name": "cub_to_cars",
        "config": "configs/d1_cub200_cosine.yaml",
        "teacher": "runs/d1_cub200_teacher/best.pth",
        "target_adapter": "cars196",
        "distilled": "runs/20260717_164702_d1_cosine_s42/best.pth",
        "before_json": "cub_to_cars_no_kd.json",
        "after_json": "cub_to_cars_distilled.json",
    },
    {
        "name": "cars_to_cub",
        "config": "configs/d2_cars196_convnext_mobilenet.yaml",
        "teacher": "runs/d2_cars196_teacher/best.pth",
        "target_adapter": "cub200",
        "distilled": "runs/20260718_084411_d2_cosine_s42/best.pth",
        "before_json": "cars_to_cub_no_kd.json",
        "after_json": "cars_to_cub_distilled.json",
    },
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="runs/revision/e1_cross_domain")
    parser.add_argument("--out", default="runs/revision/e13_failure_diagnosis.json")
    args = parser.parse_args()

    import torch

    from embedkd.diagnostics import compatibility_report, distill_report
    from embedkd.diagnostics.cka import format_report
    from embedkd.run import DistillationRun

    results_dir = Path(args.results_dir)
    reports = {}

    for case in CASES:
        print(f"\n=== {case['name']}: probing on the {case['target_adapter']} query set ===")
        run = DistillationRun.from_config(case["config"], [
            f"teacher.weights={case['teacher']}",
            "data.protocol=cross_domain",
            f"data.target.adapter={case['target_adapter']}",
        ])
        probe = run.bundle.target_query
        batch_size = run.cfg["eval"]["batch_size"]
        # compatibility_report feeds batches straight to the models, so both
        # have to sit on the device already; DistillationRun.diagnose does the
        # same move before calling it.
        teacher = run.teacher.to(run.device)
        student = run.student.to(run.device)

        # Fresh student: what the tool could report before any training.
        pre = compatibility_report(teacher, student, probe,
                                   batch_size=batch_size, device=run.device)

        # Distilled student: how close it ended up to the teacher.
        state = torch.load(case["distilled"], map_location="cpu", weights_only=True)
        student.load_state_dict(state["state_dict"])
        student = student.to(run.device)
        post = compatibility_report(teacher, student, probe,
                                    batch_size=batch_size, device=run.device)

        before = json.loads((results_dir / case["before_json"]).read_text(encoding="utf-8"))
        after = json.loads((results_dir / case["after_json"]).read_text(encoding="utf-8"))

        report = distill_report(pre, post["cka_pre"], before, after)
        report["probe"] = f"{case['target_adapter']} query split"
        report["probe_size"] = pre["probe_size"]
        report["capacity_ratio"] = pre["capacity_ratio"]
        report["risk_pre"] = pre["risk"]
        print(format_report(report))
        reports[case["name"]] = report

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(reports, indent=2), encoding="utf-8")
    print(f"\nwritten to {out}")


if __name__ == "__main__":
    main()
