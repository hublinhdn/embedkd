#!/usr/bin/env python3
"""Demo D5: assemble the diagnostics story for one teacher-student pair.

Reads existing artifacts (pre-distillation compatibility report + expected
results), measures post-distillation CKA with the trained student, classifies
the outcome via distill_report, and renders the paper figure.

Example:
  python scripts/make_d5_diagnostics.py d1 \
    --config configs/d1_cub200_cosine.yaml \
    --teacher runs/<id>_d1_teacher_resnet50/best.pth \
    --student-ckpt runs/<id>_d1_cosine_s42/best.pth \
    --pre-report runs/<id>_d1_teacher_resnet50/compatibility_report.json \
    --expected expected_results/d1_cub200.json \
    --row cosine --baseline-row no_kd --out-dir figures/d5
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("pair_id", help="label for output files, e.g. d1")
    parser.add_argument("--config", required=True)
    parser.add_argument("--teacher", required=True)
    parser.add_argument("--student-ckpt", required=True)
    parser.add_argument("--pre-report", required=True)
    parser.add_argument("--expected", required=True)
    parser.add_argument("--row", default="cosine", help="distilled row in expected results")
    parser.add_argument("--baseline-row", default="no_kd")
    parser.add_argument("--out-dir", default="figures/d5")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    args = parser.parse_args()

    import torch

    from embedkd.diagnostics import compatibility_report, distill_report
    from embedkd.run import DistillationRun

    pre = json.loads(Path(args.pre_report).read_text(encoding="utf-8"))
    expected = json.loads(Path(args.expected).read_text(encoding="utf-8"))
    after = {m: v["value"] for m, v in expected["all_rows"][args.row].items()}
    before = {m: v["value"] for m, v in expected["all_rows"][args.baseline_row].items()}

    overrides = list(args.set) + [f"teacher.weights={args.teacher}"]
    run = DistillationRun.from_config(args.config, overrides)
    state = torch.load(args.student_ckpt, map_location="cpu", weights_only=True)
    run.student.load_state_dict(state["state_dict"])
    post = compatibility_report(
        run.teacher.to(run.device), run.student.to(run.device),
        run.bundle.query, batch_size=run.cfg["eval"]["batch_size"], device=run.device,
    )

    report = distill_report(pre, post["cka_pre"], before, after, metric="map")
    report.update({
        "pair_id": args.pair_id,
        "cka_rbf_pre": pre.get("cka_rbf_pre"),
        "cka_rbf_post": post.get("cka_rbf_pre"),
        "capacity_ratio": pre.get("capacity_ratio"),
        "baseline_row": args.baseline_row,
        "distilled_row": args.row,
    })

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{args.pair_id}_distill_report.json"
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))

    try:
        from embedkd.diagnostics.plots import plot_distill_summary

        fig_path = plot_distill_summary(report, out_dir / f"{args.pair_id}_distill_summary.png")
        print(f"Figure: {fig_path}")
    except ImportError:
        print("matplotlib not installed; skipped the figure (pip install 'embedkd[plots]')")
    print(f"Report: {json_path}")


if __name__ == "__main__":
    main()
