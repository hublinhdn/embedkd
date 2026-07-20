#!/usr/bin/env python
"""Verify that the released teacher checkpoints reproduce the teacher metrics
frozen in expected_results/<demo>.json (the numbers behind Table 4's Teacher
and Gap columns, and the deployment retention).

This closes the gap that let a wrong teacher mAP reach a draft once: teacher
metrics used to be recorded by hand and were never re-checked. Now they live in
expected_results and this script re-measures them from the released checkpoints.

The teacher is evaluated through the student forward path (student.backbone set
to the teacher backbone, distill.alpha=0), the same path used to freeze the
expected values, so a clean checkout reproduces them.

Usage:
  python scripts/verify_teacher_metrics.py                    # every demo with a teacher block
  python scripts/verify_teacher_metrics.py d2_cars196         # one demo
  python scripts/verify_teacher_metrics.py d4_epillid \
      --set data.root=<ePillID classification_data dir>       # d4 needs the images root

Teacher checkpoints are fetched from the release (default tag below) if absent.
Exit code 0 = all pass, 2 = at least one mismatch.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

REPO = Path(__file__).resolve().parents[1]
EXPECTED = REPO / "expected_results"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("demo_ids", nargs="*",
                   help="default: every demo whose spec carries a teacher block")
    p.add_argument("--set", action="append", default=[],
                   help="config overrides (e.g. data.root=... for d4_epillid)")
    p.add_argument("--tag", default="v0.1.2",
                   help="release tag to fetch teacher checkpoints from if missing")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    import torch

    from embedkd.evaluation.retrieval import evaluate_model
    from embedkd.run import DistillationRun

    demos = args.demo_ids or sorted(p.stem for p in EXPECTED.glob("*.json"))
    failures: list[str] = []
    skipped: list[str] = []
    checked = 0
    for demo in demos:
        spec_path = EXPECTED / f"{demo}.json"
        if not spec_path.exists():
            print(f"{demo}: no expected-results spec", file=sys.stderr)
            failures.append(demo)
            continue
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
        teacher = spec.get("teacher")
        if not teacher:
            continue  # demo has no teacher block to verify

        ckpt = REPO / "runs" / f"{demo}_teacher" / "best.pth"
        if not ckpt.exists():
            subprocess.run(
                [sys.executable, str(REPO / "scripts" / "fetch_release_checkpoints.py"),
                 demo, "--teacher", "--tag", args.tag],
                check=False,
            )
        if not ckpt.exists():
            print(f"{demo}: teacher checkpoint not found at {ckpt}", file=sys.stderr)
            failures.append(demo)
            continue

        overrides = list(args.set) + [
            f"student.backbone={teacher['backbone']}",
            "distill.alpha=0",
            "eval.report_retention=false",
        ]
        run = DistillationRun.from_config(spec["config"], overrides)

        # Some demos (e.g. ePillID/D4 via csv_manifest) store image paths
        # relative to an images root the user must supply. Without it the loader
        # would fail deep inside the DataLoader; detect it here and skip cleanly
        # instead of crashing, telling the user exactly how to run it.
        if not run.cfg["data"].get("root"):
            print(f"[{demo}] SKIPPED: needs an images root. Re-run:\n"
                  f"    python scripts/verify_teacher_metrics.py {demo} "
                  f"--set data.root=<dir containing the images>")
            skipped.append(demo)
            continue

        checked += 1
        state = torch.load(ckpt, map_location="cpu", weights_only=True)
        run.student.load_state_dict(state["state_dict"])
        try:
            m = evaluate_model(
                run.student.to(run.device), run.bundle.gallery, run.bundle.query,
                batch_size=run.cfg["eval"]["batch_size"], device=run.device,
            )
        except FileNotFoundError as exc:
            print(f"[{demo}] SKIPPED: image not found ({exc.filename}); check "
                  f"--set data.root=<...> points at the images root.")
            skipped.append(demo)
            checked -= 1
            continue
        print(f"[{demo}] teacher ({teacher['backbone']}):")
        for name in ("map", "r1"):
            exp = teacher[name]
            got = m.get(name)
            lo, hi = exp["value"] - exp["tolerance"], exp["value"] + exp["tolerance"]
            ok = got is not None and lo <= got <= hi
            got_str = f"{got:.4f}" if got is not None else "None"
            print(f"  {name:>3}: expected {exp['value']} +/- {exp['tolerance']} "
                  f"| got {got_str} -> {'PASS' if ok else 'FAIL'}")
            if not ok:
                failures.append(f"{demo}.{name}")

    if skipped:
        print(f"\nSkipped (need --set data.root=<images root>): {skipped}")
    if failures:
        print("FAIL:", failures, file=sys.stderr)
        return 2
    if not checked:
        print("Nothing verified: every requested demo was skipped or had no teacher block.",
              file=sys.stderr)
        return 1
    print(f"All {checked} teacher metric set(s) verified against the released checkpoints.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
