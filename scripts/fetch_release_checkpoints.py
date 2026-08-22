#!/usr/bin/env python3
"""Download released checkpoints into the exact paths `embedkd reproduce`
expects, so verifying published numbers is two commands on any machine:

  python scripts/fetch_release_checkpoints.py d1_cub200
  embedkd reproduce d1_cub200 --eval-only

Asset naming convention on the GitHub Release: <demo_id>_student_best.pth
(plus <demo_id>_teacher_best.pth for retraining/diagnostics without redoing
the teacher). Placement comes from expected_results/<demo_id>.json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RELEASE_URL = "https://github.com/hublinhdn/embedkd/releases/download/{tag}/{asset}"


def fetch(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)

    def report(blocks: int, block_size: int, total: int) -> None:
        if total > 0:
            done = min(blocks * block_size / total, 1.0)
            sys.stderr.write(f"\r  {dest.name}: {done:6.1%}")
    try:
        urllib.request.urlretrieve(url, dest, reporthook=report)
    finally:
        sys.stderr.write("\n")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check(path: Path, expected: str | None, label: str) -> None:
    """A download nobody verified is not a reproduction. Fail loudly instead."""
    if not expected:
        print(f"{label}: no checksum recorded for this release, skipping verification")
        return
    got = sha256(path)
    if got != expected:
        raise SystemExit(
            f"{label}: checksum mismatch for {path}\n"
            f"  expected {expected}\n  got      {got}\n"
            "Delete the file and download it again; do not reproduce from it."
        )
    print(f"{label}: checksum verified")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("demo_ids", nargs="*",
                        help="default: every demo with an expected-results spec")
    parser.add_argument("--tag", default="v0.1.3")
    parser.add_argument("--teacher", action="store_true",
                        help="also fetch the matching teacher checkpoint into runs/<demo>_teacher/")
    args = parser.parse_args()

    specs_dir = REPO / "expected_results"
    demo_ids = args.demo_ids or sorted(p.stem for p in specs_dir.glob("d*.json"))
    for demo_id in demo_ids:
        spec_path = specs_dir / f"{demo_id}.json"
        if not spec_path.exists():
            raise SystemExit(f"No expected-results spec for '{demo_id}'")
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
        student_dest = REPO / spec["checkpoint"]
        if student_dest.exists():
            print(f"{demo_id}: student checkpoint already at {spec['checkpoint']}")
        else:
            fetch(RELEASE_URL.format(tag=args.tag, asset=f"{demo_id}_student_best.pth"),
                  student_dest)
            print(f"{demo_id}: -> {spec['checkpoint']}")
        check(student_dest, spec.get("checkpoint_sha256"), f"{demo_id} student")

        if args.teacher:
            teacher_dest = REPO / "runs" / f"{demo_id}_teacher" / "best.pth"
            if not teacher_dest.exists():
                fetch(RELEASE_URL.format(tag=args.tag, asset=f"{demo_id}_teacher_best.pth"),
                      teacher_dest)
                print(f"{demo_id}: teacher -> {teacher_dest.relative_to(REPO)}")
            check(teacher_dest, spec.get("teacher", {}).get("sha256"), f"{demo_id} teacher")


if __name__ == "__main__":
    main()
