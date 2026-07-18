#!/usr/bin/env python3
"""Aggregate finished runs into an expected-results spec for `embedkd reproduce`.

Every published number must come from a committed script (project rule); this
is that script. It reads the val_* metrics of the best epoch from each run's
log.jsonl, derives tolerances from seed replicates, and writes
expected_results/<demo_id>.json.

Example (what run_d1_cub200.sh calls):
  python scripts/make_expected_results.py d1_cub200 \
    --runs d1_cosine_s42:cosine d1_mse_s42:mse \
    --seed-runs d1_cosine_s42 d1_cosine_s43 d1_cosine_s44 \
    --config configs/d1_cub200_cosine.yaml --checkpoint-tag d1_cosine_s42
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

DEFAULT_REPO = Path(__file__).resolve().parent.parent
MIN_TOLERANCE = 0.005  # floor so a tolerance of 0.000 never fails honest reruns
REPO = DEFAULT_REPO  # overridden by --repo-root (kept global for helpers)


def latest_run_dir(tag: str) -> Path:
    candidates = sorted(REPO.glob(f"runs/*_{tag}"), key=lambda p: p.name)
    if not candidates:
        raise SystemExit(f"No run directory found for tag '{tag}' under runs/")
    return candidates[-1]


def best_val_metrics(run_dir: Path) -> dict[str, float]:
    records = [json.loads(line) for line in (run_dir / "log.jsonl").read_text().splitlines()]
    evaluated = [r for r in records if "val_map" in r]
    if not evaluated:
        raise SystemExit(f"{run_dir} has no evaluated epochs in log.jsonl")
    best = max(evaluated, key=lambda r: r["val_map"])
    return {k.removeprefix("val_"): v for k, v in best.items() if k.startswith("val_")}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("demo_id")
    parser.add_argument("--runs", nargs="+", required=True,
                        metavar="TAG:LABEL", help="runs to report, e.g. d1_cosine_s42:cosine")
    parser.add_argument("--seed-runs", nargs="*", default=[],
                        help="replicate tags of the primary run for tolerance estimation")
    parser.add_argument("--config", required=True,
                        help="config a reviewer should re-run for the primary row")
    parser.add_argument("--checkpoint-tag", default=None,
                        help="run tag whose best.pth backs 'reproduce --eval-only'")
    parser.add_argument("--metrics", nargs="*", default=["map", "r1"])
    parser.add_argument("--repo-root", default=str(DEFAULT_REPO),
                        help="repository root containing runs/ and expected_results/")
    args = parser.parse_args()
    global REPO
    REPO = Path(args.repo_root)

    # Tolerance = 2 * std across seed replicates of the primary metric set.
    tolerances: dict[str, float] = {}
    if len(args.seed_runs) >= 2:
        per_seed = [best_val_metrics(latest_run_dir(tag)) for tag in args.seed_runs]
        for metric in args.metrics:
            values = [m[metric] for m in per_seed if metric in m]
            spread = 2 * statistics.stdev(values) if len(values) >= 2 else MIN_TOLERANCE
            tolerances[metric] = round(max(spread, MIN_TOLERANCE), 4)
    else:
        tolerances = {metric: 0.02 for metric in args.metrics}  # conservative default

    rows = {}
    primary_tag = args.runs[0].split(":", 1)[0]
    for spec in args.runs:
        tag, label = spec.split(":", 1)
        metrics = best_val_metrics(latest_run_dir(tag))
        rows[label] = {
            metric: {"value": round(metrics[metric], 4), "tolerance": tolerances[metric]}
            for metric in args.metrics if metric in metrics
        }

    checkpoint_tag = args.checkpoint_tag or primary_tag
    # Repo-relative: the committed spec must not leak machine-local absolute
    # paths, and reproduce --eval-only resolves it from the repo root (the
    # released checkpoint is attached to the matching GitHub Release).
    checkpoint_rel = (latest_run_dir(checkpoint_tag) / "best.pth").relative_to(REPO)
    spec = {
        "demo_id": args.demo_id,
        "config": args.config,
        "checkpoint": str(checkpoint_rel),
        "seed_replicates": args.seed_runs,
        # 'expected' drives embedkd reproduce; it checks the primary row.
        "expected": rows[args.runs[0].split(":", 1)[1]],
        "all_rows": rows,
    }
    out = REPO / "expected_results" / f"{args.demo_id}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(spec, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out}")
    for label, metrics in rows.items():
        print(f"  {label:>10}: " + "  ".join(
            f"{m}={v['value']} +/- {v['tolerance']}" for m, v in metrics.items()))


if __name__ == "__main__":
    main()
