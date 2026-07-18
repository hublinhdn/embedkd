"""The expected-results generator: published numbers come from this script."""

import json
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "make_expected_results.py"


def _fake_run(repo: Path, tag: str, maps: list[float]) -> None:
    run_dir = repo / "runs" / f"20260717_000000_{tag}"
    run_dir.mkdir(parents=True)
    lines = [json.dumps({"epoch": i, "total": 1.0, "val_map": m, "val_r1": m + 0.1})
             for i, m in enumerate(maps)]
    (run_dir / "log.jsonl").write_text("\n".join(lines), encoding="utf-8")
    (run_dir / "best.pth").write_bytes(b"fake")


def test_generates_spec_with_seed_tolerances(tmp_path):
    _fake_run(tmp_path, "d9_cosine_s42", [0.50, 0.60, 0.58])
    _fake_run(tmp_path, "d9_cosine_s43", [0.59])
    _fake_run(tmp_path, "d9_cosine_s44", [0.62])
    _fake_run(tmp_path, "d9_mse_s42", [0.55])

    out = subprocess.run(
        [sys.executable, str(SCRIPT), "d9_demo",
         "--runs", "d9_cosine_s42:cosine", "d9_mse_s42:mse",
         "--seed-runs", "d9_cosine_s42", "d9_cosine_s43", "d9_cosine_s44",
         "--config", "configs/d1_cub200_cosine.yaml",
         "--repo-root", str(tmp_path)],
        capture_output=True, text=True,
    )
    assert out.returncode == 0, out.stderr
    spec = json.loads((tmp_path / "expected_results" / "d9_demo.json").read_text())
    # Best epoch of the primary run is 0.60; tolerance from seeds {0.60, 0.59, 0.62}.
    assert spec["expected"]["map"]["value"] == 0.6
    assert spec["expected"]["map"]["tolerance"] >= 0.005
    assert set(spec["all_rows"]) == {"cosine", "mse"}
    assert spec["all_rows"]["mse"]["map"]["value"] == 0.55
    assert spec["checkpoint"].endswith("best.pth")
    # Committed specs must be machine-portable: repo-relative, never absolute.
    assert not spec["checkpoint"].startswith("/")
    assert spec["checkpoint"].startswith("runs/")
