"""Run fingerprint: the birth certificate of every run.

Written to ``runs/<id>/fingerprint.yaml`` before training starts, so every
published number can be traced back to the exact configuration, code version
and environment that produced it.
"""

from __future__ import annotations

import platform
import socket
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import yaml


def _git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5,
            cwd=Path(__file__).resolve().parent,
        )
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def build_fingerprint(cfg: dict) -> dict:
    import numpy
    import timm
    import torch

    from .. import __version__

    return {
        "embedkd_version": __version__,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config": cfg,
        "environment": {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "timm": str(timm.__version__),
            "numpy": str(numpy.__version__),
            "cuda_available": torch.cuda.is_available(),
            "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
        },
        "git_commit": _git_commit(),
    }


def write_fingerprint(cfg: dict, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "fingerprint.yaml"
    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(build_fingerprint(cfg), fh, sort_keys=False)
    return path
