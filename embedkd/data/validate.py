"""Dataset health check: catch data problems BEFORE training, not at epoch 30.

Checks: missing or unreadable images, empty splits, classes with fewer than
K images (the PK sampler would silently oversample them), split balance.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from PIL import Image

from .adapters import build_bundle


def _class_stats(labels: list[int]) -> dict:
    counts = sorted(Counter(labels).values())
    if not counts:
        return {"classes": 0, "min": 0, "median": 0, "max": 0}
    return {
        "classes": len(counts),
        "min": counts[0],
        "median": counts[len(counts) // 2],
        "max": counts[-1],
    }


def validate_dataset(data_cfg: dict, k_samples: int | None = None) -> dict:
    """Return {'errors': [...], 'warnings': [...], 'stats': {...}}."""
    identity = lambda img: img  # noqa: E731  (no tensor conversion needed for checks)
    errors: list[str] = []
    warnings: list[str] = []

    try:
        bundle = build_bundle(data_cfg, identity, identity)
    except (FileNotFoundError, ValueError) as exc:
        return {"errors": [str(exc)], "warnings": [], "stats": {}}

    stats: dict = {"num_classes": bundle.num_classes}
    corrupt: list[str] = []
    missing: list[str] = []

    for split_name in ("train", "gallery", "query"):
        dataset = getattr(bundle, split_name)
        items = getattr(dataset, "items", None)
        stats[split_name] = {"images": len(dataset)}
        if items is None:  # synthetic datasets have nothing to check on disk
            continue
        labels = [label for _, label in items]
        stats[split_name].update(_class_stats(labels))
        for path, _ in items:
            p = Path(path)
            if not p.exists():
                missing.append(str(p))
                continue
            try:
                with Image.open(p) as img:
                    img.verify()
            except Exception:  # noqa: BLE001 (any decode failure counts)
                corrupt.append(str(p))

    if missing:
        errors.append(f"{len(missing)} missing image file(s); first: {missing[:3]}")
    if corrupt:
        errors.append(f"{len(corrupt)} unreadable image file(s); first: {corrupt[:3]}")

    k = k_samples if k_samples is not None else int(data_cfg.get("k_samples", 4))
    train_items = getattr(bundle.train, "items", None)
    if train_items is not None:
        counts = Counter(label for _, label in train_items)
        thin = [cls for cls, n in counts.items() if n < k]
        if thin:
            warnings.append(
                f"{len(thin)} train class(es) have fewer than K={k} images; the PK "
                f"sampler will oversample them with replacement (class idx: {thin[:10]})"
            )

    return {"errors": errors, "warnings": warnings, "stats": stats}


def format_validation(report: dict) -> str:
    lines: list[str] = []
    for split, values in report.get("stats", {}).items():
        if isinstance(values, dict):
            detail = ", ".join(f"{k}={v}" for k, v in values.items())
            lines.append(f"  {split:>8}: {detail}")
        else:
            lines.append(f"  {split:>12}: {values}")
    for warning in report["warnings"]:
        lines.append(f"  WARNING: {warning}")
    for error in report["errors"]:
        lines.append(f"  ERROR: {error}")
    lines.append("Result: " + ("FAIL" if report["errors"] else
                               ("OK (with warnings)" if report["warnings"] else "OK")))
    return "\n".join(lines)
