#!/usr/bin/env python3
"""Revision experiment E4: audit ONNX export parity the way the reviewers asked.

Reviewer 1 asks for real samples from every supported dataset, maximum absolute
and relative output error, mAP and Recall@k before and after export, and
coverage of the advertised input sizes and backbone families. Reviewer 2 adds
that the export declares only the batch axis dynamic while the paper discusses
variable input sizes, so that has to be implemented and tested or dropped.

This script produces the evidence for all of it.

  A. element-wise parity, per backbone, per dataset, on real images
  B. dynamic spatial axes, one export served at several resolutions
  C. retrieval parity, mAP and Recall@k recomputed through the exported graph

Runs on CPU. Usage:

  python scripts/revision/parity_audit.py --out runs/revision/parity_audit.json
"""

from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path

# Dataset name to the config that knows how to build it.
DATASET_CONFIGS = {
    "cub200": "configs/d1_cub200_cosine.yaml",
    "cars196": "configs/d2_cars196_convnext_mobilenet.yaml",
    "sop": "configs/d3_sop_cosine.yaml",
}

TIER1_BACKBONES = [
    "resnet50",
    "convnext_tiny",
    "resnet18",
    "mobilenetv3_large_100",
    "efficientnet_b0",
]


def build_run(config: str, overrides: list[str]):
    from embedkd.run import DistillationRun

    # alpha 0 keeps the teacher branch out of the way; parity tests the export
    # path, so a randomly initialised student is a valid subject except where
    # the released checkpoint is loaded explicitly below.
    return DistillationRun.from_config(config, ["distill.alpha=0", *overrides])


def section_a(datasets: list[str], backbones: list[str], n_probes: int,
              out_dir: Path) -> list[dict]:
    from embedkd.deploy import export_onnx, parity_report, real_image_probes

    rows = []
    for ds in datasets:
        config = DATASET_CONFIGS[ds]
        for backbone in backbones:
            entry = {"dataset": ds, "backbone": backbone}
            try:
                run = build_run(config, [f"student.backbone={backbone}"])
                probes = real_image_probes(run.bundle.query, n_samples=n_probes)
                path = export_onnx(run.student, out_dir / f"a_{ds}_{backbone}.onnx",
                                   input_size=int(probes.shape[-1]))
                entry.update(parity_report(run.student, path, probes=probes))
            except Exception as exc:  # noqa: BLE001 - the audit records failures
                entry["error"] = f"{type(exc).__name__}: {exc}"
                traceback.print_exc()
            print(json.dumps(entry))
            rows.append(entry)
    return rows


def section_b(backbones: list[str], sizes: list[int], n_probes: int,
              out_dir: Path) -> list[dict]:
    from embedkd.deploy import export_onnx, parity_report, real_image_probes

    rows = []
    for backbone in backbones:
        try:
            # One export, traced at the first size, then served at all of them.
            run = build_run(DATASET_CONFIGS["cub200"],
                            [f"student.backbone={backbone}",
                             f"data.input_size={sizes[0]}"])
            path = export_onnx(run.student, out_dir / f"b_{backbone}_dynamic.onnx",
                               input_size=sizes[0], dynamic_spatial=True)
        except Exception as exc:  # noqa: BLE001
            rows.append({"backbone": backbone, "traced_at": sizes[0],
                         "error": f"export failed: {type(exc).__name__}: {exc}"})
            traceback.print_exc()
            continue

        for size in sizes:
            entry = {"backbone": backbone, "traced_at": sizes[0], "served_at": size}
            try:
                probe_run = build_run(DATASET_CONFIGS["cub200"],
                                      [f"student.backbone={backbone}",
                                       f"data.input_size={size}"])
                probes = real_image_probes(probe_run.bundle.query, n_samples=n_probes)
                entry.update(parity_report(run.student, path, probes=probes))
            except Exception as exc:  # noqa: BLE001
                entry["error"] = f"{type(exc).__name__}: {exc}"
            print(json.dumps(entry))
            rows.append(entry)
    return rows


def section_c(checkpoint: str | None, max_items: int, out_dir: Path) -> dict:
    import torch

    from embedkd.deploy import export_onnx, retrieval_parity

    entry: dict = {"checkpoint": checkpoint, "max_items": max_items}
    try:
        run = build_run(DATASET_CONFIGS["cub200"], ["student.backbone=resnet18"])
        if checkpoint:
            state = torch.load(checkpoint, map_location="cpu", weights_only=True)
            run.student.load_state_dict(state["state_dict"])
        gallery, query = run.bundle.gallery, run.bundle.query
        if max_items:
            gallery = torch.utils.data.Subset(gallery, range(min(max_items, len(gallery))))
            query = torch.utils.data.Subset(query, range(min(max_items, len(query))))
        path = export_onnx(run.student, out_dir / "c_d1_student.onnx", input_size=224)
        entry.update(retrieval_parity(run.student, path, gallery, query))
    except Exception as exc:  # noqa: BLE001
        entry["error"] = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()
    print(json.dumps(entry, indent=2))
    return entry


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="*", default=["cub200", "cars196", "sop"])
    parser.add_argument("--backbones", nargs="*", default=TIER1_BACKBONES)
    parser.add_argument("--sizes", nargs="*", type=int, default=[224, 160, 288])
    parser.add_argument("--n-probes", type=int, default=16)
    parser.add_argument("--checkpoint", default="runs/20260717_164702_d1_cosine_s42/best.pth")
    parser.add_argument("--max-items", type=int, default=512,
                        help="cap gallery and query for section C; 0 uses the full split")
    parser.add_argument("--sections", default="abc")
    parser.add_argument("--out", default="runs/revision/parity_audit.json")
    parser.add_argument("--onnx-dir", default="deploy_out/parity_audit")
    args = parser.parse_args()

    out_dir = Path(args.onnx_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result: dict = {}

    if "a" in args.sections:
        print("=== A. element-wise parity on real images ===")
        result["a_elementwise"] = section_a(args.datasets, args.backbones,
                                            args.n_probes, out_dir)
    if "b" in args.sections:
        print("=== B. one dynamic export served at several resolutions ===")
        result["b_dynamic_spatial"] = section_b(args.backbones, args.sizes,
                                                args.n_probes, out_dir)
    if "c" in args.sections:
        print("=== C. retrieval metrics before and after export ===")
        result["c_retrieval"] = section_c(args.checkpoint, args.max_items, out_dir)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\nwritten to {out}")


if __name__ == "__main__":
    main()
