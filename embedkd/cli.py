"""Command line interface. Thin shell over :class:`embedkd.DistillationRun`.

Exit codes: 0 = success, 1 = user/config error, 2 = acceptance failure
(reproduce mismatch, ONNX parity failure).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import ConfigError, resolve


def _print(obj: dict) -> None:
    print(json.dumps(obj, indent=2, default=str))


def cmd_fit(args) -> int:
    from .run import DistillationRun

    run = DistillationRun(resolve(args.config, args.set))
    result = run.fit()
    _print({"best": result["best"], "checkpoints": result["checkpoints"],
            "out_dir": str(run.out_dir)})
    return 0


def cmd_eval(args) -> int:
    from .run import DistillationRun

    run = DistillationRun(resolve(args.config, args.set))
    _print(run.evaluate(checkpoint=args.checkpoint, target=args.target))
    return 0


def cmd_diagnose(args) -> int:
    import torch

    from .diagnostics.cka import format_report
    from .run import DistillationRun

    run = DistillationRun(resolve(args.config, args.set))
    if args.checkpoint:
        # Post-distillation diagnosis: measure the TRAINED student against
        # the teacher instead of the fresh initialisation.
        state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        run.student.load_state_dict(state["state_dict"])
    report = run.diagnose()
    if args.checkpoint:
        report["student_checkpoint"] = args.checkpoint
    print(format_report(report))
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0


def cmd_extract(args) -> int:
    import numpy as np
    import torch

    from .data import ImageTransform, build_bundle
    from .evaluation import extract_embeddings
    from .run import DistillationRun

    cfg = resolve(args.config, args.set)
    run = DistillationRun(cfg)
    if args.checkpoint:
        state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        run.student.load_state_dict(state["state_dict"])
    dataset = build_bundle(cfg["data"], ImageTransform(cfg["data"]["input_size"]),
                           ImageTransform(cfg["data"]["input_size"])).query
    emb, labels = extract_embeddings(run.student, dataset, cfg["eval"]["batch_size"], run.device)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, emb.numpy())
    if args.save_labels:
        np.save(out.with_name(out.stem + "_labels.npy"), labels.numpy())
    print(f"Saved {tuple(emb.shape)} L2-normalised embeddings to {out}")
    return 0


def cmd_deploy(args) -> int:
    import torch

    from .deploy import ParityError, deploy_report
    from .run import DistillationRun

    cfg = resolve(args.config, args.set)
    run = DistillationRun(cfg)
    if args.checkpoint:
        state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        run.student.load_state_dict(state["state_dict"])
    try:
        report = deploy_report(run.student.cpu(), args.out_dir,
                               input_size=cfg["data"]["input_size"])
    except ParityError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 2
    _print(report)
    return 0


def cmd_reproduce(args) -> int:
    expected_dir = Path(__file__).resolve().parent.parent / "expected_results"
    available = sorted(p.stem for p in expected_dir.glob("*.json")) if expected_dir.is_dir() else []
    if args.list or not args.demo_id:
        print("Available demos:", available or "(none packaged yet)")
        return 0
    if args.demo_id not in available:
        print(f"Unknown demo '{args.demo_id}'. Available: {available}", file=sys.stderr)
        return 1
    spec = json.loads((expected_dir / f"{args.demo_id}.json").read_text(encoding="utf-8"))
    from .run import DistillationRun

    if args.eval_only:
        # Evaluation needs neither the teacher checkpoint nor retention math;
        # the expected values are pure student retrieval metrics.
        overrides = list(args.set) + ["distill.alpha=0", "eval.report_retention=false"]
        run = DistillationRun.from_config(spec["config"], overrides)
        metrics = run.evaluate(checkpoint=spec.get("checkpoint"))
    else:
        run = DistillationRun.from_config(spec["config"], args.set)
        run.fit()
        metrics = run.evaluate()
    failures = []
    for name, entry in spec["expected"].items():
        got = metrics.get(name)
        lo = entry["value"] - entry["tolerance"]
        hi = entry["value"] + entry["tolerance"]
        status = "PASS" if got is not None and lo <= got <= hi else "FAIL"
        print(f"{name:>10}: expected {entry['value']} +/- {entry['tolerance']} | got {got} -> {status}")
        if status == "FAIL":
            failures.append(name)
    return 2 if failures else 0


def cmd_backbones(args) -> int:
    from .models import backbone_table

    for row in backbone_table():
        print(f"{row['status']:>12}  {row['name']:<32} family={row['family']}")
    return 0


def cmd_datasets(args) -> int:
    from .registry import registry

    if args.action == "list":
        from .data.datasets_builtin import DOWNLOADABLE

        print("Registered adapters:", registry.available("dataset"))
        for name, adapter in DOWNLOADABLE.items():
            doc = (adapter.__doc__ or "").strip().splitlines()[0]
            print(f"  {name:<10} {doc}")
        return 0

    if args.action == "download":
        from .data.datasets_builtin import DOWNLOADABLE
        from .data.downloads import DownloadError

        if not args.name or args.name not in DOWNLOADABLE:
            print(f"Usage: embedkd datasets download <{'|'.join(DOWNLOADABLE)}> --root data/",
                  file=sys.stderr)
            return 1
        try:
            base = DOWNLOADABLE[args.name].download(args.root)
        except DownloadError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1
        print(f"Dataset '{args.name}' ready at {base}")
        return 0

    if args.action == "validate":
        from .config import deep_update, parse_set_override
        from .data import format_validation, validate_dataset

        if args.config:
            data_cfg = resolve(args.config, args.set)["data"]
        elif args.name and ":" in args.name:
            adapter, root = args.name.split(":", 1)
            data_cfg = {"adapter": adapter, "root": root, "manifest": None,
                        "input_size": 224, "protocol": "gallery_query",
                        "split": {"mode": "auto", "gallery_ratio": 0.5},
                        "k_samples": 4, "target": None}
            for expr in args.set:  # accept both data.manifest=... and manifest=...
                nested = parse_set_override(expr)
                data_cfg = deep_update(data_cfg, nested.get("data", nested))
        else:
            print("Usage: embedkd datasets validate <adapter>:<root>  (or --config c.yaml)",
                  file=sys.stderr)
            return 1
        report = validate_dataset(data_cfg)
        print(format_validation(report))
        return 1 if report["errors"] else 0

    return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="embedkd")
    from . import __version__

    parser.add_argument("--version", action="version", version=f"embedkd {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common(p, config_required=True):
        p.add_argument("--config", required=config_required)
        p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")

    p = sub.add_parser("fit", help="train a student with knowledge distillation")
    add_common(p)
    p.set_defaults(func=cmd_fit)

    p = sub.add_parser("eval", help="retrieval evaluation (mAP / R@k)")
    add_common(p)
    p.add_argument("--checkpoint")
    p.add_argument("--target", action="store_true", help="evaluate on the cross-domain target")
    p.set_defaults(func=cmd_eval)

    p = sub.add_parser("diagnose", help="teacher-student compatibility report (run BEFORE training)")
    add_common(p)
    p.add_argument("--checkpoint", help="student checkpoint for POST-distillation diagnosis")
    p.add_argument("--out", help="also write the report as JSON")
    p.set_defaults(func=cmd_diagnose)

    p = sub.add_parser("extract", help="dump L2-normalised embeddings to .npy")
    add_common(p)
    p.add_argument("--checkpoint")
    p.add_argument("--out", required=True)
    p.add_argument("--save-labels", action="store_true")
    p.set_defaults(func=cmd_extract)

    p = sub.add_parser("deploy", help="ONNX export + parity check + CPU benchmark")
    add_common(p)
    p.add_argument("--checkpoint")
    p.add_argument("--out-dir", default="deploy_out")
    p.set_defaults(func=cmd_deploy)

    p = sub.add_parser("reproduce", help="re-run a paper demo and compare against expected results")
    p.add_argument("demo_id", nargs="?")
    p.add_argument("--list", action="store_true")
    p.add_argument("--eval-only", action="store_true")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    p.set_defaults(func=cmd_reproduce)

    p = sub.add_parser("backbones", help="list validated and experimental backbones")
    p.set_defaults(func=cmd_backbones)

    p = sub.add_parser("datasets", help="list / download / validate datasets")
    p.add_argument("action", choices=["list", "download", "validate"])
    p.add_argument("name", nargs="?")
    p.add_argument("--root", default="data")
    p.add_argument("--config")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    p.set_defaults(func=cmd_datasets)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (ConfigError, FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
