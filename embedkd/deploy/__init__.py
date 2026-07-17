"""Deployment: ONNX export with mandatory parity check + CPU benchmark.

An export whose outputs do not match the torch model is a failure, not a
warning: parity below threshold raises, and the CLI maps that to exit code 2.
"""

from __future__ import annotations

import time
from pathlib import Path

import torch
import torch.nn.functional as F


class ParityError(RuntimeError):
    pass


def _require_onnxruntime():
    try:
        import onnxruntime  # noqa: F401

        return onnxruntime
    except ImportError:
        raise ImportError(
            "Deployment features need the 'onnx' extra: pip install 'embedkd[onnx]'"
        ) from None


def export_onnx(model: torch.nn.Module, path: str | Path, input_size: int = 224,
                opset: int = 17) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model.eval()
    dummy = torch.randn(1, 3, input_size, input_size)
    torch.onnx.export(
        model, (dummy,), str(path),
        input_names=["images"], output_names=["embedding"],
        dynamic_axes={"images": {0: "batch"}, "embedding": {0: "batch"}},
        opset_version=opset, dynamo=False,
    )
    return path


@torch.no_grad()
def parity_check(model: torch.nn.Module, onnx_path: str | Path, input_size: int = 224,
                 n_samples: int = 4, threshold: float = 0.9999, seed: int = 42) -> float:
    """Minimum cosine similarity between torch and onnxruntime outputs."""
    ort = _require_onnxruntime()
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    generator = torch.Generator().manual_seed(seed)
    model.eval()
    worst = 1.0
    for _ in range(n_samples):
        x = torch.randn(1, 3, input_size, input_size, generator=generator)
        ref = model(x).float()
        out = torch.from_numpy(session.run(None, {"images": x.numpy()})[0]).float()
        cos = float(F.cosine_similarity(ref.flatten(0, -2), out.flatten(0, -2)).min())
        worst = min(worst, cos)
    if worst < threshold:
        raise ParityError(
            f"ONNX parity check failed: min cosine similarity {worst:.6f} < {threshold}. "
            "The exported graph does not reproduce the torch model."
        )
    return worst


def benchmark_cpu(onnx_path: str | Path, input_size: int = 224, n_runs: int = 100,
                  warmup: int = 10, seed: int = 42) -> dict:
    ort = _require_onnxruntime()
    import numpy as np

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((1, 3, input_size, input_size), dtype=np.float32)
    for _ in range(warmup):
        session.run(None, {"images": x})
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        session.run(None, {"images": x})
        times.append(time.perf_counter() - start)
    times_ms = np.asarray(times) * 1e3
    return {
        "latency_ms_mean": round(float(times_ms.mean()), 3),
        "latency_ms_std": round(float(times_ms.std()), 3),
        "fps": round(1000.0 / float(times_ms.mean()), 2),
        "n_runs": n_runs,
        "warmup": warmup,
        "onnx_size_mb": round(Path(onnx_path).stat().st_size / 2**20, 2),
        "input_size": input_size,
    }


def deploy_report(model: torch.nn.Module, out_dir: str | Path, input_size: int = 224,
                  parity_threshold: float = 0.9999) -> dict:
    """Export + parity + benchmark in one call. Raises ParityError on mismatch."""
    out_dir = Path(out_dir)
    onnx_path = export_onnx(model, out_dir / "model.onnx", input_size)
    parity = parity_check(model, onnx_path, input_size, threshold=parity_threshold)
    bench = benchmark_cpu(onnx_path, input_size)
    return {
        "onnx_path": str(onnx_path),
        "parity_min_cosine": round(parity, 6),
        "params_m": round(sum(p.numel() for p in model.parameters()) / 1e6, 2),
        **bench,
    }
