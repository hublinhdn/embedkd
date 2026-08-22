"""Deployment: ONNX export with mandatory parity check + CPU benchmark.

An export whose outputs do not match the torch model is a failure, not a
warning: parity below threshold raises, and the CLI maps that to exit code 2.

Cosine agreement on a handful of Gaussian probes is a weak test. It cannot see
a systematic magnitude shift, it never touches the preprocessing path that real
images take, and a cosine of 0.9999 still permits two gallery items to swap
places. This module therefore also offers probes drawn from real images,
absolute and relative error, and a comparison of the retrieval metrics
themselves before and after export, which is the property a deployment
actually depends on.
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
                opset: int = 17, dynamic_spatial: bool = False) -> Path:
    """Export to ONNX.

    dynamic_spatial also frees the height and width axes. It is off by default
    because a graph that accepts any resolution is not the same as a model that
    is correct at any resolution: pooling and interpolation behaviour has to be
    checked per backbone. Use parity_report at each size you intend to serve.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model.eval()
    dummy = torch.randn(1, 3, input_size, input_size)
    image_axes = {0: "batch"}
    if dynamic_spatial:
        image_axes = {0: "batch", 2: "height", 3: "width"}
    torch.onnx.export(
        model, (dummy,), str(path),
        input_names=["images"], output_names=["embedding"],
        dynamic_axes={"images": image_axes, "embedding": {0: "batch"}},
        opset_version=opset, dynamo=False,
    )
    return path


class OnnxEmbedder(torch.nn.Module):
    """The exported graph behind the same interface the evaluator expects.

    Wrapping it this way means the retrieval metrics after export are computed
    by exactly the code that computed them before export, rather than by a
    parallel implementation that could differ.
    """

    def __init__(self, onnx_path: str | Path) -> None:
        super().__init__()
        ort = _require_onnxruntime()
        self.session = ort.InferenceSession(str(onnx_path),
                                            providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.session.run(None, {self.input_name: x.detach().cpu().numpy()})[0]
        return torch.from_numpy(out)


def real_image_probes(dataset, n_samples: int = 16, seed: int = 42) -> torch.Tensor:
    """Take evenly spaced images from a dataset, already through its transform.

    Evenly spaced rather than random so the selection is stable across runs and
    covers the ordering of the split instead of clustering at its start.
    """
    total = len(dataset)
    if total == 0:
        raise ValueError("cannot draw probes from an empty dataset")
    n = min(n_samples, total)
    step = max(1, total // n)
    images = [dataset[i * step % total][0] for i in range(n)]
    return torch.stack(images)


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


@torch.no_grad()
def parity_report(model: torch.nn.Module, onnx_path: str | Path,
                  probes: torch.Tensor | None = None, input_size: int = 224,
                  n_samples: int = 16, threshold: float = 0.9999,
                  seed: int = 42, eps: float = 1e-6) -> dict:
    """Element-wise agreement between the torch model and the exported graph.

    Pass real images as `probes`; the Gaussian fallback exists only so the
    function still runs where no dataset is available.

    Two relative errors are reported because the element-wise one is easy to
    misread. `max_rel_error` divides by the magnitude of each coordinate, so a
    coordinate that is near zero inflates it even when the absolute error is
    at float32 resolution; it is the strict reading of the quantity but it says
    more about the embedding than about the export. `max_rel_error_norm`
    divides the largest absolute error by the norm of the reference vector,
    which is the error a cosine ranking actually sees.
    """
    ort = _require_onnxruntime()
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    name = session.get_inputs()[0].name
    model.eval()

    if probes is None:
        generator = torch.Generator().manual_seed(seed)
        probes = torch.randn(n_samples, 3, input_size, input_size, generator=generator)
        source = "gaussian"
    else:
        source = "real images"

    ref = model(probes).float()
    out = torch.from_numpy(session.run(None, {name: probes.numpy()})[0]).float()

    diff = (ref - out).abs()
    cos = F.cosine_similarity(ref, out, dim=-1)
    report = {
        "probe_source": source,
        "n_probes": int(probes.shape[0]),
        "input_size": int(probes.shape[-1]),
        "min_cosine": round(float(cos.min()), 8),
        "max_abs_error": float(diff.max()),
        "max_rel_error": float((diff / (ref.abs() + eps)).max()),
        "max_rel_error_norm": float(
            (diff.max(dim=-1).values / (ref.norm(dim=-1) + eps)).max()
        ),
        "mean_abs_error": float(diff.mean()),
        "threshold": threshold,
    }
    report["passed"] = report["min_cosine"] >= threshold
    if not report["passed"]:
        raise ParityError(
            f"ONNX parity check failed: min cosine similarity "
            f"{report['min_cosine']:.6f} < {threshold} over {report['n_probes']} "
            f"{source} probes at {report['input_size']} pixels. "
            "The exported graph does not reproduce the torch model."
        )
    return report


def retrieval_parity(model: torch.nn.Module, onnx_path: str | Path,
                     gallery, query, batch_size: int = 256,
                     max_delta: float = 1e-4) -> dict:
    """Compare the retrieval metrics themselves before and after export.

    Close embeddings do not guarantee an unchanged ranking, and the ranking is
    what a retrieval deployment serves. Both sides run through the same
    evaluator, so any difference comes from the export and nothing else.
    """
    from ..evaluation import evaluate_model

    before = evaluate_model(model, gallery, query, batch_size=batch_size, device="cpu")
    after = evaluate_model(OnnxEmbedder(onnx_path), gallery, query,
                           batch_size=batch_size, device="cpu")
    deltas = {k: float(after[k] - before[k]) for k in before
              if isinstance(before[k], float) and k in after}
    worst = max((abs(v) for v in deltas.values()), default=0.0)
    return {
        "before": before,
        "after": after,
        "delta": {k: round(v, 8) for k, v in deltas.items()},
        "max_abs_metric_delta": worst,
        "max_delta_allowed": max_delta,
        "passed": worst <= max_delta,
    }


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
                  parity_threshold: float = 0.9999, probes: torch.Tensor | None = None,
                  gallery=None, query=None, dynamic_spatial: bool = False) -> dict:
    """Export, verify, benchmark, in one call. Raises ParityError on mismatch.

    Pass `probes` to verify on real images instead of noise, and `gallery` and
    `query` to check that mAP and Recall@k survive the export.
    """
    out_dir = Path(out_dir)
    onnx_path = export_onnx(model, out_dir / "model.onnx", input_size,
                            dynamic_spatial=dynamic_spatial)
    parity = parity_report(model, onnx_path, probes=probes, input_size=input_size,
                           threshold=parity_threshold)
    bench = benchmark_cpu(onnx_path, input_size)
    report = {
        "onnx_path": str(onnx_path),
        "parity_min_cosine": parity["min_cosine"],
        "parity_max_abs_error": parity["max_abs_error"],
        "parity_max_rel_error": parity["max_rel_error"],
        "parity_probe_source": parity["probe_source"],
        "parity_n_probes": parity["n_probes"],
        "params_m": round(sum(p.numel() for p in model.parameters()) / 1e6, 2),
        **bench,
    }
    if gallery is not None and query is not None:
        report["retrieval_parity"] = retrieval_parity(model, onnx_path, gallery, query)
    return report
