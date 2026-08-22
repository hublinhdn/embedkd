import pytest

from .utils import tiny_embedding_model

pytest.importorskip("onnxruntime")
pytest.importorskip("onnx")


def test_deploy_report_end_to_end(tmp_path):
    from embedkd.deploy import deploy_report

    model = tiny_embedding_model(embed_dim=8, num_classes=None)
    report = deploy_report(model, tmp_path, input_size=64)
    assert report["parity_min_cosine"] >= 0.9999
    assert report["latency_ms_mean"] > 0
    assert report["onnx_size_mb"] > 0
    assert (tmp_path / "model.onnx").exists()


def test_parity_check_fails_on_mismatched_model(tmp_path):
    import torch

    from embedkd.deploy import ParityError, export_onnx, parity_check

    model = tiny_embedding_model(embed_dim=8, num_classes=None)
    path = export_onnx(model, tmp_path / "model.onnx", input_size=64)
    # Corrupt the torch model after export: parity must fail loudly.
    with torch.no_grad():
        model.head.proj.weight.add_(torch.randn_like(model.head.proj.weight))
    with pytest.raises(ParityError, match="parity"):
        parity_check(model, path, input_size=64, threshold=0.9999)


def test_parity_report_measures_error_not_only_cosine(tmp_path):
    from embedkd.deploy import export_onnx, parity_report

    model = tiny_embedding_model(embed_dim=8, num_classes=None)
    path = export_onnx(model, tmp_path / "model.onnx", input_size=64)
    report = parity_report(model, path, input_size=64, n_samples=4)

    assert report["passed"]
    assert report["min_cosine"] >= 0.9999
    # A cosine near one says nothing about magnitude, so both errors are
    # reported and both must be small for an export we would ship.
    assert report["max_abs_error"] < 1e-3
    assert report["max_rel_error"] < 1e-2
    # Normalised by the vector norm, which is the error a cosine ranking sees.
    assert report["max_rel_error_norm"] < 1e-4
    assert report["probe_source"] == "gaussian"
    assert report["n_probes"] == 4


def test_parity_report_accepts_real_image_probes(tmp_path):
    import torch

    from embedkd.deploy import export_onnx, parity_report, real_image_probes

    class TinySet:
        def __len__(self):
            return 6

        def __getitem__(self, i):
            return torch.full((3, 64, 64), i / 10.0), i

    model = tiny_embedding_model(embed_dim=8, num_classes=None)
    path = export_onnx(model, tmp_path / "model.onnx", input_size=64)
    probes = real_image_probes(TinySet(), n_samples=4)

    assert probes.shape == (4, 3, 64, 64)
    report = parity_report(model, path, probes=probes)
    assert report["probe_source"] == "real images"
    assert report["passed"]


def test_retrieval_parity_compares_metrics_across_the_export(tmp_path):
    import torch

    from embedkd.deploy import export_onnx, retrieval_parity

    class TinySet:
        def __init__(self, n):
            self.n = n

        def __len__(self):
            return self.n

        def __getitem__(self, i):
            torch.manual_seed(i)
            return torch.rand(3, 64, 64), i % 3

    model = tiny_embedding_model(embed_dim=8, num_classes=None)
    path = export_onnx(model, tmp_path / "model.onnx", input_size=64)
    result = retrieval_parity(model, path, TinySet(12), TinySet(6), batch_size=4)

    # Ranking, not just embedding geometry, has to survive the conversion.
    assert result["passed"]
    assert result["max_abs_metric_delta"] <= result["max_delta_allowed"]
    assert set(result["before"]) == set(result["after"])


def test_export_can_free_the_spatial_axes(tmp_path):
    import numpy as np
    import onnxruntime

    from embedkd.deploy import export_onnx

    model = tiny_embedding_model(embed_dim=8, num_classes=None)
    path = export_onnx(model, tmp_path / "dyn.onnx", input_size=64, dynamic_spatial=True)
    session = onnxruntime.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    name = session.get_inputs()[0].name

    # The graph must accept a size it was not traced at.
    out = session.run(None, {name: np.zeros((1, 3, 96, 96), dtype=np.float32)})[0]
    assert out.shape[0] == 1
