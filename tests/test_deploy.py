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
