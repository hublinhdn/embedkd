import pytest

pytest.importorskip("matplotlib")

from embedkd.diagnostics.plots import plot_cka_matrix, plot_distill_summary  # noqa: E402


def test_plot_distill_summary(tmp_path):
    report = {"cka_pre": 0.55, "cka_post": 0.81, "map_before": 0.42, "map_after": 0.51,
              "pattern": "improved"}
    out = plot_distill_summary(report, tmp_path / "summary.png")
    assert out.exists() and out.stat().st_size > 0


def test_plot_cka_matrix(tmp_path):
    matrix = [[1.0, 0.4], [0.4, 1.0]]
    out = plot_cka_matrix(matrix, ["teacher", "student"], ["teacher", "student"],
                          tmp_path / "cka.png")
    assert out.exists() and out.stat().st_size > 0
