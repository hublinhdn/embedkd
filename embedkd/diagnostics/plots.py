"""Diagnostic figures (optional 'plots' extra: pip install 'embedkd[plots]')."""

from __future__ import annotations

from pathlib import Path


def _require_matplotlib():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        return plt
    except ImportError:
        raise ImportError(
            "Plotting needs the 'plots' extra: pip install 'embedkd[plots]'"
        ) from None


def plot_distill_summary(report: dict, path: str | Path, metric: str = "map") -> Path:
    """Two-panel summary of a distill_report: CKA and retrieval metric, before vs after."""
    plt = _require_matplotlib()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0))
    pairs = [
        ("CKA(teacher, student)", report["cka_pre"], report["cka_post"], axes[0]),
        (metric.upper(), report[f"{metric}_before"], report[f"{metric}_after"], axes[1]),
    ]
    for title, before, after, ax in pairs:
        bars = ax.bar(["before", "after"], [before, after], color=["#9db8d2", "#2b6cb0"])
        ax.set_title(title, fontsize=10)
        ax.set_ylim(0, max(1.0, before, after) * 1.15)
        ax.bar_label(bars, fmt="%.3f", fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"Distillation outcome: {report['pattern']}", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def plot_cka_matrix(matrix, row_labels: list[str], col_labels: list[str],
                    path: str | Path, title: str = "CKA") -> Path:
    """Heatmap for layer-wise or pair-wise CKA matrices (paper figure D5)."""
    plt = _require_matplotlib()
    import numpy as np

    matrix = np.asarray(matrix, dtype=float)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(0.6 * len(col_labels) + 2.2, 0.6 * len(row_labels) + 1.8))
    im = ax.imshow(matrix, vmin=0.0, vmax=1.0, cmap="viridis")
    ax.set_xticks(range(len(col_labels)), col_labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(row_labels)), row_labels, fontsize=8)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center",
                    color="white" if matrix[i, j] < 0.6 else "black", fontsize=7)
    ax.set_title(title, fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path
