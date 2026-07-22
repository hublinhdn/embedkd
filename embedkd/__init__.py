"""EmbedKD: a reproducibility-first toolkit for distilling metric embeddings.

It tells you whether a teacher-student pair is worth distilling, distills it,
evaluates it with retrieval protocols, and benchmarks the deployed result.
"""

__version__ = "0.1.3"

from .registry import registry  # noqa: E402
from .objectives import DistillObjective, build_objective  # noqa: E402
from .losses import TaskLoss, build_task_loss  # noqa: E402
from .run import DistillationRun  # noqa: E402

__all__ = [
    "__version__",
    "registry",
    "DistillObjective",
    "TaskLoss",
    "DistillationRun",
    "build_objective",
    "build_task_loss",
]
