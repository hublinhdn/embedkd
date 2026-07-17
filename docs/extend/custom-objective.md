# Custom distillation objective

Register a class; never edit the core. Objectives receive L2-normalised (or
raw, your choice) embeddings and optional logits.

The example below is executed verbatim by the test suite
(`tests/test_docs_examples.py`), so this page cannot drift from the code.

```python
import torch
import torch.nn.functional as F

from embedkd import registry
from embedkd.objectives import DistillObjective


@registry.distill_objective("doc_cosine_margin")
class CosineWithMargin(DistillObjective):
    """1 - cos, but only pairs below the margin contribute."""

    def __init__(self, margin: float = 0.1):
        super().__init__()
        self.margin = float(margin)

    def forward(self, s_emb, t_emb, **_):
        cos = (F.normalize(s_emb, dim=-1) * F.normalize(t_emb, dim=-1)).sum(-1)
        return F.relu(1.0 - self.margin - cos).mean()


# Use it from a config: distill: {objective: doc_cosine_margin,
#                                 doc_cosine_margin: {margin: 0.2}}
# Sanity check:
s, t = torch.randn(4, 8), torch.randn(4, 8)
loss = CosineWithMargin(margin=0.2)(s, t)
assert loss.shape == () and torch.isfinite(loss)
assert float(CosineWithMargin(margin=0.0)(s, s)) < 1e-6  # identical => 0
```

Notes:

- Declare `needs_fp32 = True` if your objective is numerically unsafe in
  fp16 (all losses already run fp32 in the trainer; the flag documents and
  enforces it for direct callers).
- Per-objective parameters live in the config under the objective's own name.
