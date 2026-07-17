# Custom task loss

Task losses train the student's own metric-learning task. Signature:
`forward(emb, logits, labels)`; `logits` is None unless the model has a
classifier head.

This example is executed by the test suite.

```python
import torch
import torch.nn.functional as F

from embedkd import registry
from embedkd.losses import TaskLoss


@registry.task_loss("doc_center_pull")
class CenterPull(TaskLoss):
    """Pull each embedding toward its in-batch class centroid."""

    def __init__(self, embed_dim: int = 0, num_classes: int = 0, weight: float = 1.0):
        super().__init__()
        self.weight = float(weight)

    def forward(self, emb, logits, labels):
        loss = emb.new_zeros(())
        for cls in labels.unique():
            members = emb[labels == cls]
            if len(members) > 1:
                loss = loss + F.mse_loss(members, members.mean(0, keepdim=True).expand_as(members))
        return self.weight * loss


# Use it from a config: head: {losses: {doc_center_pull: 0.5}}
emb = torch.randn(8, 16)
labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
value = CenterPull()(emb, None, labels)
assert torch.isfinite(value) and value >= 0
```

If your loss needs `embed_dim`/`num_classes` (like ArcFace's weight matrix),
accept them as constructor arguments; the builder injects them for losses
listed in `embedkd.losses._NEEDS_DIMS`.
