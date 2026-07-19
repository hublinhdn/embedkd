# Custom dataset adapter

An adapter turns your storage layout into a `DataBundle` with `train`,
`gallery` and `query` datasets. Adapters never choose transforms; the engine
passes them in.

This example is executed by the test suite.

```python
import torch
from torch.utils.data import Dataset

from embedkd import registry
from embedkd.data import DataBundle


class RandomBlobs(Dataset):
    """Stand-in for your real storage access."""

    def __init__(self, n: int, num_classes: int, size: int, seed: int):
        self.n, self.num_classes, self.size, self.seed = n, num_classes, size, seed

    def __len__(self):
        return self.n

    @property
    def labels(self):
        return [i % self.num_classes for i in range(self.n)]

    def __getitem__(self, idx):
        gen = torch.Generator().manual_seed(self.seed + idx)
        return torch.randn(3, self.size, self.size, generator=gen), idx % self.num_classes


@registry.dataset("doc_blobs")
class BlobAdapter:
    @staticmethod
    def build(data_cfg, train_tf, eval_tf) -> DataBundle:
        size = int(data_cfg.get("input_size", 64))
        train = RandomBlobs(32, 4, size, seed=1)
        return DataBundle(
            train=train,
            gallery=RandomBlobs(8, 4, size, seed=2),
            query=RandomBlobs(8, 4, size, seed=3),
            num_classes=4,
            train_labels=train.labels,
        )


# Use it from a config: data: {adapter: doc_blobs}
from embedkd.data import build_bundle

bundle = build_bundle({"adapter": "doc_blobs", "input_size": 32, "protocol": "gallery_query",
                       "target": None}, None, None)
assert bundle.num_classes == 4 and len(bundle.train) == 32
```

Contract checklist:

1. Each dataset yields `(image_tensor, label_index)`.
2. `train_labels` powers the PK sampler.
3. If your items are file paths, expose them as `.items` so
   `embedkd datasets validate` can health-check them.
4. Freeze any generated split to disk so reruns see identical data.
