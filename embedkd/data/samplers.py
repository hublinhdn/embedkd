"""Batch samplers for metric-learning losses."""

from __future__ import annotations

from collections import defaultdict

import torch
from torch.utils.data import Sampler


class PKSampler(Sampler[list[int]]):
    """Yield batches of P distinct classes x K samples per class.

    An epoch covers approximately the whole dataset: it emits
    ``num_images // (P * K)`` batches, cycling through reshuffled class
    permutations as needed. (A single pass over the class list would touch
    only ``num_classes // P`` batches per epoch, silently shrinking training
    by an order of magnitude on datasets with many images per class.)

    Classes with fewer than K images are sampled with replacement;
    ``embedkd datasets validate`` warns about them before training.
    """

    def __init__(self, labels: list[int], p_classes: int, k_samples: int, seed: int = 42):
        self.by_class: dict[int, list[int]] = defaultdict(list)
        for idx, label in enumerate(labels):
            self.by_class[int(label)].append(idx)
        self.classes = sorted(self.by_class)
        self.num_images = len(labels)
        self.p = min(int(p_classes), len(self.classes))
        self.k = int(k_samples)
        self.seed = seed
        self.epoch = 0
        if self.p < 2:
            raise ValueError("PKSampler needs at least 2 classes in the training split")

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def _class_chunks(self, gen: torch.Generator):
        while True:  # endless stream of shuffled P-sized class groups
            order = [self.classes[i] for i in torch.randperm(len(self.classes), generator=gen)]
            for start in range(0, len(order) - self.p + 1, self.p):
                yield order[start:start + self.p]

    def __iter__(self):
        gen = torch.Generator().manual_seed(self.seed + self.epoch)
        chunks = self._class_chunks(gen)
        for _ in range(len(self)):
            batch: list[int] = []
            for cls in next(chunks):
                pool = self.by_class[cls]
                if len(pool) >= self.k:
                    picks = torch.randperm(len(pool), generator=gen)[: self.k]
                else:
                    picks = torch.randint(len(pool), (self.k,), generator=gen)
                batch.extend(pool[i] for i in picks.tolist())
            yield batch

    def __len__(self) -> int:
        return max(1, self.num_images // (self.p * self.k))
