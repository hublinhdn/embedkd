"""Minimal image transforms (PIL + torch only; no torchvision dependency)."""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class ImageTransform:
    """Resize, optional horizontal flip (train), tensor conversion, normalise."""

    def __init__(self, size: int, train: bool = False,
                 mean: tuple = IMAGENET_MEAN, std: tuple = IMAGENET_STD) -> None:
        self.size = int(size)
        self.train = train
        self.mean = torch.tensor(mean).view(3, 1, 1)
        self.std = torch.tensor(std).view(3, 1, 1)

    def __call__(self, img: Image.Image) -> torch.Tensor:
        img = img.convert("RGB").resize((self.size, self.size), Image.BILINEAR)
        if self.train and torch.rand(()) < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1)
        return (tensor - self.mean) / self.std
