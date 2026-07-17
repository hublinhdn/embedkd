"""Minimal image transforms (PIL + torch only; no torchvision dependency)."""

from __future__ import annotations

import math

import numpy as np
import torch
from PIL import Image

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _random_resized_crop(img: Image.Image, size: int,
                         scale: tuple[float, float], ratio: tuple[float, float]) -> Image.Image:
    """torchvision-style RandomResizedCrop using torch RNG (seed-controlled)."""
    width, height = img.size
    area = width * height
    log_ratio = (math.log(ratio[0]), math.log(ratio[1]))
    for _ in range(10):
        target_area = area * (scale[0] + (scale[1] - scale[0]) * float(torch.rand(())))
        aspect = math.exp(log_ratio[0] + (log_ratio[1] - log_ratio[0]) * float(torch.rand(())))
        crop_w = int(round(math.sqrt(target_area * aspect)))
        crop_h = int(round(math.sqrt(target_area / aspect)))
        if 0 < crop_w <= width and 0 < crop_h <= height:
            left = int(torch.randint(0, width - crop_w + 1, ()))
            top = int(torch.randint(0, height - crop_h + 1, ()))
            img = img.crop((left, top, left + crop_w, top + crop_h))
            return img.resize((size, size), Image.BILINEAR)
    # Fallback: center crop of the largest fitting square.
    side = min(width, height)
    left, top = (width - side) // 2, (height - side) // 2
    return img.crop((left, top, left + side, top + side)).resize((size, size), Image.BILINEAR)


class ImageTransform:
    """Train: RandomResizedCrop + horizontal flip. Eval: plain resize.

    Random crops are the standard guard against open-set overfitting in
    fine-grained retrieval; without them validation mAP peaks within a few
    epochs and decays while the train loss keeps falling.
    """

    def __init__(self, size: int, train: bool = False,
                 mean: tuple = IMAGENET_MEAN, std: tuple = IMAGENET_STD,
                 crop_scale: tuple[float, float] = (0.8, 1.0),
                 crop_ratio: tuple[float, float] = (3 / 4, 4 / 3),
                 jitter: tuple[float, float] = (0.2, 0.1)) -> None:
        self.size = int(size)
        self.train = train
        self.crop_scale = crop_scale
        self.crop_ratio = crop_ratio
        self.jitter = jitter  # (brightness, contrast) strengths
        self.mean = torch.tensor(mean).view(3, 1, 1)
        self.std = torch.tensor(std).view(3, 1, 1)

    @staticmethod
    def _color_jitter(img: Image.Image, brightness: float, contrast: float) -> Image.Image:
        from PIL import ImageEnhance

        if brightness > 0:
            factor = 1.0 + (2.0 * float(torch.rand(())) - 1.0) * brightness
            img = ImageEnhance.Brightness(img).enhance(factor)
        if contrast > 0:
            factor = 1.0 + (2.0 * float(torch.rand(())) - 1.0) * contrast
            img = ImageEnhance.Contrast(img).enhance(factor)
        return img

    def __call__(self, img: Image.Image) -> torch.Tensor:
        img = img.convert("RGB")
        if self.train:
            img = _random_resized_crop(img, self.size, self.crop_scale, self.crop_ratio)
            if torch.rand(()) < 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
            img = self._color_jitter(img, *self.jitter)
        else:
            img = img.resize((self.size, self.size), Image.BILINEAR)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1)
        return (tensor - self.mean) / self.std
