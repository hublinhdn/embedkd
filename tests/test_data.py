import csv

import torch
from PIL import Image

from embedkd.data import (
    ImageTransform,
    PKSampler,
    SyntheticDataset,
    build_bundle,
)
from embedkd.data.adapters import SPLIT_FILENAME


def test_pk_sampler_batches_have_p_classes_k_samples():
    labels = [i // 10 for i in range(60)]  # 6 classes x 10
    sampler = PKSampler(labels, p_classes=3, k_samples=4, seed=1)
    batches = list(sampler)
    # An epoch covers ~the whole dataset: 60 images / (3*4) = 5 batches,
    # NOT num_classes // P = 2 (regression guard: that bug shrank training
    # by an order of magnitude on many-images-per-class datasets).
    assert len(batches) == len(sampler) == 5
    for batch in batches:
        assert len(batch) == 12
        classes = {labels[i] for i in batch}
        assert len(classes) == 3
        for cls in classes:
            assert sum(labels[i] == cls for i in batch) == 4


def test_pk_sampler_epoch_changes_order_deterministically():
    labels = [i // 5 for i in range(40)]
    sampler = PKSampler(labels, 4, 2, seed=7)
    sampler.set_epoch(0)
    first = list(sampler)
    sampler.set_epoch(0)
    assert list(sampler) == first
    sampler.set_epoch(1)
    assert list(sampler) != first


def test_synthetic_dataset_is_deterministic_and_class_conditional():
    ds = SyntheticDataset(num_classes=3, per_class=4, size=16, seed=42)
    x1, y1 = ds[0]
    x2, _ = ds[0]
    assert torch.equal(x1, x2)
    assert y1 == 0
    assert ds[4][1] == 1


def _make_image_folder(root, classes=3, per_class=8):
    for c in range(classes):
        d = root / f"class_{c}"
        d.mkdir(parents=True)
        for i in range(per_class):
            Image.new("RGB", (20, 20), color=(40 * c, 10 * i, 100)).save(d / f"img_{i}.jpg")


def test_image_folder_adapter_split_is_frozen_to_disk(tmp_path):
    _make_image_folder(tmp_path)
    cfg = {"adapter": "image_folder", "root": str(tmp_path), "input_size": 32,
           "protocol": "gallery_query", "split": {"mode": "auto", "gallery_ratio": 0.5}}
    tf = ImageTransform(32)
    bundle = build_bundle(cfg, tf, tf)
    assert bundle.num_classes == 3
    assert (tmp_path / SPLIT_FILENAME).exists()
    sizes = (len(bundle.train), len(bundle.gallery), len(bundle.query))
    # Second build must reuse the frozen split file, not re-randomise.
    bundle2 = build_bundle(cfg, tf, tf)
    assert (len(bundle2.train), len(bundle2.gallery), len(bundle2.query)) == sizes
    assert bundle2.train.items == bundle.train.items
    image, label = bundle.train[0]
    assert image.shape == (3, 32, 32)
    assert 0 <= label < 3


def test_csv_manifest_adapter_with_explicit_split(tmp_path):
    _make_image_folder(tmp_path, classes=2, per_class=4)
    manifest = tmp_path / "manifest.csv"
    rows = []
    for c in range(2):
        for i in range(4):
            split = "train" if i < 2 else ("gallery" if i == 2 else "query")
            rows.append({"path": f"class_{c}/img_{i}.jpg", "label": f"c{c}", "split": split})
    with open(manifest, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["path", "label", "split"])
        writer.writeheader()
        writer.writerows(rows)
    cfg = {"adapter": "csv_manifest", "root": str(tmp_path), "manifest": str(manifest),
           "input_size": 32, "protocol": "gallery_query", "split": {"mode": "auto"}}
    tf = ImageTransform(32)
    bundle = build_bundle(cfg, tf, tf)
    assert len(bundle.train) == 4
    assert len(bundle.gallery) == 2
    assert len(bundle.query) == 2
