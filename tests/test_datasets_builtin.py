"""Built-in dataset adapters, tested on miniature fakes of the official layouts."""

import csv

import pytest
from PIL import Image

from embedkd.data import ImageTransform, build_bundle

TF = ImageTransform(32)


def _img(path, color=(120, 30, 200)):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (16, 16), color=color).save(path)


def _fake_cub(root, n_train_classes=2, n_test_classes=2, per_class=4):
    base = root / "CUB_200_2011"
    lines = []
    idx = 1
    for c in range(1, n_train_classes + n_test_classes + 1):
        # Real CUB train classes are ids 1..100; fake uses 1..n_train as <=100
        cid = c if c <= n_train_classes else 100 + c
        cname = f"{cid:03d}.Fake_Bird_{cid}"
        for i in range(per_class):
            rel = f"{cname}/img_{i}.jpg"
            _img(base / "images" / rel)
            lines.append(f"{idx} {rel}")
            idx += 1
    (base / "images.txt").write_text("\n".join(lines), encoding="utf-8")
    return base


def test_cub200_open_set_split(tmp_path):
    _fake_cub(tmp_path)
    cfg = {"adapter": "cub200", "root": str(tmp_path), "protocol": "gallery_query",
           "target": None}
    bundle = build_bundle(cfg, TF, TF)
    # 2 train classes x 4 images; 2 test classes split 2 gallery / 2 query each.
    assert len(bundle.train) == 8
    assert len(bundle.gallery) == 4
    assert len(bundle.query) == 4
    train_labels = set(bundle.train.labels)
    test_labels = set(bundle.gallery.labels) | set(bundle.query.labels)
    assert train_labels.isdisjoint(test_labels)  # open-set: disjoint classes


def test_cub200_missing_root_message(tmp_path):
    cfg = {"adapter": "cub200", "root": str(tmp_path), "protocol": "gallery_query",
           "target": None}
    with pytest.raises(FileNotFoundError, match="datasets download cub200"):
        build_bundle(cfg, TF, TF)


def _fake_sop(root):
    base = root / "Stanford_Online_Products"
    base.mkdir(parents=True)
    header = "image_id class_id super_class_id path"
    train_rows, test_rows = [header], [header]
    idx = 1
    for cls in (1, 2):
        for i in range(3):
            rel = f"bicycle_final/train_{cls}_{i}.jpg"
            _img(base / rel)
            train_rows.append(f"{idx} {cls} 1 {rel}")
            idx += 1
    for cls in (3, 4):
        for i in range(4):
            rel = f"cabinet_final/test_{cls}_{i}.jpg"
            _img(base / rel)
            test_rows.append(f"{idx} {cls} 2 {rel}")
            idx += 1
    # Singleton test class: must land in gallery only, never crash the split.
    rel = "chair_final/test_5_0.jpg"
    _img(base / rel)
    test_rows.append(f"{idx} 5 3 {rel}")
    (base / "Ebay_train.txt").write_text("\n".join(train_rows), encoding="utf-8")
    (base / "Ebay_test.txt").write_text("\n".join(test_rows), encoding="utf-8")


def test_sop_official_lists_and_singleton_class(tmp_path):
    _fake_sop(tmp_path)
    cfg = {"adapter": "sop", "root": str(tmp_path), "protocol": "gallery_query",
           "target": None}
    bundle = build_bundle(cfg, TF, TF)
    assert len(bundle.train) == 6
    assert len(bundle.gallery) == 2 * 2 + 1  # even halves + singleton
    assert len(bundle.query) == 2 * 2


def test_cars196_manual_layout(tmp_path):
    base = tmp_path / "cars196"
    rows = []
    for cls in range(1, 5):
        for i in range(2):
            rel = f"car_ims/{cls:03d}_{i}.jpg"
            _img(base / rel)
            rows.append({"path": rel, "label": str(cls)})
    with open(base / "cars_annos.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["path", "label"])
        writer.writeheader()
        writer.writerows(rows)
    cfg = {"adapter": "cars196", "root": str(tmp_path), "protocol": "gallery_query",
           "target": None}
    bundle = build_bundle(cfg, TF, TF)
    # 4 classes: first half train, second half split gallery/query.
    assert len(bundle.train) == 4
    assert len(bundle.gallery) == 2
    assert len(bundle.query) == 2


def test_cars196_helpful_error_without_data(tmp_path):
    cfg = {"adapter": "cars196", "root": str(tmp_path), "protocol": "gallery_query",
           "target": None}
    with pytest.raises(FileNotFoundError, match="cars_annos.csv"):
        build_bundle(cfg, TF, TF)
