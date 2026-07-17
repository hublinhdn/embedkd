from PIL import Image

from embedkd.data import format_validation, validate_dataset


def _folder(root, classes=3, per_class=6):
    for c in range(classes):
        d = root / f"class_{c}"
        d.mkdir(parents=True)
        for i in range(per_class):
            Image.new("RGB", (16, 16), color=(c * 50, i * 20, 10)).save(d / f"i{i}.jpg")


def _cfg(root):
    return {"adapter": "image_folder", "root": str(root), "manifest": None,
            "input_size": 32, "protocol": "gallery_query",
            "split": {"mode": "auto", "gallery_ratio": 0.5}, "k_samples": 4,
            "target": None}


def test_validate_healthy_dataset(tmp_path):
    _folder(tmp_path)
    report = validate_dataset(_cfg(tmp_path))
    assert report["errors"] == []
    assert report["stats"]["num_classes"] == 3
    assert "OK" in format_validation(report)


def test_validate_flags_corrupt_image(tmp_path):
    _folder(tmp_path)
    bad = tmp_path / "class_0" / "broken.jpg"
    bad.write_bytes(b"not a real jpeg")
    report = validate_dataset(_cfg(tmp_path))
    assert any("unreadable" in e for e in report["errors"])
    assert "FAIL" in format_validation(report)


def test_validate_warns_on_thin_classes(tmp_path):
    _folder(tmp_path, classes=3, per_class=3)  # train split gets 1-2 images per class
    report = validate_dataset(_cfg(tmp_path))
    assert any("fewer than K" in w for w in report["warnings"])


def test_validate_reports_missing_root(tmp_path):
    report = validate_dataset(_cfg(tmp_path / "nope"))
    assert report["errors"]
