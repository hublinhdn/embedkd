"""Dataset adapters. Every adapter builds a :class:`DataBundle`.

Contract (docs: guides/bring-your-own-dataset):
  * an adapter is registered with ``@registry.dataset(name)`` and implements
    ``build(data_cfg, train_transform, eval_transform) -> DataBundle``;
  * adapters never choose transforms themselves, the engine passes them in;
  * auto splits are generated once, written to disk and re-read afterwards,
    so a split is frozen the first time it is materialised.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset

from ..registry import registry

IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLIT_FILENAME = "embedkd_generated_split.csv"


@dataclass
class DataBundle:
    train: Dataset
    gallery: Dataset
    query: Dataset
    num_classes: int
    train_labels: list[int]
    target_gallery: Dataset | None = None
    target_query: Dataset | None = None


class ImageListDataset(Dataset):
    """(path, label_idx) items loaded with PIL and a fixed transform."""

    def __init__(self, items: list[tuple[str, int]], transform) -> None:
        self.items = items
        self.transform = transform

    def __len__(self) -> int:
        return len(self.items)

    @property
    def labels(self) -> list[int]:
        return [label for _, label in self.items]

    def __getitem__(self, idx: int):
        path, label = self.items[idx]
        with Image.open(path) as img:
            return self.transform(img), label


def _auto_split(records: list[tuple[str, str]], split_cfg: dict, seed: int) -> list[tuple[str, str, str]]:
    """Stratified per-class split into train / gallery / query."""
    gallery_ratio = float(split_cfg.get("gallery_ratio", 0.5))
    by_class: dict[str, list[str]] = {}
    for path, label in records:
        by_class.setdefault(label, []).append(path)
    gen = torch.Generator().manual_seed(seed)
    out: list[tuple[str, str, str]] = []
    for label in sorted(by_class):
        paths = sorted(by_class[label])
        order = torch.randperm(len(paths), generator=gen).tolist()
        n = len(paths)
        n_train = max(1, n // 2)
        rest = order[n_train:]
        n_gallery = max(1, int(round(len(rest) * gallery_ratio))) if rest else 0
        for pos, i in enumerate(order):
            if pos < n_train:
                split = "train"
            elif pos - n_train < n_gallery:
                split = "gallery"
            else:
                split = "query"
            out.append((paths[i], label, split))
    return out


def _load_or_create_split(root: Path, records: list[tuple[str, str]],
                          split_cfg: dict, seed: int) -> list[tuple[str, str, str]]:
    split_file = root / SPLIT_FILENAME
    if split_file.exists():
        with open(split_file, newline="", encoding="utf-8") as fh:
            return [(r["path"], r["label"], r["split"]) for r in csv.DictReader(fh)]
    rows = _auto_split(records, split_cfg, seed)
    with open(split_file, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["path", "label", "split"])
        writer.writeheader()
        for path, label, split in rows:
            writer.writerow({"path": path, "label": label, "split": split})
    return rows


def _bundle_from_rows(rows: list[tuple[str, str, str]], train_tf, eval_tf) -> DataBundle:
    labels = sorted({label for _, label, _ in rows})
    label_to_idx = {label: i for i, label in enumerate(labels)}
    per_split: dict[str, list[tuple[str, int]]] = {"train": [], "gallery": [], "query": []}
    for path, label, split in rows:
        if split not in per_split:
            raise ValueError(f"Unknown split '{split}' (expected train/gallery/query)")
        per_split[split].append((path, label_to_idx[label]))
    for split in ("train", "gallery", "query"):
        if not per_split[split]:
            raise ValueError(f"Split '{split}' is empty; check your split settings")
    train = ImageListDataset(per_split["train"], train_tf)
    return DataBundle(
        train=train,
        gallery=ImageListDataset(per_split["gallery"], eval_tf),
        query=ImageListDataset(per_split["query"], eval_tf),
        num_classes=len(labels),
        train_labels=train.labels,
    )


@registry.dataset("image_folder")
class ImageFolderAdapter:
    """root/<class_name>/<image> layout; zero code required."""

    @staticmethod
    def build(data_cfg: dict, train_tf, eval_tf) -> DataBundle:
        root = Path(data_cfg["root"] or "")
        if not root.is_dir():
            raise FileNotFoundError(f"data.root '{root}' is not a directory")
        records: list[tuple[str, str]] = []
        for class_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            for img in sorted(class_dir.rglob("*")):
                if img.suffix.lower() in IMG_EXTENSIONS:
                    records.append((str(img), class_dir.name))
        if not records:
            raise ValueError(f"No images found under '{root}' (extensions: {sorted(IMG_EXTENSIONS)})")
        rows = _load_or_create_split(root, records, data_cfg.get("split") or {}, seed=42)
        return _bundle_from_rows(rows, train_tf, eval_tf)


@registry.dataset("csv_manifest")
class CsvManifestAdapter:
    """CSV with columns: path,label[,split][,domain]. See docs for the contract."""

    @staticmethod
    def build(data_cfg: dict, train_tf, eval_tf) -> DataBundle:
        manifest = Path(data_cfg.get("manifest") or "")
        if not manifest.is_file():
            raise FileNotFoundError(f"data.manifest '{manifest}' not found")
        root = Path(data_cfg.get("root") or manifest.parent)
        with open(manifest, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            if reader.fieldnames is None or not {"path", "label"} <= set(reader.fieldnames):
                raise ValueError("csv_manifest needs at least the columns: path,label")
            has_split = "split" in reader.fieldnames
            raw = [(str((root / r["path"])) if not Path(r["path"]).is_absolute() else r["path"],
                    r["label"], r.get("split", "")) for r in reader]
        if has_split:
            rows = [(p, lb, sp) for p, lb, sp in raw]
        else:
            rows = _load_or_create_split(root, [(p, lb) for p, lb, _ in raw],
                                         data_cfg.get("split") or {}, seed=42)
        return _bundle_from_rows(rows, train_tf, eval_tf)


class SyntheticDataset(Dataset):
    """Deterministic class-conditional images; used by the CPU quickstart and tests.

    Each class has a fixed colour pattern plus seeded noise, so the task is
    learnable within a couple of epochs on CPU and fully reproducible.
    """

    def __init__(self, num_classes: int, per_class: int, size: int, seed: int = 42):
        self.num_classes = num_classes
        self.per_class = per_class
        self.size = size
        self.seed = seed

    def __len__(self) -> int:
        return self.num_classes * self.per_class

    @property
    def labels(self) -> list[int]:
        return [i // self.per_class for i in range(len(self))]

    def __getitem__(self, idx: int):
        label = idx // self.per_class
        gen = torch.Generator().manual_seed(self.seed * 100_003 + idx)
        base_gen = torch.Generator().manual_seed(1_000 + label)
        base = torch.rand((3, 4, 4), generator=base_gen)
        pattern = torch.nn.functional.interpolate(
            base.unsqueeze(0), size=(self.size, self.size), mode="nearest"
        ).squeeze(0)
        noise = 0.3 * torch.randn((3, self.size, self.size), generator=gen)
        return (pattern + noise), label


@registry.dataset("synthetic")
class SyntheticAdapter:
    """No-download bundle for smoke tests and the quickstart config."""

    @staticmethod
    def build(data_cfg: dict, train_tf, eval_tf) -> DataBundle:
        params = data_cfg.get("synthetic") or {}
        num_classes = int(params.get("num_classes", 4))
        per_class = int(params.get("per_class", 32))
        size = int(data_cfg.get("input_size", 64))
        train = SyntheticDataset(num_classes, per_class, size, seed=42)
        gallery = SyntheticDataset(num_classes, max(2, per_class // 4), size, seed=43)
        query = SyntheticDataset(num_classes, max(2, per_class // 4), size, seed=44)
        return DataBundle(
            train=train, gallery=gallery, query=query,
            num_classes=num_classes, train_labels=train.labels,
        )


def build_bundle(data_cfg: dict, train_tf, eval_tf) -> DataBundle:
    adapter = registry.get("dataset", data_cfg["adapter"])
    bundle = adapter.build(data_cfg, train_tf, eval_tf)
    target_cfg = data_cfg.get("target")
    if data_cfg.get("protocol") == "cross_domain" and target_cfg:
        target_bundle = registry.get("dataset", target_cfg["adapter"]).build(
            {**data_cfg, **target_cfg, "protocol": "gallery_query", "target": None},
            train_tf, eval_tf,
        )
        bundle.target_gallery = target_bundle.gallery
        bundle.target_query = target_bundle.query
    return bundle
