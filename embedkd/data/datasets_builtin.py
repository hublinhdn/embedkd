"""Built-in public datasets with frozen open-set retrieval splits.

Split protocol (documented divergence from leave-one-out evaluation): the
standard metric-learning class splits are used for train vs test classes,
and each test class's images are split deterministically into disjoint
gallery and query halves (even index -> gallery, odd -> query, after sorting
by file name). The resulting split is written to disk on first use, so it is
frozen and shipped with the reproduction demos.

  * CUB-200-2011: classes 1-100 train, 101-200 test (Wah et al., 2011).
  * Cars196: classes 1-98 train, 99-196 test (Krause et al., 2013).
  * Stanford Online Products: official Ebay_train/Ebay_test lists
    (Oh Song et al., CVPR 2016).
"""

from __future__ import annotations

from pathlib import Path

from ..registry import registry
from .adapters import DataBundle, _bundle_from_rows, _load_or_create_split  # noqa: F401
from .downloads import DownloadError, download_file, extract_archive

CUB_URL = "https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz"
CUB_MD5 = "97eceeb196236b17998738112f37df78"
SOP_URL = "http://ftp.cs.stanford.edu/cs/cvgl/Stanford_Online_Products.zip"
SOP_MD5 = "7f73d41a2f44250d4779881525aea32e"


def _even_odd_split(paths: list[str]) -> list[str]:
    """Deterministic gallery/query assignment for test-class images."""
    return ["gallery" if i % 2 == 0 else "query" for i, _ in enumerate(sorted(paths))]


def _open_set_rows(items: list[tuple[str, str]], train_classes: set[str]) -> list[tuple[str, str, str]]:
    rows: list[tuple[str, str, str]] = []
    by_class: dict[str, list[str]] = {}
    for path, label in items:
        by_class.setdefault(label, []).append(path)
    for label in sorted(by_class):
        paths = sorted(by_class[label])
        if label in train_classes:
            rows.extend((p, label, "train") for p in paths)
        else:
            splits = _even_odd_split(paths)
            rows.extend((p, label, s) for p, s in zip(paths, splits))
    return rows


@registry.dataset("cub200")
class Cub200Adapter:
    """CUB-200-2011. data.root should contain (or receive) CUB_200_2011/."""

    @staticmethod
    def download(root: str | Path) -> Path:
        root = Path(root)
        base = root / "CUB_200_2011"
        if (base / "images.txt").exists():
            return base
        archive = download_file(CUB_URL, root / "CUB_200_2011.tgz", md5=CUB_MD5)
        extract_archive(archive, root)
        if not (base / "images.txt").exists():
            raise DownloadError(f"Extraction finished but {base}/images.txt is missing")
        return base

    @staticmethod
    def build(data_cfg: dict, train_tf, eval_tf) -> DataBundle:
        base = Path(data_cfg["root"] or "") / "CUB_200_2011"
        images_txt = base / "images.txt"
        if not images_txt.exists():
            raise FileNotFoundError(
                f"CUB-200-2011 not found under '{base}'. "
                "Run: embedkd datasets download cub200 --root <root>"
            )
        items: list[tuple[str, str]] = []
        with open(images_txt, encoding="utf-8") as fh:
            for line in fh:
                _, rel = line.split()
                class_name = rel.split("/")[0]  # e.g. 001.Black_footed_Albatross
                items.append((str(base / "images" / rel), class_name))
        train_classes = {c for _, c in items if int(c.split(".")[0]) <= 100}
        rows = _open_set_rows(items, train_classes)
        return _bundle_from_rows(rows, train_tf, eval_tf)


@registry.dataset("sop")
class SOPAdapter:
    """Stanford Online Products. data.root should contain Stanford_Online_Products/."""

    @staticmethod
    def download(root: str | Path) -> Path:
        root = Path(root)
        base = root / "Stanford_Online_Products"
        if (base / "Ebay_train.txt").exists():
            return base
        archive = download_file(SOP_URL, root / "Stanford_Online_Products.zip", md5=SOP_MD5)
        extract_archive(archive, root)
        if not (base / "Ebay_train.txt").exists():
            raise DownloadError(f"Extraction finished but {base}/Ebay_train.txt is missing")
        return base

    @staticmethod
    def _read_list(path: Path) -> list[tuple[str, str]]:
        items = []
        with open(path, encoding="utf-8") as fh:
            next(fh)  # header: image_id class_id super_class_id path
            for line in fh:
                _, class_id, _, rel = line.split()
                items.append((str(path.parent / rel), class_id))
        return items

    @classmethod
    def build(cls, data_cfg: dict, train_tf, eval_tf) -> DataBundle:
        base = Path(data_cfg["root"] or "") / "Stanford_Online_Products"
        if not (base / "Ebay_train.txt").exists():
            raise FileNotFoundError(
                f"Stanford Online Products not found under '{base}'. "
                "Run: embedkd datasets download sop --root <root>"
            )
        train_items = cls._read_list(base / "Ebay_train.txt")
        test_items = cls._read_list(base / "Ebay_test.txt")
        rows = [(p, label, "train") for p, label in train_items]
        by_class: dict[str, list[str]] = {}
        for path, label in test_items:
            by_class.setdefault(label, []).append(path)
        for label in sorted(by_class):
            paths = sorted(by_class[label])
            if len(paths) == 1:  # singleton class: gallery only
                rows.append((paths[0], label, "gallery"))
                continue
            splits = _even_odd_split(paths)
            rows.extend((p, label, s) for p, s in zip(paths, splits))
        return _bundle_from_rows(rows, train_tf, eval_tf)


@registry.dataset("cars196")
class Cars196Adapter:
    """Cars196. The original Stanford URLs are frequently offline, so this
    adapter expects a manual download; see the error message for the layout.
    """

    LAYOUT_HELP = (
        "Expected layout under data.root:\n"
        "  cars196/\n"
        "    car_ims/000001.jpg ... 016185.jpg\n"
        "    cars_annos.csv  (columns: path,label; one row per image)\n"
        "The official archive (car_ims.tgz + cars_annos.mat) is distributed via "
        "the Stanford AI mirror or Kaggle ('cars196'). Convert cars_annos.mat to "
        "cars_annos.csv with scripts/convert_cars196.py in the EmbedKD repo."
    )

    @staticmethod
    def download(root: str | Path) -> Path:
        raise DownloadError(
            "Cars196 has no stable official mirror; manual download is required.\n"
            + Cars196Adapter.LAYOUT_HELP
        )

    @staticmethod
    def build(data_cfg: dict, train_tf, eval_tf) -> DataBundle:
        import csv

        base = Path(data_cfg["root"] or "") / "cars196"
        annos = base / "cars_annos.csv"
        if not annos.exists():
            raise FileNotFoundError(
                f"Cars196 annotations not found at '{annos}'.\n" + Cars196Adapter.LAYOUT_HELP
            )
        items: list[tuple[str, str]] = []
        with open(annos, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                items.append((str(base / row["path"]), row["label"]))
        labels_sorted = sorted({label for _, label in items}, key=lambda x: int(x))
        train_classes = set(labels_sorted[: len(labels_sorted) // 2])
        rows = _open_set_rows(items, train_classes)
        return _bundle_from_rows(rows, train_tf, eval_tf)


DOWNLOADABLE = {"cub200": Cub200Adapter, "sop": SOPAdapter, "cars196": Cars196Adapter}
