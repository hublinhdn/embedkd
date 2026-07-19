"""Download helpers for the built-in public datasets.

No dataset images are ever redistributed with EmbedKD; files are fetched from
their official sources and verified by checksum where one is published.
"""

from __future__ import annotations

import hashlib
import sys
import tarfile
import urllib.request
import zipfile
from pathlib import Path


class DownloadError(RuntimeError):
    pass


def _md5(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.md5()
    with open(path, "rb") as fh:
        while block := fh.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def download_file(url: str, dest: Path, md5: str | None = None, quiet: bool = False) -> Path:
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        if md5 is None or _md5(dest) == md5:
            return dest
        dest.unlink()  # checksum mismatch: re-download

    def report(blocks: int, block_size: int, total: int) -> None:
        if quiet or total <= 0:
            return
        done = min(blocks * block_size / total, 1.0)
        sys.stderr.write(f"\r  downloading {dest.name}: {done:6.1%}")
        sys.stderr.flush()

    try:
        urllib.request.urlretrieve(url, dest, reporthook=report)
    except OSError as exc:
        raise DownloadError(
            f"Could not download {url} ({exc}). If the mirror is unavailable, "
            f"download the archive manually, place it at {dest}, and re-run."
        ) from exc
    finally:
        if not quiet:
            sys.stderr.write("\n")
    if md5 is not None and (actual := _md5(dest)) != md5:
        raise DownloadError(
            f"Checksum mismatch for {dest.name}: expected {md5}, got {actual}. "
            "The download may be corrupted or the source may have changed."
        )
    return dest


def extract_archive(archive: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    name = archive.name.lower()
    if name.endswith((".tgz", ".tar.gz", ".tar")):
        with tarfile.open(archive) as tar:
            tar.extractall(dest, filter="data")
    elif name.endswith(".zip"):
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(dest)
    else:
        raise DownloadError(f"Unsupported archive format: {archive.name}")
