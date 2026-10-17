"""Safe zip extract and sha256 dedupe."""
from __future__ import annotations

import hashlib
import os
import zipfile


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_extract(zf: zipfile.ZipFile, dest: str) -> None:
    """Extract a zip without writing outside dest."""
    root = os.path.abspath(dest)
    for info in zf.infolist():
        name = info.filename.replace("\\", "/")
        if name.startswith("/") or name.startswith("../") or "/../" in name:
            raise ValueError(f"zip entry escapes the library: {info.filename}")
        target = os.path.abspath(os.path.join(root, name))
        if target != root and not target.startswith(root + os.sep):
            raise ValueError(f"zip entry escapes the library: {info.filename}")
    zf.extractall(root)


def extract_zip(zip_path: str, dest: str) -> None:
    os.makedirs(dest, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        safe_extract(zf, dest)
