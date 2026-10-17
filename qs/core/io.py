"""Image read/write and header verification."""
from __future__ import annotations

import os

import cv2
import numpy as np


def imread(path):
    data = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise IOError(f"could not read image: {path}")
    if img.dtype == np.uint8:
        f = img.astype(np.float32) / 255.0
    elif img.dtype == np.uint16:
        f = img.astype(np.float32) / 65535.0
    else:
        f = np.clip(np.nan_to_num(img.astype(np.float32)), 0.0, None)
    if f.ndim == 3:
        if f.shape[2] == 4:
            f = f[..., :3]
        f = f[..., ::-1].copy()
    return f


def load_rgb(path):
    f = imread(path)
    if f.ndim == 2:
        f = np.repeat(f[..., None], 3, axis=2)
    return f


def load_gray(path):
    f = imread(path)
    if f.ndim == 3:
        f = f.mean(axis=2)
    return f


def imwrite(path, rgb_or_gray_u8):
    a = rgb_or_gray_u8
    if path.lower().endswith(".tga"):
        from PIL import Image
        mode = {2: "L", 3: "RGB", 4: "RGBA"}[a.ndim if a.ndim == 2 else a.shape[2]]
        Image.fromarray(np.ascontiguousarray(a), mode).save(path, format="TGA", compression=None)
        return
    if a.ndim == 3:
        a = a[..., [2, 1, 0, 3]] if a.shape[2] == 4 else a[..., ::-1]
    ok, buf = cv2.imencode(".png", np.ascontiguousarray(a))
    if not ok:
        raise IOError(f"could not encode {path}")
    buf.tofile(path)


def to_u8(x):
    return (np.clip(np.nan_to_num(x), 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def resize(img, w, h):
    ih, iw = img.shape[:2]
    if (iw, ih) == (w, h):
        return img.astype(np.float32)
    interp = cv2.INTER_AREA if (w < iw or h < ih) else cv2.INTER_LANCZOS4
    return cv2.resize(img.astype(np.float32), (w, h), interpolation=interp)


def png_color_type(path):
    with open(path, "rb") as fh:
        b = fh.read(26)
    if b[:8] != b"\x89PNG\r\n\x1a\n" or b[12:16] != b"IHDR":
        return None
    return b[24], b[25]


def verify_written(path, channels, size):
    from PIL import Image
    with Image.open(path) as im:
        mode, dims = im.mode, im.size
    want_mode = {1: "L", 3: "RGB", 4: "RGBA"}[channels]
    if mode != want_mode or tuple(dims) != tuple(size):
        raise RuntimeError(f"wrote {path} as {mode} {dims}, expected {want_mode} {tuple(size)}")
    desc = f"{mode} {dims[0]}x{dims[1]}"
    if path.lower().endswith(".png"):
        depth, ctype = png_color_type(path)
        if ctype != {1: 0, 3: 2, 4: 6}[channels] or depth != 8:
            raise RuntimeError(f"{path}: PNG IHDR colour type {ctype}, depth {depth}")
        desc += f" IHDR type {ctype}"
    elif path.lower().endswith(".tga"):
        with open(path, "rb") as fh:
            bpp = fh.read(18)[16]
        if bpp != {1: 8, 3: 24, 4: 32}[channels]:
            raise RuntimeError(f"{path}: TGA is {bpp}-bit")
        desc += f" TGA {bpp}-bit"
    return desc


def resolve_out_dir(set_folder, out_dir=None):
    if out_dir is not None and str(out_dir).strip():
        od = os.path.abspath(os.path.expandvars(os.path.expanduser(str(out_dir).strip().strip('"'))))
        if os.path.normcase(os.path.normpath(od)) == os.path.normcase(os.path.normpath(set_folder)):
            od = os.path.join(od, "Stylized")
        return od
    return os.path.join(set_folder, "Stylized")
