"""Normal decode/encode, height reconstruction, convention detection."""
from __future__ import annotations

import cv2
import numpy as np

from qs.core.filter import gblur, resize
from qs.core.io import to_u8


def decode_normal(rgb):
    n = rgb * 2.0 - 1.0
    n[..., 2] = np.maximum(n[..., 2], 0.0)
    return normalize_normal(n)


def normalize_normal(n):
    n = n.astype(np.float32).copy()
    n[..., 2] = np.maximum(n[..., 2], 1e-3)
    ln = np.sqrt((n * n).sum(axis=2, keepdims=True))
    return n / np.maximum(ln, 1e-6)


def encode_normal(n):
    return to_u8(n * 0.5 + 0.5)


def _ddx(a, tile):
    if tile:
        return (np.roll(a, -1, 1) - np.roll(a, 1, 1)) * 0.5
    return cv2.Sobel(a, cv2.CV_32F, 1, 0, ksize=1, borderType=cv2.BORDER_REPLICATE) * 0.5


def _ddy(a, tile):
    if tile:
        return (np.roll(a, -1, 0) - np.roll(a, 1, 0)) * 0.5
    return cv2.Sobel(a, cv2.CV_32F, 0, 1, ksize=1, borderType=cv2.BORDER_REPLICATE) * 0.5


def normal_from_height(h, strength, tile):
    hx, hy = _ddx(h, tile), _ddy(h, tile)
    n = np.stack([-hx * strength, -hy * strength, np.ones_like(h)], axis=2)
    return normalize_normal(n)


def _corr(a, b):
    a = a - a.mean()
    b = b - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else 0.0


def detect_normal_convention(n, h, tile):
    s = 512.0 / max(h.shape)
    if s < 1:
        w, hh = max(int(h.shape[1] * s), 8), max(int(h.shape[0] * s), 8)
        h = resize(h, w, hh)
        n = resize(n, w, hh)
    h = gblur(h, 1.0, tile)
    cy = _corr(n[..., 1], -_ddy(h, tile))
    cx = _corr(n[..., 0], -_ddx(h, tile))
    return cx, cy


def curvature_from_normal(n_dx, tile):
    return _ddx(n_dx[..., 0], tile) + _ddy(n_dx[..., 1], tile)


def normalise_signed(c):
    c = c - np.median(c)
    s = np.percentile(np.abs(c), 98) + 1e-6
    return np.clip(c / s, -1.0, 1.0).astype(np.float32)
