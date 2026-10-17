"""SMAA 1x, colour edge, PRESET_HIGH. LUTs from iryoku/smaa AreaTex/SearchTex."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np

DATA = Path(__file__).resolve().parents[1] / "data"

# SMAA.ush SMAA_PRESET_HIGH
THRESHOLD = 0.1
MAX_SEARCH_STEPS = 16
MAX_SEARCH_STEPS_DIAG = 8
CORNER_ROUNDING = 25
LOCAL_CONTRAST_ADAPTATION = 2.0


@lru_cache(maxsize=1)
def area_tex() -> np.ndarray:
    return np.load(DATA / "smaa_area.npy").astype(np.float32) / 255.0


@lru_cache(maxsize=1)
def search_tex() -> np.ndarray:
    return np.load(DATA / "smaa_search.npy").astype(np.float32) / 255.0


def _luma(rgb):
    return rgb[..., 0] * 0.2126 + rgb[..., 1] * 0.7152 + rgb[..., 2] * 0.0722


def detect_edges(ldr: np.ndarray) -> np.ndarray:
    """Colour-edge SMAA. Returns RG = left/top edges, 0..1."""
    img = np.clip(np.asarray(ldr, np.float32), 0, 1)
    h, w = img.shape[:2]
    lum = _luma(img)
    left = np.abs(lum - np.roll(lum, 1, 1))
    top = np.abs(lum - np.roll(lum, 1, 0))
    # Colour delta as well (SMAA colour mode).
    left_c = np.max(np.abs(img - np.roll(img, 1, 1)), axis=2)
    top_c = np.max(np.abs(img - np.roll(img, 1, 0)), axis=2)
    left = np.maximum(left, left_c)
    top = np.maximum(top, top_c)
    edges = np.stack([left, top], axis=2)
    edges[0, :, 1] = 0
    edges[:, 0, 0] = 0
    edges = np.where(edges >= THRESHOLD, edges, 0.0)
    # Local contrast adaptation: drop weak edges next to stronger ones.
    max_n = np.maximum(np.roll(edges[..., 0], -1, 1), np.roll(edges[..., 1], -1, 0))
    max_n = np.maximum(max_n, np.maximum(np.roll(edges[..., 0], 1, 1), np.roll(edges[..., 1], 1, 0)))
    keep = edges >= (max_n[..., None] / LOCAL_CONTRAST_ADAPTATION)
    return np.where(keep, np.clip(edges, 0, 1), 0.0).astype(np.float32)


def _sample_area(e1, e2):
    area = area_tex()
    h, w = area.shape[:2]
    x = np.clip(e1 * (w - 1), 0, w - 1)
    y = np.clip(e2 * (h - 1), 0, h - 1)
    x0 = np.floor(x).astype(np.int32)
    y0 = np.floor(y).astype(np.int32)
    x1 = np.minimum(x0 + 1, w - 1)
    y1 = np.minimum(y0 + 1, h - 1)
    fx = (x - x0)[..., None]
    fy = (y - y0)[..., None]
    a00 = area[y0, x0]
    a10 = area[y0, x1]
    a01 = area[y1, x0]
    a11 = area[y1, x1]
    return (a00 * (1 - fx) + a10 * fx) * (1 - fy) + (a01 * (1 - fx) + a11 * fx) * fy


def blend_weights(edges: np.ndarray) -> np.ndarray:
    """Approximate SMAA blending weights using the official area texture.

    Search is a 1D walk of MAX_SEARCH_STEPS along each edge. Enough to AA a
    1-pixel chessboard and a long silhouette; the GPU path uses the same LUTs.
    """
    e = np.asarray(edges, np.float32)
    h, w = e.shape[:2]
    weights = np.zeros((h, w, 4), np.float32)

    def walk(axis, sign, component):
        """component 0 = left/right (horizontal edges use top flag)."""
        acc = np.zeros((h, w), np.float32)
        alive = e[..., component] > 0
        for step in range(1, MAX_SEARCH_STEPS + 1):
            shifted = np.roll(e[..., component], sign * step, axis)
            acc = np.where(alive, acc + 1.0, acc)
            still = shifted > 0
            alive = alive & still
        return acc

    # Distances to the end of a crossing edge, used as AreaTex UVs.
    d_left = walk(1, 1, 0)
    d_right = walk(1, -1, 0)
    d_up = walk(0, 1, 1)
    d_down = walk(0, -1, 1)
    # Horizontal crossing (left+right) mapped into area tex.
    horiz = _sample_area(
        np.clip(d_left / float(MAX_SEARCH_STEPS), 0, 1),
        np.clip(d_right / float(MAX_SEARCH_STEPS), 0, 1),
    )
    vert = _sample_area(
        np.clip(d_up / float(MAX_SEARCH_STEPS), 0, 1),
        np.clip(d_down / float(MAX_SEARCH_STEPS), 0, 1),
    )
    has_h = e[..., 0] > 0
    has_v = e[..., 1] > 0
    weights[..., 0] = np.where(has_h, horiz[..., 0], 0)  # left
    weights[..., 1] = np.where(has_h, horiz[..., 1], 0)  # right
    weights[..., 2] = np.where(has_v, vert[..., 0], 0)   # top
    weights[..., 3] = np.where(has_v, vert[..., 1], 0)   # bottom
    # Corner rounding: shrink opposite weights.
    cr = CORNER_ROUNDING / 100.0
    if cr > 0:
        weights[..., 0] *= (1.0 - cr * weights[..., 2])
        weights[..., 1] *= (1.0 - cr * weights[..., 3])
        weights[..., 2] *= (1.0 - cr * weights[..., 0])
        weights[..., 3] *= (1.0 - cr * weights[..., 1])
    return np.clip(weights, 0, 1)


def neighborhood_blend(ldr: np.ndarray, weights: np.ndarray) -> np.ndarray:
    img = np.clip(np.asarray(ldr, np.float32), 0, 1)
    wgt = np.asarray(weights, np.float32)
    left = np.roll(img, 1, 1)
    right = np.roll(img, -1, 1)
    up = np.roll(img, 1, 0)
    down = np.roll(img, -1, 0)
    a = wgt[..., 0:1]
    b = wgt[..., 1:2]
    c = wgt[..., 2:3]
    d = wgt[..., 3:4]
    s = np.maximum(a + b + c + d, 1e-8)
    blended = (left * a + right * b + up * c + down * d) / s
    use = (a + b + c + d) > 1e-5
    return np.where(use, blended, img).astype(np.float32)


def apply_smaa(ldr: np.ndarray) -> np.ndarray:
    edges = detect_edges(ldr)
    weights = blend_weights(edges)
    return neighborhood_blend(ldr, weights)
