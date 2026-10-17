"""CPU Kuwahara (match oracle) plus median/bilateral and seam fade."""
from __future__ import annotations

import cv2
import numpy as np

from qs.core.io import resize  # noqa: F401  re-export for normals


class Cancelled(Exception):
    """A newer request made this paint stale."""


def _pad(img, p, tile):
    return cv2.copyMakeBorder(img, p, p, p, p, cv2.BORDER_WRAP if tile else cv2.BORDER_REFLECT_101)


_KCACHE = {}


def kuwahara_kernels(r, classic=False):
    key = (r, classic)
    if key in _KCACHE:
        return _KCACHE[key]
    y, x = np.mgrid[-r:r + 1, -r:r + 1].astype(np.float32)
    full = []
    if classic:
        for sx, sy in ((-1, -1), (1, -1), (-1, 1), (1, 1)):
            k = ((x * sx >= 0) & (y * sy >= 0)).astype(np.float32)
            full.append(k / k.sum())
    else:
        ang = np.arctan2(y, x)
        dist = np.sqrt(x * x + y * y)
        sigma = max(r / 2.0, 0.5)
        g = np.exp(-(dist ** 2) / (2 * sigma ** 2)).astype(np.float32)
        g[dist > r + 0.5] = 0.0
        n = 8
        for i in range(n):
            d = np.angle(np.exp(1j * (ang - 2 * np.pi * i / n)))
            w = np.where(np.abs(d) < np.pi / 4, np.cos(2 * d) ** 2, 0.0).astype(np.float32)
            w[r, r] = 1.0
            k = w * g
            full.append((k / k.sum()).astype(np.float32))
    ks = []
    for k in full:
        k = k.astype(np.float32)
        rows, cols = np.nonzero(k)
        y0, y1, x0, x1 = rows.min(), rows.max(), cols.min(), cols.max()
        ks.append((np.ascontiguousarray(k[y0:y1 + 1, x0:x1 + 1]), (int(r - x0), int(r - y0))))
    _KCACHE[key] = ks
    return ks


def kuwahara_pass(img, guides, r, q, tile, classic=False, cancel=None):
    ks = kuwahara_kernels(r, classic)
    p = r + 1
    H, W = img.shape[:2]
    C = img.shape[2]
    border = cv2.BORDER_WRAP if tile else cv2.BORDER_REFLECT_101

    def padp(a):
        return cv2.copyMakeBorder(np.ascontiguousarray(a, dtype=np.float32), p, p, p, p, border)

    def filt(plane_p, ka):
        return cv2.filter2D(plane_p, cv2.CV_32F, ka[0], anchor=ka[1],
                            borderType=cv2.BORDER_REFLECT_101)[p:H + p, p:W + p]

    ap = [padp(img[..., c]) for c in range(C)]
    a2p = [padp(img[..., c] * img[..., c]) for c in range(C)]
    means = []
    logw = np.empty((len(ks), H, W), np.float32)
    for i, ka in enumerate(ks):
        var = np.zeros((H, W), np.float32)
        mi = []
        for c in range(C):
            m = filt(ap[c], ka)
            var += filt(a2p[c], ka)
            var -= m * m
            mi.append(m)
        means.append(mi)
        np.maximum(var, 0, out=var)
        var += 1e-5
        np.log(var, out=logw[i])
        logw[i] *= -1.0
    if cancel and cancel():
        raise Cancelled()
    if classic:
        w = (logw == logw.max(axis=0, keepdims=True)).astype(np.float32)
    else:
        logw *= (q / 2.0)
        logw -= logw.max(axis=0, keepdims=True)
        w = np.exp(logw)
    w /= w.sum(axis=0, keepdims=True)
    outc = [np.zeros((H, W), np.float32) for _ in range(C)]
    for i in range(len(ks)):
        for c in range(C):
            cv2.accumulateProduct(np.ascontiguousarray(means[i][c]), w[i], outc[c])
    del means
    planes, layout = [], []
    for g_ in guides:
        if g_.ndim == 2:
            layout.append(1)
            planes.append(padp(g_))
        else:
            layout.append(g_.shape[2])
            planes.extend(padp(g_[..., c]) for c in range(g_.shape[2]))
    acc = [np.zeros((H, W), np.float32) for _ in planes]
    for i, ka in enumerate(ks):
        if cancel and i == 4 and cancel():
            raise Cancelled()
        for j, pl in enumerate(planes):
            cv2.accumulateProduct(np.ascontiguousarray(filt(pl, ka)), w[i], acc[j])
    gout, o = [], 0
    for n_ in layout:
        gout.append(acc[o] if n_ == 1 else np.dstack(acc[o:o + n_]))
        o += n_
    return np.dstack(outc).astype(np.float32), gout


def _median_float(channel, k):
    channel = np.ascontiguousarray(channel, dtype=np.float32)
    if k <= 5:
        return cv2.medianBlur(channel, k)
    pad = k // 2
    padded = np.pad(channel, pad, mode="edge")
    out = np.empty_like(channel)
    H = channel.shape[0]
    step = 8
    for y in range(0, H, step):
        y1 = min(y + step, H)
        slab = padded[y:y1 + 2 * pad]
        win = np.lib.stride_tricks.sliding_window_view(slab, (k, k))
        out[y:y1] = np.median(win, axis=(-1, -2))
    return out


def simple_filter(img, method, r, tile):
    p = r + 1
    ip = _pad(img, p, tile)
    k = 2 * r + 1
    if method == "median":
        if ip.ndim == 3:
            out = np.stack([_median_float(ip[..., i], k) for i in range(ip.shape[2])], axis=2)
        else:
            out = _median_float(ip, k)
    else:
        out = cv2.bilateralFilter(ip.astype(np.float32), k, 0.12, max(r, 1))
    return out[p:-p, p:-p].astype(np.float32)


def gblur(img, sigma, tile):
    if sigma <= 0.05:
        return np.array(img, np.float32, copy=True)
    p = int(np.ceil(sigma * 3)) + 1
    ip = _pad(img, p, tile)
    return cv2.GaussianBlur(ip, (0, 0), sigma)[p:-p, p:-p]


def seam_metrics(alb, normal):
    def sides(img):
        a = img[:, 0].astype(np.float32) - img[:, -1].astype(np.float32)
        b = img[0, :].astype(np.float32) - img[-1, :].astype(np.float32)
        return float(np.mean(np.abs(a))), float(np.mean(np.abs(b)))
    d_lr, d_tb = sides(alb)
    n_lr, n_tb = sides(normal)
    return {"diffuse_lr": d_lr, "diffuse_tb": d_tb, "normal_lr": n_lr, "normal_tb": n_tb}


def kuwahara_pass_islands(img, guides, r, q, island_ids, classic=False, cancel=None):
    """Run Kuwahara independently on each UV island (no wrap across gutters)."""
    ids = np.asarray(island_ids)
    ih, iw = img.shape[:2]
    if ids.shape[:2] != (ih, iw):
        ids = np.round(resize(ids.astype(np.float32), iw, ih)).astype(np.int32)
    out = np.array(img, np.float32, copy=True)
    gout = [np.array(g, np.float32, copy=True) for g in guides]
    for iid in np.unique(ids):
        if iid <= 0:
            continue
        if cancel and cancel():
            raise Cancelled()
        mask = ids == iid
        ys, xs = np.where(mask)
        y0, y1 = max(ys.min() - r, 0), min(ys.max() + r + 1, img.shape[0])
        x0, x1 = max(xs.min() - r, 0), min(xs.max() + r + 1, img.shape[1])
        crop = img[y0:y1, x0:x1]
        cg = [g[y0:y1, x0:x1] for g in guides]
        painted, pg = kuwahara_pass(crop, cg, r, q, tile=False, classic=classic, cancel=cancel)
        sub = mask[y0:y1, x0:x1]
        if sub.shape[:2] != painted.shape[:2]:
            continue
        if painted.ndim == 3:
            out[y0:y1, x0:x1][sub] = painted[sub]
        else:
            out[y0:y1, x0:x1][sub] = painted[sub]
        for i, g in enumerate(pg):
            if g.shape[:2] != sub.shape[:2]:
                continue
            if g.ndim == 3:
                gout[i][y0:y1, x0:x1][sub] = g[sub]
            else:
                gout[i][y0:y1, x0:x1][sub] = g[sub]
    return out, gout


def dilate_islands(img, island_ids, radius: int):
    """Spread island texels into the 0-gutter so mip filtering does not pick black.

    One dilate of the packed atlas, not one pass per island. The stump has ~100
    islands; the per-island loop froze the UI for over ten seconds on open.
    """
    if radius <= 0 or island_ids is None:
        return img
    import cv2
    ids = np.asarray(island_ids)
    if ids.shape[:2] != img.shape[:2]:
        ids = np.round(resize(ids.astype(np.float32), img.shape[1], img.shape[0])).astype(np.int32)
    gutter = ids <= 0
    covered = ids > 0
    if not np.any(gutter) or not np.any(covered):
        return img
    k = 2 * int(radius) + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    out = np.asarray(img, np.float32)
    mask = covered.astype(np.float32)
    count = cv2.dilate(mask, kernel)
    if out.ndim == 2:
        filled = cv2.dilate(out * mask, kernel) / np.maximum(count, 1e-6)
        return np.where(gutter, filled, out).astype(np.float32)
    filled = cv2.dilate(out * mask[..., None], kernel) / np.maximum(count[..., None], 1e-6)
    return np.where(gutter[..., None], filled, out).astype(np.float32)


def seam_fade(img, strength, tile):
    if not tile or strength <= 0 or img is None:
        return img
    out = np.array(img, np.float32, copy=True)
    h, w = out.shape[:2]
    span = min(h, w)
    bw = int(round(max(2.0, span * (0.01 + 0.04 * float(strength)))))
    bw = max(1, min(bw, w // 4, h // 4))
    ramp = np.linspace(float(strength), 0.0, bw, dtype=np.float32)
    if out.ndim == 2:
        ramp_x = ramp[None, :]
        ramp_y = ramp[:, None]
        right = out[:, -bw:][:, ::-1]
        out[:, :bw] = out[:, :bw] * (1.0 - ramp_x) + right * ramp_x
        bottom = out[-bw:, :][::-1, :]
        out[:bw, :] = out[:bw, :] * (1.0 - ramp_y) + bottom * ramp_y
    else:
        ramp_x = ramp[None, :, None]
        ramp_y = ramp[:, None, None]
        right = out[:, -bw:][:, ::-1]
        out[:, :bw] = out[:, :bw] * (1.0 - ramp_x) + right * ramp_x
        bottom = out[-bw:, :][::-1, :]
        out[:bw, :] = out[:bw, :] * (1.0 - ramp_y) + bottom * ramp_y
    return out
