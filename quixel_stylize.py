#!/usr/bin/env python3
"""
QuixelStylizer - turn a Quixel Megascans / Fab PBR surface set into a hand-painted
(Dishonored-style) material and write packed maps for Unreal Engine 5.

Outputs per set (8-bit PNG):
  <Name>_D.png     RGB = stylised diffuse (sRGB), no alpha
  <Name>_MROD.png  R = Metallic, G = Roughness, B = AO, A = Displacement/height (all linear)
  <Name>_N.png     stylised tangent-space normal (DirectX / green-down by default)
  <Name>_stylize.json  settings + detected inputs, for reproducibility
  <Name>_preview.png   before/after preview (albedo + normal)

Run with no arguments (or --gui) for the GUI, or pass folders for the CLI.
"""
import os
import sys
import re
import json
import time
import argparse

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
import numpy as np
import cv2

TOOL_VERSION = "1.3.0"
MODULE_PATH = os.path.abspath(__file__)
TOOL_DIR = os.path.dirname(MODULE_PATH)
WINDOW_TITLE_PREFIX = "Quixel Stylizer v"
LOG_PATH = os.path.join(TOOL_DIR, "stylizer.log")


def log_line(msg):
    """Append a timestamped line (or traceback) to stylizer.log next to the script."""
    try:
        with open(LOG_PATH, "a", encoding="utf-8") as fh:
            for ln in str(msg).rstrip().splitlines() or [""]:
                fh.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {ln}\n")
    except OSError:
        pass


def version_banner():
    return f"Quixel Stylizer v{TOOL_VERSION}  code: {MODULE_PATH}"


def other_tool_windows():
    """Titles of OTHER open Quixel Stylizer windows whose version differs from this code (Windows only).
    Old windows keep running the code they started with, so they still write the old file layout."""
    if os.name != "nt":
        return []
    try:
        import ctypes
        from ctypes import wintypes
        u = ctypes.windll.user32
        found = []

        def cb(h, _l):
            n = u.GetWindowTextLengthW(h)
            if n and u.IsWindowVisible(h):
                b = ctypes.create_unicode_buffer(n + 1)
                u.GetWindowTextW(h, b, n + 1)
                t = b.value
                if t.startswith(WINDOW_TITLE_PREFIX) and not t.startswith(f"{WINDOW_TITLE_PREFIX}{TOOL_VERSION} "):
                    pid = wintypes.DWORD()
                    u.GetWindowThreadProcessId(h, ctypes.byref(pid))
                    found.append(f"{t}  (PID {pid.value})")
            return True
        u.EnumWindows(ctypes.WINFUNCTYPE(ctypes.c_bool, wintypes.HWND, wintypes.LPARAM)(cb), 0)
        return found
    except Exception:
        return []


def old_window_warning():
    w = other_tool_windows()
    if not w:
        return ""
    return ("An older Quixel Stylizer window is still open:\n  " + "\n  ".join(w) +
            f"\nIt keeps running the old code it was started with and writes the OLD file layout "
            f"(e.g. _MROE + RGBA _D). Close it and only use v{TOOL_VERSION}.")


def png_color_type(path):
    """IHDR (bit depth, colour type) of a PNG: type 2 = RGB, 6 = RGBA, 0 = grey."""
    with open(path, "rb") as fh:
        b = fh.read(26)
    if b[:8] != b"\x89PNG\r\n\x1a\n" or b[12:16] != b"IHDR":
        return None
    return b[24], b[25]


def verify_written(path, channels, size):
    """Read the file header back and fail loudly if it is not what we meant to write."""
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


LEGACY_MASK_KEYS = ("MRO", "MROE")  # layouts written by v1.2.1 and earlier; removed from the tool's own output folder


def remove_legacy_outputs(out_dir, name, set_folder, source_files=()):
    """Delete <name>_MRO/_MROE .png/.tga left by older versions, ONLY when it is clearly this tool's output:
    in the output folder (never the source set folder), next to <name>_stylize.json, and not a source map."""
    if os.path.normcase(os.path.abspath(out_dir)) == os.path.normcase(os.path.abspath(set_folder)):
        return []
    if not os.path.isfile(os.path.join(out_dir, f"{name}_stylize.json")):
        return []
    src = {os.path.normcase(os.path.abspath(x)) for x in source_files}
    removed = []
    for k in LEGACY_MASK_KEYS:
        for e in (".png", ".tga"):
            pth = os.path.join(out_dir, f"{name}_{k}{e}")
            if os.path.isfile(pth) and os.path.normcase(os.path.abspath(pth)) not in src:
                try:
                    os.remove(pth)
                    removed.append(os.path.basename(pth))
                    log_line(f"removed old-layout output {pth}")
                except OSError as ex:
                    log_line(f"could not remove old-layout output {pth}: {ex}")
    return removed


def resolve_out_dir(set_folder, out_dir=None):
    """Default <set>\\Stylized. A chosen folder is made absolute; if it IS the source set folder,
    write into its Stylized subfolder so source maps are never mixed with or overwritten by outputs."""
    if out_dir is not None and str(out_dir).strip():
        od = os.path.abspath(os.path.expandvars(os.path.expanduser(str(out_dir).strip().strip('"'))))
        if os.path.normcase(os.path.normpath(od)) == os.path.normcase(os.path.normpath(set_folder)):
            od = os.path.join(od, "Stylized")
        return od
    return os.path.join(set_folder, "Stylized")
REF_SIZE = 1024  # radii / blur widths are specified in pixels at this size and scaled

DEFAULTS = {
    "size": 1024,                 # output size (longest side): 256/512/1024/2048
    "filter": "kuwahara",         # kuwahara | kuwahara_classic | anisotropic | median | bilateral
    "radius": 6.0,                # paint radius, px at 1024 (scaled with size)
    "passes": 2,                  # filter passes
    "sharpness": 8.0,             # Kuwahara sector selectivity (q); higher = harder edges
    "saturation": 1.15,           # 1 = unchanged
    "value_compression": 0.15,    # 0..1, squeeze luminance toward its mean
    "posterize": 0,               # luminance levels, 0 = off (try 8-16 for a light effect)
    "tint_strength": 0.35,        # warm-light / cool-shadow split, 0 = off
    "ao_strength": 0.5,           # AO multiplied into the diffuse, 0..1
    "edge_strength": 0.25,        # curvature edge highlight / crevice darkening, 0..1
    "edge_blur": 2.0,             # blur of curvature mask, px at 1024
    "curvature_source": "auto",   # auto (cavity > normal) | cavity | normal | height
    "roughness_flatten": 0.8,     # 0..1, pull roughness toward its mean
    "roughness_bias": 0.0,        # added to roughness after flattening
    "metal_binarize": 0.5,        # 0..1, push metallic toward 0/1
    "normal_soften": 0.5,         # 0..1, extra low-pass on the normal after paint filter
    "normal_strength": 1.0,       # scales normal XY before renormalising
    "normal_mode": "filter",      # filter (paint-filter the normal) | height (rebuild from filtered height)
    "normal_in": "auto",          # auto | dx | gl  (source normal convention)
    "flip_green": False,          # invert source normal green (override if detection is wrong)
    "normal_out": "dx",           # dx (Unreal) | gl
    "height_soften": 0.5,         # 0..1, extra low-pass on height after paint filter
    "tile": True,                 # wrap borders (Megascans surfaces are tileable)
    "preview": True,              # write <Name>_preview.png
    "format": "png",              # png | tga  (texture outputs; preview stays PNG)
    # 0 = keep the source map, 1 = fully painted. Albedo stays the paint guide.
    "mix_normal": 1.0,
    "mix_height": 1.0,
    "mix_rough": 1.0,
    "mix_metal": 1.0,
    "mix_ao": 1.0,
    "detail_restore": 0.0,        # add a high-pass of the source albedo back after the paint
    "palette": [],                # up to 8 sRGB swatches, 0..1. Empty = no snap
    "palette_strength": 0.0,      # 0 leaves the Oklab grade unsnapped
    "palette_hardness": 8.0,      # higher = flatter poster colours
    "tint_warm": [0.96, 0.78, 0.55],
    "tint_cool": [0.55, 0.68, 0.82],
    "seam_fade": 0.0,             # 0 = off. Cross-fade a band across the wrap
    "exposure": 0.0,              # stops on the lit view, 2^exposure
    "shade_ao": False,            # multiply the lit view by AO again (AO is already in the diffuse)
}
SIZES = [256, 512, 1024, 2048]
FILTERS = ["kuwahara", "kuwahara_classic", "anisotropic", "median", "bilateral"]
# Built-in Oklab palette starting points. Taste, not measurements. Strength stays 0 until a preset sets it.
PALETTES = {
    "plaster": [
        [0.93, 0.90, 0.84],
        [0.86, 0.78, 0.66],
        [0.72, 0.58, 0.44],
        [0.55, 0.40, 0.32],
        [0.40, 0.32, 0.28],
    ],
    "stone": [
        [0.78, 0.80, 0.81],
        [0.62, 0.64, 0.66],
        [0.45, 0.49, 0.52],
        [0.32, 0.35, 0.38],
        [0.24, 0.26, 0.28],
    ],
    "moss": [
        [0.55, 0.58, 0.32],
        [0.40, 0.48, 0.22],
        [0.24, 0.34, 0.18],
        [0.16, 0.22, 0.13],
        [0.30, 0.24, 0.16],
    ],
}

# --------------------------------------------------------------------------------------
# Map detection
# --------------------------------------------------------------------------------------
IMG_EXT_RANK = {".png": 0, ".tif": 1, ".tiff": 1, ".tga": 2, ".jpg": 3, ".jpeg": 3,
                ".bmp": 4, ".exr": 5}

SUFFIX_TYPES = {
    "albedo": ["albedo", "basecolor", "base_color", "basecolour", "base_colour", "diffuse",
               "diff", "color", "colour", "col"],
    "roughness": ["roughness", "rough", "rgh"],
    "gloss": ["gloss", "glossiness"],
    "metallic": ["metalness", "metallic", "metal"],
    "ao": ["ao", "ambientocclusion", "ambient_occlusion", "occlusion", "occ"],
    "cavity": ["cavity"],
    "normal_dx": ["normaldx", "normal_dx", "nor_dx", "nrm_dx", "normal_directx", "normaldirectx"],
    "normal_gl": ["normalgl", "normal_gl", "nor_gl", "nrm_gl", "normal_opengl", "normalopengl"],
    "normal": ["normal", "normals", "nrm", "nor", "norm"],
    "height": ["displacement", "disp", "height", "heightmap", "displace"],
    "bump": ["bump"],
    "opacity": ["opacity", "alpha"],
    "orm": ["orm", "arm", "occlusionroughnessmetallic", "occlusion_roughness_metallic"],
    "specular": ["specular", "spec"],
    "translucency": ["translucency", "transmission"],
}
_SUFFIXES = sorted(((s, t) for t, ss in SUFFIX_TYPES.items() for s in ss),
                   key=lambda x: -len(x[0]))
_RES_RE = re.compile(r"(_(\d{1,2}k|\d{3,5}|\d{3,5}x\d{3,5}|lod\d+))+$")
OUTPUT_MARKERS = ("_D", "_MROD", "_N", "_preview")


def _norm_stem(stem):
    return re.sub(r"[\s\-.]", "_", stem).lower()


def classify_file(fname):
    """Return (map_type, set_name) or (None, None)."""
    stem, ext = os.path.splitext(fname)
    if ext.lower() not in IMG_EXT_RANK:
        return None, None
    n = _norm_stem(stem)
    m = _RES_RE.search(n)
    if m:
        n2, stem2 = n[:m.start()], stem[:m.start()]
    else:
        n2, stem2 = n, stem
    for suf, typ in _SUFFIXES:
        if n2 == suf or n2.endswith("_" + suf):
            name = stem2[:len(stem2) - len(suf)].rstrip("_- .")
            nn = _norm_stem(name)
            m2 = _RES_RE.search(nn)
            if m2:
                name = name[:m2.start()]
            return typ, (name.rstrip("_- .") or "Material")
    return None, None


def detect_maps(folder):
    """Detect Megascans-style maps in one folder (non-recursive)."""
    found = {}
    names = {}
    ignored = []
    try:
        entries = sorted(os.listdir(folder))
    except OSError:
        return None
    for f in entries:
        p = os.path.join(folder, f)
        if not os.path.isfile(p):
            continue
        typ, name = classify_file(f)
        if typ is None:
            if os.path.splitext(f)[1].lower() in IMG_EXT_RANK:
                ignored.append(f)
            continue
        rank = (IMG_EXT_RANK[os.path.splitext(f)[1].lower()], -os.path.getsize(p))
        found.setdefault(typ, []).append((rank, p))
        names.setdefault(typ, name)
    if "albedo" not in found:
        return None
    maps = {t: sorted(v)[0][1] for t, v in found.items()}
    extra = {t: [p for _, p in sorted(v)[1:]] for t, v in found.items() if len(v) > 1}
    return {"folder": os.path.abspath(folder), "name": names["albedo"], "maps": maps,
            "alternates": extra, "ignored": ignored}


def find_sets(paths, recursive=False):
    sets, seen = [], set()

    def add(folder):
        key = os.path.normcase(os.path.abspath(folder))
        if key in seen:
            return
        seen.add(key)
        s = detect_maps(folder)
        if s:
            sets.append(s)

    for p in paths:
        p = os.path.abspath(p.strip().strip('"'))
        if os.path.isfile(p):
            p = os.path.dirname(p)
        if not os.path.isdir(p):
            print(f"[warn] not a folder: {p}")
            continue
        here = detect_maps(p)
        if here:
            add(p)
        if recursive:
            for root, dirs, _ in os.walk(p):
                dirs[:] = [d for d in dirs if d.lower() != "stylized"]
                if os.path.normcase(root) != os.path.normcase(p):
                    add(root)
        elif not here:
            for d in sorted(os.listdir(p)):
                sub = os.path.join(p, d)
                if os.path.isdir(sub) and d.lower() != "stylized":
                    add(sub)
    return sets


# --------------------------------------------------------------------------------------
# Image IO
# --------------------------------------------------------------------------------------
def imread(path):
    data = np.fromfile(path, dtype=np.uint8)  # unicode-safe on Windows
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
        f = f[..., ::-1].copy()  # BGR -> RGB
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
    """Write RGB/RGBA/grey uint8 (RGB order in memory). .tga via Pillow (32-bit RGBA / 24-bit RGB,
    uncompressed); everything else via OpenCV, which needs BGR(A) order."""
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


# --------------------------------------------------------------------------------------
# Filters
# --------------------------------------------------------------------------------------
def _pad(img, p, tile):
    return cv2.copyMakeBorder(img, p, p, p, p, cv2.BORDER_WRAP if tile else cv2.BORDER_REFLECT_101)


def _conv(img_padded, k):
    """filter2D in chunks of <=4 channels."""
    if img_padded.ndim == 2:
        return cv2.filter2D(img_padded, cv2.CV_32F, k, borderType=cv2.BORDER_REFLECT_101)
    out = [cv2.filter2D(np.ascontiguousarray(img_padded[..., i:i + 4]), cv2.CV_32F, k,
                        borderType=cv2.BORDER_REFLECT_101)
           for i in range(0, img_padded.shape[2], 4)]
    out = [o[..., None] if o.ndim == 2 else o for o in out]
    return np.concatenate(out, axis=2)


_KCACHE = {}


def kuwahara_kernels(r, classic=False):
    """Sector kernels, each cropped to its non-zero bounding box: list of (kernel, anchor)."""
    key = (r, classic)
    if key in _KCACHE:
        return _KCACHE[key]
    y, x = np.mgrid[-r:r + 1, -r:r + 1].astype(np.float32)
    full = []
    if classic:  # 4 overlapping square quadrants (box)
        for sx, sy in ((-1, -1), (1, -1), (-1, 1), (1, 1)):
            k = ((x * sx >= 0) & (y * sy >= 0)).astype(np.float32)
            full.append(k / k.sum())
    else:  # generalized Kuwahara: 8 smooth, overlapping sectors x Gaussian (Papari et al.)
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
    """One Kuwahara pass on img (HxWxC). The same per-pixel sector weights are applied
    to every array in `guides` so all maps share identical paint edges.
    Planar implementation: every channel is a contiguous 2D plane, convolved with cropped
    sector kernels via cv2.filter2D and accumulated with cv2.accumulateProduct (fast, threaded)."""
    ks = kuwahara_kernels(r, classic)
    # Pad one past the kernel. filter2D still consults its own border mode on the
    # last pad pixel, which would break the wrap. The extra pixel keeps every tap
    # inside the wrapped pad, matching the GPU's explicit wrap.
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
    # guides: weighted sector means with the albedo's weights
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
    """Same-size median. ksize 3 and 5 stay on OpenCV's float path. Larger kernels are
    true float medians in row chunks: OpenCV's large-kernel medianBlur is 8-bit only,
    and routing a float plane through that quantize/dequantize step banded the map."""
    channel = np.ascontiguousarray(channel, dtype=np.float32)
    if k <= 5:
        return cv2.medianBlur(channel, k)
    pad = k // 2
    # The caller already padded for the tile/reflect border. This pad is only so the
    # window view's interior matches `channel`. edge here is unused: we crop it off.
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
    else:  # bilateral
        out = cv2.bilateralFilter(ip.astype(np.float32), k, 0.12, max(r, 1))
    return out[p:-p, p:-p].astype(np.float32)


def gblur(img, sigma, tile):
    if sigma <= 0.05:
        return img
    p = int(np.ceil(sigma * 3)) + 1
    ip = _pad(img, p, tile)
    return cv2.GaussianBlur(ip, (0, 0), sigma)[p:-p, p:-p]


# --------------------------------------------------------------------------------------
# Normals / curvature
# --------------------------------------------------------------------------------------
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


def _ddy(a, tile):  # derivative along image rows (down)
    if tile:
        return (np.roll(a, -1, 0) - np.roll(a, 1, 0)) * 0.5
    return cv2.Sobel(a, cv2.CV_32F, 0, 1, ksize=1, borderType=cv2.BORDER_REPLICATE) * 0.5


def normal_from_height(h, strength, tile):
    """DirectX convention: nx = -dh/dx, ny = -dh/drow."""
    hx, hy = _ddx(h, tile), _ddy(h, tile)
    n = np.stack([-hx * strength, -hy * strength, np.ones_like(h)], axis=2)
    return normalize_normal(n)


def _corr(a, b):
    a = a - a.mean()
    b = b - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else 0.0


def detect_normal_convention(n, h, tile):
    """Compare normal Y with the height gradient. DX: ny ~ -dh/drow, GL: ny ~ +dh/drow."""
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
    """Divergence of the (DirectX) normal XY = -Laplacian(height); convex > 0."""
    return _ddx(n_dx[..., 0], tile) + _ddy(n_dx[..., 1], tile)


def normalise_signed(c):
    c = c - np.median(c)
    s = np.percentile(np.abs(c), 98) + 1e-6
    return np.clip(c / s, -1.0, 1.0).astype(np.float32)


# --------------------------------------------------------------------------------------
# Colour
# --------------------------------------------------------------------------------------
def luma(rgb):
    return rgb[..., 0] * 0.2126 + rgb[..., 1] * 0.7152 + rgb[..., 2] * 0.0722


def srgb_to_linear(c):
    c = np.clip(c, 0, 1)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4).astype(np.float32)


def linear_to_srgb(c):
    c = np.clip(c, 0, 1)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * np.power(c, 1 / 2.4) - 0.055).astype(np.float32)


def smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0, 1)
    return t * t * (3 - 2 * t)


# Björn Ottosson, Oklab. sRGB linear -> LMS, cube root, then Lab. Round-trips white to L=1.
_OK_M1 = np.array([
    [0.4122214708, 0.5363325363, 0.0514459929],
    [0.2119034982, 0.6806995451, 0.1073969566],
    [0.0883024619, 0.2817188376, 0.6299787005],
], np.float64)
_OK_M2 = np.array([
    [0.2104542553, 0.7936177850, -0.0040720468],
    [1.9779984951, -2.4285922050, 0.4505937099],
    [0.0259040371, 0.7827717662, -0.8086757660],
], np.float64)
_OK_M2_INV = np.linalg.inv(_OK_M2)
_OK_M1_INV = np.linalg.inv(_OK_M1)


def srgb_to_oklab(rgb):
    """rgb is sRGB 0..1, any shape (..., 3). Returns Oklab float32."""
    lin = srgb_to_linear(np.asarray(rgb, np.float32)).astype(np.float64)
    lms = np.matmul(lin, _OK_M1.T)
    np.maximum(lms, 0, out=lms)
    lms = np.cbrt(lms)
    return np.matmul(lms, _OK_M2.T).astype(np.float32)


def oklab_to_srgb(lab):
    lab64 = np.asarray(lab, np.float64)
    lms = np.matmul(lab64, _OK_M2_INV.T)
    lin = np.matmul(lms * lms * lms, _OK_M1_INV.T)
    return linear_to_srgb(lin.astype(np.float32))


def _as_rgb_list(value, fallback):
    try:
        rgb = [float(value[0]), float(value[1]), float(value[2])]
    except (TypeError, ValueError, IndexError):
        rgb = list(fallback)
    return np.clip(np.array(rgb, np.float32), 0, 1)


def colour_grade(rgb, st):
    """Grade in Oklab so lightness, chroma, and hue come apart.

    Posterize steps L and chroma. Hue stays continuous unless a palette is set,
    in which case pixels soften toward those swatches. palette_strength 0 is the
    grade alone, which is how DEFAULTS opens.
    """
    lab = srgb_to_oklab(np.clip(rgb, 0, 1))
    vc = float(st["value_compression"])
    if vc:
        L = lab[..., 0]
        Lm = float(L.mean())
        lab[..., 0] = Lm + (L - Lm) * (1.0 - vc)
    sat = float(st["saturation"])
    if sat != 1.0:
        lab[..., 1:] *= sat
    ts = float(st["tint_strength"])
    if ts:
        L = lab[..., 0]
        Lm, Ls = float(L.mean()), float(L.std()) + 1e-4
        t = smoothstep(-1.0, 1.0, (L - Lm) / (1.5 * Ls))
        warm = srgb_to_oklab(_as_rgb_list(st.get("tint_warm"), DEFAULTS["tint_warm"]))
        cool = srgb_to_oklab(_as_rgb_list(st.get("tint_cool"), DEFAULTS["tint_cool"]))
        target_ab = t[..., None] * warm[1:] + (1.0 - t)[..., None] * cool[1:]
        lab[..., 1:] += (target_ab - lab[..., 1:]) * ts
    lv = int(st["posterize"])
    if lv and lv > 1:
        # Chroma is quantized against a fixed Oklab range so the steps don't
        # depend on whichever pixel happens to be the most saturated.
        chroma_max = 0.4
        L = np.clip(lab[..., 0], 0, 1)
        C = np.hypot(lab[..., 1], lab[..., 2])
        hue_a = lab[..., 1] / (C + 1e-8)
        hue_b = lab[..., 2] / (C + 1e-8)
        Lq = np.floor(L * lv) / float(lv)
        Cq = np.floor(np.clip(C / chroma_max, 0, 1) * lv) / float(lv) * chroma_max
        lab[..., 0] = Lq
        lab[..., 1] = hue_a * Cq
        lab[..., 2] = hue_b * Cq
    strength = float(st.get("palette_strength", 0) or 0)
    palette = st.get("palette") or []
    if strength > 0 and palette:
        cols = []
        for swatch in list(palette)[:8]:
            try:
                cols.append(srgb_to_oklab(_as_rgb_list(swatch, (0.5, 0.5, 0.5))))
            except (TypeError, ValueError):
                continue
        if cols:
            cols = np.stack(cols).astype(np.float32)  # N,3
            dist = np.linalg.norm(lab[..., None, :] - cols, axis=-1)
            hardness = max(float(st.get("palette_hardness", 8) or 8), 0.01)
            w = np.exp(-dist * hardness).astype(np.float32)
            w /= np.maximum(w.sum(axis=-1, keepdims=True), 1e-8)
            snapped = np.einsum("hwn,nc->hwc", w, cols)
            lab = lab * (1.0 - strength) + snapped * strength
    return np.clip(oklab_to_srgb(lab), 0, 1).astype(np.float32)


# --------------------------------------------------------------------------------------
# Main processing
# --------------------------------------------------------------------------------------
def scaled(v, size):
    return float(v) * size / REF_SIZE


class Cancelled(Exception):
    """Raised inside the paint stage when a newer preview request makes this one stale."""


LOAD_KEYS = ("size", "normal_in", "flip_green", "tile")
PAINT_KEYS = ("filter", "radius", "passes", "sharpness")


def stage_keys(s, st):
    """Cache keys for the load and paint stages (finish stage = everything else)."""
    lk = (s["folder"],) + tuple(st[k] for k in LOAD_KEYS)
    pk = lk + tuple(st[k] for k in PAINT_KEYS)
    return lk, pk


def load_stage(s, st):
    """Load all maps for a set, resize to st['size'] (longest side), resolve defaults/normal convention."""
    st = dict(DEFAULTS, **st)
    tile = bool(st["tile"])
    maps = s["maps"]
    report = {"defaulted": [], "notes": [], "used": {}}
    alb0 = load_rgb(maps["albedo"])
    ih, iw = alb0.shape[:2]
    size = int(st["size"])
    sc = size / float(max(iw, ih))
    W, H = max(int(round(iw * sc)), 1), max(int(round(ih * sc)), 1)
    alb_src = np.clip(resize(alb0, W, H), 0, 1)
    report["used"]["albedo"] = maps["albedo"]
    del alb0

    def g(typ):
        report["used"][typ] = maps[typ]
        return np.clip(resize(load_gray(maps[typ]), W, H), 0, 1)

    orm = None
    if "orm" in maps:
        orm = np.clip(resize(load_rgb(maps["orm"]), W, H), 0, 1)
        report["used"]["orm"] = maps["orm"]
    if "roughness" in maps:
        rough = g("roughness")
    elif "gloss" in maps:
        rough = 1.0 - g("gloss")
        report["notes"].append("roughness = 1 - gloss")
    elif orm is not None:
        rough = orm[..., 1].copy()
        report["notes"].append("roughness from ORM/ARM green")
    else:
        rough = np.full((H, W), 0.6, np.float32)
        report["defaulted"].append("roughness (0.6)")
    if "metallic" in maps:
        metal = g("metallic")
    elif orm is not None:
        metal = orm[..., 2].copy()
        report["notes"].append("metallic from ORM/ARM blue")
    else:
        metal = np.zeros((H, W), np.float32)
        report["defaulted"].append("metallic (0)")
    if "ao" in maps:
        ao = g("ao")
    elif orm is not None:
        ao = orm[..., 0].copy()
        report["notes"].append("AO from ORM/ARM red")
    else:
        ao = np.ones((H, W), np.float32)
        report["defaulted"].append("ao (1, white)")
    height_real = True
    if "height" in maps:
        height = g("height")
    elif "bump" in maps:
        height = g("bump")
        report["notes"].append("height taken from Bump map (no Displacement found)")
    else:
        height = np.full((H, W), 0.5, np.float32)
        height_real = False
        report["defaulted"].append("height (0.5 flat)")
    cavity = g("cavity") if "cavity" in maps else None
    opacity = g("opacity") if "opacity" in maps else None
    for t in ("specular", "translucency"):
        if t in maps:
            report["notes"].append(f"{t} map ignored: {os.path.basename(maps[t])}")

    conv = None
    if "normal_dx" in maps:
        nkey, conv = "normal_dx", "dx"
        if "normal_gl" in maps:
            report["notes"].append("both NormalDX and NormalGL found; using DirectX")
    elif "normal" in maps:
        nkey = "normal"
    elif "normal_gl" in maps:
        nkey, conv = "normal_gl", "gl"
    else:
        nkey = None
    if nkey:
        report["used"]["normal"] = maps[nkey]
        n_src = decode_normal(np.clip(resize(load_rgb(maps[nkey]), W, H), 0, 1))
        if st["normal_in"] in ("dx", "gl"):
            conv = st["normal_in"]
            report["notes"].append(f"normal convention forced: {conv}")
        elif conv is None:
            if height_real:
                cx, cy = detect_normal_convention(n_src, height, tile)
                if abs(cy) >= 0.05:
                    conv = "dx" if cy > 0 else "gl"
                    report["notes"].append(
                        f"normal convention auto-detected from height gradient: {conv.upper()} "
                        f"(corrY={cy:+.2f}, corrX={cx:+.2f})")
                else:
                    conv = "gl"
                    report["notes"].append(f"normal convention inconclusive (corrY={cy:+.2f}); assumed GL")
            else:
                conv = "gl"
                report["notes"].append("unsuffixed Normal with no height to check against; assumed OpenGL")
        else:
            report["notes"].append(f"normal convention from filename: {conv.upper()}")
        if conv == "gl":
            n_src[..., 1] *= -1.0  # work in DirectX internally
        if st["flip_green"]:
            n_src[..., 1] *= -1.0
            report["notes"].append("--flip-green applied to source normal")
    elif height_real:
        n_src = normal_from_height(height, 32.0 * max(W, H) / REF_SIZE, tile)
        report["notes"].append("no normal map: derived from height")
        report["defaulted"].append("normal (derived from height)")
    else:
        n_src = np.zeros((H, W, 3), np.float32)
        n_src[..., 2] = 1.0
        report["defaulted"].append("normal (flat)")
    report["normal_convention_in"] = conv
    return {"W": W, "H": H, "alb": alb_src, "rough": rough, "metal": metal, "ao": ao,
            "height": height, "height_real": height_real, "cavity": cavity,
            "opacity": opacity, "normal": n_src, "nkey": nkey, "conv": conv, "report": report,
            "name": s["name"], "folder": s["folder"]}


def paint_stage(src, st, cancel=None):
    """The expensive edge-preserving paint filter. Albedo drives the sector weights; the same
    weights are applied to normal/height/roughness/metal/AO/cavity."""
    st = dict(DEFAULTS, **st)
    tile = bool(st["tile"])
    W, H = src["W"], src["H"]
    r_px = max(1, int(round(scaled(st["radius"], max(W, H)))))
    passes = max(1, int(st["passes"]))
    cav = src["cavity"]
    guides = [src["normal"], src["height"], src["rough"], src["metal"], src["ao"]] + \
             ([cav] if cav is not None else [])
    alb = src["alb"].copy()
    method = st["filter"]
    for _ in range(passes):
        if cancel and cancel():
            raise Cancelled()
        if method in ("kuwahara", "kuwahara_classic"):
            alb, guides = kuwahara_pass(alb, guides, r_px, float(st["sharpness"]), tile,
                                        classic=(method == "kuwahara_classic"), cancel=cancel)
        else:
            alb = simple_filter(alb, method, r_px, tile)
            guides = [simple_filter(gi, method, r_px, tile) for gi in guides]
    return {"alb": alb, "normal": guides[0], "height": guides[1], "rough": guides[2],
            "metal": guides[3], "ao": guides[4], "cavity": guides[5] if cav is not None else None,
            "r_px": r_px}


def finish_stage(src, painted, st):
    """Cheap stages: per-map mix, softening, colour grade, AO + edge bake, seam fade, mask.

    The mix and the mask are lerps on maps the paint cache already produced, so dragging
    them does not re-run Kuwahara.
    """
    st = dict(DEFAULTS, **st)
    tile = bool(st["tile"])
    W, H = src["W"], src["H"]
    r_px = painted["r_px"]
    alb_p = painted["alb"]
    dr = float(st.get("detail_restore", 0) or 0)
    if dr:
        # High-pass of the source colour, so a large brush can stay painterly without going flat.
        hp = src["alb"] - gblur(src["alb"], max(r_px * 0.35, 0.6), tile)
        alb_p = np.clip(alb_p + hp * dr, 0, 1)
    n_p = _mix_map(painted["normal"], src["normal"], st.get("mix_normal", 1))
    height_p = _mix_map(painted["height"], src["height"], st.get("mix_height", 1))
    rough_p = _mix_map(painted["rough"], src["rough"], st.get("mix_rough", 1))
    metal_p = _mix_map(painted["metal"], src["metal"], st.get("mix_metal", 1))
    ao_p = _mix_map(painted["ao"], src["ao"], st.get("mix_ao", 1))
    height_f = gblur(height_p, float(st["height_soften"]) * r_px * 0.5, tile)
    n_f = n_p
    if st["normal_mode"] == "height" and src["height_real"]:
        n_f = normal_from_height(height_f, 32.0 * max(W, H) / REF_SIZE, tile)
    n_f = np.array(gblur(n_f, float(st["normal_soften"]) * r_px * 0.5, tile), np.float32)
    n_f[..., :2] *= float(st["normal_strength"])
    n_f = normalize_normal(n_f)

    rough_f = rough_p
    rm = float(rough_f.mean())
    rough_o = np.clip(rm + (rough_f - rm) * (1 - float(st["roughness_flatten"]))
                      + float(st["roughness_bias"]), 0, 1)
    mb = float(st["metal_binarize"])
    metal_f = metal_p
    metal_o = np.clip(metal_f * (1 - mb) + smoothstep(0.35, 0.65, metal_f) * mb, 0, 1)
    ao_o = np.clip(ao_p, 0, 1)

    cav_f = painted["cavity"]
    cs = st["curvature_source"]
    if cs == "auto":
        cs = "cavity" if cav_f is not None else ("normal" if src["nkey"] or src["height_real"] else "none")
    if cs == "cavity" and cav_f is None:
        cs = "normal"
    if cs == "cavity":
        curv = normalise_signed(cav_f)
    elif cs == "height" and src["height_real"]:
        curv = normalise_signed(-cv2.Laplacian(gblur(height_f, 1.0, tile), cv2.CV_32F))
    elif cs in ("normal", "height"):
        cs = "normal"
        curv = normalise_signed(curvature_from_normal(n_f, tile))
    else:
        curv = np.zeros((H, W), np.float32)
    curv = gblur(curv, scaled(st["edge_blur"], max(W, H)), tile)

    alb = colour_grade(np.clip(alb_p, 0, 1), st)
    aos = float(st["ao_strength"])
    if aos:
        alb = linear_to_srgb(srgb_to_linear(alb) * (1 - aos + aos * ao_o)[..., None])
    es = float(st["edge_strength"])
    if es:
        c = curv[..., None] * es
        alb = np.where(c > 0, alb + (1 - alb) * c * 0.6, alb * (1 + c * 0.75))
    alb = np.clip(alb, 0, 1).astype(np.float32)
    fade = float(st.get("seam_fade", 0) or 0)
    alb = seam_fade(alb, fade, tile)
    n_f = seam_fade(n_f, fade, tile)
    height_f = seam_fade(np.clip(height_f, 0, 1), fade, tile)
    rough_o = seam_fade(rough_o, fade, tile)
    metal_o = seam_fade(metal_o, fade, tile)
    ao_o = seam_fade(ao_o, fade, tile)
    n_f = normalize_normal(np.array(n_f, np.float32))
    mask = st.get("_mask")
    if mask is not None:
        alb = apply_stylize_mask(alb, src["alb"], mask, (H, W))
        n_f = normalize_normal(apply_stylize_mask(n_f, src["normal"], mask, (H, W)))
        height_f = apply_stylize_mask(height_f, src["height"], mask, (H, W))
        rough_o = apply_stylize_mask(rough_o, src["rough"], mask, (H, W))
        metal_o = apply_stylize_mask(metal_o, src["metal"], mask, (H, W))
        ao_o = apply_stylize_mask(ao_o, src["ao"], mask, (H, W))
    return {"alb": np.clip(alb, 0, 1).astype(np.float32), "normal": n_f,
            "height": np.clip(height_f, 0, 1).astype(np.float32),
            "rough": np.clip(rough_o, 0, 1).astype(np.float32),
            "metal": np.clip(metal_o, 0, 1).astype(np.float32),
            "ao": np.clip(ao_o, 0, 1).astype(np.float32),
            "curvature_source": cs}


def _smith_g1(nd, a2):
    nd = np.maximum(nd, 0)
    return (2.0 * nd) / np.maximum(nd + np.sqrt(a2 + (1.0 - a2) * nd * nd), 1e-6)


def _env_radiance(direction):
    """A fixed studio gradient. z is out of the surface (toward the camera). Not a measured HDRI."""
    z = np.clip(direction[..., 2], -1, 1)
    sky = np.array([0.55, 0.62, 0.75], np.float32)
    horizon = np.array([0.90, 0.82, 0.70], np.float32)
    ground = np.array([0.16, 0.14, 0.12], np.float32)
    up = np.clip(z, 0, 1)[..., None]
    down = np.clip(-z, 0, 1)[..., None]
    hor = (1.0 - np.abs(z))[..., None]
    col = sky * up + ground * down + horizon * hor
    # Keep the sum of weights from blowing the white point on the horizon.
    return col / np.maximum(up + down + hor, 1e-4)


def shade_lit(alb_srgb, n_dx, rough, metal, az_deg=135.0, el_deg=45.0, ao=None,
              exposure=0.0, shade_ao=False):
    """Preview shading in the tangent frame. One directional light plus a small environment.

    Smith GGX (D, G, Schlick F). The environment is what metal and rough read as when the
    key light is edge-on. AO is already baked into the diffuse; shade_ao multiplies it
    again and is off unless the checkbox is on. exposure is stops (2^exposure).
    """
    az, el = np.radians(float(az_deg)), np.radians(float(el_deg))
    L = np.array([np.cos(el) * np.cos(az), -np.cos(el) * np.sin(az), np.sin(el)], np.float32)
    V = np.array([0, 0, 1], np.float32)
    H = L + V
    H = H / np.linalg.norm(H)
    n = n_dx.astype(np.float32)
    ndl = np.clip(n @ L, 0, 1)
    ndv = np.clip(n[..., 2], 0, 1)
    ndh = np.clip(n @ H, 0, 1)
    vdh = float(np.clip(H[2], 0, 1))
    a = np.maximum(rough, 0.045) ** 2
    a2 = a * a
    D = a2 / (np.pi * ((ndh * ndh) * (a2 - 1.0) + 1.0) ** 2 + 1e-6)
    G = _smith_g1(ndl, a2) * _smith_g1(ndv, a2)
    base = srgb_to_linear(alb_srgb)
    m = metal[..., None]
    f0 = 0.04 * (1.0 - m) + base * m
    fres = (1.0 - vdh) ** 5
    F = f0 + (1.0 - f0) * fres
    spec = (D * G)[..., None] * F / np.maximum((4.0 * ndl * ndv)[..., None], 1e-4) * ndl[..., None]
    kd = (1.0 - F) * (1.0 - m)
    diff = kd * base * (ndl / np.pi)[..., None]
    irr = _env_radiance(n)
    diff_amb = base * (1.0 - m) * irr * 0.45
    R = n * (2.0 * ndv)[..., None] - V
    # Rougher surfaces see a flatter environment, so the reflection fades toward the diffuse ambient.
    spec_amb = _env_radiance(R) * (0.04 * (1.0 - m) + m) * (1.0 - np.clip(rough, 0, 1))[..., None]
    lin = diff + spec + diff_amb * 0.35 + spec_amb * 0.55
    if shade_ao and ao is not None:
        lin = lin * np.clip(ao, 0, 1)[..., None]
    lin = lin * np.float32(2.0 ** float(exposure))
    return linear_to_srgb(np.clip(lin, 0, 1))


def seam_metrics(alb, normal):
    """Mean absolute difference across the wrap. Measured, no pass/fail line."""
    def sides(img):
        a = img[:, 0].astype(np.float32) - img[:, -1].astype(np.float32)
        b = img[0, :].astype(np.float32) - img[-1, :].astype(np.float32)
        return float(np.mean(np.abs(a))), float(np.mean(np.abs(b)))
    d_lr, d_tb = sides(alb)
    n_lr, n_tb = sides(normal)
    return {"diffuse_lr": d_lr, "diffuse_tb": d_tb, "normal_lr": n_lr, "normal_tb": n_tb}


def seam_fade(img, strength, tile):
    """Pull a band inside each edge toward the opposite edge so a near-tileable set joins.

    One-sided: the left band moves toward the right edge (and the top band toward the bottom).
    Doing both sides would average the seam twice. strength 0 leaves the image alone.
    """
    if not tile or strength <= 0 or img is None:
        return img
    out = np.array(img, np.float32, copy=True)
    h, w = out.shape[:2]
    span = min(h, w)
    bw = int(round(max(2.0, span * (0.01 + 0.04 * float(strength)))))
    bw = max(1, min(bw, w // 4, h // 4))
    ramp = np.linspace(float(strength), 0.0, bw, dtype=np.float32)  #  strongest on the seam
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


def _mix_map(painted, source, t):
    t = float(np.clip(t, 0.0, 1.0))
    if t >= 1.0:
        return painted
    if t <= 0.0:
        return source
    return source * (1.0 - t) + painted * t


def apply_stylize_mask(stylized, source, mask, shape_hw):
    """mask 1 keeps the stylized pixel, 0 returns the source. Resized to this image."""
    if mask is None:
        return stylized
    m = np.asarray(mask, np.float32)
    if m.ndim == 3:
        m = m[..., 0]
    H, W = shape_hw
    if m.shape != (H, W):
        m = resize(m, W, H)
    m = np.clip(m, 0, 1)
    if stylized.ndim == 2:
        return source * (1.0 - m) + stylized * m
    return source * (1.0 - m[..., None]) + stylized * m[..., None]


def paint_dispatch(src, st, cancel=None):
    """GPU paint for the Kuwahara filters when ModernGL is up. CPU otherwise.

    Anisotropic Kuwahara lives in the GPU kernel. With no GPU it falls through to the
    generalized Kuwahara and says so on the report.
    """
    st = dict(DEFAULTS, **st)
    method = st["filter"]
    if method in ("kuwahara", "kuwahara_classic", "anisotropic"):
        try:
            import qs_gpu
            if qs_gpu.available():
                painted = qs_gpu.paint(src, st, cancel)
                notes = src.setdefault("report", {}).setdefault("notes", [])
                note = "paint ran on the GPU (ModernGL compute)"
                if note not in notes:
                    notes.append(note)
                return painted
        except Exception as exc:
            log_line(f"GPU paint failed, using CPU: {exc}")
    if method == "anisotropic":
        st = dict(st, filter="kuwahara")
        note = "anisotropic Kuwahara needs the GPU path; this render used kuwahara"
        notes = src.setdefault("report", {}).setdefault("notes", [])
        if note not in notes:
            notes.append(note)
    return paint_stage(src, st, cancel)


def json_settings(st):
    """Preset/JSON view of settings. Drops the in-memory mask array."""
    out = {}
    for key in DEFAULTS:
        if key not in st:
            continue
        value = st[key]
        if isinstance(value, np.ndarray):
            continue
        if isinstance(value, np.generic):
            value = value.item()
        out[key] = value
    return out


def process_set(s, st, write=True, out_dir=None, log=print):
    t0 = time.time()
    st = dict(DEFAULTS, **st)
    src = load_stage(s, st)
    painted = paint_dispatch(src, st)
    fin = finish_stage(src, painted, st)
    report = dict(src["report"])
    report["curvature_source"] = cs = fin["curvature_source"]
    W, H, name, conv, r_px = src["W"], src["H"], s["name"], src["conv"], painted["r_px"]
    alb = fin["alb"]
    n_out = fin["normal"].copy()
    if st["normal_out"] == "gl":
        n_out[..., 1] *= -1
    D = to_u8(alb)  # plain RGB diffuse
    # MROD alpha = displacement. Clamped to >= 1/255 so no pixel is ever fully transparent:
    # UE's PNG importer infill only rewrites pixels that are exactly (255,255,255,0).
    MROD = np.dstack([to_u8(fin["metal"]), to_u8(fin["rough"]), to_u8(fin["ao"]),
                      (np.maximum(to_u8(fin["height"]), 1) if src["height_real"]
                       else np.full(fin["height"].shape, 128, np.uint8))])  # no height map -> exact mid-grey
    N = encode_normal(n_out)
    result = {"name": name, "albedo_before": src["alb"], "albedo_after": alb, "D": D, "MROD": MROD,
              "N": N, "normal_before": src["normal"], "report": report, "size": (W, H), "radius_px": r_px}
    opacity = src["opacity"]

    if write:
        out_dir = resolve_out_dir(s["folder"], out_dir)
        os.makedirs(out_dir, exist_ok=True)
        removed = remove_legacy_outputs(out_dir, name, s["folder"], s["maps"].values())
        if removed:
            report["notes"].append("removed old-layout output(s) from an earlier version: " + ", ".join(removed))
        files, checked = {}, {}
        ext = ".tga" if str(st.get("format", "png")).lower() == "tga" else ".png"
        assert D.ndim == 3 and D.shape[2] == 3, D.shape
        assert MROD.ndim == 3 and MROD.shape[2] == 4, MROD.shape
        for key, arr, ch in (("D", D, 3), ("MROD", MROD, 4), ("N", N, 3)):
            files[key] = os.path.join(out_dir, f"{name}_{key}{ext}")
            imwrite(files[key], arr)
            checked[key] = verify_written(files[key], ch, (W, H))
        if opacity is not None:
            files["Opacity"] = os.path.join(out_dir, f"{name}_Opacity{ext}")
            imwrite(files["Opacity"], to_u8(opacity))
            checked["Opacity"] = verify_written(files["Opacity"], 1, (W, H))
        if st["preview"]:
            files["preview"] = os.path.join(out_dir, f"{name}_preview.png")
            imwrite(files["preview"], make_preview(result))
            checked["preview"] = verify_written(files["preview"], 3, (W, H))
        result["checked"] = checked
        log_line(f"wrote {name} v{TOOL_VERSION} size={st['size']} -> {out_dir}: "
                 + "; ".join(f"_{k}{'.png' if k == 'preview' else ext} {v}" for k, v in checked.items())
                 + ("" if src["height_real"] else "  (no height map: MROD alpha = 128)"))
        files["json"] = os.path.join(out_dir, f"{name}_stylize.json")
        meta = {"tool": "QuixelStylizer", "version": TOOL_VERSION,
                "created": time.strftime("%Y-%m-%d %H:%M:%S"), "set_name": name,
                "source_folder": s["folder"], "code": MODULE_PATH, "output_size": [W, H], "radius_px": r_px,
                "files_checked": checked,
                "settings": json_settings(st), "inputs": report["used"], "defaulted": report["defaulted"],
                "notes": report["notes"], "normal_convention_in": conv,
                "normal_convention_out": st["normal_out"], "curvature_source": cs,
                "outputs": files,
                "packing": {"_D": "RGB=diffuse (sRGB), no alpha",
                            "_MROD": "R=metallic G=roughness B=AO A=displacement/height (linear, min 1/255)",
                            "_N": f"tangent normal ({st['normal_out'].upper()})"}}
        with open(files["json"], "w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=2)
        result["files"] = files
    result["seconds"] = time.time() - t0
    return result


def _label(img, text):
    img = img.copy()
    fs = max(0.3, min(0.7, img.shape[1] / 700.0))
    th = max(1, int(round(fs * 2.5)))
    (tw, tht), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fs, th)
    cv2.rectangle(img, (0, 0), (min(img.shape[1], tw + 12), tht + 14), (0, 0, 0), -1)
    cv2.putText(img, text, (6, tht + 7), cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), th, cv2.LINE_AA)
    return img


def make_preview(res):
    """2x2 sheet (albedo before/after, normal before/after) with exactly the output dimensions."""
    a0 = to_u8(res["albedo_before"])
    a1 = to_u8(res["albedo_after"])
    n0 = encode_normal(res["normal_before"])
    n1 = res["N"]
    h, w = a0.shape[:2]
    lw, rw, th, bh = w // 2, w - w // 2, h // 2, h - h // 2
    rs = lambda x, ww, hh: cv2.resize(x, (ww, hh), interpolation=cv2.INTER_AREA)
    top = np.hstack([_label(rs(a0, lw, th), "Albedo before"), _label(rs(a1, rw, th), "Stylized _D")])
    bot = np.hstack([_label(rs(n0, lw, bh), "Normal before"), _label(rs(n1, rw, bh), "Stylized _N")])
    return np.ascontiguousarray(np.vstack([top, bot]))


def preview_pair(res):
    """before|after albedo side by side (uint8 RGB) for the GUI."""
    return np.hstack([to_u8(res["albedo_before"]), to_u8(res["albedo_after"])])


# --------------------------------------------------------------------------------------
# Presets / CLI
# --------------------------------------------------------------------------------------
def load_preset(path):
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    if "settings" in data and isinstance(data["settings"], dict):  # accept a _stylize.json too
        data = data["settings"]
    return {k: v for k, v in data.items() if k in DEFAULTS}


def save_preset(path, st):
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(json_settings(st), fh, indent=2)


def mask_path_for(preset_path):
    base, _ext = os.path.splitext(preset_path)
    return base + "_mask.png"


def save_mask(path, mask):
    imwrite(path, to_u8(np.clip(mask, 0, 1)))


def load_mask(path):
    if not path or not os.path.isfile(path):
        return None
    img = load_gray(path)
    return np.clip(img, 0, 1).astype(np.float32)


def build_parser():
    p = argparse.ArgumentParser(description="Convert Megascans/Fab surfaces to a hand-painted look "
                                            "with UE5 packed maps (_D, _MROD, _N).")
    p.add_argument("inputs", nargs="*", help="set folder(s) or parent folder(s)")
    p.add_argument("--recursive", action="store_true", help="search all subfolders for sets")
    p.add_argument("--out", help="output folder (default: <set folder>\\Stylized)")
    p.add_argument("--preset", help="load settings from a preset JSON (or a _stylize.json)")
    p.add_argument("--save-preset", help="save the effective settings to this JSON and continue")
    p.add_argument("--list", action="store_true", help="only list detected sets/maps")
    p.add_argument("--gui", action="store_true", help="open the GUI")
    a = p.add_argument
    a("--size", type=int, choices=SIZES)
    a("--filter", choices=FILTERS)
    a("--radius", type=float, help="paint radius in px at 1024 (default 6)")
    a("--passes", type=int)
    a("--sharpness", type=float)
    a("--saturation", type=float)
    a("--value-compression", dest="value_compression", type=float)
    a("--posterize", type=int, help="luminance levels, 0=off")
    a("--tint-strength", dest="tint_strength", type=float)
    a("--ao-strength", dest="ao_strength", type=float)
    a("--edge-strength", dest="edge_strength", type=float)
    a("--edge-blur", dest="edge_blur", type=float)
    a("--curvature-source", dest="curvature_source", choices=["auto", "cavity", "normal", "height"])
    a("--roughness-flatten", dest="roughness_flatten", type=float)
    a("--roughness-bias", dest="roughness_bias", type=float)
    a("--metal-binarize", dest="metal_binarize", type=float)
    a("--normal-soften", dest="normal_soften", type=float)
    a("--normal-strength", dest="normal_strength", type=float)
    a("--normal-mode", dest="normal_mode", choices=["filter", "height"])
    a("--normal-in", dest="normal_in", choices=["auto", "dx", "gl"])
    a("--flip-green", dest="flip_green", action=argparse.BooleanOptionalAction, default=None)
    a("--normal-out", dest="normal_out", choices=["dx", "gl"])
    a("--height-soften", dest="height_soften", type=float)
    a("--format", dest="format", choices=["png", "tga"], help="texture file format (default png)")
    a("--tile", dest="tile", action=argparse.BooleanOptionalAction, default=None)
    a("--preview", dest="preview", action=argparse.BooleanOptionalAction, default=None)
    return p


def settings_from_args(args):
    st = dict(DEFAULTS)
    if args.preset:
        st.update(load_preset(args.preset))
    for k in DEFAULTS:
        v = getattr(args, k, None)
        if v is not None:
            st[k] = v
    return st


def run_cli(args):
    st = settings_from_args(args)
    print(version_banner())
    print(f"Output size {st['size']} px, format {st['format']}")
    log_line(f"CLI start v{TOOL_VERSION} code={MODULE_PATH}: inputs={args.inputs} out={args.out!r} "
             f"size={st['size']} format={st['format']}")
    warn = old_window_warning()
    if warn:
        print("\n!! " + warn.replace("\n", "\n!! ") + "\n")
        log_line("WARNING " + warn)
    if args.save_preset:
        save_preset(args.save_preset, st)
        print(f"preset saved: {args.save_preset}")
    sets = find_sets(args.inputs, args.recursive)
    if not sets:
        msg = ("No material sets found in: " + ", ".join(args.inputs) +
               "\n(need an image ending in _Albedo/_BaseColor/_Diffuse/... in the folder or its subfolders)")
        print(msg)
        log_line("CLI: " + msg)
        return 1
    print(f"Found {len(sets)} set(s).")
    for s in sets:
        print(f"\n== {s['name']}  ({s['folder']})")
        for t, pth in sorted(s["maps"].items()):
            print(f"   {t:10s} {os.path.basename(pth)}")
        if s["ignored"]:
            print(f"   ignored: {', '.join(s['ignored'])}")
    if args.list:
        return 0
    fails, written = 0, []
    for s in sets:
        try:
            r = process_set(s, st, write=True, out_dir=args.out)
            rep = r["report"]
            print(f"\n[ok] {r['name']}  {r['size'][0]}x{r['size'][1]}  radius={r['radius_px']}px  "
                  f"{r['seconds']:.1f}s")
            for n in rep["notes"]:
                print(f"   note: {n}")
            if rep["defaulted"]:
                print(f"   defaulted: {', '.join(rep['defaulted'])}")
            print(f"   curvature from: {rep['curvature_source']}")
            for k, v in r["files"].items():
                print(f"   {k:8s} {v}" + (f"   [{r['checked'][k]}]" if k in r.get("checked", {}) else ""))
            written.append((r["name"], os.path.dirname(r["files"]["D"])))
            log_line(f"CLI ok: {r['name']} -> {os.path.dirname(r['files']['D'])}")
        except Exception as e:  # keep batch going
            fails += 1
            import traceback
            tb = traceback.format_exc()
            print(tb)
            print(f"[FAILED] {s['folder']}: {e}")
            log_line(f"CLI FAILED: {s['folder']}\n{tb}")
    print("\n" + "=" * 70)
    print(f"Done: {len(written)} converted, {fails} failed.")
    for name, d in written:
        print(f"  {name}  ->  {d}")
    if fails:
        print(f"Errors were logged to {LOG_PATH}")
    print("=" * 70)
    return 1 if fails else 0


def main(argv=None):
    # run as a script this module is "__main__"; make "import quixel_stylize" (GUI) reuse it, not load a 2nd copy
    if __name__ == "__main__":
        sys.modules.setdefault("quixel_stylize", sys.modules["__main__"])
    args = build_parser().parse_args(argv)
    try:
        if args.gui or not args.inputs:
            from stylizer_gui import run_gui
            return run_gui(settings_from_args(args), args.inputs)
        return run_cli(args)
    except Exception:
        import traceback
        tb = traceback.format_exc()
        log_line("FATAL\n" + tb)
        print(tb)
        print(f"Logged to {LOG_PATH}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
