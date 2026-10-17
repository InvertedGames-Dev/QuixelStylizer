"""Filename-suffix map detection for Quixel / Fab surface sets."""
from __future__ import annotations

import os
import re

IMG_EXT_RANK = {
    ".png": 0, ".tif": 1, ".tiff": 1, ".tga": 2, ".jpg": 3, ".jpeg": 3,
    ".bmp": 4, ".exr": 5,
}

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


def _norm_stem(stem):
    return re.sub(r"[\s\-.]", "_", stem).lower()


def classify_file(fname):
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
