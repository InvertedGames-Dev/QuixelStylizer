"""Self-imposed PBR 360 poly / texture ceilings, per asset class."""
from __future__ import annotations

import json
from pathlib import Path

from qs.version import TOOL_DIR

DEFAULTS = {
    "near": 0.85,
    "classes": {
        "hero": {"tris": 20000, "tex": 2048},
        "prop": {"tris": 3000, "tex": 1024},
        "foliage": {"tris": 800, "tex": 512},
        "modular": {"tris": 1500, "tex": 1024},
        "terrain": {"tris": 5000, "tex": 1024},
        "surface": {"tris": 2, "tex": 1024},
    },
}

# Display order on the library meter (cheap classes first).
CLASS_ORDER = ("surface", "foliage", "modular", "prop", "terrain", "hero")


def path() -> Path:
    return Path(TOOL_DIR) / "budgets.json"


def load() -> dict:
    p = path()
    if not p.is_file():
        save(DEFAULTS)
        return dict(DEFAULTS)
    data = json.loads(p.read_text(encoding="utf-8"))
    data.setdefault("classes", {})
    data.setdefault("near", 0.85)
    for k, v in DEFAULTS["classes"].items():
        data["classes"].setdefault(k, v)
    return data


def save(data: dict) -> None:
    path().write_text(json.dumps(data, indent=2), encoding="utf-8")


def class_table() -> dict:
    """Merged class ceilings (file overrides, defaults fill missing keys)."""
    data = load()
    classes = {k: dict(v) for k, v in DEFAULTS["classes"].items()}
    for name, cap in (data.get("classes") or {}).items():
        row = classes.setdefault(name, {})
        if isinstance(cap, dict):
            row.update(cap)
    return classes


def ceiling(asset_class: str) -> dict:
    return class_table().get(asset_class, DEFAULTS["classes"]["prop"])


def status(tris, tex, asset_class: str) -> str:
    """ok / near / over."""
    cap = ceiling(asset_class)
    near = float(load().get("near", 0.85))
    tcap, xcap = cap["tris"], cap["tex"]
    t = 0 if tris is None else int(tris)
    x = 0 if tex is None else int(tex)
    if t > tcap or x > xcap:
        return "over"
    if (tcap and t >= tcap * near) or (xcap and x >= xcap * near):
        return "near"
    return "ok"
