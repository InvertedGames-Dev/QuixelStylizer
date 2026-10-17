"""Typed recipe. Unknown keys from old JSON are ignored; missing keys take defaults."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from typing import Any

REF_SIZE = 1024
SIZES = [256, 512, 1024, 2048]
FILTERS = ["kuwahara", "kuwahara_classic", "anisotropic", "median", "bilateral"]


@dataclass
class Recipe:
    size: int = 1024
    filter: str = "kuwahara"
    radius: float = 0.0
    passes: int = 2
    sharpness: float = 8.0
    saturation: float = 1.0
    value_compression: float = 0.0
    posterize: int = 0
    ao_strength: float = 0.0
    edge_strength: float = 0.0
    edge_blur: float = 2.0
    curvature_source: str = "auto"
    roughness_flatten: float = 0.0
    roughness_bias: float = 0.0
    metal_binarize: float = 0.0
    normal_soften: float = 0.0
    normal_strength: float = 1.0
    normal_mode: str = "filter"
    normal_in: str = "auto"
    flip_green: bool = False
    normal_out: str = "dx"
    height_soften: float = 0.0
    tile: bool = True
    format: str = "png"
    mix_normal: float = 1.0
    mix_height: float = 1.0
    mix_rough: float = 1.0
    mix_metal: float = 1.0
    mix_ao: float = 1.0
    detail_restore: float = 0.0
    palette: list = field(default_factory=list)
    palette_strength: float = 0.0
    palette_hardness: float = 8.0
    seam_fade: float = 0.0
    grade_exposure: float = 0.0
    temp: float = 0.0
    grade_tint: float = 0.0
    contrast: float = 0.0
    gradient: bool = False
    gradient_root: list = field(default_factory=list)
    level_overrides: list = field(default_factory=list)
    stroke: float = 0.0
    packing: str = "mrod"
    island_mask: str | None = None
    # Kept so v1.x JSON loads; colour_grade does not apply it.
    tint_strength: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict | None) -> "Recipe":
        data = dict(data or {})
        known = {f.name for f in fields(cls)}
        kwargs = {k: v for k, v in data.items() if k in known}
        rec = cls(**kwargs)
        if rec.filter not in FILTERS:
            rec.filter = "kuwahara"
        if rec.size not in SIZES:
            rec.size = min(SIZES, key=lambda s: abs(s - int(rec.size or 1024)))
        if rec.packing not in ("mrod", "orm", "ormd"):
            rec.packing = "mrod"
        return rec


DEFAULTS = Recipe().to_dict()


def scaled(v, size) -> float:
    return float(v) * size / REF_SIZE


def json_settings(st: dict) -> dict:
    rec = Recipe.from_dict(st)
    return rec.to_dict()
