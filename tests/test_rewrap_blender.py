"""Real-Blender rewrap bake on the real ram skull.

This is the test whose absence let a rotated, black-patched bake ship:
the Blender-internal renders looked fine while the exported GLB came
back in the wrong frame and a fifth of its texels had never been
baked. It exercises the whole path — PyMeshLab decimate, OBJ out,
Blender unwrap + bake, GLB back through the app's own loader — and
checks the two things the viewport cares about: the mesh returns in
the frame it left in, and the baked maps are actually written.
Skipped when Blender or the library skull is not present.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from qs.blender import client, find_blender, rewrap_bake
from qs.core.maps import detect_maps
from qs.core.mesh import decimate, load_mesh, rasterize_islands, write_obj

SKULL_DIR = (Path(__file__).resolve().parent.parent / "library" / "sources"
             / "ram_skull_tiiief2fa_low_345e19b6")
SKULL = SKULL_DIR / "mesh" / "Ram_Skull_tiiief2fa_Low.glb"

pytestmark = pytest.mark.skipif(
    find_blender() is None or not SKULL.is_file(),
    reason="needs Blender and the ram skull library asset")


def test_rewrap_bake_returns_same_frame_and_written_maps(tmp_path):
    srcset = detect_maps(str(SKULL_DIR))
    src = load_mesh(str(SKULL))
    low = decimate(src, 0.35)
    low_obj = write_obj(low, str(tmp_path / "low_input.obj"))
    try:
        result = rewrap_bake(
            str(SKULL), low_obj, str(tmp_path / "baked"),
            str(tmp_path / "baked" / "low.glb"),
            maps={"albedo": srcset["maps"]["albedo"],
                  "normal": srcset["maps"]["normal"]},
            size=256, base_name=srcset["name"])
    finally:
        client().stop()

    out = load_mesh(result["mesh"])
    assert out["tris"] == low["tris"]
    # The exported mesh must come back in the decimated mesh's frame:
    # same bounds, not a quarter-turn out or at node scale.
    pin = np.asarray(low["positions"], float)
    pout = np.asarray(out["positions"], float)
    assert np.allclose(pout.min(axis=0), pin.min(axis=0), atol=0.05)
    assert np.allclose(pout.max(axis=0), pin.max(axis=0), atol=0.05)

    # And the bake must actually land: near-zero black texels inside
    # the new UV islands (the broken build left ~17% black).
    ids = rasterize_islands(out["uvs"], out["indices"], 256, 256)
    inside = ids > 0
    assert inside.mean() > 0.03
    alb = np.asarray(Image.open(result["files"]["albedo"]).convert("RGB"),
                     dtype=np.float32)
    if alb.shape[:2] != (256, 256):
        alb = np.asarray(Image.open(result["files"]["albedo"]).convert("RGB")
                         .resize((256, 256)), dtype=np.float32)
    black = float(((alb.sum(axis=2) < 12) & inside).sum() / inside.sum())
    assert black < 0.02, f"{black:.3f} of in-island texels baked black"
