import numpy as np
from PIL import Image

from qs.core.maps import detect_maps
from qs.core.recipe import DEFAULTS
from qs.core.stylize import finish_stage, load_stage, paint_copy, process_set


def _tiny_set(folder):
    Image.new("RGB", (32, 32), (180, 90, 40)).save(folder / "Tiny_1K_Albedo.png")
    Image.new("L", (32, 32), 140).save(folder / "Tiny_1K_Roughness.png")
    Image.new("L", (32, 32), 10).save(folder / "Tiny_1K_Metallic.png")
    Image.new("L", (32, 32), 200).save(folder / "Tiny_1K_AO.png")
    Image.new("RGB", (32, 32), (128, 128, 255)).save(folder / "Tiny_1K_NormalDX.png")
    Image.new("L", (32, 32), 128).save(folder / "Tiny_1K_Displacement.png")
    return detect_maps(str(folder))


def test_process_set_identity_write(tmp_path):
    s = _tiny_set(tmp_path)
    out = tmp_path / "export"
    st = dict(DEFAULTS, size=32, radius=0)
    result = process_set(s, st, write=True, out_dir=str(out), write_meta=True)
    assert (out / "Tiny_D.png").is_file()
    assert (out / "Tiny_MROD.png").is_file()
    assert (out / "Tiny_N.png").is_file()
    assert "RGB" in result["checked"]["D"]
    before = np.asarray(s)  # noqa: F841 — just ensure detect worked
    assert result["size"] == (32, 32)


def test_normal_strength_changes_slope(tmp_path):
    s = _tiny_set(tmp_path)
    n = np.dstack([
        np.full((32, 32), 180, np.uint8),
        np.full((32, 32), 70, np.uint8),
        np.full((32, 32), 220, np.uint8),
    ])
    Image.fromarray(n, "RGB").save(tmp_path / "Tiny_1K_NormalDX.png")
    s = detect_maps(str(tmp_path))
    src = load_stage(s, dict(DEFAULTS, size=32, radius=0))
    painted = paint_copy(src)
    flat = finish_stage(src, painted, dict(DEFAULTS, size=32, radius=0, normal_strength=0.0))
    hard = finish_stage(src, painted, dict(DEFAULTS, size=32, radius=0, normal_strength=2.0))
    xy_flat = float(np.abs(flat["normal"][..., :2]).mean())
    xy_hard = float(np.abs(hard["normal"][..., :2]).mean())
    assert xy_hard > xy_flat + 0.02
    soft = finish_stage(src, painted, dict(DEFAULTS, size=32, radius=0, normal_soften=1.0, normal_strength=1.0))
    assert float(np.abs(soft["normal"] - src["normal"]).mean()) > 0.0


def test_recipe_drops_unknown_keys():
    from qs.core.recipe import Recipe
    rec = Recipe.from_dict({"emissive_mode": "keep", "saturation": 1.2, "bogus": 1})
    d = rec.to_dict()
    assert "emissive_mode" not in d
    assert abs(d["saturation"] - 1.2) < 1e-9
