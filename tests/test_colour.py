import numpy as np

from qs.core.colour import colour_grade, grade_is_identity, oklab_to_srgb, srgb_to_oklab
from qs.core.recipe import DEFAULTS


def test_oklab_roundtrip_white():
    white = np.array([[[1.0, 1.0, 1.0]]], np.float32)
    lab = srgb_to_oklab(white)
    assert abs(float(lab[0, 0, 0]) - 1.0) < 1e-4
    back = oklab_to_srgb(lab)
    assert float(np.max(np.abs(back - white))) < 1e-4


def test_grade_identity_is_copy():
    rng = np.random.RandomState(0)
    rgb = rng.rand(16, 16, 3).astype(np.float32)
    st = dict(DEFAULTS)
    assert grade_is_identity(st)
    out = colour_grade(rgb, st)
    assert float(np.max(np.abs(out - rgb))) == 0.0


def test_saturation_changes_picture():
    rng = np.random.RandomState(0)
    rgb = rng.rand(16, 16, 3).astype(np.float32)
    st = dict(DEFAULTS, saturation=1.4)
    out = colour_grade(rgb, st)
    assert float(np.max(np.abs(out - rgb))) > 1e-4


def _mean_chroma(rgb):
    x = np.asarray(rgb, np.float32)
    return float((x.max(axis=-1) - x.min(axis=-1)).mean())


def test_posterize_keeps_chroma():
    rgb = np.zeros((8, 8, 3), np.float32)
    rgb[..., 0] = 0.78
    rgb[..., 1] = 0.29
    rgb[..., 2] = 0.20
    src_c = _mean_chroma(rgb)
    out = colour_grade(rgb, dict(DEFAULTS, posterize=6))
    out_c = _mean_chroma(out)
    assert src_c > 0.4
    assert out_c > src_c * 0.5, "posterize quantized chroma to grey"
    lab = srgb_to_oklab(out)
    unique_l = len(np.unique(np.round(lab[..., 0], 5)))
    assert unique_l <= 6


def test_palette_strength_snaps_towards_swatch():
    rgb = np.full((4, 4, 3), (0.8, 0.25, 0.2), np.float32)
    pal = [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    out = colour_grade(rgb, dict(DEFAULTS, palette=pal, palette_strength=1.0, palette_hardness=32))
    assert float(out[..., 0].mean()) > 0.7
    assert float(out[..., 2].mean()) < 0.25
