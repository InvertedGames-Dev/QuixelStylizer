import numpy as np

from qs.core.exodus_look import (
    LookSettings,
    apply_agx_punchy,
    d_ggx_dual,
    evaluate_brdf,
    load_look,
    render_png,
    srgb_to_ap1,
    tonemap_agx,
)
from qs.core.smaa import apply_smaa, detect_edges


def test_ini_parser_reads_exodus_defines():
    look = load_look()
    assert look.spec_mode == 1
    assert abs(look.spec_primary_weight - 0.7) < 1e-6
    assert abs(look.spec_primary_roughness_scale - 0.6) < 1e-6
    assert look.diffuse_mode == 2
    assert look.tonemap == "agx_punchy"
    assert look.agx_look == 2


def test_dual_lobe_d_is_blend_of_two_ggx():
    look = LookSettings()
    a2 = 0.09
    noh = 0.8
    d = d_ggx_dual(a2, noh, look)
    look1 = LookSettings(spec_mode=0)
    d_single = d_ggx_dual(a2, noh, look1)
    assert d > 0
    assert abs(d - d_single) > 1e-6


def test_brdf_front_lit_dielectric_is_finite():
    n = np.array([0, 0, 1], np.float32)
    v = n
    l = n
    rgb = evaluate_brdf((0.5, 0.1, 0.05), 0.4, 0.0, n, v, l)
    assert np.all(np.isfinite(rgb))
    assert float(rgb.max()) > 0


def test_agx_maps_mid_grey_into_display_range():
    mid = np.array([[[0.18, 0.18, 0.18]]], np.float32)
    out = tonemap_agx(mid)
    assert 0.0 < float(out.mean()) < 1.0
    punchy = apply_agx_punchy(srgb_to_ap1(mid), 2)
    default = apply_agx_punchy(srgb_to_ap1(mid), 0)
    assert float(np.max(np.abs(punchy - default))) > 1e-4


def test_smaa_softens_a_hard_edge():
    img = np.zeros((32, 32, 3), np.float32)
    img[:, 16:] = 1.0
    out = apply_smaa(img)
    seam = out[:, 15:17].mean()
    assert 0.05 < float(seam) < 0.95
    edges = detect_edges(img)
    assert float(edges[:, 16, 0].max()) >= 0.1


def test_np_eye_dtype_kwonly():
    import numpy as np
    # np.eye(4, np.float32) is illegal in numpy 2 (second arg is M).
    m = np.eye(4, dtype=np.float32)
    assert m.shape == (4, 4)


def test_viewport_glsl_compiles():
    """The live viewport died on HLSL saturate; catch that before show()."""
    import pytest
    from qs.app.viewport import FRAG, VERT
    try:
        import moderngl
        ctx = moderngl.create_standalone_context(require=330)
    except Exception as exc:
        pytest.skip(f"no GL context ({exc})")
    try:
        ctx.program(vertex_shader=VERT, fragment_shader=FRAG)
    finally:
        ctx.release()


def test_render_png(tmp_path):
    path = tmp_path / "sphere.png"
    render_png(str(path), width=32, height=32, smaa=False)
    assert path.is_file()
    assert path.stat().st_size > 100
