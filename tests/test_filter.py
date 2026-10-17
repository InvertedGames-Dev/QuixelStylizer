import numpy as np

from qs.core.filter import kuwahara_pass, seam_fade


def test_kuwahara_preserves_shape():
    rng = np.random.RandomState(0)
    alb = rng.rand(32, 32, 3).astype(np.float32)
    guide = rng.rand(32, 32).astype(np.float32)
    out, guides = kuwahara_pass(alb, [guide], 2, 8.0, True, classic=False)
    assert out.shape == alb.shape
    assert guides[0].shape == guide.shape
    assert float(np.max(out)) <= 1.0 + 1e-5
    assert float(np.min(out)) >= -1e-5


def test_seam_fade_zero_is_copy():
    rng = np.random.RandomState(0)
    img = rng.rand(24, 24, 3).astype(np.float32)
    out = seam_fade(img, 0.0, True)
    assert out is img or float(np.max(np.abs(out - img))) == 0.0


def test_gpu_kuwahara_matches_cpu():
    from qs.core import gpu as qs_gpu
    if not qs_gpu.available():
        import pytest
        pytest.skip(qs_gpu.status().get("reason") or "no GPU")
    diff = qs_gpu.status()["diff"]
    assert diff is not None
    assert diff <= 2.0 / 255.0
