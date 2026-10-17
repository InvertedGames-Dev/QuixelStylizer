import numpy as np

from qs.core.filter import dilate_islands, kuwahara_pass_islands
from qs.core.mesh import rasterize_islands


def test_kuwahara_does_not_cross_islands():
    h = w = 32
    img = np.zeros((h, w, 3), np.float32)
    img[:, :16] = (1, 0, 0)
    img[:, 16:] = (0, 0, 1)
    ids = np.zeros((h, w), np.int32)
    ids[:, :16] = 1
    ids[:, 16:] = 2
    guides = [np.zeros((h, w), np.float32)]
    out, _g = kuwahara_pass_islands(img, guides, r=4, q=8, island_ids=ids)
    # A pixel just inside the red island should stay red-dominant.
    assert out[16, 8, 0] > out[16, 8, 2]
    assert out[16, 24, 2] > out[16, 24, 0]


def test_dilate_fills_gutter():
    img = np.zeros((16, 16), np.float32)
    ids = np.zeros((16, 16), np.int32)
    ids[4:12, 4:12] = 1
    img[4:12, 4:12] = 0.8
    out = dilate_islands(img, ids, radius=2)
    assert out[3, 8] > 0.1
    assert out[0, 0] == 0.0


def test_dilate_islands_is_fast_with_many_islands():
    import time
    h = w = 512
    img = np.random.rand(h, w, 3).astype(np.float32)
    ids = np.zeros((h, w), np.int32)
    n = 100
    for i in range(n):
        y0 = (i * 7) % (h - 16)
        x0 = (i * 11) % (w - 16)
        ids[y0:y0 + 12, x0:x0 + 12] = i + 1
        img[y0:y0 + 12, x0:x0 + 12] = 0.6
    t0 = time.perf_counter()
    out = dilate_islands(img, ids, radius=4)
    elapsed = time.perf_counter() - t0
    assert out.shape == img.shape
    assert elapsed < 0.5, f"gutter dilate took {elapsed:.2f}s"


def test_kuwahara_islands_resizes_mask():
    img = np.zeros((32, 32, 3), np.float32)
    img[:, :16] = (1, 0, 0)
    img[:, 16:] = (0, 0, 1)
    ids = np.zeros((20, 18), np.int32)
    ids[:, :9] = 1
    ids[:, 9:] = 2
    guides = [np.zeros((32, 32), np.float32)]
    out, _g = kuwahara_pass_islands(img, guides, r=2, q=8, island_ids=ids)
    assert out.shape == img.shape
    assert float(out.mean()) > 0.0


def test_gpu_kuwahara_does_not_cross_islands():
    from qs.core import gpu as qs_gpu
    if not qs_gpu.available():
        import pytest
        pytest.skip(qs_gpu.status().get("reason") or "no GPU")
    h = w = 32
    img = np.zeros((h, w, 3), np.float32)
    img[:, :16] = (1, 0, 0)
    img[:, 16:] = (0, 0, 1)
    ids = np.zeros((h, w), np.int32)
    ids[:, :16] = 1
    ids[:, 16:] = 2
    nrm = np.zeros((h, w, 3), np.float32)
    pack = np.zeros((h, w, 4), np.float32)
    cav = np.zeros((h, w), np.float32)
    gpu = qs_gpu._thread_gpu()
    alb, _n, _p, _c = gpu.kuwahara_pass(
        img, nrm, pack, cav, 4, 8.0, False, False, False, island_ids=ids)
    assert alb[16, 8, 0] > alb[16, 8, 2]
    assert alb[16, 24, 2] > alb[16, 24, 0]


def test_rasterize_two_triangles():
    uv = np.array([[0.1, 0.1], [0.4, 0.1], [0.1, 0.4],
                   [0.6, 0.6], [0.9, 0.6], [0.6, 0.9]], np.float32)
    faces = np.array([[0, 1, 2], [3, 4, 5]], np.int32)
    canvas = rasterize_islands(uv, faces, 64, 64)
    assert canvas.max() == 2
    assert (canvas == 1).sum() > 10
    assert (canvas == 2).sum() > 10
