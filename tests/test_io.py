import numpy as np

from qs.core.io import imwrite, to_u8, verify_written


def test_png_write_verify(tmp_path):
    rgb = to_u8(np.full((16, 16, 3), 0.4, np.float32))
    path = tmp_path / "t_D.png"
    imwrite(str(path), rgb)
    desc = verify_written(str(path), 3, (16, 16))
    assert "RGB" in desc
    assert "type 2" in desc


def test_rgba_write_verify(tmp_path):
    rgba = to_u8(np.full((8, 8, 4), 0.5, np.float32))
    rgba[..., 3] = 1
    path = tmp_path / "t_MROD.png"
    imwrite(str(path), rgba)
    desc = verify_written(str(path), 4, (8, 8))
    assert "RGBA" in desc
