import numpy as np

from qs.core.packing import pack_maps


def _fin():
    h = w = 8
    return {
        "alb": np.full((h, w, 3), 0.4, np.float32),
        "metal": np.full((h, w), 0.2, np.float32),
        "rough": np.full((h, w), 0.5, np.float32),
        "ao": np.full((h, w), 0.8, np.float32),
        "height": np.full((h, w), 0.1, np.float32),
        "normal": np.dstack([np.zeros((h, w)), np.zeros((h, w)), np.ones((h, w))]).astype(np.float32),
    }


def test_mrod_default():
    packed = pack_maps(_fin(), True, "mrod")
    assert packed["packed_key"] == "MROD"
    mrod = packed["MROD"]
    assert mrod.shape[2] == 4
    assert mrod[0, 0, 0] < mrod[0, 0, 1]  # metal < roughness at these values
    assert int(mrod[0, 0, 3]) >= 1


def test_orm():
    packed = pack_maps(_fin(), True, "orm")
    assert packed["packed_key"] == "ORM"
    assert packed["ORM"].shape[2] == 3
