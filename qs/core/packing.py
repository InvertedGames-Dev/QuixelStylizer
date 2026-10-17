"""Channel packing for Unreal. Default MROD matches the Exodus master material."""
from __future__ import annotations

import numpy as np

from qs.core.io import to_u8
from qs.core.normals import encode_normal

PACKINGS = {
    "mrod": "R metallic, G roughness, B AO, A displacement (min 1/255)",
    "orm": "R AO, G roughness, B metallic",
    "ormd": "R AO, G roughness, B metallic, A displacement (min 1/255)",
}


def pack_maps(fin: dict, height_real: bool, packing: str = "mrod") -> dict[str, np.ndarray]:
    packing = (packing or "mrod").lower()
    metal = to_u8(fin["metal"])
    rough = to_u8(fin["rough"])
    ao = to_u8(fin["ao"])
    if height_real:
        height = np.maximum(to_u8(fin["height"]), 1)
    else:
        height = np.full(fin["height"].shape, 128, np.uint8)
    if packing == "orm":
        packed = np.dstack([ao, rough, metal])
        key = "ORM"
    elif packing == "ormd":
        packed = np.dstack([ao, rough, metal, height])
        key = "ORMD"
    else:
        packed = np.dstack([metal, rough, ao, height])
        key = "MROD"
    return {
        "D": to_u8(fin["alb"]),
        key: packed,
        "N": encode_normal(fin["normal"]),
        "packing": packing,
        "packed_key": key,
    }
