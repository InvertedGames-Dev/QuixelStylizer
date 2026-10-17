import numpy as np

from qs.core.mesh import (
    _triangles, count_uv_islands, decimate, find_original_mesh_file,
    is_reuv_path, load_mesh, snap_coincident, texel_density, vertex_islands,
    vertex_tangents,
)


def _two_islands():
    # Two disconnected triangles in UV.
    uv = np.array([[0, 0], [0.4, 0], [0, 0.4], [0.6, 0.6], [1, 0.6], [0.6, 1]], np.float32)
    faces = np.array([[0, 1, 2], [3, 4, 5]], np.int32)
    pos = np.array([[0, 0, 0], [100, 0, 0], [0, 0, 100],
                    [200, 0, 0], [300, 0, 0], [200, 0, 100]], np.float32)
    return uv, faces, pos


def test_find_original_mesh_file_strips_reuv_suffixes(tmp_path):
    original = tmp_path / "mesh" / "Prop_Low.glb"
    original.parent.mkdir()
    original.write_bytes(b"glb")
    first = tmp_path / "mesh" / "Prop_Low_reuv.glb"
    second = tmp_path / "mesh" / "Prop_Low_reuv_reuv.glb"
    first.write_bytes(b"glb")
    second.write_bytes(b"glb")
    (tmp_path / "Other.glb").write_bytes(b"glb")

    assert is_reuv_path(str(second))
    assert not is_reuv_path(str(original))
    assert find_original_mesh_file(str(tmp_path), str(second)) == str(original)
    assert find_original_mesh_file(str(tmp_path), str(first)) == str(original)
    assert find_original_mesh_file(str(tmp_path), str(original)) == str(original)


def test_snap_coincident_merges_duplicates():
    p = np.array([[0, 0, 0], [0, 0, 0], [1, 0, 0]], np.float32)
    n = np.array([[0, 1, 0], [0, 0, 1], [0, 1, 0]], np.float32)
    p2, n2 = snap_coincident(p, n, 1e-4)
    assert np.allclose(p2[0], p2[1])
    assert np.allclose(n2[1], [0, 0, 1])
    assert np.allclose(p2[2], [1, 0, 0])


def test_count_uv_islands():
    uv, faces, _ = _two_islands()
    assert count_uv_islands(uv, faces) == 2
    ids = vertex_islands(uv, faces)
    assert set(ids.tolist()) == {1, 2}
    assert ids[0] == ids[1] == ids[2]
    assert ids[3] == ids[4] == ids[5]
    assert ids[0] != ids[3]


def test_texel_density_positive():
    uv, faces, pos = _two_islands()
    d = texel_density(uv, pos, faces, 1024)
    assert d > 0


def test_triangles_splits_quads():
    pos = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], np.float32)
    quads = np.array([[0, 1, 2, 3]], np.int32)
    tris = _triangles(pos, quads)
    assert tris.shape == (2, 3)
    assert set(tris[0].tolist() + tris[1].tolist()) == {0, 1, 2, 3}


def test_vertex_tangents_mismatched_normals():
    pos = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], np.float32)
    nrm = np.array([[0, 0, 1]] * 3, np.float32)  # short on purpose
    uv = np.array([[0, 0], [1, 0], [1, 1]], np.float32)
    idx = np.array([0, 1, 2, 0, 2, 3], np.uint32)
    tan = vertex_tangents(pos, nrm, uv, idx)
    assert tan.shape == (4, 4)


def test_vertex_tangents_follow_uv():
    pos = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], np.float32)
    nrm = np.array([[0, 0, 1]] * 4, np.float32)
    uv = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], np.float32)
    idx = np.array([0, 1, 2, 0, 2, 3], np.uint32)
    tan = vertex_tangents(pos, nrm, uv, idx)
    assert tan.shape == (4, 4)
    # +U is +X on this plane.
    assert float(tan[:, 0].mean()) > 0.9
    assert abs(float(tan[:, 1].mean())) < 0.1
    assert abs(float(tan[:, 2].mean())) < 0.1


def test_load_and_decimate_obj(tmp_path):
    lines = []
    n = 12
    for j in range(n + 1):
        for i in range(n + 1):
            u, v = i / n, j / n
            lines.append(f"v {(u - 0.5) * 100.0:.3f} {(v - 0.5) * 100.0:.3f} 0.0")
            lines.append(f"vt {u:.4f} {v:.4f}")
    for j in range(n):
        for i in range(n):
            a = j * (n + 1) + i + 1
            b = a + 1
            c = a + (n + 1)
            d = c + 1
            lines.append(f"f {a}/{a} {b}/{b} {c}/{c}")
            lines.append(f"f {b}/{b} {d}/{d} {c}/{c}")
    obj = tmp_path / "grid.obj"
    obj.write_text("\n".join(lines), encoding="utf-8")
    m = load_mesh(str(obj))
    assert m is not None
    assert m["tris"] == 2 * n * n
    assert m["normals"].shape == m["positions"].shape
    assert m["uvs"].shape == (len(m["positions"]), 2)
    dec = decimate(m, ratio=0.4)
    assert dec["tris"] < m["tris"]
    assert dec["uvs"].shape == (len(dec["positions"]), 2)


def _write_mixed_uv_obj(path, n=20):
    """Some faces have no vt. Quixel GLBs fail MeshLab's texture-preserving collapse this way."""
    lines = []
    for j in range(n + 1):
        for i in range(n + 1):
            u, v = i / n, j / n
            lines.append(f"v {(u - 0.5) * 100.0:.4f} {(v - 0.5) * 100.0:.4f} 0.0")
            lines.append(f"vt {u:.5f} {v:.5f}")
    for j in range(n):
        for i in range(n):
            a = j * (n + 1) + i + 1
            b, d, c = a + 1, a + (n + 1), a + (n + 1) + 1
            if (i + j) % 3 == 0:
                lines.append(f"f {a} {b} {c}")
                lines.append(f"f {a} {c} {d}")
            else:
                lines.append(f"f {a}/{a} {b}/{b} {c}/{c}")
                lines.append(f"f {a}/{a} {c}/{c} {d}/{d}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_decimate_inconsistent_face_uvs(tmp_path):
    obj = tmp_path / "mixed.obj"
    _write_mixed_uv_obj(obj, n=20)
    m = load_mesh(str(obj))
    assert m is not None and m["tris"] > 100
    d1 = decimate(m, ratio=0.5)
    d2 = decimate(m, ratio=0.25)
    assert d1["tris"] < m["tris"]
    assert d2["tris"] < d1["tris"]
    assert d2["uvs"].shape[0] == d2["verts"]
    assert d2["normals"].shape[0] == d2["verts"]


def _geometric_boundary_edges(mesh) -> int:
    from collections import Counter
    p = np.asarray(mesh["positions"], np.float32)
    f = mesh["indices"].reshape(-1, 3)
    keys = np.round(p * 1e5).astype(np.int64)
    _uniq, inv = np.unique(keys, axis=0, return_inverse=True)
    inv = np.asarray(inv).reshape(-1)
    gf = inv[f]
    edges = Counter()
    for a, b, c in gf:
        for x, y in ((a, b), (b, c), (c, a)):
            if x != y:
                edges[tuple(sorted((int(x), int(y))))] += 1
    return sum(1 for count in edges.values() if count == 1)


def test_decimate_quixel_glb_if_present():
    from pathlib import Path
    glb = Path(r"C:\Users\BigMoney\Projects\QuixelStylizer\library\sources\ram_skull_tiiief2fa_low_345e19b6\mesh\Ram_Skull_tiiief2fa_Low.glb")
    if not glb.is_file():
        import pytest
        pytest.skip("user ram skull GLB not in this library")
    m = load_mesh(str(glb))
    assert m is not None
    d = decimate(m, ratio=0.4)
    assert d["tris"] < m["tris"]
    d2 = decimate(m, ratio=0.2)
    assert d2["tris"] <= d["tris"]
    # A closed source must stay closed: no edge used by only one triangle.
    assert _geometric_boundary_edges(d) == 0
    assert _geometric_boundary_edges(d2) == 0


def test_decimate_stump_keeps_uv_islands_if_present():
    from pathlib import Path
    glb = Path(
        r"C:\Users\BigMoney\Projects\QuixelStylizer\library\sources"
        r"\old_tree_stump_wd0qcaobw_low_8fe9d185\mesh\Old_Tree_Stump_wd0qcaobw_Low.glb"
    )
    if not glb.is_file():
        import pytest
        pytest.skip("user tree stump GLB not in this library")
    m = load_mesh(str(glb))
    d = decimate(m, ratio=0.25)
    assert d["tris"] < m["tris"] * 0.5
    # The stump source is open at the base; decimate must not open it further.
    assert _geometric_boundary_edges(d) <= _geometric_boundary_edges(m)

