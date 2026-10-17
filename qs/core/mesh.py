"""In-process PyMeshLab decimate + mesh counts. Blender convert is a separate job."""
from __future__ import annotations

import threading
from pathlib import Path

import numpy as np

from qs.logging import log_line

MESH_SUFFIXES = (".gltf", ".glb", ".fbx", ".obj")
# MeshLab's native heap is not re-entrant. One MeshSet at a time.
_MESHLAB_LOCK = threading.Lock()


def _pymeshlab():
    try:
        import pymeshlab
        return pymeshlab
    except ImportError:
        return None


def find_mesh_file(folder: str) -> str | None:
    root = Path(folder)
    if not root.is_dir():
        return None
    ranked = []
    for suf in MESH_SUFFIXES:
        ranked.extend(root.rglob(f"*{suf}"))
    if not ranked:
        return None
    prefer = {".gltf": 0, ".glb": 1, ".obj": 2, ".fbx": 3}
    ranked.sort(key=lambda p: (prefer.get(p.suffix.lower(), 9), len(str(p))))
    return str(ranked[0])


def is_reuv_path(path: str | None) -> bool:
    return bool(path) and Path(path).stem.lower().endswith("_reuv")


def find_original_mesh_file(folder: str, current: str | None = None) -> str | None:
    """Recover the pre-Re-UV mesh for `current` inside a source folder.

    Re-UV writes ``<stem>_reuv.glb`` next to its input (and repeats the
    suffix when clicked twice), so stripping that suffix names the file
    we want. Older libraries only persisted the re-UV path, which is why
    this cannot rely on a stored original alone.
    """
    root = Path(folder) if folder else None
    files: list[Path] = []
    if root is not None and root.is_dir():
        for suf in MESH_SUFFIXES:
            files.extend(root.rglob(f"*{suf}"))
    prefer = {".gltf": 0, ".glb": 1, ".obj": 2, ".fbx": 3}

    def rank(p: Path):
        return (prefer.get(p.suffix.lower(), 9), len(str(p)))

    if current:
        cur = Path(current)
        if not is_reuv_path(current):
            return str(cur) if cur.is_file() else (str(cur) if not files else None)
        base = cur.stem
        while base.lower().endswith("_reuv"):
            base = base[: -len("_reuv")]
        exact = [p for p in files
                 if p.stem.lower() == base.lower() and not is_reuv_path(str(p))]
        if exact:
            exact.sort(key=lambda p: (p.parent != cur.parent, rank(p)))
            return str(exact[0])
    originals = [p for p in files if not is_reuv_path(str(p))]
    if originals:
        originals.sort(key=rank)
        return str(originals[0])
    return None


def write_obj(mesh: dict, path: str) -> str:
    """Serialize the in-memory mesh (current decimated state) as an OBJ.

    Decimation happens in memory and the mesh dict's ``path`` still points
    at the original source file, so the rewrap-and-bake flow cannot hand
    Blender that path. This writes the actual positions, indices, UVs and
    normals Blender must unwrap. Faces are written as ``v/vt/vn`` corners.
    """
    p = np.asarray(mesh["positions"], np.float32)
    n = _align_attr(mesh.get("normals"), len(p), 0.0, 3)
    uv = _align_attr(mesh.get("uvs"), len(p), 0.0, 2)[:, :2]
    f = _triangles(p, mesh["indices"])
    if len(p) == 0 or len(f) == 0:
        raise ValueError("cannot write an empty mesh")
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", encoding="utf-8") as fh:
        fh.write("o QS_Low\n")
        for row in p:
            fh.write(f"v {row[0]:.6f} {row[1]:.6f} {row[2]:.6f}\n")
        for row in uv:
            fh.write(f"vt {row[0]:.6f} {row[1]:.6f}\n")
        for row in n:
            fh.write(f"vn {row[0]:.6f} {row[1]:.6f} {row[2]:.6f}\n")
        for a, b, c in f:
            corners = " ".join(f"{int(i) + 1}/{int(i) + 1}/{int(i) + 1}" for i in (a, b, c))
            fh.write(f"f {corners}\n")
    return str(dest)


def _triangles(v, f) -> np.ndarray:
    """Force an (N, 3) index buffer. Quads become two triangles; padded -1 faces drop."""
    f = np.asarray(f)
    if f.size == 0 or len(v) == 0:
        return np.zeros((0, 3), np.int32)
    if f.ndim == 1:
        cut = f.size - (f.size % 3)
        f = f[:cut].reshape(-1, 3)
    cols = int(f.shape[1])
    if cols >= 4:
        a, b, c, d = f[:, 0], f[:, 1], f[:, 2], f[:, 3]
        quad = (d >= 0)
        tris = np.vstack([
            np.stack([a, b, c], axis=1),
            np.stack([a[quad], c[quad], d[quad]], axis=1),
        ])
        f = tris
    elif cols != 3:
        f = f[:, :3] if cols > 3 else np.zeros((0, 3), np.int32)
    nv = len(v)
    ok = (f.min(axis=1) >= 0) & (f.max(axis=1) < nv)
    return f[ok].astype(np.int32, copy=False)


def _align_attr(arr, nverts, fill=0.0, width=3):
    a = np.asarray(arr, np.float32)
    if a.ndim == 1:
        a = a.reshape(-1, 1)
    if a.shape[0] == nverts:
        return a
    out = np.full((nverts, a.shape[1] if a.ndim == 2 else width), fill, np.float32)
    n = min(len(a), nverts)
    if n:
        out[:n] = a[:n]
    return out


def _extract_mesh_arrays(ms) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int]:
    try:
        ms.meshing_poly_to_tri()
    except Exception:
        pass
    m = ms.current_mesh()
    if not m.has_vertex_tex_coord() and m.has_wedge_tex_coord():
        try:
            ms.compute_texcoord_transfer_wedge_to_vertex()
            m = ms.current_mesh()
        except Exception:
            pass
    v = np.array(m.vertex_matrix(), np.float32)
    f = _triangles(v, np.array(m.face_matrix(), np.int32))
    try:
        n = np.array(m.vertex_normal_matrix(), np.float32)
        n = _align_attr(n, len(v), 0.0, 3)
        if n.shape != v.shape or not np.any(n):
            n = _face_normals(v, f)
    except Exception:
        n = _face_normals(v, f)
    if m.has_vertex_tex_coord():
        uv = _align_attr(np.array(m.vertex_tex_coord_matrix(), np.float32), len(v), 0.0, 2)
        if uv.shape[1] < 2:
            uv = np.zeros((len(v), 2), np.float32)
        else:
            uv = uv[:, :2]
    else:
        uv = np.zeros((len(v), 2), np.float32)
    return v, f, n, uv, int(len(f)), int(len(v))


def _extract_wedge_split(ms) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int]:
    """Extract a render mesh split by wedge UV, sharing the underlying positions.

    A welded MeshLab vertex can carry a different UV on each face corner.
    Transferring wedge UVs to one vertex UV throws those seams away; splitting
    by (vertex, UV) keeps the atlas while every duplicate keeps the exact same
    position, so the surface cannot open along a seam.
    """
    m = ms.current_mesh()
    if not m.has_wedge_tex_coord():
        return _extract_mesh_arrays(ms)
    v = np.array(m.vertex_matrix(), np.float32)
    f = _triangles(v, np.array(m.face_matrix(), np.int32))
    wedge = np.asarray(m.wedge_tex_coord_matrix(), np.float32).reshape(-1, 2)
    if len(f) == 0 or wedge.shape[0] != len(f) * 3:
        return _extract_mesh_arrays(ms)
    normals = _face_normals(v, f)
    corner_v = f.reshape(-1).astype(np.int32, copy=False)
    key = np.empty(len(corner_v), dtype=[("v", "<i4"), ("u", "<f4"), ("w", "<f4")])
    key["v"] = corner_v
    key["u"] = wedge[:, 0]
    key["w"] = wedge[:, 1]
    uniq, inverse = np.unique(key, return_inverse=True)
    pos = v[uniq["v"]]
    nrm = normals[uniq["v"]]
    uv = np.column_stack([uniq["u"], uniq["w"]]).astype(np.float32)
    faces = np.asarray(inverse).reshape(-1, 3).astype(np.int32)
    return pos, faces, nrm, uv, int(len(faces)), int(len(pos))


def _welded_meshset(ml, src: str):
    """Load a mesh as one geometric surface, with UVs kept per face corner.

    Quixel GLBs split a closed surface into many UV patches. MeshLab sees
    those patches as open sheets and quadric collapse shrinks each one on its
    own, which is the tearing seen in the viewport. Welding coincident
    positions first gives the collapse a single closed surface to work on.
    """
    loader = ml.MeshSet()
    loader.load_new_mesh(src)
    try:
        loader.meshing_poly_to_tri()
    except Exception:
        pass
    m = loader.current_mesh()
    v = np.array(m.vertex_matrix(), np.float64)
    f = _triangles(v.astype(np.float32), np.array(m.face_matrix(), np.int32)).astype(np.int32)
    if m.has_wedge_tex_coord():
        wedge = np.asarray(m.wedge_tex_coord_matrix(), np.float64).reshape(-1, 2)
        if wedge.shape[0] != len(f) * 3:
            wedge = np.zeros((len(f) * 3, 2), np.float64)
    elif m.has_vertex_tex_coord():
        v_uv_src = _align_attr(np.array(m.vertex_tex_coord_matrix(), np.float32), len(v), 0.0, 2)[:, :2]
        wedge = np.asarray(v_uv_src[f.reshape(-1)], np.float64)
    else:
        wedge = np.zeros((len(f) * 3, 2), np.float64)
    keys = np.round(v * 1e5).astype(np.int64)
    _uniq_keys, inv = np.unique(keys, axis=0, return_inverse=True)
    inv = np.asarray(inv).reshape(-1)
    welded = np.zeros((int(inv.max()) + 1 if len(inv) else 0, 3), np.float64)
    counts = np.bincount(inv).astype(np.float64)
    np.add.at(welded, inv, v)
    if len(welded):
        welded /= counts[:, None]
    wf = inv[f].astype(np.int32)
    v_uv = np.zeros((len(welded), 2), np.float64)
    seen = np.zeros(len(welded), bool)
    wedge3 = wedge.reshape(-1, 3, 2)
    for fi, face in enumerate(wf):
        for corner, wid in enumerate(face):
            if not seen[wid]:
                v_uv[wid] = wedge3[fi, corner]
                seen[wid] = True
    mesh = ml.Mesh(
        vertex_matrix=welded,
        face_matrix=wf,
        v_tex_coords_matrix=v_uv,
        w_tex_coords_matrix=wedge,
    )
    out = ml.MeshSet()
    out.add_mesh(mesh, "welded")
    return out


def load_mesh(path: str) -> dict | None:
    path = str(path)
    if not Path(path).is_file():
        return None
    ml = _pymeshlab()
    if ml is None:
        return None
    with _MESHLAB_LOCK:
        ms = ml.MeshSet()
        ms.load_new_mesh(path)
        v, f, n, uv, tris, verts = _extract_mesh_arrays(ms)
    return {
        "positions": v,
        "normals": n,
        "uvs": uv,
        "indices": np.ascontiguousarray(f.reshape(-1), np.uint32),
        "tris": tris,
        "verts": verts,
        "islands": count_uv_islands(uv, f),
        "path": path,
        "source_path": path,
        "source_tris": tris,
    }


def _face_normals(v, f):
    n = np.zeros_like(v)
    tri = v[f]
    fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    for i in range(3):
        np.add.at(n, f[:, i], fn)
    ln = np.linalg.norm(n, axis=1, keepdims=True)
    return n / np.maximum(ln, 1e-8)


def vertex_tangents(positions, normals, uvs, indices) -> np.ndarray:
    """Per-vertex UV tangents as (N, 4). xyz is T; w is 1. Shader bitangent is cross(N, T).

    Screen-space dFdx frames go to zero on some NVIDIA MSAA FBOs, which made Lit
    ignore the normal map. These are computed from the mesh, so bump survives.
    """
    p = np.asarray(positions, np.float32)
    n = _align_attr(normals, len(p), 0.0, 3)
    uv = _align_attr(uvs, len(p), 0.0, 2)[:, :2]
    if len(p) == 0:
        return np.zeros((0, 4), np.float32)
    f = _triangles(p, indices)
    tan = np.zeros_like(p)
    if len(f):
        p0, p1, p2 = p[f[:, 0]], p[f[:, 1]], p[f[:, 2]]
        w0, w1, w2 = uv[f[:, 0]], uv[f[:, 1]], uv[f[:, 2]]
        e1 = p1 - p0
        e2 = p2 - p0
        d1 = w1 - w0
        d2 = w2 - w0
        det = d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]
        r = np.divide(1.0, det, out=np.zeros_like(det), where=np.abs(det) > 1e-12)
        t = (e1 * d2[:, 1:2] - e2 * d1[:, 1:2]) * r[:, None]
        np.add.at(tan, f[:, 0], t)
        np.add.at(tan, f[:, 1], t)
        np.add.at(tan, f[:, 2], t)
    tan = tan - n * (tan * n).sum(axis=1, keepdims=True)
    ln = np.linalg.norm(tan, axis=1, keepdims=True)
    missing = ln.ravel() < 1e-8
    tan = tan / np.maximum(ln, 1e-8)
    if np.any(missing):
        nn = n[missing]
        axis = np.zeros_like(nn)
        axis[:, 2] = 1.0
        axis[np.abs(nn[:, 2]) >= 0.999] = (1.0, 0.0, 0.0)
        fb = np.cross(axis, nn)
        fb = fb / np.maximum(np.linalg.norm(fb, axis=1, keepdims=True), 1e-8)
        tan[missing] = fb
    w = np.ones((len(p), 1), np.float32)
    return np.hstack([tan.astype(np.float32), w])


def _faces3(faces) -> np.ndarray:
    f = np.asarray(faces)
    if f.ndim == 1:
        f = f.reshape(-1, 3)
    return f.astype(np.int32, copy=False)


def vertex_islands(uv, faces) -> np.ndarray:
    """Union-find over UV-mesh connectivity. Returns an id per vertex (0 if unused)."""
    n = len(uv)
    parent = np.arange(n, dtype=np.int32)
    f = _faces3(faces)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def uni(a, b):
        ra, rb = find(int(a)), find(int(b))
        if ra != rb:
            parent[rb] = ra

    used = np.zeros(n, dtype=bool)
    for a, b, c in f:
        uni(a, b)
        uni(b, c)
        used[a] = used[b] = used[c] = True
    roots = {}
    nxt = 1
    ids = np.zeros(n, dtype=np.int32)
    for i in range(n):
        if not used[i]:
            continue
        r = find(i)
        if r not in roots:
            roots[r] = nxt
            nxt += 1
        ids[i] = roots[r]
    return ids


def count_uv_islands(uv, faces) -> int:
    if uv is None or len(uv) == 0 or faces is None or len(faces) == 0:
        return 0
    ids = vertex_islands(uv, faces)
    return int(ids.max()) if ids.size else 0


def rasterize_islands(uv, faces, width: int, height: int) -> np.ndarray:
    """Integer island-id image (0 = gutter). Used to clamp Kuwahara to UV islands."""
    import cv2
    canvas = np.zeros((height, width), np.int32)
    if uv is None or faces is None or len(uv) == 0:
        return canvas
    ids = vertex_islands(uv, faces)
    f = _faces3(faces)
    scale = np.array([width - 1, height - 1], np.float32)
    pts = np.clip(np.asarray(uv, np.float32) * scale, 0, [width - 1, height - 1])
    pts[:, 1] = (height - 1) - pts[:, 1]
    for a, b, c in f:
        iid = int(ids[a])
        if iid <= 0:
            continue
        tri = np.array([pts[a], pts[b], pts[c]], np.int32)
        cv2.fillConvexPoly(canvas, tri, iid)
    return canvas


def snap_coincident(positions, normals, epsilon: float = 1e-4):
    """Force UV-split duplicates to share a position so tessellation cannot crack."""
    p = np.array(positions, np.float32, copy=True)
    n = np.array(normals, np.float32, copy=True)
    if len(p) == 0:
        return p, n
    scale = 1.0 / max(float(epsilon), 1e-8)
    keys = np.round(p * scale).astype(np.int64)
    buckets: dict[tuple, list[int]] = {}
    for i, key in enumerate(map(tuple, keys)):
        buckets.setdefault(key, []).append(i)
    for idxs in buckets.values():
        if len(idxs) < 2:
            continue
        ix = np.asarray(idxs)
        p[ix] = p[ix].mean(axis=0)
        # Positions only. Averaging inner/outer normals on a thin shell
        # makes displacement shoot the surface apart.
    return p, n


def _median_edge(positions, faces) -> float:
    p = np.asarray(positions, np.float32)
    f = _triangles(p, faces)
    if len(f) == 0:
        return 1.0
    e = p[f[:, 1]] - p[f[:, 0]]
    return float(np.median(np.linalg.norm(e, axis=1)))


def decimate(mesh: dict, ratio: float | None = None, target_tris: int | None = None) -> dict:
    """UV-preserving quadric collapse. Ratio 1 is a copy."""
    src = mesh.get("source_path") or mesh.get("path")
    tris = int(mesh["tris"])
    if target_tris is not None:
        ratio = min(1.0, max(0.02, float(target_tris) / max(tris, 1)))
    ratio = 1.0 if ratio is None else float(np.clip(ratio, 0.02, 1.0))
    if ratio >= 0.999:
        return dict(mesh, ratio=1.0)
    ml = _pymeshlab()
    if ml is None or not src or not Path(src).is_file():
        return dict(mesh, ratio=ratio, notes="pymeshlab not installed; mesh unchanged")
    target = max(4, int(round(tris * ratio)))
    with _MESHLAB_LOCK:
        # Collapse the welded surface, not the UV patches. Plain quadric
        # collapse updates wedge UVs as it goes; the texture-preserving
        # variant treats every UV seam as a boundary and shrinks patches
        # independently, opening the cracks.
        ms = _welded_meshset(ml, src)
        ms.meshing_decimation_quadric_edge_collapse(
            targetfacenum=target, preservenormal=True, qualitythr=0.3,
            optimalplacement=True)
        v, f, n, uv, out_tris, out_verts = _extract_wedge_split(ms)
    return {
        "positions": v,
        "normals": n,
        "uvs": uv,
        "indices": np.ascontiguousarray(f.reshape(-1), np.uint32),
        "tris": out_tris,
        "verts": out_verts,
        "islands": count_uv_islands(uv, f),
        "path": src,
        "source_path": mesh.get("source_path", src),
        "source_tris": int(mesh.get("source_tris") or mesh.get("tris") or tris),
        "ratio": ratio,
    }


def texel_density(uv, positions, faces, tex_size: int) -> float:
    """Median texels per metre. Positions are centimetres (Unreal / Quixel)."""
    p = np.asarray(positions, np.float32) / 100.0
    f = _triangles(p, faces)
    t = _align_attr(uv, len(p), 0.0, 2)[:, :2]
    if len(f) == 0 or len(p) == 0:
        return 0.0
    tri_p = p[f]
    tri_t = t[f]
    world = np.linalg.norm(np.cross(tri_p[:, 1] - tri_p[:, 0], tri_p[:, 2] - tri_p[:, 0]), axis=1) * 0.5
    e1 = tri_t[:, 1] - tri_t[:, 0]
    e2 = tri_t[:, 2] - tri_t[:, 0]
    uv_area = np.abs(e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]) * 0.5
    valid = (world > 1e-10) & (uv_area > 1e-12)
    if not np.any(valid):
        return 0.0
    dens = (np.sqrt(uv_area[valid]) * float(tex_size)) / np.sqrt(world[valid])
    return float(np.median(dens))


def fit_to_budget(mat: dict, cap: dict) -> dict | None:
    mesh_path = mat.get("mesh_path") or find_mesh_file(mat.get("source_folder") or "")
    if not mesh_path:
        return None
    mesh = load_mesh(mesh_path)
    if mesh is None:
        return None
    out = decimate(mesh, target_tris=int(cap["tris"]))
    out["cap_tris"] = cap["tris"]
    out["texel_density"] = texel_density(
        out["uvs"], out["positions"], out["indices"], int(cap.get("tex") or 1024))
    return out
