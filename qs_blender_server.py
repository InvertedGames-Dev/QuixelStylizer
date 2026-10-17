"""Warm headless Blender. JSON-lines on 127.0.0.1. Started as `blender -b --python`."""
from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import traceback

DEFAULT_PORT = 18765
# Bump when the job protocol changes. The client restarts a warm server
# that reports anything else, so new bake/re-UV code cannot be shadowed
# by a Blender process started by an older build.
SERVER_VERSION = "rewrap-bake-6"


def _import_any(path: str, clear: bool = True):
    import bpy
    ext = os.path.splitext(path)[1].lower()
    if clear:
        bpy.ops.wm.read_factory_settings(use_empty=True)
    if ext in (".glb", ".gltf"):
        bpy.ops.import_scene.gltf(filepath=path)
    elif ext == ".fbx":
        bpy.ops.import_scene.fbx(filepath=path)
    elif ext == ".obj":
        bpy.ops.wm.obj_import(filepath=path)
    else:
        raise ValueError(f"unsupported mesh: {ext}")


def _meshes():
    import bpy
    return [o for o in bpy.context.scene.objects if o.type == "MESH"]


def op_convert(job: dict) -> dict:
    import bpy
    src, dest = job["src"], job["dest"]
    _import_any(src)
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    bpy.ops.export_scene.gltf(filepath=dest, export_format="GLB" if dest.lower().endswith(".glb") else "GLTF_SEPARATE")
    return {"dest": dest}


def op_reuv(job: dict) -> dict:
    import bpy
    src, dest = job["src"], job["dest"]
    method = job.get("method") or "smart"
    _import_any(src)
    bpy.ops.object.select_all(action="DESELECT")
    for obj in _meshes():
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.select_all(action="SELECT")
        if method == "smart":
            bpy.ops.uv.smart_project(angle_limit=1.15192, island_margin=0.02)
        else:
            bpy.ops.uv.unwrap(method="ANGLE_BASED", margin=0.02)
        bpy.ops.object.mode_set(mode="OBJECT")
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    bpy.ops.export_scene.gltf(filepath=dest, export_format="GLB" if dest.lower().endswith(".glb") else "GLTF_SEPARATE")
    return {"dest": dest}


_BAKE_SUFFIXES = {
    "albedo": "BaseColor",
    "normal": "Normal",
    "normal_dx": "Normal",
    "normal_gl": "Normal",
    "roughness": "Roughness",
    "metallic": "Metallic",
    "ao": "AO",
    "height": "Displacement",
    "bump": "Displacement",
}


def _emission_material(name: str, image_path: str, srgb: bool):
    """A material that simply emits one source map through the high mesh's UVs.

    Baking EMIT selected-to-active is Blender's texture-transfer path: the
    low mesh asks, per point on its own surface, what the high mesh emits
    there, and writes the answer into the low mesh's UV layout.
    """
    import bpy
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    nodes.clear()
    out = nodes.new("ShaderNodeOutputMaterial")
    emit = nodes.new("ShaderNodeEmission")
    tex = nodes.new("ShaderNodeTexImage")
    tex.image = bpy.data.images.load(image_path)
    try:
        tex.image.colorspace_settings.name = "sRGB" if srgb else "Non-Color"
    except Exception:
        pass
    mat.node_tree.links.new(tex.outputs["Color"], emit.inputs["Color"])
    mat.node_tree.links.new(emit.outputs["Emission"], out.inputs["Surface"])
    return mat


def op_bake(job: dict) -> dict:
    import bpy
    high, low, dest = job["high"], job["low"], job["dest"]
    maps = job.get("maps") or {}
    maps = {k: v for k, v in maps.items() if k in _BAKE_SUFFIXES and v and os.path.isfile(v)}
    if not maps:
        raise ValueError("bake needs at least one source map to transfer")
    if os.path.abspath(high) == os.path.abspath(low):
        raise ValueError("bake needs two different meshes: the original and the re-UV mesh")
    cage = float(job.get("cage_offset") or 0.02)
    os.makedirs(dest, exist_ok=True)
    _import_any(high)
    high_objs = list(_meshes())
    for o in high_objs:
        o.name = "QS_HIGH_" + o.name
    # The second import must not reset the scene, or the high mesh is gone
    # before the bake starts (the old behaviour, and why Bake did nothing).
    _import_any(low, clear=False)
    low_objs = [o for o in _meshes() if not o.name.startswith("QS_HIGH_")]
    if not high_objs or not low_objs:
        raise ValueError(f"bake needs a mesh in both files (high={len(high_objs)}, low={len(low_objs)})")

    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.samples = 8
    scene.cycles.use_denoising = False
    bake = scene.render.bake
    bake.use_selected_to_active = True
    bake.use_cage = True
    bake.cage_extrusion = cage
    bake.margin = 16
    try:
        bake.margin_type = "EXTEND"
    except Exception:
        pass

    low_mat = bpy.data.materials.new("QS_BakeTarget")
    low_mat.use_nodes = True
    target_node = low_mat.node_tree.nodes.new("ShaderNodeTexImage")
    low_mat.node_tree.nodes.active = target_node
    target_node.select = True
    for obj in low_objs:
        obj.data.materials.clear()
        obj.data.materials.append(low_mat)

    base = "".join(c if c.isalnum() or c in "-_" else "_" for c in str(job.get("base_name") or "Baked"))
    size = max(64, int(job.get("size") or 1024))
    written = {}
    for kind, src_path in maps.items():
        emit_mat = _emission_material(f"QS_SRC_{kind}", src_path, srgb=(kind == "albedo"))
        for obj in high_objs:
            obj.data.materials.clear()
            obj.data.materials.append(emit_mat)
        img = bpy.data.images.new(
            f"QS_{kind}", width=size, height=size, alpha=False,
            float_buffer=(kind == "height"))
        target_node.image = img
        for obj in low_objs:
            bpy.ops.object.select_all(action="DESELECT")
            for h in high_objs:
                h.select_set(True)
            obj.select_set(True)
            bpy.context.view_layer.objects.active = obj
            bpy.ops.object.mode_set(mode="OBJECT")
            bpy.ops.object.bake(type="EMIT", use_selected_to_active=True, cage_extrusion=cage)
        path = os.path.join(dest, f"{base}_1K_{_BAKE_SUFFIXES[kind]}.png")
        img.filepath_raw = path
        img.file_format = "PNG"
        img.save()
        written[kind] = path
    return {"files": written, "dest": dest}


_NORMAL_KINDS = ("normal", "normal_dx", "normal_gl")


def _unwrap_low(low_objs, method: str = "smart"):
    import bpy
    for obj in low_objs:
        bpy.ops.object.select_all(action="DESELECT")
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.select_all(action="SELECT")
        if method == "smart":
            bpy.ops.uv.smart_project(angle_limit=1.15192, island_margin=0.02)
        else:
            bpy.ops.uv.unwrap(method="ANGLE_BASED", margin=0.02)
        bpy.ops.object.mode_set(mode="OBJECT")


def _world_bbox(objs):
    import bpy
    from mathutils import Vector
    bpy.context.view_layer.update()
    corners = []
    for obj in objs:
        corners.extend(obj.matrix_world @ Vector(c) for c in obj.bound_box)
    lo = Vector((min(c.x for c in corners), min(c.y for c in corners), min(c.z for c in corners)))
    hi = Vector((max(c.x for c in corners), max(c.y for c in corners), max(c.z for c in corners)))
    return (lo + hi) * 0.5, (hi - lo).length


def _align_low_to_high(high_objs, low_objs):
    """Overlay the low mesh on the hi-poly mesh before baking.

    The two files arrive through different importers: a Quixel GLB carries
    a 0.01 node scale, while the serialized low OBJ arrives at raw size.
    Selected-to-active rays would never meet across that gap. The low mesh
    is a decimated copy of the same geometry, so matching world bounding
    box centre and diagonal overlays it on the hi-poly mesh.
    """
    import bpy
    from mathutils import Vector
    high_centre, high_diag = _world_bbox(high_objs)
    _low_centre, low_diag = _world_bbox(low_objs)
    if low_diag <= 0 or high_diag <= 0:
        return 1.0, Vector((0.0, 0.0, 0.0))
    factor = high_diag / low_diag
    if abs(factor - 1.0) > 1e-6:
        for obj in low_objs:
            obj.scale = obj.scale * factor
    bpy.context.view_layer.update()
    low_centre, _ = _world_bbox(low_objs)
    delta = high_centre - low_centre
    if delta.length > 1e-9:
        for obj in low_objs:
            obj.location = obj.location + delta
    bpy.context.view_layer.update()
    return factor, delta


def _surface_gap(high_objs, low_objs, limit=400):
    """Median distance from low-mesh vertices to the hi-poly surface."""
    import bpy
    from mathutils.bvhtree import BVHTree
    dg = bpy.context.evaluated_depsgraph_get()
    trees = [(BVHTree.FromObject(h, dg), h.matrix_world, h.matrix_world.inverted())
             for h in high_objs]
    dists = []
    for obj in low_objs:
        verts = list(obj.data.vertices)
        step = max(1, len(verts) // limit)
        mw = obj.matrix_world
        for v in verts[::step]:
            world = mw @ v.co
            best = None
            for tree, hmw, hinv in trees:
                loc, _n, _i, _d = tree.find_nearest(hinv @ world)
                if loc is not None:
                    d = (hmw @ loc - world).length
                    best = d if best is None else min(best, d)
            if best is not None:
                dists.append(best)
    if not dists:
        return float("inf")
    dists.sort()
    return dists[len(dists) // 2]


def _orient_low_to_high(high_objs, low_objs):
    """Align the low mesh onto the hi-poly and prove that it sits there.

    The hi-poly (glTF) and the serialized low mesh (OBJ) arrive through
    different importers whose world orientations differ by a quarter
    turn about X, which a bounding-box alignment cannot see: the boxes
    coincide while the surfaces do not, and the bake then samples a
    rotated copy. So after aligning, measure the actual surface gap,
    try the other quarter turns, and keep whichever orientation truly
    overlays the hi-poly. Returns (factor, delta, quarter_turns_degrees)
    so the export can undo the same rotation.
    """
    import bpy
    from mathutils import Matrix
    factor, delta = _align_low_to_high(high_objs, low_objs)
    best_gap = _surface_gap(high_objs, low_objs)
    best_theta = 0
    for theta in (90.0, -90.0, 180.0):
        rot = Matrix.Rotation(theta * 3.141592653589793 / 180.0, 4, "X")
        for obj in low_objs:
            obj.matrix_world = rot @ obj.matrix_world
        bpy.context.view_layer.update()
        f2, d2 = _align_low_to_high(high_objs, low_objs)
        gap = _surface_gap(high_objs, low_objs)
        if gap < best_gap:
            best_gap, factor, delta = gap, f2, d2
            best_theta += theta
        else:
            back = Matrix.Rotation(-theta * 3.141592653589793 / 180.0, 4, "X")
            for obj in low_objs:
                obj.matrix_world = back @ obj.matrix_world
            bpy.context.view_layer.update()
            factor, delta = _align_low_to_high(high_objs, low_objs)
    return factor, delta, best_theta


def _restore_low_frame(low_objs, import_mats):
    """Put the baked low mesh back into the caller's coordinate frame.

    The caller (PyMeshLab) reads a GLB's raw vertex data and ignores node
    transforms, and the source Quixel GLBs keep their rotation and 0.01
    scale on the node. So the exported file must carry raw vertices in
    the frame the low OBJ arrived in, under an identity node, or the
    mesh comes back rotated and shrunk. Rather than bookkeeping the
    alignment's scale/rotation steps (which drift as soon as an
    orientation trial re-runs the alignment), this puts each low object
    back under the exact matrix it was imported with and bakes that
    into the vertices: the glTF exporter's own axis conversion is that
    import transform's inverse, so the raw data lands exactly on the
    frame the OBJ arrived in.
    """
    import bpy
    bpy.ops.object.select_all(action="DESELECT")
    for obj in low_objs:
        obj.select_set(True)
    for obj in low_objs:
        start = import_mats.get(obj.name)
        if start is not None:
            obj.matrix_world = start.copy()
    bpy.context.view_layer.update()
    if low_objs:
        bpy.context.view_layer.objects.active = low_objs[0]
        bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    bpy.context.view_layer.update()


def _make_cages(high_objs, low_objs):
    """Build an inflated cage copy of each low mesh for the bake.

    Ticking "use cage" without a cage object changes nothing in Blender,
    and rays that start on the low surface miss the hi-poly wherever a
    decimated face chords across a curve or a socket — those texels bake
    black. A cage pushed out along the low mesh's normals, far enough to
    enclose the hi-poly surface, makes the inward rays land everywhere.
    The push is measured, not guessed: the farthest low vertex from the
    hi-poly surface, plus a quarter again.
    """
    import bpy
    from mathutils.bvhtree import BVHTree
    dg = bpy.context.evaluated_depsgraph_get()
    trees = []
    for h in high_objs:
        trees.append((BVHTree.FromObject(h, dg), h.matrix_world,
                      h.matrix_world.inverted(), h.matrix_world.to_3x3().inverted()))
    gap = 0.0
    for obj in low_objs:
        mw = obj.matrix_world
        for v in obj.data.vertices:
            world = mw @ v.co
            best = None
            for tree, hmw, hinv, _r in trees:
                loc, _n, _i, _d = tree.find_nearest(hinv @ world)
                if loc is not None:
                    d = (hmw @ loc - world).length
                    best = d if best is None else min(best, d)
            if best is not None:
                gap = max(gap, best)
    push = gap * 1.25 + 1e-4
    cages = {}
    for obj in low_objs:
        cage = obj.copy()
        cage.data = obj.data.copy()
        cage.name = "QS_CAGE_" + obj.name
        bpy.context.scene.collection.objects.link(cage)
        mesh = cage.data
        mesh.update()
        for v in mesh.vertices:
            v.co = v.co + v.normal * push
        mesh.update()
        cages[obj.name] = cage
    return cages


def op_rewrap_bake(job: dict) -> dict:
    """Unwrap the current low-poly mesh fresh, then bake hi-poly onto it.

    One job, one result: the low mesh is smart-projected, the colour /
    roughness / metallic / AO / height maps are transferred from the
    hi-poly mesh by EMIT selected-to-active, and the normal map is a real
    tangent-space NORMAL bake of the hi-poly geometry (an EMIT transfer of
    the old tangent-space normal map would be meaningless on new UVs).
    The unwrapped low mesh is exported beside the baked maps, so the caller
    only ever displays the low mesh together with maps made for its UVs.
    """
    import bpy
    high, low, dest = job["high"], job["low"], job["dest"]
    dest_mesh = job.get("dest_mesh") or os.path.join(dest, "low.glb")
    maps = job.get("maps") or {}
    maps = {k: v for k, v in maps.items() if k in _BAKE_SUFFIXES and v and os.path.isfile(v)}
    if not maps:
        raise ValueError("rewrap bake needs at least one source map to transfer")
    if os.path.abspath(high) == os.path.abspath(low):
        raise ValueError("rewrap bake needs the hi-poly original and the low-poly mesh")
    cage = float(job.get("cage_offset") or 0.02)
    os.makedirs(dest, exist_ok=True)
    os.makedirs(os.path.dirname(dest_mesh) or ".", exist_ok=True)

    _import_any(high)
    high_objs = list(_meshes())
    for o in high_objs:
        o.name = "QS_HIGH_" + o.name
    _import_any(low, clear=False)
    low_objs = [o for o in _meshes() if not o.name.startswith("QS_HIGH_")]
    if not high_objs or not low_objs:
        raise ValueError(f"rewrap bake needs a mesh in both files (high={len(high_objs)}, low={len(low_objs)})")

    import_mats = {o.name: o.matrix_world.copy() for o in low_objs}
    _orient_low_to_high(high_objs, low_objs)
    _unwrap_low(low_objs, job.get("method") or "smart")
    cages = _make_cages(high_objs, low_objs)

    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.samples = 8
    scene.cycles.use_denoising = False
    bake_settings = scene.render.bake
    bake_settings.use_selected_to_active = True
    bake_settings.use_cage = True
    bake_settings.cage_extrusion = cage
    bake_settings.margin = 16
    try:
        bake_settings.margin_type = "EXTEND"
    except Exception:
        pass
    try:
        bake_settings.normal_space = "TANGENT"
    except Exception:
        pass

    low_mat = bpy.data.materials.new("QS_BakeTarget")
    low_mat.use_nodes = True
    target_node = low_mat.node_tree.nodes.new("ShaderNodeTexImage")
    low_mat.node_tree.nodes.active = target_node
    target_node.select = True
    for obj in low_objs:
        obj.data.materials.clear()
        obj.data.materials.append(low_mat)

    base = "".join(c if c.isalnum() or c in "-_" else "_" for c in str(job.get("base_name") or "Baked"))
    size = max(64, int(job.get("size") or 1024))
    written = {}

    def run_bake(kind, bake_type):
        img = bpy.data.images.new(
            f"QS_{kind}", width=size, height=size, alpha=False,
            float_buffer=(kind == "height"))
        target_node.image = img
        for obj in low_objs:
            bake_settings.cage_object = cages.get(obj.name)
            bpy.ops.object.select_all(action="DESELECT")
            for h in high_objs:
                h.select_set(True)
            obj.select_set(True)
            bpy.context.view_layer.objects.active = obj
            bpy.ops.object.mode_set(mode="OBJECT")
            bpy.ops.object.bake(type=bake_type, use_selected_to_active=True, cage_extrusion=cage)
        path = os.path.join(dest, f"{base}_1K_{_BAKE_SUFFIXES[kind]}.png")
        img.filepath_raw = path
        img.file_format = "PNG"
        img.save()
        written["normal" if kind in _NORMAL_KINDS else kind] = path

    # Geometry normals first: a true NORMAL bake from hi-poly to low-poly.
    run_bake("normal", "NORMAL")
    # Then EMIT texture transfer for every other source map. The old
    # tangent-space normal map is deliberately not transferred as a normal.
    for kind, src_path in maps.items():
        if kind in _NORMAL_KINDS:
            continue
        emit_mat = _emission_material(f"QS_SRC_{kind}", src_path, srgb=(kind == "albedo"))
        for obj in high_objs:
            obj.data.materials.clear()
            obj.data.materials.append(emit_mat)
        run_bake(kind, "EMIT")

    # Export only the unwrapped low mesh, with its new UVs, back in the
    # caller's coordinate frame (identity node, raw vertices as loaded).
    _restore_low_frame(low_objs, import_mats)
    bpy.ops.object.select_all(action="DESELECT")
    for obj in low_objs:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = low_objs[0]
    bpy.ops.export_scene.gltf(
        filepath=dest_mesh, export_format="GLB" if dest_mesh.lower().endswith(".glb") else "GLTF_SEPARATE",
        use_selection=True)
    tris = sum(len(o.data.polygons) for o in low_objs)
    return {"files": written, "dest": dest, "mesh": dest_mesh, "tris": tris}


def op_export_fbx(job: dict) -> dict:
    import bpy
    src, dest = job["src"], job["dest"]
    _import_any(src)
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    def _fbx(path):
        bpy.ops.export_scene.fbx(
            filepath=path,
            use_selection=False,
            apply_scale_options="FBX_SCALE_ALL",
            mesh_smooth_type="FACE",
            add_leaf_bones=False,
        )
    bpy.ops.export_scene.fbx(
        filepath=dest,
        use_selection=False,
        apply_scale_options="FBX_SCALE_ALL",
        mesh_smooth_type="FACE",
        add_leaf_bones=False,
    )
    lods = job.get("lods") or []
    extra = []
    for i, ratio in enumerate(lods):
        mods = []
        for obj in _meshes():
            mod = obj.modifiers.new("QS_DEC", "DECIMATE")
            mod.ratio = max(0.05, min(1.0, float(ratio)))
            mods.append((obj, mod))
        lod_path = dest[:-4] + f"_LOD{i + 1}.fbx" if dest.lower().endswith(".fbx") else dest + f"_LOD{i + 1}.fbx"
        _fbx(lod_path)
        extra.append(lod_path)
        for obj, mod in mods:
            obj.modifiers.remove(mod)
    return {"dest": dest, "lods": extra}


def op_render(job: dict) -> dict:
    import bpy
    import math
    src, dest = job["src"], job["dest"]
    _import_any(src)
    scene = bpy.context.scene
    try:
        scene.render.engine = "BLENDER_EEVEE_NEXT"
    except Exception:
        scene.render.engine = "BLENDER_EEVEE"
    scene.render.resolution_x = int(job.get("width") or 512)
    scene.render.resolution_y = int(job.get("height") or 512)
    scene.render.filepath = dest
    scene.render.image_settings.file_format = "PNG"
    cam = bpy.data.cameras.new("QS_Cam")
    cam_obj = bpy.data.objects.new("QS_Cam", cam)
    scene.collection.objects.link(cam_obj)
    scene.camera = cam_obj
    cam_obj.location = (2.4, -2.4, 1.6)
    cam_obj.rotation_euler = (math.radians(70), 0, math.radians(45))
    light_data = bpy.data.lights.new("QS_Key", "SUN")
    light_data.energy = 4.0
    light = bpy.data.objects.new("QS_Key", light_data)
    scene.collection.objects.link(light)
    light.rotation_euler = (math.radians(50), 0, math.radians(30))
    alb = job.get("alb")
    if alb and os.path.isfile(alb):
        img = bpy.data.images.load(alb)
        for obj in _meshes():
            mat = bpy.data.materials.new("QS_Ref")
            mat.use_nodes = True
            nt = mat.node_tree
            tex = nt.nodes.new("ShaderNodeTexImage")
            tex.image = img
            bsdf = nt.nodes.get("Principled BSDF")
            if bsdf:
                nt.links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
            obj.data.materials.clear()
            obj.data.materials.append(mat)
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    bpy.ops.render.render(write_still=True)
    return {"dest": dest}


OPS = {
    "convert": op_convert,
    "reuv": op_reuv,
    "bake": op_bake,
    "rewrap_bake": op_rewrap_bake,
    "export_fbx": op_export_fbx,
    "render": op_render,
}


def handle_job(job: dict) -> dict:
    op = job.get("op")
    if op == "quit":
        return {"ok": True, "result": {"bye": True}}
    if op == "ping":
        return {"ok": True, "result": {"pong": True, "version": SERVER_VERSION}}
    fn = OPS.get(op)
    if fn is None:
        return {"ok": False, "error": f"unknown op {op}"}
    try:
        return {"ok": True, "result": fn(job)}
    except Exception as exc:
        return {"ok": False, "error": f"{exc}\n{traceback.format_exc()}"}


def serve(port: int = DEFAULT_PORT):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("127.0.0.1", port))
    sock.listen(8)
    while True:
        conn, _addr = sock.accept()
        with conn:
            buf = b""
            while b"\n" not in buf:
                chunk = conn.recv(65536)
                if not chunk:
                    break
                buf += chunk
            if not buf.strip():
                continue
            job = {}
            try:
                job = json.loads(buf.decode("utf-8"))
                reply = handle_job(job)
            except json.JSONDecodeError as exc:
                reply = {"ok": False, "error": str(exc)}
            conn.sendall((json.dumps(reply) + "\n").encode("utf-8"))
            if job.get("op") == "quit":
                return


def main(argv=None):
    argv = list(sys.argv if argv is None else argv)
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = parser.parse_args(argv)
    serve(args.port)


if __name__ == "__main__":
    main()
