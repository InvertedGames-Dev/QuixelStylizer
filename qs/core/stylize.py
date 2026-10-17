"""load / paint / finish / write. GPU paint when available, CPU otherwise."""
from __future__ import annotations

import os
import time

import cv2
import numpy as np

from qs.core.colour import colour_grade, linear_to_srgb, srgb_to_linear, smoothstep
from qs.core.filter import Cancelled, gblur, kuwahara_pass, seam_fade, simple_filter
from qs.core.io import imwrite, load_gray, load_rgb, resize, resolve_out_dir, to_u8, verify_written
from qs.core.maps import find_sets
from qs.core.normals import (
    curvature_from_normal,
    decode_normal,
    detect_normal_convention,
    encode_normal,
    normal_from_height,
    normalise_signed,
    normalize_normal,
)
from qs.core.packing import pack_maps
from qs.core.recipe import DEFAULTS, REF_SIZE, Recipe, json_settings, scaled
from qs.logging import log_line
from qs.version import TOOL_VERSION


def _mix_map(painted, source, t):
    t = float(np.clip(t, 0.0, 1.0))
    if t >= 1.0:
        return painted
    if t <= 0.0:
        return source
    return source * (1.0 - t) + painted * t


def apply_stylize_mask(stylized, source, mask, shape_hw):
    if mask is None:
        return stylized
    m = np.asarray(mask, np.float32)
    if m.ndim == 3:
        m = m[..., 0]
    H, W = shape_hw
    if m.shape != (H, W):
        m = resize(m, W, H)
    m = np.clip(m, 0, 1)
    if stylized.ndim == 2:
        return source * (1.0 - m) + stylized * m
    return source * (1.0 - m[..., None]) + stylized * m[..., None]


def load_stage(s, st):
    st = dict(DEFAULTS, **st)
    tile = bool(st["tile"])
    maps = s["maps"]
    report = {"defaulted": [], "notes": [], "used": {}}
    alb0 = load_rgb(maps["albedo"])
    ih, iw = alb0.shape[:2]
    size = int(st["size"])
    sc = size / float(max(iw, ih))
    W, H = max(int(round(iw * sc)), 1), max(int(round(ih * sc)), 1)
    alb_src = np.clip(resize(alb0, W, H), 0, 1)
    report["used"]["albedo"] = maps["albedo"]
    del alb0

    def g(typ):
        report["used"][typ] = maps[typ]
        return np.clip(resize(load_gray(maps[typ]), W, H), 0, 1)

    orm = None
    if "orm" in maps:
        orm = np.clip(resize(load_rgb(maps["orm"]), W, H), 0, 1)
        report["used"]["orm"] = maps["orm"]
    if "roughness" in maps:
        rough = g("roughness")
    elif "gloss" in maps:
        rough = 1.0 - g("gloss")
        report["notes"].append("roughness = 1 - gloss")
    elif orm is not None:
        rough = orm[..., 1].copy()
        report["notes"].append("roughness from ORM/ARM green")
    else:
        rough = np.full((H, W), 0.6, np.float32)
        report["defaulted"].append("roughness (0.6)")
    if "metallic" in maps:
        metal = g("metallic")
    elif orm is not None:
        metal = orm[..., 2].copy()
        report["notes"].append("metallic from ORM/ARM blue")
    else:
        metal = np.zeros((H, W), np.float32)
        report["defaulted"].append("metallic (0)")
    if "ao" in maps:
        ao = g("ao")
    elif orm is not None:
        ao = orm[..., 0].copy()
        report["notes"].append("AO from ORM/ARM red")
    else:
        ao = np.ones((H, W), np.float32)
        report["defaulted"].append("ao (1, white)")
    height_real = True
    if "height" in maps:
        height = g("height")
    elif "bump" in maps:
        height = g("bump")
        report["notes"].append("height taken from Bump map (no Displacement found)")
    else:
        height = np.full((H, W), 0.5, np.float32)
        height_real = False
        report["defaulted"].append("height (0.5 flat)")
    cavity = g("cavity") if "cavity" in maps else None
    opacity = g("opacity") if "opacity" in maps else None
    for t in ("specular", "translucency"):
        if t in maps:
            report["notes"].append(f"{t} map ignored: {os.path.basename(maps[t])}")

    conv = None
    if "normal_dx" in maps:
        nkey, conv = "normal_dx", "dx"
        if "normal_gl" in maps:
            report["notes"].append("both NormalDX and NormalGL found; using DirectX")
    elif "normal" in maps:
        nkey = "normal"
    elif "normal_gl" in maps:
        nkey, conv = "normal_gl", "gl"
    else:
        nkey = None
    if nkey:
        report["used"]["normal"] = maps[nkey]
        n_src = decode_normal(np.clip(resize(load_rgb(maps[nkey]), W, H), 0, 1))
        if st["normal_in"] in ("dx", "gl"):
            conv = st["normal_in"]
            report["notes"].append(f"normal convention forced: {conv}")
        elif conv is None:
            if height_real:
                cx, cy = detect_normal_convention(n_src, height, tile)
                if abs(cy) >= 0.05:
                    conv = "dx" if cy > 0 else "gl"
                    report["notes"].append(
                        f"normal convention auto-detected from height gradient: {conv.upper()} "
                        f"(corrY={cy:+.2f}, corrX={cx:+.2f})")
                else:
                    conv = "gl"
                    report["notes"].append(f"normal convention inconclusive (corrY={cy:+.2f}); assumed GL")
            else:
                conv = "gl"
                report["notes"].append("unsuffixed Normal with no height to check against; assumed OpenGL")
        else:
            report["notes"].append(f"normal convention from filename: {conv.upper()}")
        if conv == "gl":
            n_src[..., 1] *= -1.0
        if st["flip_green"]:
            n_src[..., 1] *= -1.0
            report["notes"].append("--flip-green applied to source normal")
    elif height_real:
        n_src = normal_from_height(height, 32.0 * max(W, H) / REF_SIZE, tile)
        report["notes"].append("no normal map: derived from height")
        report["defaulted"].append("normal (derived from height)")
    else:
        n_src = np.zeros((H, W, 3), np.float32)
        n_src[..., 2] = 1.0
        report["defaulted"].append("normal (flat)")
    report["normal_convention_in"] = conv
    return {"W": W, "H": H, "alb": alb_src, "rough": rough, "metal": metal, "ao": ao,
            "height": height, "height_real": height_real, "cavity": cavity,
            "opacity": opacity, "normal": n_src, "nkey": nkey, "conv": conv, "report": report,
            "name": s["name"], "folder": s["folder"]}


def paint_copy(src):
    cav = src["cavity"]
    return {
        "alb": np.array(src["alb"], np.float32, copy=True),
        "normal": np.array(src["normal"], np.float32, copy=True),
        "height": np.array(src["height"], np.float32, copy=True),
        "rough": np.array(src["rough"], np.float32, copy=True),
        "metal": np.array(src["metal"], np.float32, copy=True),
        "ao": np.array(src["ao"], np.float32, copy=True),
        "cavity": None if cav is None else np.array(cav, np.float32, copy=True),
        "r_px": 0,
    }


def paint_stage(src, st, cancel=None):
    st = dict(DEFAULTS, **st)
    if float(st.get("radius") or 0) <= 0:
        return paint_copy(src)
    tile = bool(st["tile"])
    W, H = src["W"], src["H"]
    r_px = max(1, int(round(scaled(st["radius"], max(W, H)))))
    passes = max(1, int(st["passes"]))
    cav = src["cavity"]
    guides = [src["normal"], src["height"], src["rough"], src["metal"], src["ao"]] + \
             ([cav] if cav is not None else [])
    alb = src["alb"].copy()
    method = st["filter"]
    island_ids = st.get("_island_ids")
    if island_ids is None and st.get("island_mask"):
        island_ids = np.round(load_gray(st["island_mask"]) * 255).astype(np.int32)
    if island_ids is not None:
        island_ids = np.asarray(island_ids)
        if island_ids.shape[:2] != (H, W):
            island_ids = np.round(resize(island_ids.astype(np.float32), W, H)).astype(np.int32)
    if island_ids is not None:
        from qs.core.filter import dilate_islands, kuwahara_pass_islands
        tile = False
    for _ in range(passes):
        if cancel and cancel():
            raise Cancelled()
        if method in ("kuwahara", "kuwahara_classic"):
            if island_ids is not None:
                alb, guides = kuwahara_pass_islands(
                    alb, guides, r_px, float(st["sharpness"]), island_ids,
                    classic=(method == "kuwahara_classic"), cancel=cancel)
            else:
                alb, guides = kuwahara_pass(
                    alb, guides, r_px, float(st["sharpness"]), tile,
                    classic=(method == "kuwahara_classic"), cancel=cancel)
        else:
            alb = simple_filter(alb, method, r_px, tile)
            guides = [simple_filter(gi, method, r_px, tile) for gi in guides]
    if island_ids is not None:
        from qs.core.filter import dilate_islands
        alb = dilate_islands(alb, island_ids, max(2, r_px // 2))
        guides = [dilate_islands(g, island_ids, max(2, r_px // 2)) for g in guides]
    return {"alb": alb, "normal": guides[0], "height": guides[1], "rough": guides[2],
            "metal": guides[3], "ao": guides[4], "cavity": guides[5] if cav is not None else None,
            "r_px": r_px}


def paint_dispatch(src, st, cancel=None):
    st = dict(DEFAULTS, **st)
    if float(st.get("radius") or 0) <= 0:
        return paint_copy(src)
    method = st["filter"]
    stroke = float(st.get("stroke") or 0)
    wants_stroke = stroke > 0 and method in ("kuwahara", "kuwahara_classic", "anisotropic")
    if method in ("kuwahara", "kuwahara_classic", "anisotropic"):
        try:
            from qs.core import gpu as qs_gpu
            if qs_gpu.available():
                painted = qs_gpu.paint(src, st, cancel)
                notes = src.setdefault("report", {}).setdefault("notes", [])
                note = "paint ran on the GPU (ModernGL compute)"
                if note not in notes:
                    notes.append(note)
                return painted
        except Exception as exc:
            log_line(f"GPU paint failed, using CPU: {exc}")
    notes = src.setdefault("report", {}).setdefault("notes", [])
    if wants_stroke or method == "anisotropic":
        st = dict(st, filter="kuwahara")
        note = "anisotropic/stroke needs the GPU path; this render used kuwahara"
        if note not in notes:
            notes.append(note)
    return paint_stage(src, st, cancel)


def pre_grade(src, painted, st):
    st = dict(DEFAULTS, **st)
    tile = bool(st["tile"])
    W, H = src["W"], src["H"]
    r_px = painted["r_px"]
    alb_p = painted["alb"]
    dr = float(st.get("detail_restore", 0) or 0)
    if dr:
        hp = src["alb"] - gblur(src["alb"], max(r_px * 0.35, 0.6), tile)
        alb_p = np.clip(alb_p + hp * dr, 0, 1)
    n_p = _mix_map(painted["normal"], src["normal"], st.get("mix_normal", 1))
    height_p = _mix_map(painted["height"], src["height"], st.get("mix_height", 1))
    rough_p = _mix_map(painted["rough"], src["rough"], st.get("mix_rough", 1))
    metal_p = _mix_map(painted["metal"], src["metal"], st.get("mix_metal", 1))
    ao_p = _mix_map(painted["ao"], src["ao"], st.get("mix_ao", 1))
    # Soften is 0–1 of an 8 px kernel at 1024, not tied to paint radius
    # (radius 0 used to make Normal soften a no-op).
    blur_px = scaled(8.0, max(W, H))
    height_f = gblur(height_p, float(st["height_soften"]) * blur_px, tile)
    n_f = n_p
    if st["normal_mode"] == "height" and src["height_real"]:
        n_f = normal_from_height(height_f, 32.0 * max(W, H) / REF_SIZE, tile)
    n_f = np.array(gblur(n_f, float(st["normal_soften"]) * blur_px, tile), np.float32)
    n_f[..., :2] *= float(st["normal_strength"])
    n_f = normalize_normal(n_f)

    rm = float(rough_p.mean())
    rough_o = np.clip(rm + (rough_p - rm) * (1 - float(st["roughness_flatten"]))
                      + float(st["roughness_bias"]), 0, 1)
    mb = float(st["metal_binarize"])
    metal_o = np.clip(metal_p * (1 - mb) + smoothstep(0.35, 0.65, metal_p) * mb, 0, 1)
    ao_o = np.clip(ao_p, 0, 1)

    cav_f = painted["cavity"]
    cs = st["curvature_source"]
    if cs == "auto":
        cs = "cavity" if cav_f is not None else ("normal" if src["nkey"] or src["height_real"] else "none")
    if cs == "cavity" and cav_f is None:
        cs = "normal"
    es = float(st["edge_strength"])
    if es:
        if cs == "cavity":
            curv = normalise_signed(cav_f)
        elif cs == "height" and src["height_real"]:
            curv = normalise_signed(-cv2.Laplacian(gblur(height_f, 1.0, tile), cv2.CV_32F))
        elif cs in ("normal", "height"):
            cs = "normal"
            curv = normalise_signed(curvature_from_normal(n_f, tile))
        else:
            curv = np.zeros((H, W), np.float32)
        curv = gblur(curv, scaled(st["edge_blur"], max(W, H)), tile)
    else:
        curv = None
    return {
        "alb_p": np.clip(alb_p, 0, 1).astype(np.float32),
        "normal": n_f, "height": np.clip(height_f, 0, 1).astype(np.float32),
        "rough": np.clip(rough_o, 0, 1).astype(np.float32),
        "metal": np.clip(metal_o, 0, 1).astype(np.float32),
        "ao": np.clip(ao_o, 0, 1).astype(np.float32),
        "curv": curv, "curvature_source": cs, "tile": tile,
    }


def _bake_albedo(alb, ao, curv, es, aos):
    if aos:
        alb = linear_to_srgb(srgb_to_linear(alb) * (1 - aos + aos * ao)[..., None])
    if es and curv is not None:
        c = curv[..., None] * es
        alb = np.where(c > 0, alb + (1 - alb) * c * 0.6, alb * (1 + c * 0.75))
    return np.clip(alb, 0, 1).astype(np.float32)


def finish_stage(src, painted, st):
    st = dict(DEFAULTS, **st)
    prep = pre_grade(src, painted, st)
    tile = prep["tile"]
    alb = colour_grade(prep["alb_p"], st)
    alb = _bake_albedo(alb, prep["ao"], prep["curv"], float(st["edge_strength"]), float(st["ao_strength"]))
    n_f = prep["normal"]
    height_f = prep["height"]
    rough_o = prep["rough"]
    metal_o = prep["metal"]
    ao_o = prep["ao"]
    fade = float(st.get("seam_fade", 0) or 0)
    alb = seam_fade(alb, fade, tile)
    n_f = seam_fade(n_f, fade, tile)
    height_f = seam_fade(np.clip(height_f, 0, 1), fade, tile)
    rough_o = seam_fade(rough_o, fade, tile)
    metal_o = seam_fade(metal_o, fade, tile)
    ao_o = seam_fade(ao_o, fade, tile)
    n_f = normalize_normal(np.array(n_f, np.float32))
    mask = st.get("_mask")
    if mask is not None:
        shape = (src["H"], src["W"])
        alb = apply_stylize_mask(alb, src["alb"], mask, shape)
        n_f = normalize_normal(apply_stylize_mask(n_f, src["normal"], mask, shape))
        height_f = apply_stylize_mask(height_f, src["height"], mask, shape)
        rough_o = apply_stylize_mask(rough_o, src["rough"], mask, shape)
        metal_o = apply_stylize_mask(metal_o, src["metal"], mask, shape)
        ao_o = apply_stylize_mask(ao_o, src["ao"], mask, shape)
    return {"alb": np.clip(alb, 0, 1).astype(np.float32), "normal": n_f,
            "height": np.clip(height_f, 0, 1).astype(np.float32),
            "rough": np.clip(rough_o, 0, 1).astype(np.float32),
            "metal": np.clip(metal_o, 0, 1).astype(np.float32),
            "ao": np.clip(ao_o, 0, 1).astype(np.float32),
            "curvature_source": prep["curvature_source"]}


def finish_dispatch(src, painted, st):
    try:
        from qs.core import grade_gpu
        fin = grade_gpu.finish_readback(src, painted, st)
        if fin is not None:
            notes = src.setdefault("report", {}).setdefault("notes", [])
            note = "finish ran on the GPU"
            if note not in notes:
                notes.append(note)
            return fin
    except Exception as exc:
        log_line(f"GPU finish failed, using CPU: {exc}")
    return finish_stage(src, painted, st)


def stylize(src, st, cancel=None):
    """Paint then finish. `src` is a load_stage dict; `st` is a recipe dict."""
    painted = paint_dispatch(src, st, cancel)
    return finish_dispatch(src, painted, st)


def process_set(s, st, write=True, out_dir=None, log=print, write_meta=True):
    t0 = time.time()
    st = dict(DEFAULTS, **st)
    src = load_stage(s, st)
    painted = paint_dispatch(src, st)
    fin = finish_dispatch(src, painted, st)
    report = dict(src["report"])
    report["curvature_source"] = cs = fin["curvature_source"]
    W, H, name, conv, r_px = src["W"], src["H"], s["name"], src["conv"], painted["r_px"]
    n_out = fin["normal"].copy()
    if st["normal_out"] == "gl":
        n_out[..., 1] *= -1
    packed = pack_maps({**fin, "normal": n_out}, src["height_real"], st.get("packing", "mrod"))
    D = packed["D"]
    key = packed["packed_key"]
    packed_img = packed[key]
    N = packed["N"]
    result = {
        "name": name, "albedo_before": src["alb"], "albedo_after": fin["alb"],
        "D": D, key: packed_img, "N": N,
        "normal_before": src["normal"], "report": report, "size": (W, H), "radius_px": r_px,
        "packing": packed["packing"],
    }
    opacity = src["opacity"]
    if write:
        out_dir = resolve_out_dir(s["folder"], out_dir)
        os.makedirs(out_dir, exist_ok=True)
        files, checked = {}, {}
        ext = ".tga" if str(st.get("format", "png")).lower() == "tga" else ".png"
        for k, arr, ch in (("D", D, 3), (key, packed_img, packed_img.shape[2]), ("N", N, 3)):
            files[k] = os.path.join(out_dir, f"{name}_{k}{ext}")
            imwrite(files[k], arr)
            checked[k] = verify_written(files[k], ch, (W, H))
        if opacity is not None:
            files["Opacity"] = os.path.join(out_dir, f"{name}_Opacity{ext}")
            imwrite(files["Opacity"], to_u8(opacity))
            checked["Opacity"] = verify_written(files["Opacity"], 1, (W, H))
        result["checked"] = checked
        log_line(f"wrote {name} v{TOOL_VERSION} size={st['size']} -> {out_dir}")
        if write_meta:
            import json
            files["json"] = os.path.join(out_dir, f"{name}_stylize.json")
            meta = {
                "tool": "QuixelStylizer", "version": TOOL_VERSION,
                "created": time.strftime("%Y-%m-%d %H:%M:%S"), "set_name": name,
                "source_folder": s["folder"], "output_size": [W, H], "radius_px": r_px,
                "files_checked": checked, "settings": json_settings(st),
                "inputs": report["used"], "defaulted": report["defaulted"],
                "notes": report["notes"], "normal_convention_in": conv,
                "normal_convention_out": st["normal_out"], "curvature_source": cs,
                "outputs": files, "packing": packed["packing"],
            }
            with open(files["json"], "w", encoding="utf-8") as fh:
                json.dump(meta, fh, indent=2)
        result["files"] = files
    result["seconds"] = time.time() - t0
    return result


def load_preset(path):
    import json
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    settings = data.get("settings", data)
    return Recipe.from_dict(settings).to_dict()
