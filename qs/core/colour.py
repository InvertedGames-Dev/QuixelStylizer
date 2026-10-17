"""Oklab grade, posterize, palette snap. Ported from v1.5 colour_grade."""
from __future__ import annotations

import numpy as np

_OK_M1 = np.array([
    [0.4122214708, 0.5363325363, 0.0514459929],
    [0.2119034982, 0.6806995451, 0.1073969566],
    [0.0883024619, 0.2817188376, 0.6299787005],
], np.float64)
_OK_M2 = np.array([
    [0.2104542553, 0.7936177850, -0.0040720468],
    [1.9779984951, -2.4285922050, 0.4505937099],
    [0.0259040371, 0.7827717662, -0.8086757660],
], np.float64)
_OK_M2_INV = np.linalg.inv(_OK_M2)
_OK_M1_INV = np.linalg.inv(_OK_M1)


def luma(rgb):
    return rgb[..., 0] * 0.2126 + rgb[..., 1] * 0.7152 + rgb[..., 2] * 0.0722


def srgb_to_linear(c):
    c = np.clip(c, 0, 1)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4).astype(np.float32)


def linear_to_srgb(c):
    c = np.clip(c, 0, 1)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * np.power(c, 1 / 2.4) - 0.055).astype(np.float32)


def smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0, 1)
    return t * t * (3 - 2 * t)


def srgb_to_oklab(rgb):
    lin = srgb_to_linear(np.asarray(rgb, np.float32)).astype(np.float64)
    lms = np.matmul(lin, _OK_M1.T)
    np.maximum(lms, 0, out=lms)
    lms = np.cbrt(lms)
    return np.matmul(lms, _OK_M2.T).astype(np.float32)


def oklab_to_srgb(lab):
    lab64 = np.asarray(lab, np.float64)
    lms = np.matmul(lab64, _OK_M2_INV.T)
    lin = np.matmul(lms * lms * lms, _OK_M1_INV.T)
    return linear_to_srgb(lin.astype(np.float32))


def _as_rgb_list(value, fallback):
    try:
        rgb = [float(value[0]), float(value[1]), float(value[2])]
    except (TypeError, ValueError, IndexError):
        rgb = list(fallback)
    return np.clip(np.array(rgb, np.float32), 0, 1)


def _posterize_quantize(lab, levels):
    """Step lightness only. Quantizing chroma against a 0.4 ceiling floored typical
    material C (often < 0.05) to zero, which is why Posterize looked grey."""
    lab = lab.copy()
    L = np.clip(lab[..., 0], 0, 1)
    lab[..., 0] = np.floor(L * levels) / float(levels)
    return lab


def copy_overrides(value):
    out = []
    for item in list(value or []):
        if item is None or item == []:
            out.append(None)
            continue
        try:
            out.append([float(item[0]), float(item[1]), float(item[2])])
        except (TypeError, ValueError, IndexError):
            out.append(None)
    return out


def _generated_ramp(levels, root_srgb):
    root = srgb_to_oklab(_as_rgb_list(root_srgb, (0.5, 0.5, 0.5)))
    lights = np.linspace(0.12, 0.92, int(levels)).astype(np.float32)
    anchor = int(np.argmin(np.abs(lights - float(root[0]))))
    lights = lights.copy()
    lights[anchor] = float(root[0])
    ramp = np.zeros((int(levels), 3), np.float32)
    root_l = max(float(root[0]), 1e-3)
    for i, light in enumerate(lights):
        scale = 1.0 if light >= root_l else max(0.25, float(light) / root_l)
        ramp[i, 0] = light
        ramp[i, 1] = float(root[1]) * scale
        ramp[i, 2] = float(root[2]) * scale
    ramp[anchor] = root
    return ramp, lights


def level_ramp(st):
    lv = int(st.get("posterize") or 0)
    root = st.get("gradient_root") or []
    if not (lv > 1 and len(root) >= 3 and bool(st.get("gradient", True))):
        return None
    generated, lights = _generated_ramp(lv, root)
    labs = generated.copy()
    overrides = copy_overrides(st.get("level_overrides"))
    edited = []
    for i in range(lv):
        ov = overrides[i] if i < len(overrides) else None
        edited.append(ov is not None)
        if ov is not None:
            labs[i] = srgb_to_oklab(_as_rgb_list(ov, (0.5, 0.5, 0.5)))
    swatches = np.clip(oklab_to_srgb(labs), 0, 1).astype(np.float32)
    return {"labs": labs, "lights": lights, "swatches": swatches, "edited": edited}


def _posterize_gradient(lab, levels, root_srgb, overrides=None):
    got = level_ramp({
        "posterize": int(levels), "gradient": True, "gradient_root": root_srgb,
        "level_overrides": overrides or [],
    })
    ramp, lights = got["labs"], got["lights"]
    mids = (lights[:-1] + lights[1:]) * 0.5 if levels > 1 else lights
    bins = np.digitize(np.clip(lab[..., 0], 0, 1), mids)
    return ramp[bins]


def grade_is_identity(st):
    if float(st.get("grade_exposure") or 0):
        return False
    if float(st.get("contrast") or 0):
        return False
    if float(st.get("temp") or 0):
        return False
    if float(st.get("grade_tint") or 0):
        return False
    if float(st.get("value_compression") or 0):
        return False
    if abs(float(st.get("saturation", 1)) - 1.0) > 1e-6:
        return False
    if int(st.get("posterize") or 0) > 1:
        return False
    if float(st.get("palette_strength") or 0) > 0 and (st.get("palette") or []):
        return False
    return True


def colour_grade(rgb, st):
    rgb = np.clip(np.asarray(rgb, np.float32), 0, 1)
    if grade_is_identity(st):
        return rgb.astype(np.float32, copy=True)
    stops = float(st.get("grade_exposure", 0) or 0)
    if stops:
        rgb = linear_to_srgb(np.clip(srgb_to_linear(rgb) * np.float32(2.0 ** stops), 0, 1))
    lab = srgb_to_oklab(rgb)
    contrast = float(st.get("contrast", 0) or 0)
    if contrast:
        gain = max(0.0, 1.0 + contrast)
        lab[..., 0] = np.clip(0.5 + (lab[..., 0] - 0.5) * gain, 0, 1)
    temp = float(st.get("temp", 0) or 0)
    if temp:
        lab[..., 2] = lab[..., 2] + temp * 0.15
    tint = float(st.get("grade_tint", 0) or 0)
    if tint:
        lab[..., 1] = lab[..., 1] + tint * 0.12
    vc = float(st.get("value_compression") or 0)
    if vc:
        L = lab[..., 0]
        Lm = float(L.mean())
        lab[..., 0] = Lm + (L - Lm) * (1.0 - vc)
    sat = float(st.get("saturation", 1))
    if sat != 1.0:
        lab[..., 1:] *= sat
    lv = int(st.get("posterize") or 0)
    root = st.get("gradient_root") or []
    use_gradient = bool(st.get("gradient")) and lv and lv > 1 and len(root) >= 3
    if use_gradient:
        lab = _posterize_gradient(lab, lv, root, st.get("level_overrides"))
    elif lv and lv > 1:
        lab = _posterize_quantize(lab, lv)
    strength = float(st.get("palette_strength", 0) or 0)
    palette = st.get("palette") or []
    if strength > 0 and palette:
        cols = []
        for swatch in list(palette)[:8]:
            try:
                cols.append(srgb_to_oklab(_as_rgb_list(swatch, (0.5, 0.5, 0.5))))
            except (TypeError, ValueError):
                continue
        if cols:
            cols = np.stack(cols).astype(np.float32)
            dist = np.linalg.norm(lab[..., None, :] - cols, axis=-1)
            hardness = max(float(st.get("palette_hardness", 8) or 8), 0.01)
            w = np.exp(-dist * hardness).astype(np.float32)
            w /= np.maximum(w.sum(axis=-1, keepdims=True), 1e-8)
            snapped = np.einsum("hwn,nc->hwc", w, cols)
            lab = lab * (1.0 - strength) + snapped * strength
    return np.clip(oklab_to_srgb(lab), 0, 1).astype(np.float32)
