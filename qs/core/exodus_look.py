"""Exodus viewport look: ShaderShift Burley + dual-lobe GGX, AgX Punchy, SMAA 1x.

Constants come from Exodus/Config/DefaultShaderShift.ini when that file exists.
"""
from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from qs.core.colour import linear_to_srgb, srgb_to_linear
from qs.core.io import imwrite, to_u8
from qs.core.smaa import apply_smaa

PI = math.pi

# Fallbacks matching DefaultShaderShift.ini Defines Value= (not the template defaults).
DEFAULT_SPEC_PRIMARY_WEIGHT = 0.7
DEFAULT_SPEC_PRIMARY_ROUGHNESS_SCALE = 0.6
DEFAULT_SPEC_SECONDARY_ROUGHNESS_SCALE = 2.2
DEFAULT_SPEC_SECONDARY_MAX_ROUGHNESS = 0.95


@dataclass
class LookSettings:
    diffuse_mode: int = 2  # Burley
    spec_mode: int = 1     # dual-lobe
    spec_primary_weight: float = DEFAULT_SPEC_PRIMARY_WEIGHT
    spec_primary_roughness_scale: float = DEFAULT_SPEC_PRIMARY_ROUGHNESS_SCALE
    spec_secondary_roughness_scale: float = DEFAULT_SPEC_SECONDARY_ROUGHNESS_SCALE
    spec_secondary_max_roughness: float = DEFAULT_SPEC_SECONDARY_MAX_ROUGHNESS
    tonemap: str = "agx_punchy"
    agx_look: int = 2
    auto_exposure: bool = False
    exposure_bias: float = 1.0
    aperture: float = 4.0
    shutter: float = 1.0 / 60.0
    iso: float = 100.0
    smaa_quality: int = 2
    working_color_space: str = "sRGB"


def _exodus_root() -> Path | None:
    env = os.environ.get("EXODUS_ROOT")
    if env and Path(env).is_dir():
        return Path(env)
    guess = Path(r"C:\Users\BigMoney\Projects\Exodus")
    return guess if guess.is_dir() else None


def _define_values(text: str) -> dict[str, str]:
    found = {}
    for m in re.finditer(r'DefineName="([^"]+)".*?Value="([^"]*)"', text, re.S):
        found[m.group(1)] = m.group(2)
    return found


def load_look(exodus_root: Path | None = None) -> LookSettings:
    look = LookSettings()
    root = exodus_root or _exodus_root()
    if root is None:
        return look
    shift = root / "Config" / "DefaultShaderShift.ini"
    engine = root / "Config" / "DefaultEngine.ini"
    if shift.is_file():
        text = shift.read_text(encoding="utf-8", errors="replace")
        defs = _define_values(text)
        look.diffuse_mode = int(float(defs.get("CUSTOM_DIFFUSE_MODE", look.diffuse_mode)))
        look.spec_mode = int(float(defs.get("CUSTOM_SPEC_MODE", look.spec_mode)))
        look.spec_primary_weight = float(defs.get("CUSTOM_SPEC_PRIMARY_WEIGHT", look.spec_primary_weight))
        look.spec_primary_roughness_scale = float(
            defs.get("CUSTOM_SPEC_PRIMARY_ROUGHNESS_SCALE", look.spec_primary_roughness_scale))
        look.spec_secondary_roughness_scale = float(
            defs.get("CUSTOM_SPEC_SECONDARY_ROUGHNESS_SCALE", look.spec_secondary_roughness_scale))
        look.spec_secondary_max_roughness = float(
            defs.get("CUSTOM_SPEC_SECONDARY_MAX_ROUGHNESS", look.spec_secondary_max_roughness))
        look.agx_look = int(float(defs.get("AGX_LOOK", look.agx_look)))
        if "TonemapMode=AgX" in text or defs.get("CUSTOM_TONEMAP_MODE") == "1":
            look.tonemap = "agx_punchy"
    if engine.is_file():
        et = engine.read_text(encoding="utf-8", errors="replace")
        m = re.search(r"r\.DefaultFeature\.AutoExposure=(\w+)", et)
        if m:
            look.auto_exposure = m.group(1).strip().lower() in ("true", "1")
        m = re.search(r"r\.DefaultFeature\.AutoExposure\.Bias=([0-9.]+)", et)
        if m:
            look.exposure_bias = float(m.group(1))
        m = re.search(r"r\.AntiAliasingMethod=(\d+)", et)
        if m and int(m.group(1)) == 5:
            look.smaa_quality = 2
        if "WorkingColorSpaceChoice=sRGB" in et:
            look.working_color_space = "sRGB"
    return look


def ev100(aperture: float, shutter: float, iso: float) -> float:
    return math.log2((aperture * aperture) * 100.0 / (shutter * iso))


def exposure_scale(look: LookSettings, extra_stops: float = 0.0) -> float:
    """Unreal ConvertEV100ToExposure, with AutoExposureBias as extra stops (brighter)."""
    ev = ev100(look.aperture, look.shutter, look.iso) - look.exposure_bias - extra_stops
    max_lum = 1.2 * (2.0 ** ev)
    return 1.0 / max_lum


def pow5(x):
    x2 = x * x
    return x2 * x2 * x


def diffuse_burley(diffuse, roughness, nov, nol, voh):
    fd90 = 0.5 + 2.0 * voh * voh * roughness
    fdv = 1.0 + (fd90 - 1.0) * pow5(1.0 - nov)
    fdl = 1.0 + (fd90 - 1.0) * pow5(1.0 - nol)
    return diffuse * ((1.0 / PI) * fdv * fdl)


def sample_specular_dir_albedo(roughness, nov):
    c0 = np.array([-1.0, -0.0275], np.float32)
    c1 = np.array([1.0, 0.0425], np.float32)
    r = roughness * c0 + c1
    return float(min(r[0] * r[0], math.exp2(-9.28 * np.clip(nov, 0, 1))) * r[0] + r[1])


def dual_lobe_diffuse_scale(roughness, nov, look: LookSettings):
    if look.spec_mode != 1:
        return 1.0
    rp = float(np.clip(roughness * look.spec_primary_roughness_scale, 0, 1))
    rs = min(roughness * look.spec_secondary_roughness_scale, look.spec_secondary_max_roughness)
    ep = sample_specular_dir_albedo(rp, nov)
    es = sample_specular_dir_albedo(rs, nov)
    e = es + (ep - es) * look.spec_primary_weight
    return 1.0 - float(np.clip(e, 0, 1))


def d_ggx_dual(a2, noh, look: LookSettings):
    if look.spec_mode != 1:
        d = (noh * a2 - noh) * noh + 1.0
        return a2 / (PI * d * d)
    alpha = math.sqrt(max(a2, 1e-8))
    ap = alpha * (look.spec_primary_roughness_scale ** 2)
    a2p = ap * ap
    dp = (noh * a2p - noh) * noh + 1.0
    dp = a2p / (PI * dp * dp)
    max_a = look.spec_secondary_max_roughness ** 2
    as_ = min(alpha * (look.spec_secondary_roughness_scale ** 2), max_a)
    a2s = as_ * as_
    ds = (noh * a2s - noh) * noh + 1.0
    ds = a2s / (PI * ds * ds)
    w1 = look.spec_primary_weight
    return w1 * dp + (1.0 - w1) * ds


def vis_smith_joint_approx(a2, nov, nol):
    a = math.sqrt(max(a2, 1e-12))
    vis_v = nol * (nov * (1 - a) + a)
    vis_l = nov * (nol * (1 - a) + a)
    return 0.5 / max(vis_v + vis_l, 1e-6)


def f_schlick(specular_color, voh):
    fc = pow5(1.0 - voh)
    spec = np.asarray(specular_color, np.float32)
    # Unreal: saturate(50 * SpecularColor.g) * Fc + (1 - Fc) * SpecularColor
    scale = float(np.clip(50.0 * spec[1] if spec.ndim == 1 else spec[..., 1], 0, 1))
    return scale * fc + (1.0 - fc) * spec


def env_brdf_approx(specular_color, roughness, nov):
    c0 = np.array([-1.0, -0.0275, -0.572, 0.022], np.float32)
    c1 = np.array([1.0, 0.0425, 1.04, -0.04], np.float32)
    r = roughness * c0 + c1
    a004 = min(r[0] * r[0], math.exp2(-9.28 * nov)) * r[0] + r[1]
    ab = np.array([-1.04, 1.04], np.float32) * a004 + r[2:]
    spec = np.asarray(specular_color, np.float32)
    return spec * ab[0] + ab[1]


def evaluate_brdf(base_srgb, roughness, metallic, n, v, l, look: LookSettings | None = None,
                  specular=0.5):
    """Single-pixel / broadcast analytic light. Vectors unit length. Returns linear RGB."""
    look = look or LookSettings()
    n = np.asarray(n, np.float32)
    v = np.asarray(v, np.float32)
    l = np.asarray(l, np.float32)
    h = l + v
    h = h / max(float(np.linalg.norm(h)), 1e-6)
    nov = float(np.clip(np.dot(n, v), 0, 1))
    nol = float(np.clip(np.dot(n, l), 0, 1))
    noh = float(np.clip(np.dot(n, h), 0, 1))
    voh = float(np.clip(np.dot(v, h), 0, 1))
    base = srgb_to_linear(np.asarray(base_srgb, np.float32))
    metal = float(metallic)
    rough = float(np.clip(roughness, 0.045, 1.0))
    a = rough * rough
    a2 = a * a
    f0 = 0.08 * specular * (1.0 - metal) + base * metal
    d = d_ggx_dual(a2, noh, look)
    vis = vis_smith_joint_approx(a2, nov, nol)
    f = f_schlick(f0, voh)
    spec = d * vis * f * nol
    kd = (1.0 - f) * (1.0 - metal)
    diff = diffuse_burley(base, rough, nov, nol, voh) * kd * dual_lobe_diffuse_scale(rough, nov, look) * nol
    # Unreal's BxDF already includes (1/PI) in Burley; directional intensity is applied by the light.
    return np.clip(diff + spec, 0, None).astype(np.float32)


# --- AgX Punchy (ShaderShift PostProcessCombineLUTs.usf) ---

_AGX_MAT = np.array([
    [0.842479062253094, 0.0423282422610123, 0.0423756549057051],
    [0.0784335999999992, 0.878468636469772, 0.0784336],
    [0.0792237451477643, 0.0791661274605434, 0.879142973793104],
], np.float64)
_AGX_MAT_INV = np.array([
    [1.19687900512017, -0.0528968517574562, -0.0529716355144438],
    [-0.0980208811401368, 1.15190312990417, -0.0980434501171241],
    [-0.0990297440797205, -0.0989611768448433, 1.15107367264116],
], np.float64)

# AP1 <-> XYZ (ACEScg), XYZ <-> sRGB D65. No CAT, matching ApplyAgX.
_AP1_2_XYZ = np.array([
    [0.6624541811, 0.1340042065, 0.1561876870],
    [0.2722287168, 0.6740817658, 0.0536895174],
    [-0.0055746495, 0.0040607335, 1.0103391003],
], np.float64)
_XYZ_2_SRGB = np.array([
    [3.2404542, -1.5371385, -0.4985314],
    [-0.9692660, 1.8760108, 0.0415560],
    [0.0556434, -0.2040259, 1.0572252],
], np.float64)
_SRGB_2_XYZ = np.linalg.inv(_XYZ_2_SRGB)
_XYZ_2_AP1 = np.linalg.inv(_AP1_2_XYZ)
AP1_2_LINEAR = _XYZ_2_SRGB @ _AP1_2_XYZ
LINEAR_2_AP1 = _XYZ_2_AP1 @ _SRGB_2_XYZ


def agx_contrast_approx(x):
    x2 = x * x
    x4 = x2 * x2
    return (15.5 * x4 * x2 - 40.14 * x4 * x + 31.96 * x4
            - 6.868 * x2 * x + 0.4298 * x2 + 0.1191 * x - 0.00232)


def agx_inset(val):
    val = val @ _AGX_MAT.T
    min_ev, max_ev = -12.47393, 4.026069
    val = np.clip(np.log2(np.maximum(val, 1e-6)), min_ev, max_ev)
    val = (val - min_ev) / (max_ev - min_ev)
    return agx_contrast_approx(val)


def agx_look(val, look_id=2):
    offset = np.zeros(3)
    slope = np.ones(3)
    power = np.ones(3)
    sat = 1.0
    if look_id == 1:
        slope = np.array([1.0, 0.9, 0.5])
        power = np.array([0.8, 0.8, 0.8])
        sat = 0.8
    elif look_id == 2:
        power = np.array([1.35, 1.35, 1.35])
        sat = 1.4
    val = np.power(np.maximum(val * slope + offset, 0.0), power)
    lw = np.array([0.2126, 0.7152, 0.0722])
    luma = val @ lw
    return luma[..., None] + sat * (val - luma[..., None])


def agx_eotf(val):
    val = val @ _AGX_MAT_INV.T
    val = np.maximum(val, 0.0)
    return np.power(val, 2.2)


def apply_agx_punchy(color_ap1, look_id=2):
    linear_srgb = np.asarray(color_ap1, np.float64) @ AP1_2_LINEAR.T
    linear_srgb = agx_inset(linear_srgb)
    linear_srgb = agx_look(linear_srgb, look_id)
    linear_srgb = agx_eotf(linear_srgb)
    return (linear_srgb @ LINEAR_2_AP1.T).astype(np.float32)


def srgb_to_ap1(linear_srgb):
    return np.asarray(linear_srgb, np.float64) @ LINEAR_2_AP1.T


def ap1_to_srgb(ap1):
    return np.asarray(ap1, np.float64) @ AP1_2_LINEAR.T


def tonemap_agx(linear_srgb, look: LookSettings | None = None):
    look = look or LookSettings()
    ap1 = srgb_to_ap1(np.clip(linear_srgb, 0, None))
    mapped = apply_agx_punchy(ap1, look.agx_look)
    # ApplyAgX returns AP1; output encode as sRGB display.
    return np.clip(ap1_to_srgb(mapped), 0, None).astype(np.float32)


def studio_env(direction):
    z = np.clip(direction[..., 2] if direction.ndim > 1 else direction[2], -1, 1)
    sky = np.array([4.0, 4.4, 5.2], np.float32)
    horizon = np.array([6.0, 5.2, 4.0], np.float32)
    ground = np.array([0.4, 0.35, 0.3], np.float32)
    up = np.clip(z, 0, 1)[..., None]
    down = np.clip(-z, 0, 1)[..., None]
    hor = (1.0 - np.abs(z))[..., None]
    col = sky * up + ground * down + horizon * hor
    return (col / np.maximum(up + down + hor, 1e-4)).astype(np.float32)


def shade_exodus(base_srgb, n_dx, rough, metal, light_dir, view_dir=(0, 0, 1),
                 look: LookSettings | None = None, extra_stops=0.0, light_intensity=4.0):
    """Image-shaped arrays. Tangent-space n_dx, z toward camera."""
    look = look or LookSettings()
    n = np.asarray(n_dx, np.float32)
    ln = np.sqrt((n * n).sum(axis=-1, keepdims=True))
    n = n / np.maximum(ln, 1e-6)
    v = np.asarray(view_dir, np.float32)
    v = v / max(float(np.linalg.norm(v)), 1e-6)
    l = np.asarray(light_dir, np.float32)
    l = l / max(float(np.linalg.norm(l)), 1e-6)
    h = n.shape[:2]
    lin = np.zeros(n.shape, np.float32)
    # Vectorised directional term.
    nov = np.clip(n[..., 2], 0, 1)
    nol = np.clip(n @ l, 0, 1)
    hh = l + v
    hh = hh / max(float(np.linalg.norm(hh)), 1e-6)
    noh = np.clip(n @ hh, 0, 1)
    voh = float(np.clip(np.dot(v, hh), 0, 1))
    base = srgb_to_linear(np.clip(base_srgb, 0, 1))
    m = metal[..., None] if np.asarray(metal).ndim == 2 else np.asarray(metal, np.float32)
    r = np.clip(np.asarray(rough, np.float32), 0.045, 1.0)
    a2 = (r * r) ** 2
    f0 = 0.08 * 0.5 * (1.0 - m) + base * m
    # Dual-lobe D per pixel.
    alpha = np.sqrt(np.maximum(a2, 1e-8))
    w1 = look.spec_primary_weight
    ap = alpha * (look.spec_primary_roughness_scale ** 2)
    a2p = ap * ap
    dp = (noh * a2p - noh) * noh + 1.0
    dp = a2p / (PI * dp * dp)
    max_a = look.spec_secondary_max_roughness ** 2
    as_ = np.minimum(alpha * (look.spec_secondary_roughness_scale ** 2), max_a)
    a2s = as_ * as_
    ds = (noh * a2s - noh) * noh + 1.0
    ds = a2s / (PI * ds * ds)
    d = w1 * dp + (1.0 - w1) * ds
    a = np.sqrt(np.maximum(a2, 1e-12))
    vis_v = nol * (nov * (1 - a) + a)
    vis_l = nov * (nol * (1 - a) + a)
    vis = 0.5 / np.maximum(vis_v + vis_l, 1e-6)
    fc = pow5(1.0 - voh)
    scale = np.clip(50.0 * f0[..., 1], 0, 1)[..., None]
    f = scale * fc + (1.0 - fc) * f0
    spec = (d * vis)[..., None] * f * nol[..., None]
    fd90 = 0.5 + 2.0 * voh * voh * r
    fdv = 1.0 + (fd90 - 1.0) * pow5(1.0 - nov)
    fdl = 1.0 + (fd90 - 1.0) * pow5(1.0 - nol)
    burley = base * ((1.0 / PI) * fdv * fdl)[..., None]
    kd = (1.0 - f) * (1.0 - m)
    # Dual-lobe diffuse scale (Lazarov, F0=0.04).
    rp = np.clip(r * look.spec_primary_roughness_scale, 0, 1)
    rs = np.minimum(r * look.spec_secondary_roughness_scale, look.spec_secondary_max_roughness)
    def laz(rr, nv):
        c0x, c0y = -1.0, -0.0275
        c1x, c1y = 1.0, 0.0425
        rx = rr * c0x + c1x
        ry = rr * c0y + c1y
        return np.minimum(rx * rx, np.exp2(-9.28 * nv)) * rx + ry
    e = laz(rs, nov) + (laz(rp, nov) - laz(rs, nov)) * w1
    dscale = (1.0 - np.clip(e, 0, 1))[..., None]
    diff = burley * kd * dscale * nol[..., None]
    irr = studio_env(n)
    R = n * (2.0 * nov)[..., None] - v
    c0 = np.array([-1.0, -0.0275, -0.572, 0.022], np.float32)
    c1 = np.array([1.0, 0.0425, 1.04, -0.04], np.float32)
    rr = r[..., None] * c0 + c1
    a004 = np.minimum(rr[..., 0] * rr[..., 0], np.exp2(-9.28 * nov)) * rr[..., 0] + rr[..., 1]
    abx = -1.04 * a004 + rr[..., 2]
    aby = 1.04 * a004 + rr[..., 3]
    spec_ibl = studio_env(R) * (f0 * abx[..., None] + aby[..., None])
    diff_ibl = base * (1.0 - m) * irr * 0.25
    lin = (diff + spec) * light_intensity + diff_ibl + spec_ibl * 0.5
    lin = lin * np.float32(exposure_scale(look, extra_stops))
    return lin.astype(np.float32)


def render_sphere(width, height, base=(0.7, 0.15, 0.08), roughness=0.35, metallic=0.0,
                  look: LookSettings | None = None, extra_stops=0.0, smaa=True):
    look = look or load_look()
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    u = (xx + 0.5) / width * 2 - 1
    v = 1 - (yy + 0.5) / height * 2
    aspect = width / max(height, 1)
    u *= aspect
    r2 = u * u + v * v
    mask = r2 <= 1.0
    z = np.sqrt(np.clip(1.0 - r2, 0, 1))
    n = np.stack([u, v, z], axis=2)
    n[~mask] = (0, 0, 1)
    alb = np.broadcast_to(np.array(base, np.float32), (height, width, 3)).copy()
    rough = np.full((height, width), roughness, np.float32)
    metal = np.full((height, width), metallic, np.float32)
    # Epic HQ directional: pitch -40, yaw 128
    pitch, yaw = math.radians(-40), math.radians(128)
    light = np.array([
        math.cos(pitch) * math.cos(yaw),
        math.cos(pitch) * math.sin(yaw),
        math.sin(-pitch),
    ], np.float32)
    # Camera looks +Z in tangent preview; remap light into view-ish frame.
    light = light / np.linalg.norm(light)
    lin = shade_exodus(alb, n, rough, metal, light, look=look, extra_stops=extra_stops)
    tm = np.clip(tonemap_agx(lin, look), 0, 1)
    tm[~mask] = (0.02, 0.02, 0.025)
    if smaa:
        tm = apply_smaa(tm)
        tm[~mask] = (0.02, 0.02, 0.025)
    return tm.astype(np.float32)


def render_png(path, width=512, height=512, hdri=None, **kwargs):
    img = render_sphere(width, height, **kwargs)
    imwrite(path, to_u8(img))
    return path


# GLSL used by the Qt viewport. Column-major mat3 like the rest of the GPU path.
EXODUS_GLSL = r"""
const float PI = 3.141592653589793;
float saturate(float x) { return clamp(x, 0.0, 1.0); }
float pow5(float x) { float x2 = x * x; return x2 * x2 * x; }
vec3 srgb_to_linear(vec3 c) {
    c = clamp(c, 0.0, 1.0);
    return mix(pow((c + vec3(0.055)) / 1.055, vec3(2.4)), c / 12.92, step(c, vec3(0.04045)));
}
vec3 linear_to_srgb(vec3 c) {
    c = clamp(c, 0.0, 1.0);
    return mix(1.055 * pow(c, vec3(1.0/2.4)) - 0.055, c * 12.92, step(c, vec3(0.0031308)));
}
vec3 diffuse_burley(vec3 albedo, float rough, float NoV, float NoL, float VoH) {
    float FD90 = 0.5 + 2.0 * VoH * VoH * rough;
    float FdV = 1.0 + (FD90 - 1.0) * pow5(1.0 - NoV);
    float FdL = 1.0 + (FD90 - 1.0) * pow5(1.0 - NoL);
    return albedo * ((1.0 / PI) * FdV * FdL);
}
float D_GGX_dual(float a2, float NoH, float w1, float sp, float ss, float smax) {
    float alpha = sqrt(max(a2, 1e-8));
    float ap = alpha * (sp * sp);
    float a2p = ap * ap;
    float dp = (NoH * a2p - NoH) * NoH + 1.0;
    dp = a2p / (PI * dp * dp);
    float as_ = min(alpha * (ss * ss), smax * smax);
    float a2s = as_ * as_;
    float ds = (NoH * a2s - NoH) * NoH + 1.0;
    ds = a2s / (PI * ds * ds);
    return mix(ds, dp, w1);
}
float Vis_SmithJointApprox(float a2, float NoV, float NoL) {
    float a = sqrt(max(a2, 1e-12));
    float VisV = NoL * (NoV * (1.0 - a) + a);
    float VisL = NoV * (NoL * (1.0 - a) + a);
    return 0.5 / max(VisV + VisL, 1e-6);
}
vec3 F_Schlick(vec3 F0, float VoH) {
    float Fc = pow5(1.0 - VoH);
    return saturate(50.0 * F0.g) * Fc + (1.0 - Fc) * F0;
}
"""
