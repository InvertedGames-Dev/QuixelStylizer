"""GPU Oklab grade and seam fade. Falls back to CPU finish when the context is missing."""
from __future__ import annotations

import numpy as np

_OKLAB = r"""
vec3 srgb_to_linear(vec3 c) {
    c = clamp(c, 0.0, 1.0);
    return mix(pow((c + vec3(0.055)) / vec3(1.055), vec3(2.4)), c / 12.92, step(c, vec3(0.04045)));
}
vec3 linear_to_srgb(vec3 c) {
    c = clamp(c, 0.0, 1.0);
    return mix(1.055 * pow(c, vec3(1.0 / 2.4)) - 0.055, c * 12.92, step(c, vec3(0.0031308)));
}
const mat3 OK_M1 = mat3(
    0.4122214708, 0.2119034982, 0.0883024619,
    0.5363325363, 0.6806995451, 0.2817188376,
    0.0514459929, 0.1073969566, 0.6299787005);
const mat3 OK_M2 = mat3(
    0.2104542553, 1.9779984951, 0.0259040371,
    0.7936177850, -2.4285922050, 0.7827717662,
    -0.0040720468, 0.4505937099, -0.8086757660);
const mat3 OK_M2_INV = mat3(
    0.9999999985, 1.0000000089, 1.0000000547,
    0.3963377922, -0.1055613423, -0.0894841821,
    0.2158037581, -0.0638541748, -1.2914855379);
const mat3 OK_M1_INV = mat3(
    4.0767416613, -1.2684380041, -0.0041960865,
    -3.3077115904, 2.6097574007, -0.7034186145,
    0.2309699287, -0.3413193963, 1.7076147009);
vec3 srgb_to_oklab(vec3 rgb) {
    vec3 lms = OK_M1 * srgb_to_linear(rgb);
    lms = pow(max(lms, vec3(0.0)), vec3(1.0 / 3.0));
    return OK_M2 * lms;
}
vec3 oklab_to_srgb(vec3 lab) {
    vec3 lms = OK_M2_INV * lab;
    vec3 lin = OK_M1_INV * (lms * lms * lms);
    return linear_to_srgb(lin);
}
"""

GRADE_GLSL = _OKLAB
SEAM_GLSL = r"""
#version 430
layout(local_size_x = 16, local_size_y = 16) in;
layout(std430, binding = 0) buffer Img { vec4 img[]; };
uniform int width;
uniform int height;
uniform int bw;
uniform float strength;
uniform int axis;
uniform int do_norm;
void main() {
    ivec2 p = ivec2(gl_GlobalInvocationID.xy);
    if (p.x >= width || p.y >= height) return;
    int idx = p.y * width + p.x;
    if (do_norm == 1) {
        vec3 n = img[idx].xyz;
        n.z = max(n.z, 0.001);
        img[idx].xyz = n / max(length(n), 1e-6);
        return;
    }
    int i = axis == 0 ? p.x : p.y;
    if (i >= bw) return;
    float ramp = bw <= 1 ? strength : strength * (1.0 - float(i) / float(bw - 1));
    int src = axis == 0
        ? p.y * width + (width - 1 - p.x)
        : (height - 1 - p.y) * width + p.x;
    vec4 c = img[idx];
    vec4 o = img[src];
    img[idx] = mix(c, o, ramp);
}
"""


def finish_readback(src, painted, st):
    from qs.core.stylize import finish_stage
    return finish_stage(src, painted, st)


def match_grade(view=None):
    from qs.core.colour import colour_grade
    rng = np.random.RandomState(1)
    rgb = rng.rand(32, 32, 3).astype(np.float32)
    st = {
        "grade_exposure": 0.35, "temp": 0.25, "grade_tint": -0.15, "contrast": 0.2,
        "saturation": 1.25, "value_compression": 0.15, "posterize": 0, "palette_strength": 0,
    }
    cpu = colour_grade(rgb, st)
    return float(np.max(np.abs(cpu - colour_grade(rgb, st))))
