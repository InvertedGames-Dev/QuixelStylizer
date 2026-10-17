"""QOpenGLWidget viewport. One GL context on the Qt thread. Exodus BRDF + AgX."""
from __future__ import annotations

import math

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QMouseEvent, QSurfaceFormat, QWheelEvent
from PySide6.QtOpenGLWidgets import QOpenGLWidget

from qs.core.exodus_look import EXODUS_GLSL, load_look
from qs.core.mesh import vertex_tangents
from qs.logging import log_line

# Pass-through for the tessellation pipeline. Position is written in the TES
# after the height map displaces the subdivided patch.
VERT = """
#version 430
in vec3 in_pos;
in vec3 in_nrm;
in vec2 in_uv;
in vec4 in_tan;
out vec3 vs_pos;
out vec3 vs_nrm;
out vec2 vs_uv;
out vec4 vs_tan;
void main() {
    vs_pos = in_pos;
    vs_nrm = in_nrm;
    vs_uv = in_uv;
    vs_tan = in_tan;
}
"""

TCS = """
#version 430
layout(vertices = 3) out;
in vec3 vs_pos[];
in vec3 vs_nrm[];
in vec2 vs_uv[];
in vec4 vs_tan[];
out vec3 tc_pos[];
out vec3 tc_nrm[];
out vec2 tc_uv[];
out vec4 tc_tan[];
uniform float tess_level;
void main() {
    tc_pos[gl_InvocationID] = vs_pos[gl_InvocationID];
    tc_nrm[gl_InvocationID] = vs_nrm[gl_InvocationID];
    tc_uv[gl_InvocationID] = vs_uv[gl_InvocationID];
    tc_tan[gl_InvocationID] = vs_tan[gl_InvocationID];
    if (gl_InvocationID == 0) {
        float L = max(tess_level, 1.0);
        gl_TessLevelOuter[0] = L;
        gl_TessLevelOuter[1] = L;
        gl_TessLevelOuter[2] = L;
        gl_TessLevelInner[0] = L;
    }
}
"""

TES = """
#version 430
layout(triangles, fractional_odd_spacing, ccw) in;
in vec3 tc_pos[];
in vec3 tc_nrm[];
in vec2 tc_uv[];
in vec4 tc_tan[];
out vec3 v_pos;
out vec3 v_nrm;
out vec2 v_uv;
out vec4 v_tan;
uniform mat4 mvp;
uniform mat4 model;
uniform sampler2D tex_pack;
uniform sampler2D tex_pack0;
uniform int compare_mode;
uniform float disp_scale;
uniform int disp_from_tan;
void main() {
    vec3 p = gl_TessCoord.x * tc_pos[0] + gl_TessCoord.y * tc_pos[1] + gl_TessCoord.z * tc_pos[2];
    vec3 n = gl_TessCoord.x * tc_nrm[0] + gl_TessCoord.y * tc_nrm[1] + gl_TessCoord.z * tc_nrm[2];
    vec2 uv = gl_TessCoord.x * tc_uv[0] + gl_TessCoord.y * tc_uv[1] + gl_TessCoord.z * tc_uv[2];
    vec4 tn = gl_TessCoord.x * tc_tan[0] + gl_TessCoord.y * tc_tan[1] + gl_TessCoord.z * tc_tan[2];
    n = normalize(n);
    float h = (disp_from_tan == 1) ? tn.w
        : ((compare_mode == 1) ? texture(tex_pack0, uv).a : texture(tex_pack, uv).a);
    p += n * (h - 0.5) * disp_scale;
    v_pos = (model * vec4(p, 1.0)).xyz;
    v_nrm = mat3(model) * n;
    v_uv = uv;
    v_tan = tn;
    gl_Position = mvp * vec4(p, 1.0);
}
"""

# No tessellation: same varyings as the TES so the fragment shader can be shared.
VERT_FLAT = """
#version 430
in vec3 in_pos;
in vec3 in_nrm;
in vec2 in_uv;
in vec4 in_tan;
uniform mat4 mvp;
uniform mat4 model;
out vec3 v_nrm;
out vec2 v_uv;
out vec3 v_pos;
out vec4 v_tan;
void main() {
    v_pos = (model * vec4(in_pos, 1.0)).xyz;
    v_nrm = mat3(model) * in_nrm;
    v_uv = in_uv;
    v_tan = vec4(mat3(model) * in_tan.xyz, in_tan.w);
    gl_Position = mvp * vec4(in_pos, 1.0);
}
"""

FRAG = """
#version 430
""" + EXODUS_GLSL + r"""
in vec3 v_nrm;
in vec2 v_uv;
in vec3 v_pos;
in vec4 v_tan;
uniform sampler2D tex_alb;
uniform sampler2D tex_n;
uniform sampler2D tex_pack; // MROD after
uniform sampler2D tex_alb0;
uniform sampler2D tex_n0;
uniform sampler2D tex_pack0; // MROD before
uniform vec3 light_dir;
uniform vec3 cam_pos;
uniform float exposure;
uniform float w1;
uniform float sp;
uniform float ss;
uniform float smax;
uniform int view_channel; // 0=Lit, 1=Diffuse, 2=Normal, 3=Roughness, 4=AO, 5=Displacement, 6=Metallic
uniform int compare_mode; // 0=After, 1=Before, 2=Split
uniform float split_px;
out vec4 f_color;

vec3 studio_env(vec3 d) {
    float y = clamp(d.y, -1.0, 1.0);
    vec3 sky = vec3(3.2, 3.6, 4.2);
    vec3 horizon = vec3(4.2, 3.8, 3.2);
    vec3 ground = vec3(0.45, 0.40, 0.35);
    float up = clamp(y, 0.0, 1.0);
    float down = clamp(-y, 0.0, 1.0);
    float hor = 1.0 - abs(y);
    return (sky * up + ground * down + horizon * hor) / max(up + down + hor, 1e-4);
}

vec3 env_brdf_approx(vec3 spec_col, float rough, float NoV) {
    vec4 c0 = vec4(-1.0, -0.0275, -0.572, 0.022);
    vec4 c1 = vec4(1.0, 0.0425, 1.04, -0.04);
    vec4 r = rough * c0 + c1;
    float a004 = min(r.x * r.x, exp2(-9.28 * NoV)) * r.x + r.y;
    vec2 ab = vec2(-1.04, 1.04) * a004 + r.zw;
    return spec_col * ab.x + ab.y;
}

float laz_albedo(float rr, float NoV) {
    float rx = -rr + 1.0;
    float ry = -0.0275 * rr + 0.0425;
    return min(rx * rx, exp2(-9.28 * clamp(NoV, 0.0, 1.0))) * rx + ry;
}

float dual_lobe_diffuse_scale(float rough, float NoV, float w1_, float sp_, float ss_, float smax_) {
    float rp = clamp(rough * sp_, 0.0, 1.0);
    float rs = min(rough * ss_, smax_);
    float ep = laz_albedo(rp, NoV);
    float es = laz_albedo(rs, NoV);
    float e = es + (ep - es) * w1_;
    return 1.0 - clamp(e, 0.0, 1.0);
}

mat3 shading_tbn(vec3 N_geom, vec3 T_in, vec3 p, vec2 uv) {
    // Prefer the UV tangent from the mesh. Screen-space dFdx frames go to zero
    // on some NVIDIA MSAA FBOs; whiteout z=1 then becomes N_geom and Lit
    // ignores the normal map (strength/soften look like they do nothing).
    vec3 N = normalize(N_geom);
    vec3 T = T_in - N * dot(N, T_in);
    float t2 = dot(T, T);
    vec3 B;
    if (t2 > 1e-10) {
        T *= inversesqrt(t2);
        B = cross(N, T);
    } else {
        vec3 dp1 = dFdx(p);
        vec3 dp2 = dFdy(p);
        vec2 duv1 = dFdx(uv);
        vec2 duv2 = dFdy(uv);
        vec3 dp2perp = cross(dp2, N);
        vec3 dp1perp = cross(N, dp1);
        T = dp2perp * duv1.x + dp1perp * duv2.x;
        B = dp2perp * duv1.y + dp1perp * duv2.y;
        float tt = dot(T, T);
        float bb = dot(B, B);
        if (max(tt, bb) < 1e-12) {
            vec3 axis = (abs(N.z) < 0.999) ? vec3(0.0, 0.0, 1.0) : vec3(1.0, 0.0, 0.0);
            T = normalize(cross(axis, N));
            B = cross(N, T);
        } else {
            float invmax = inversesqrt(max(tt, bb) + 1e-20);
            T *= invmax;
            B *= invmax;
            T = normalize(T - N * dot(N, T));
            B = normalize(B - N * dot(N, B) - T * dot(T, B));
        }
    }
    return mat3(T, B, N);
}

vec3 mapped_normal(mat3 TBN, vec3 nt_dx) {
    // CPU already renormalised after strength/soften. Keep that Z; flipping Y
    // puts DirectX green into the OpenGL UV frame.
    vec3 nts = vec3(nt_dx.x, -nt_dx.y, max(nt_dx.z, 0.0));
    return normalize(TBN * nts);
}

vec3 agx_contrast(vec3 x) {
    vec3 x2 = x * x;
    vec3 x4 = x2 * x2;
    return 15.5*x4*x2 - 40.14*x4*x + 31.96*x4 - 6.868*x2*x + 0.4298*x2 + 0.1191*x - 0.00232;
}
vec3 apply_agx(vec3 lin) {
    const mat3 agx = mat3(
        0.842479062253094, 0.0784335999999992, 0.0792237451477643,
        0.0423282422610123, 0.878468636469772, 0.0791661274605434,
        0.0423756549057051, 0.0784336, 0.879142973793104);
    const mat3 agx_inv = mat3(
        1.19687900512017, -0.0980208811401368, -0.0990297440797205,
        -0.0528968517574562, 1.15190312990417, -0.0989611768448433,
        -0.0529716355144438, -0.0980434501171241, 1.15107367264116);
    vec3 val = agx * lin;
    const float min_ev = -12.47393;
    const float max_ev = 4.026069;
    val = clamp(log2(max(val, vec3(1e-6))), min_ev, max_ev);
    val = (val - min_ev) / (max_ev - min_ev);
    val = agx_contrast(val);
    val = pow(max(val, vec3(0.0)), vec3(1.35));
    float luma = dot(val, vec3(0.2126, 0.7152, 0.0722));
    val = luma + 1.4 * (val - luma);
    val = max(agx_inv * val, vec3(0.0));
    return pow(val, vec3(2.2));
}

vec4 shade_maps(sampler2D alb_tex, sampler2D n_tex, sampler2D pack_tex, vec2 vuv, vec3 N_geom, vec3 V, vec3 p) {
    mat3 TBN = shading_tbn(N_geom, v_tan.xyz, p, vuv);
    vec2 uv = vuv;

    vec3 alb = texture(alb_tex, uv).rgb;
    vec3 nt_raw = texture(n_tex, uv).xyz;
    vec3 nt = nt_raw * 2.0 - 1.0;
    vec4 pack = texture(pack_tex, uv);
    float metal = clamp(pack.r, 0.0, 1.0);
    float rough = clamp(pack.g, 0.045, 1.0);
    float ao = clamp(pack.b, 0.0, 1.0);
    float height = clamp(pack.a, 0.0, 1.0);

    if (view_channel == 1) return vec4(alb, 1.0);
    if (view_channel == 2) return vec4(nt_raw, 1.0);
    if (view_channel == 3) return vec4(vec3(pack.g), 1.0);
    if (view_channel == 4) return vec4(vec3(ao), 1.0);
    if (view_channel == 5) return vec4(vec3(height), 1.0);
    if (view_channel == 6) return vec4(vec3(metal), 1.0);

    vec3 N = mapped_normal(TBN, nt);
    vec3 L = normalize(light_dir);
    vec3 H = normalize(V + L);
    float NoV = max(dot(N, V), 1e-4);
    float NoL = max(dot(N, L), 0.0);
    float NoH = max(dot(N, H), 0.0);
    float VoH = max(dot(V, H), 0.0);
    vec3 base = srgb_to_linear(alb);
    vec3 F0 = mix(vec3(0.04), base, metal);
    float a = rough * rough;
    float a2 = a * a;
    float D = D_GGX_dual(a2, NoH, w1, sp, ss, smax);
    float Vis = Vis_SmithJointApprox(a2, NoV, NoL);
    vec3 F = F_Schlick(F0, VoH);
    vec3 spec = D * Vis * F * NoL;
    vec3 kd = (1.0 - F) * (1.0 - metal);
    float dscale = dual_lobe_diffuse_scale(rough, NoV, w1, sp, ss, smax);
    vec3 diff = diffuse_burley(base, rough, NoV, NoL, VoH) * kd * dscale * NoL;
    vec3 R = reflect(-V, N);
    vec3 diff_ibl = base * (1.0 - metal) * studio_env(N) * 0.25 * ao;
    vec3 spec_ibl = studio_env(R) * env_brdf_approx(F0, rough, NoV) * 0.40 * ao;
    float dir_ao = mix(0.35, 1.0, ao);
    vec3 lin = ((diff + spec) * 3.8 * dir_ao + diff_ibl + spec_ibl) * exp2(exposure);
    vec3 tm = linear_to_srgb(apply_agx(max(lin, vec3(0.0))));
    return vec4(clamp(tm, 0.0, 1.0), 1.0);
}

void main() {
    vec3 N_geom = normalize(v_nrm);
    vec3 V = normalize(cam_pos - v_pos);
    if (compare_mode == 1) {
        f_color = shade_maps(tex_alb0, tex_n0, tex_pack0, v_uv, N_geom, V, v_pos);
        return;
    }
    if (compare_mode == 2) {
        if (gl_FragCoord.x < split_px) {
            f_color = shade_maps(tex_alb0, tex_n0, tex_pack0, v_uv, N_geom, V, v_pos);
        } else {
            f_color = shade_maps(tex_alb, tex_n, tex_pack, v_uv, N_geom, V, v_pos);
        }
        if (abs(gl_FragCoord.x - split_px) < 1.25)
            f_color = vec4(0.95, 0.82, 0.18, 1.0);
        return;
    }
    f_color = shade_maps(tex_alb, tex_n, tex_pack, v_uv, N_geom, V, v_pos);
}
"""


def _cube():
    # 24 verts with per-face UV 0-1
    faces = [
        ((-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1), (0, 0, 1)),
        ((1, -1, -1), (-1, -1, -1), (-1, 1, -1), (1, 1, -1), (0, 0, -1)),
        ((-1, -1, -1), (-1, -1, 1), (-1, 1, 1), (-1, 1, -1), (-1, 0, 0)),
        ((1, -1, 1), (1, -1, -1), (1, 1, -1), (1, 1, 1), (1, 0, 0)),
        ((-1, 1, 1), (1, 1, 1), (1, 1, -1), (-1, 1, -1), (0, 1, 0)),
        ((-1, -1, -1), (1, -1, -1), (1, -1, 1), (-1, -1, 1), (0, -1, 0)),
    ]
    pos, nrm, uv, idx = [], [], [], []
    for _fi, (a, b, c, d, n) in enumerate(faces):
        base = len(pos)
        for p, t in ((a, (0, 1)), (b, (1, 1)), (c, (1, 0)), (d, (0, 0))):
            pos.append(p)
            nrm.append(n)
            uv.append(t)
        idx.extend([base, base + 1, base + 2, base, base + 2, base + 3])
    return (np.array(pos, np.float32), np.array(nrm, np.float32),
            np.array(uv, np.float32), np.array(idx, np.uint32))


def _sphere(stacks=48, slices=64):
    pos, nrm, uv, idx = [], [], [], []
    for i in range(stacks + 1):
        v = i / stacks
        phi = math.pi * v
        for j in range(slices + 1):
            u = j / slices
            th = 2 * math.pi * u
            x = math.sin(phi) * math.cos(th)
            y = math.cos(phi)
            z = math.sin(phi) * math.sin(th)
            pos.append((x, y, z))
            nrm.append((x, y, z))
            uv.append((u, 1 - v))
    for i in range(stacks):
        for j in range(slices):
            a = i * (slices + 1) + j
            b = a + slices + 1
            idx.extend([a, b, a + 1, a + 1, b, b + 1])
    return (np.array(pos, np.float32), np.array(nrm, np.float32),
            np.array(uv, np.float32), np.array(idx, np.uint32))


def _plane():
    pos = np.array([
        [-1.1, -1.1, 0.0],
        [1.1, -1.1, 0.0],
        [1.1, 1.1, 0.0],
        [-1.1, 1.1, 0.0],
    ], np.float32)
    nrm = np.array([
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0],
    ], np.float32)
    uv = np.array([
        [0.0, 1.0],
        [1.0, 1.0],
        [1.0, 0.0],
        [0.0, 0.0],
    ], np.float32)
    idx = np.array([0, 1, 2, 0, 2, 3], np.uint32)
    return pos, nrm, uv, idx


def _packed_vao(pos, nrm, uv, idx, disp=None):
    tan = vertex_tangents(pos, nrm, uv, idx)
    if disp is not None:
        tan[:, 3] = np.asarray(disp, np.float32)
    packed = np.ascontiguousarray(np.hstack([pos, nrm, uv, tan]), np.float32)
    ibo = np.ascontiguousarray(idx, np.uint32)
    return packed, ibo


def _look(eye, target, up):
    f = target - eye
    f = f / np.linalg.norm(f)
    s = np.cross(f, up)
    s = s / np.linalg.norm(s)
    u = np.cross(s, f)
    m = np.eye(4, dtype=np.float32)
    m[0, :3] = s
    m[1, :3] = u
    m[2, :3] = -f
    t = np.eye(4, dtype=np.float32)
    t[:3, 3] = -eye
    return m @ t


def _persp(fovy, aspect, near, far):
    f = 1.0 / math.tan(math.radians(fovy) / 2)
    m = np.zeros((4, 4), np.float32)
    m[0, 0] = f / max(aspect, 1e-4)
    m[1, 1] = f
    m[2, 2] = (far + near) / (near - far)
    m[2, 3] = (2 * far * near) / (near - far)
    m[3, 2] = -1
    return m


def default_gl_format() -> QSurfaceFormat:
    fmt = QSurfaceFormat()
    fmt.setVersion(4, 3)
    fmt.setProfile(QSurfaceFormat.OpenGLContextProfile.CoreProfile)
    fmt.setDepthBufferSize(24)
    fmt.setSamples(4)
    fmt.setSwapBehavior(QSurfaceFormat.SwapBehavior.DoubleBuffer)
    return fmt


class Viewport(QOpenGLWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFormat(default_gl_format())
        self.yaw = 0.6
        self.pitch = 0.4
        self.dist = 3.2
        self.shape = "sphere"
        self.channel = "Lit"
        self.compare = "After"
        self.maps = None
        self.source_maps = None
        self.mesh = None
        self.look = load_look()
        self.exposure = 0.0
        self.displace = True
        self.light_dir = (0.5, 0.7, 0.4)
        self._drag = None
        self._gl_ok = False
        self._prog = None
        self._tess = False
        self._dummy_tex = None
        self._tex = {}
        self._vao = {}
        self._mesh_bufs = []
        self._mesh_disp = False
        self._u8_scratch = None
        self._src_token = None
        self._prim_bufs = []
        self.tex_creates = 0
        self.tex_writes = 0
        self.setMinimumSize(400, 400)
        self.setMouseTracking(True)
        self.setToolTip(
            "Left-drag orbits. Right-drag aims the key light. Wheel zooms. "
            "Displace tessellates the mesh and pushes it with the height map.")

    def initializeGL(self):
        try:
            import moderngl
            self.ctx = moderngl.create_context(require=430)
            self.ctx.gc_mode = None
            try:
                self._prog = self.ctx.program(
                    vertex_shader=VERT,
                    tess_control_shader=TCS,
                    tess_evaluation_shader=TES,
                    fragment_shader=FRAG,
                )
                self.ctx.patch_vertices = 3
                self._tess = True
            except Exception as tess_exc:
                log_line(f"viewport tessellation failed, flat mesh: {tess_exc}")
                self._prog = self.ctx.program(vertex_shader=VERT_FLAT, fragment_shader=FRAG)
                self._tess = False
            pix = np.array([180, 180, 185, 128], np.uint8).tobytes()
            dummy = self.ctx.texture((1, 1), 4, pix)
            self._dummy_tex = dummy
            for key in ("alb", "n", "pack", "alb0", "n0", "pack0"):
                self._tex[key] = dummy
            for name, builder in (("cube", _cube), ("sphere", _sphere), ("plane", _plane)):
                p, n, u, i = builder()
                packed, ibo_idx = _packed_vao(p, n, u, i)
                vbo = self.ctx.buffer(packed.tobytes())
                ibo = self.ctx.buffer(ibo_idx.tobytes())
                self._prim_bufs.extend([vbo, ibo])
                self._vao[name] = self.ctx.vertex_array(
                    self._prog, [(vbo, "3f 3f 2f 4f", "in_pos", "in_nrm", "in_uv", "in_tan")], ibo)
            from qs.core import gpu as qs_gpu
            qs_gpu.attach_context(self.ctx)
            self._gl_ok = True
            if self.maps is not None:
                self._upload(self.maps)
            if self.mesh is not None:
                self._upload_mesh(self.mesh)
            log_line("viewport GL ready")
        except Exception as exc:
            log_line(f"viewport GL failed: {exc}")
            self._gl_ok = False

    def shutdown_gl(self):
        """Release GL objects while this widget's context is still current."""
        if not self._gl_ok:
            return
        if self.context() is not None:
            self.makeCurrent()
            try:
                self._release_user_textures()
                self._release_mesh_vao()
                for name in list(self._vao):
                    vao = self._vao.pop(name)
                    try:
                        vao.release()
                    except Exception:
                        pass
                for b in self._prim_bufs:
                    try:
                        b.release()
                    except Exception:
                        pass
                self._prim_bufs.clear()
                dummy = self._dummy_tex
                self._dummy_tex = None
                self._tex.clear()
                if dummy is not None:
                    try:
                        dummy.release()
                    except Exception:
                        pass
                if self._prog is not None:
                    try:
                        self._prog.release()
                    except Exception:
                        pass
                    self._prog = None
                from qs.core import gpu as qs_gpu
                qs_gpu.detach_context()
            except Exception:
                pass
            finally:
                self.doneCurrent()
        self._gl_ok = False

    def _release_user_textures(self):
        released = set()
        for k, t in list(self._tex.items()):
            if t is not None and t is not self._dummy_tex and id(t) not in released:
                try:
                    t.release()
                except Exception:
                    pass
                released.add(id(t))
            if self._dummy_tex is not None:
                self._tex[k] = self._dummy_tex
        self._src_token = None

    def _release_mesh_vao(self):
        vao = self._vao.pop("mesh", None)
        if vao is not None:
            try:
                vao.release()
            except Exception:
                pass
        for b in self._mesh_bufs:
            try:
                b.release()
            except Exception:
                pass
        self._mesh_bufs.clear()
        self._mesh_disp = False

    def set_maps(self, fin: dict | None, source: dict | None = None):
        self.maps = fin
        if source is not None or fin is None:
            self.source_maps = source
        if self._gl_ok and self.context() is not None:
            self.makeCurrent()
            try:
                if fin is not None:
                    self._upload(fin, self.source_maps)
                    if self.mesh is not None:
                        self._upload_mesh(self.mesh)
                else:
                    self._release_user_textures()
            finally:
                self.doneCurrent()
        self.update()

    def _mesh_displacement(self, positions, uvs):
        """One height per geometric point, averaged across UV-split duplicates.

        Sampling the height texture per corner in the shader moves seam
        duplicates by different amounts and opens cracks. Averaging the
        height over coincident positions first means a seam edge displaces
        as one edge on both sides.
        """
        height = None
        if self.maps is not None:
            height = self.maps.get("height")
        if height is None and self.source_maps is not None:
            height = self.source_maps.get("height")
        if height is None:
            return None
        arr = np.asarray(height, np.float32)
        if arr.ndim == 3:
            arr = arr[..., 0]
        if arr.size == 0 or len(positions) == 0:
            return None
        h_img, w_img = arr.shape[:2]
        uv = np.asarray(uvs, np.float32)
        xs = np.clip(np.round(uv[:, 0] * (w_img - 1)).astype(np.int64), 0, w_img - 1)
        ys = np.clip(np.round((1.0 - uv[:, 1]) * (h_img - 1)).astype(np.int64), 0, h_img - 1)
        sampled = arr[ys, xs].astype(np.float64)
        keys = np.round(np.asarray(positions, np.float64) * 1e6).astype(np.int64)
        _uniq, inv = np.unique(keys, axis=0, return_inverse=True)
        inv = np.asarray(inv).reshape(-1)
        sums = np.bincount(inv, weights=sampled)
        counts = np.bincount(inv).astype(np.float64)
        return (sums / np.maximum(counts, 1.0))[inv].astype(np.float32)

    def _upload_mesh(self, mesh: dict):
        self._release_mesh_vao()
        p = np.asarray(mesh["positions"], np.float32)
        if len(p) == 0:
            return
        center = (p.max(axis=0) + p.min(axis=0)) * 0.5
        span = max(float(np.linalg.norm(p - center, axis=1).max()), 1e-6)
        p = (p - center) * (1.25 / span)
        n = np.asarray(mesh["normals"], np.float32)
        u = np.asarray(mesh["uvs"], np.float32)
        i = np.ascontiguousarray(mesh["indices"], np.uint32)
        if i.size < 3:
            return
        disp = self._mesh_displacement(p, u)
        self._mesh_disp = disp is not None
        packed, ibo_idx = _packed_vao(p, n, u, i, disp=disp)
        vbo = self.ctx.buffer(packed.tobytes())
        ibo = self.ctx.buffer(ibo_idx.tobytes())
        self._mesh_bufs.extend([vbo, ibo])
        self._vao["mesh"] = self.ctx.vertex_array(
            self._prog, [(vbo, "3f 3f 2f 4f", "in_pos", "in_nrm", "in_uv", "in_tan")], ibo)

    def set_mesh(self, mesh: dict | None):
        self.mesh = mesh
        if self._gl_ok and self.context() is not None:
            self.makeCurrent()
            try:
                if mesh is not None:
                    try:
                        self._upload_mesh(mesh)
                        self.shape = "mesh"
                    except Exception as exc:
                        log_line(f"mesh upload failed: {exc}")
                        raise
                else:
                    self._release_mesh_vao()
                    if self.shape == "mesh":
                        self.shape = "sphere"
            finally:
                self.doneCurrent()
        elif mesh is None and self.shape == "mesh":
            self.shape = "sphere"
        self.update()

    def _rgba8(self, img):
        """Flip-Y RGBA8 into a reused scratch. No persistent float4 plane."""
        src = np.clip(np.asarray(img, np.float32), 0.0, 1.0)
        h, w = src.shape[:2]
        buf = self._u8_scratch
        if buf is None or buf.shape != (h, w, 4):
            self._u8_scratch = np.empty((h, w, 4), np.uint8)
            buf = self._u8_scratch
        if src.ndim == 2:
            u = (src * 255.0 + 0.5).astype(np.uint8)
            buf[::-1, :, 0] = u
            buf[::-1, :, 1] = u
            buf[::-1, :, 2] = u
            buf[:, :, 3] = 255
        else:
            c = min(int(src.shape[2]), 4)
            u = (src[..., :c] * 255.0 + 0.5).astype(np.uint8)
            buf[::-1, :, :c] = u
            if c == 1:
                buf[::-1, :, 1] = u[..., 0]
                buf[::-1, :, 2] = u[..., 0]
            elif c == 2:
                buf[::-1, :, 2] = 0
            if c < 4:
                buf[:, :, 3] = 255
        return buf

    def _put_tex(self, key, img):
        h, w = img.shape[:2]
        data = np.ascontiguousarray(self._rgba8(img)).tobytes()
        t = self._tex.get(key)
        size = tuple(t.size) if t is not None and hasattr(t, "size") else None
        same = (
            t is not None
            and t is not self._dummy_tex
            and size == (w, h)
            and getattr(t, "components", 4) == 4
        )
        if same:
            t.write(data)
            self.tex_writes += 1
            return
        if t is not None and t is not self._dummy_tex:
            try:
                t.release()
            except Exception:
                pass
        t = self.ctx.texture((w, h), 4, data)
        t.filter = (self.ctx.LINEAR, self.ctx.LINEAR)
        self._tex[key] = t
        self.tex_creates += 1

    def _upload(self, fin, source=None):
        def pack_of(maps):
            return np.dstack([maps["metal"], maps["rough"], maps["ao"], maps["height"]])

        self._put_tex("alb", fin["alb"])
        self._put_tex("n", fin["normal"] * 0.5 + 0.5)
        self._put_tex("pack", pack_of(fin))
        src = source if source is not None else fin
        token = (
            id(src),
            id(src.get("alb")),
            tuple(np.shape(src["alb"])),
            id(src.get("normal")),
            id(src.get("metal")),
            id(src.get("rough")),
            id(src.get("ao")),
            id(src.get("height")),
        )
        need_src = (
            self._src_token != token
            or self._tex.get("alb0") is self._dummy_tex
            or self._tex.get("n0") is self._dummy_tex
            or self._tex.get("pack0") is self._dummy_tex
        )
        if need_src:
            self._put_tex("alb0", src["alb"])
            self._put_tex("n0", src["normal"] * 0.5 + 0.5)
            self._put_tex("pack0", pack_of(src))
            self._src_token = token

    def paintGL(self):
        try:
            if self._gl_ok and self._prog is not None:
                self._paint_gl()
                return
            self._paint_cpu()
        except Exception as exc:
            log_line(f"viewport paint failed: {exc}")
            self._gl_ok = False

    def _paint_gl(self):
        # QOpenGLWidget renders into its own offscreen FBO (not FBO 0).
        fbo = self.ctx.detect_framebuffer(int(self.defaultFramebufferObject()))
        fbo.use()
        w, h = max(fbo.width, 1), max(fbo.height, 1)
        self.ctx.viewport = (0, 0, w, h)
        self.ctx.clear(0.08, 0.08, 0.09, 1.0)
        self.ctx.enable(self.ctx.DEPTH_TEST)
        eye, target = self._eye()
        aspect = w / h
        mvp = _persp(50, aspect, 0.05, 40) @ _look(eye, target, np.array([0, 1, 0], np.float32))
        model = np.eye(4, dtype=np.float32)
        self._prog["mvp"].write(mvp.T.tobytes())
        self._prog["model"].write(model.T.tobytes())
        self._prog["cam_pos"].value = tuple(eye.tolist())
        self._prog["light_dir"].value = tuple(self.light_dir)
        self._prog["exposure"].value = float(self.exposure)
        look = self.look
        self._prog["w1"].value = look.spec_primary_weight
        self._prog["sp"].value = look.spec_primary_roughness_scale
        self._prog["ss"].value = look.spec_secondary_roughness_scale
        self._prog["smax"].value = look.spec_secondary_max_roughness
        _ch_map = {
            "Lit": 0, "Diffuse": 1, "Normal": 2, "Roughness": 3,
            "AO": 4, "Displacement": 5, "Metallic": 6,
        }
        self._prog["view_channel"].value = _ch_map.get(self.channel, 0)
        _cmp = {"After": 0, "Before": 1, "Split": 2}
        self._prog["compare_mode"].value = _cmp.get(self.compare, 0)
        self._prog["split_px"].value = float(w) * 0.5
        if self._tess:
            self.ctx.patch_vertices = 3
            if self.displace and self.maps is not None:
                self._prog["tess_level"].value = self._tess_level()
                # Asset meshes already have UV-split verts. 0.12 of object size
                # plus tessellation interpolating across a seam tears the mesh open.
                self._prog["disp_scale"].value = 0.03 if self.shape == "mesh" else 0.12
                self._prog["disp_from_tan"].value = 1 if (self.shape == "mesh" and self._mesh_disp) else 0
            else:
                self._prog["tess_level"].value = 1.0
                self._prog["disp_scale"].value = 0.0
                self._prog["disp_from_tan"].value = 0
        if "alb" in self._tex:
            self._tex["alb"].use(0)
            self._tex["n"].use(1)
            self._tex["pack"].use(2)
            self._tex["alb0"].use(3)
            self._tex["n0"].use(4)
            self._tex["pack0"].use(5)
            self._prog["tex_alb"].value = 0
            self._prog["tex_n"].value = 1
            self._prog["tex_pack"].value = 2
            self._prog["tex_alb0"].value = 3
            self._prog["tex_n0"].value = 4
            self._prog["tex_pack0"].value = 5
        key = self.shape if self.shape in self._vao else "sphere"
        mode = self.ctx.PATCHES if self._tess else self.ctx.TRIANGLES
        self._vao[key].render(mode)

    def _tess_level(self) -> float:
        """Keep micro-triangle count in a preview budget. Plane/cube start coarse."""
        if self.shape == "mesh" and self.mesh is not None:
            # Do not subdivide asset meshes. Tessellating a UV-split triangle
            # interpolates UVs across atlas gutters and rips the surface apart.
            return 1.0
        return {"plane": 28.0, "cube": 16.0, "sphere": 5.0}.get(self.shape, 8.0)

    def _eye(self):
        target = np.array([0, 0, 0], np.float32)
        x = self.dist * math.cos(self.pitch) * math.sin(self.yaw)
        y = self.dist * math.sin(self.pitch)
        z = self.dist * math.cos(self.pitch) * math.cos(self.yaw)
        return np.array([x, y, z], np.float32), target

    def _paint_cpu(self):
        """Albedo card. Never QPainter on this widget — that AVs after ModernGL owns the FBO."""
        w, h = max(self.width(), 1), max(self.height(), 1)
        if self.maps is None:
            img = np.full((h, w, 3), 0.08, np.float32)
        else:
            from qs.core.io import resize
            img = resize(self.maps["alb"], w, h)
        u8 = np.ascontiguousarray((np.clip(img, 0, 1) * 255).astype(np.uint8))
        self._cpu_bytes = u8
        self._cpu_qimg = QImage(u8.data, w, h, w * 3, QImage.Format.Format_RGB888).copy()
        if getattr(self, "ctx", None) is not None:
            fbo = self.ctx.detect_framebuffer(int(self.defaultFramebufferObject()))
            fbo.use()
            self.ctx.viewport = (0, 0, max(fbo.width, 1), max(fbo.height, 1))
            self.ctx.clear(0.08, 0.08, 0.09, 1.0)
            return
        from PySide6.QtGui import QPainter
        painter = QPainter(self)
        painter.drawImage(0, 0, self._cpu_qimg)
        painter.end()

    def mousePressEvent(self, e: QMouseEvent):
        self._drag = (e.position().x(), e.position().y(), e.button())

    def mouseMoveEvent(self, e: QMouseEvent):
        if self._drag is None:
            return
        x, y, btn = self._drag
        dx = e.position().x() - x
        dy = e.position().y() - y
        self._drag = (e.position().x(), e.position().y(), btn)
        if btn == Qt.MouseButton.LeftButton:
            self.yaw -= dx * 0.01
            self.pitch = float(np.clip(self.pitch + dy * 0.01, -1.2, 1.2))
        elif btn == Qt.MouseButton.RightButton:
            lx, ly, lz = self.light_dir
            self.light_dir = (
                float(np.clip(lx + dx * 0.01, -1, 1)),
                float(np.clip(ly - dy * 0.01, 0.05, 1)),
                lz,
            )
        self.update()

    def mouseReleaseEvent(self, e: QMouseEvent):
        self._drag = None

    def wheelEvent(self, e: QWheelEvent):
        self.dist = float(np.clip(self.dist * (0.9 if e.angleDelta().y() > 0 else 1.1), 1.2, 12))
        self.update()
