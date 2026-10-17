"""Fused Kuwahara on the GPU. SSBOs are reused so a pass does not leak VRAM."""
from __future__ import annotations

import threading

import numpy as np

from qs.core.filter import Cancelled
from qs.core.recipe import DEFAULTS, scaled

_STATE = {"tried": False, "ok": False, "reason": "", "diff": None}
# When the Qt widget attaches, compute stays on that thread. Workers never
# create a second GL context (that is the v1.5 deadlock class).
_ATTACHED = False
_ATTACH_IDENT = None

_TENSOR = r"""
#version 430
layout(local_size_x = 16, local_size_y = 16) in;
layout(std430, binding = 0) readonly buffer InA { vec4 A[]; };
layout(std430, binding = 1) writeonly buffer OutG { vec4 G[]; };
layout(std430, binding = 2) readonly buffer IslIn { vec4 IslI[]; };
uniform int width;
uniform int height;
uniform int tile_wrap;
uniform int use_islands;

int wrap_index(int x, int n) {
    if (tile_wrap == 1) {
        return int(floor(mod(float(x), float(n))));
    }
    if (n <= 1) return 0;
    int period = 2 * (n - 1);
    int m = int(floor(mod(float(x), float(period))));
    return m < n ? m : period - m;
}

int island_id(int x, int y) {
    if (use_islands == 0) return 1;
    if (x < 0 || y < 0 || x >= width || y >= height) return -1;
    return int(floor(IslI[y * width + x].x + 0.5));
}

vec3 at(int x, int y) {
    x = wrap_index(x, width);
    y = wrap_index(y, height);
    return A[y * width + x].rgb;
}

void main() {
    ivec2 pix = ivec2(gl_GlobalInvocationID.xy);
    if (pix.x >= width || pix.y >= height) return;
    int my_id = island_id(pix.x, pix.y);
    if (use_islands == 1 && my_id <= 0) {
        G[pix.y * width + pix.x] = vec4(0.0);
        return;
    }
    float jxx = 0.0, jxy = 0.0, jyy = 0.0;
    float wsum = 0.0;
    for (int oy = -2; oy <= 2; oy++) {
        for (int ox = -2; ox <= 2; ox++) {
            int x = pix.x + ox;
            int y = pix.y + oy;
            if (use_islands == 1 && island_id(x, y) != my_id) continue;
            vec3 c1 = at(x + 1, y);
            vec3 c0 = at(x - 1, y);
            vec3 r1 = at(x, y + 1);
            vec3 r0 = at(x, y - 1);
            float gx = dot(c1 - c0, vec3(0.2126, 0.7152, 0.0722)) * 0.5;
            float gy = dot(r1 - r0, vec3(0.2126, 0.7152, 0.0722)) * 0.5;
            jxx += gx * gx;
            jxy += gx * gy;
            jyy += gy * gy;
            wsum += 1.0;
        }
    }
    wsum = max(wsum, 1.0);
    jxx /= wsum; jxy /= wsum; jyy /= wsum;
    float ang = 0.5 * atan(2.0 * jxy, jxx - jyy) + 1.57079632679;
    G[pix.y * width + pix.x] = vec4(ang, 0.0, 0.0, 0.0);
}
"""

_KUWAHARA = r"""
#version 430
layout(local_size_x = 16, local_size_y = 16) in;
layout(std430, binding = 0) readonly buffer AlbIn { vec4 AlbI[]; };
layout(std430, binding = 1) writeonly buffer AlbOut { vec4 AlbO[]; };
layout(std430, binding = 2) readonly buffer NrmIn { vec4 NrmI[]; };
layout(std430, binding = 3) writeonly buffer NrmOut { vec4 NrmO[]; };
layout(std430, binding = 4) readonly buffer PakIn { vec4 PakI[]; };
layout(std430, binding = 5) writeonly buffer PakOut { vec4 PakO[]; };
layout(std430, binding = 6) readonly buffer CavIn { vec4 CavI[]; };
layout(std430, binding = 7) writeonly buffer CavOut { vec4 CavO[]; };
layout(std430, binding = 8) readonly buffer AngIn { vec4 AngI[]; };
layout(std430, binding = 9) readonly buffer IslIn { vec4 IslI[]; };
uniform int width;
uniform int height;
uniform int radius;
uniform float sharpness;
uniform int tile_wrap;
uniform int classic;
uniform int anisotropic;
uniform float stretch;
uniform int use_islands;

int wrap_index(int x, int n) {
    if (tile_wrap == 1) {
        return int(floor(mod(float(x), float(n))));
    }
    if (n <= 1) return 0;
    int period = 2 * (n - 1);
    int m = int(floor(mod(float(x), float(period))));
    return m < n ? m : period - m;
}

int idx(int x, int y) {
    return wrap_index(y, height) * width + wrap_index(x, width);
}

const float PI = 3.14159265358979323846;

float sector_weight(vec2 d, int sector, int nsec) {
    if (classic == 1) {
        int sx = (sector % 2 == 0) ? -1 : 1;
        int sy = (sector < 2) ? -1 : 1;
        if (d.x * float(sx) >= 0.0 && d.y * float(sy) >= 0.0) return 1.0;
        return 0.0;
    }
    float ang = atan(d.y, d.x);
    float delta = ang - (2.0 * PI * float(sector) / float(nsec));
    delta = atan(sin(delta), cos(delta));
    float dist = length(d);
    float sigma = max(float(radius) * 0.5, 0.5);
    float g = exp(-(dist * dist) / (2.0 * sigma * sigma));
    if (dist > float(radius) + 0.5) g = 0.0;
    float w = 0.0;
    if (abs(delta) < (PI * 0.25)) w = pow(cos(2.0 * delta), 2.0);
    if (d.x == 0.0 && d.y == 0.0) w = 1.0;
    return w * g;
}

vec2 rot(vec2 d, float a) {
    float c = cos(a), s = sin(a);
    return vec2(c * d.x - s * d.y, s * d.x + c * d.y);
}

int island_at(int x, int y) {
    if (use_islands == 0) return 1;
    if (x < 0 || y < 0 || x >= width || y >= height) return -1;
    return int(floor(IslI[y * width + x].x + 0.5));
}

void main() {
    ivec2 pix = ivec2(gl_GlobalInvocationID.xy);
    if (pix.x >= width || pix.y >= height) return;
    int my_id = island_at(pix.x, pix.y);
    if (use_islands == 1 && my_id <= 0) {
        int o = pix.y * width + pix.x;
        AlbO[o] = AlbI[o];
        NrmO[o] = NrmI[o];
        PakO[o] = PakI[o];
        CavO[o] = CavI[o];
        return;
    }
    int nsec = (classic == 1) ? 4 : 8;
    int r = radius;
    float tang = 0.0;
    if (anisotropic == 1) tang = AngI[pix.y * width + pix.x].x;

    float logw[8];
    vec3 meanA[8];
    vec3 meanN[8];
    vec4 meanP[8];
    float meanC[8];
    for (int s = 0; s < 8; s++) {
        logw[s] = 0.0;
        meanA[s] = vec3(0.0);
        meanN[s] = vec3(0.0);
        meanP[s] = vec4(0.0);
        meanC[s] = 0.0;
    }

    for (int s = 0; s < nsec; s++) {
        vec3 acc = vec3(0.0), acc2 = vec3(0.0);
        vec3 accN = vec3(0.0);
        vec4 accP = vec4(0.0);
        float accC = 0.0;
        float wsum = 0.0;
        for (int oy = -r; oy <= r; oy++) {
            for (int ox = -r; ox <= r; ox++) {
                vec2 d = vec2(float(ox), float(oy));
                if (anisotropic == 1) {
                    d = rot(d, -tang);
                    d.y *= stretch;
                }
                float w = sector_weight(d, s, nsec);
                if (w <= 0.0) continue;
                int sx = pix.x + ox;
                int sy = pix.y + oy;
                int id;
                if (use_islands == 1) {
                    if (island_at(sx, sy) != my_id) continue;
                    id = sy * width + sx;
                } else {
                    id = idx(sx, sy);
                }
                vec3 c = AlbI[id].rgb;
                acc += c * w;
                acc2 += c * c * w;
                accN += NrmI[id].rgb * w;
                accP += PakI[id] * w;
                accC += CavI[id].x * w;
                wsum += w;
            }
        }
        wsum = max(wsum, 1e-8);
        vec3 m = acc / wsum;
        vec3 m2 = acc2 / wsum;
        float var = max(dot(m2 - m * m, vec3(1.0)), 0.0) + 1e-5;
        logw[s] = -log(var);
        meanA[s] = m;
        meanN[s] = accN / wsum;
        meanP[s] = accP / wsum;
        meanC[s] = accC / wsum;
    }

    float mx = logw[0];
    for (int s = 1; s < nsec; s++) mx = max(mx, logw[s]);
    float wsum = 0.0;
    float ww[8];
    for (int s = 0; s < nsec; s++) {
        float w;
        if (classic == 1) w = (logw[s] == mx) ? 1.0 : 0.0;
        else w = exp((logw[s] - mx) * (sharpness * 0.5));
        ww[s] = w;
        wsum += w;
    }
    wsum = max(wsum, 1e-8);
    vec3 oA = vec3(0.0);
    vec3 oN = vec3(0.0);
    vec4 oP = vec4(0.0);
    float oC = 0.0;
    for (int s = 0; s < nsec; s++) {
        float w = ww[s] / wsum;
        oA += meanA[s] * w;
        oN += meanN[s] * w;
        oP += meanP[s] * w;
        oC += meanC[s] * w;
    }
    int o = pix.y * width + pix.x;
    AlbO[o] = vec4(oA, 1.0);
    NrmO[o] = vec4(oN, 0.0);
    PakO[o] = oP;
    CavO[o] = vec4(oC, 0.0, 0.0, 0.0);
}
"""


def _pack4(img, channels):
    arr = np.asarray(img, np.float32)
    if arr.ndim == 2:
        arr = arr[..., None]
    h, w = arr.shape[:2]
    out = np.zeros((h, w, 4), np.float32)
    n = min(channels, arr.shape[2], 4)
    out[..., :n] = arr[..., :n]
    return np.ascontiguousarray(out)


class Gpu:
    def __init__(self, ctx):
        self.ctx = ctx
        self.tensor = ctx.compute_shader(_TENSOR)
        self.kuwa = ctx.compute_shader(_KUWAHARA)
        self._bufs = {}

    def _ssbo(self, key, nbytes):
        buf = self._bufs.get(key)
        if buf is None or buf.size != nbytes:
            if buf is not None:
                buf.release()
            buf = self.ctx.buffer(reserve=nbytes)
            self._bufs[key] = buf
        return buf

    def _write4(self, key, array):
        packed = _pack4(array, array.shape[2] if array.ndim == 3 else 1)
        buf = self._ssbo(key, packed.nbytes)
        buf.write(packed.tobytes())
        return buf

    def _wait_fence(self):
        """Park until the last dispatch is done. Yield to Qt between polls."""
        yfn = getattr(_local, "yield_fn", None)
        try:
            fence = self.ctx.fence()
        except Exception:
            self.ctx.finish()
            return
        if yfn is None:
            fence.wait()
            return
        while not fence.wait(timeout=0):
            yfn()

    def _read4(self, buf, h, w):
        self._wait_fence()
        return np.frombuffer(buf.read(), np.float32).reshape(h, w, 4).copy()

    def release_buffers(self):
        for buf in self._bufs.values():
            try:
                buf.release()
            except Exception:
                pass
        self._bufs.clear()

    def kuwahara_pass(self, alb, normal, pack, cavity, radius, sharpness, tile, classic, anisotropic,
                      stretch=1.0, island_ids=None):
        h, w = alb.shape[:2]
        nbytes = h * w * 16
        alb_i = self._write4("alb_i", alb)
        alb_o = self._ssbo("alb_o", nbytes)
        nrm_i = self._write4("nrm_i", normal)
        nrm_o = self._ssbo("nrm_o", nbytes)
        pak_i = self._write4("pak_i", pack)
        pak_o = self._ssbo("pak_o", nbytes)
        cav_i = self._write4("cav_i", cavity)
        cav_o = self._ssbo("cav_o", nbytes)
        ang = self._ssbo("ang", nbytes)
        use_isl = island_ids is not None
        if use_isl:
            isl = self._write4("isl", np.asarray(island_ids, np.float32))
        else:
            isl = self._ssbo("isl", 16)
        gx = (w + 15) // 16
        gy = (h + 15) // 16
        if anisotropic:
            alb_i.bind_to_storage_buffer(0)
            ang.bind_to_storage_buffer(1)
            isl.bind_to_storage_buffer(2)
            self.tensor["width"].value = w
            self.tensor["height"].value = h
            self.tensor["tile_wrap"].value = 1 if tile else 0
            self.tensor["use_islands"].value = 1 if use_isl else 0
            self.tensor.run(group_x=gx, group_y=gy)
            self._wait_fence()
        for binding, buf in (
            (0, alb_i), (1, alb_o), (2, nrm_i), (3, nrm_o),
            (4, pak_i), (5, pak_o), (6, cav_i), (7, cav_o), (8, ang),
        ):
            buf.bind_to_storage_buffer(binding)
        isl.bind_to_storage_buffer(9)
        k = self.kuwa
        k["width"].value = w
        k["height"].value = h
        k["radius"].value = int(radius)
        k["sharpness"].value = float(sharpness)
        k["tile_wrap"].value = 1 if tile else 0
        k["classic"].value = 1 if classic else 0
        k["anisotropic"].value = 1 if anisotropic else 0
        k["stretch"].value = float(stretch if anisotropic else 1.0)
        k["use_islands"].value = 1 if use_isl else 0
        k.run(group_x=gx, group_y=gy)
        self._wait_fence()
        alb_n = self._read4(alb_o, h, w)[..., :3]
        nrm_n = self._read4(nrm_o, h, w)[..., :3]
        pak_n = self._read4(pak_o, h, w)
        cav_n = self._read4(cav_o, h, w)[..., 0]
        return alb_n, nrm_n, pak_n, cav_n


_local = threading.local()
_lock = threading.Lock()


def _thread_gpu():
    return getattr(_local, "gpu", None)


def release():
    gpu = _thread_gpu()
    if gpu is None:
        return
    try:
        gpu.release_buffers()
        if not _ATTACHED:
            gpu.ctx.release()
    except Exception:
        pass
    _local.gpu = None


def detach_context():
    """Drop compute SSBOs for the Qt widget. Do not glDelete a dead context's names."""
    global _ATTACHED, _ATTACH_IDENT
    gpu = _thread_gpu()
    if gpu is not None:
        try:
            gpu.release_buffers()
            for sh in (gpu.tensor, gpu.kuwa):
                try:
                    sh.release()
                except Exception:
                    pass
        except Exception:
            pass
        _local.gpu = None
    if threading.get_ident() == _ATTACH_IDENT:
        _ATTACHED = False
        _ATTACH_IDENT = None


def attach_context(ctx):
    """Use an existing ModernGL context (the Qt widget's). Call from that thread."""
    global _ATTACHED, _ATTACH_IDENT
    old = _thread_gpu()
    # Only delete SSBOs if they belong to this same live context. A previous
    # QOpenGLWidget's Gpu still sitting on _local would glDelete recycled names
    # on the new widget (pytest smoke then e2e in one process).
    if old is not None and old.ctx is ctx:
        try:
            old.release_buffers()
        except Exception:
            pass
    _local.gpu = Gpu(ctx)
    _ATTACHED = True
    _ATTACH_IDENT = threading.get_ident()
    with _lock:
        _STATE["tried"] = True
        _STATE["ok"] = True
        _STATE["reason"] = "attached Qt/ModernGL context"


def set_yield(fn):
    """Install a no-arg callback invoked while a compute fence is outstanding."""
    _local.yield_fn = fn


def attached() -> bool:
    return _ATTACHED and threading.get_ident() == _ATTACH_IDENT


def _new_gpu():
    import moderngl
    ctx = moderngl.create_standalone_context(require=430)
    return Gpu(ctx)


def _ensure():
    if _ATTACHED:
        return attached() and _thread_gpu() is not None
    if _thread_gpu() is not None:
        return _STATE["ok"]
    with _lock:
        first = not _STATE["tried"]
        if first:
            _STATE["tried"] = True
            try:
                _local.gpu = _new_gpu()
            except ImportError:
                _STATE["reason"] = "moderngl is not installed"
                return False
            except Exception as exc:
                _STATE["reason"] = f"no OpenGL 4.3 compute context ({exc})"
                return False
            diff = _match_cpu()
            _STATE["diff"] = diff
            if diff is None:
                _STATE["reason"] = "GPU match check did not run"
                release()
                return False
            if diff > (2.0 / 255.0):
                _STATE["reason"] = f"GPU Kuwahara differed from the CPU path by {diff:.5f}"
                release()
                return False
            _STATE["ok"] = True
            _STATE["reason"] = f"ModernGL compute, CPU match max abs {diff:.3g}"
            return True
        ok = _STATE["ok"]
    if not ok:
        return False
    try:
        _local.gpu = _new_gpu()
    except Exception as exc:
        _STATE["reason"] = f"no OpenGL 4.3 compute context ({exc})"
        return False
    return True


def available():
    return _ensure()


def status():
    _ensure()
    return dict(_STATE)


def _match_cpu():
    from qs.core.filter import kuwahara_pass as cpu_pass
    rng = np.random.RandomState(0)
    h = w = 48
    alb = rng.rand(h, w, 3).astype(np.float32)
    guides = [rng.rand(h, w, 3).astype(np.float32), rng.rand(h, w).astype(np.float32)]
    r = 3
    try:
        cpu_alb, _cpu_guides = cpu_pass(alb, guides, r, 8.0, True, classic=False)
        gpu_alb, _n, _p, _c = _thread_gpu().kuwahara_pass(
            alb, guides[0], np.dstack([guides[1], guides[1], guides[1], guides[1]]),
            np.zeros((h, w), np.float32), r, 8.0, True, False, False)
    except Exception as exc:
        _STATE["reason"] = f"match check failed ({exc})"
        return None
    return float(np.max(np.abs(cpu_alb - gpu_alb)))


def paint(src, st, cancel=None):
    from qs.core.stylize import paint_copy
    if not _ensure() or _thread_gpu() is None:
        raise RuntimeError(_STATE["reason"] or "GPU paint unavailable")
    st = dict(DEFAULTS, **st)
    if float(st.get("radius") or 0) <= 0:
        return paint_copy(src)
    tile = bool(st["tile"])
    W, H = src["W"], src["H"]
    r_px = max(1, int(round(scaled(st["radius"], max(W, H)))))
    r_px = min(r_px, 48)
    passes = max(1, int(st["passes"]))
    method = st["filter"]
    classic = method == "kuwahara_classic"
    stroke = float(st.get("stroke") or 0)
    if stroke > 0 and method in ("kuwahara", "kuwahara_classic", "anisotropic"):
        anisotropic = True
        stretch = 1.0 + 4.0 * min(max(stroke, 0.0), 1.0)
    elif method == "anisotropic":
        anisotropic = True
        stretch = 2.0
    else:
        anisotropic = False
        stretch = 1.0
    ids = st.get("_island_ids")
    if ids is None and st.get("island_mask"):
        from qs.core.io import load_gray
        ids = np.round(load_gray(st["island_mask"]) * 255).astype(np.int32)
    if ids is not None:
        from qs.core.io import resize
        ids = np.asarray(ids)
        if ids.shape[:2] != (H, W):
            ids = np.round(resize(ids.astype(np.float32), W, H)).astype(np.int32)
        tile = False
    def _f32(img):
        arr = np.asarray(img, np.float32)
        return arr if arr.flags.c_contiguous else np.ascontiguousarray(arr)

    alb = _f32(src["alb"])
    normal = _f32(src["normal"])
    height = _f32(src["height"])
    rough = _f32(src["rough"])
    metal = _f32(src["metal"])
    ao = _f32(src["ao"])
    cav = src["cavity"]
    cav_img = np.zeros((H, W), np.float32) if cav is None else _f32(cav)
    sharpness = float(st["sharpness"])
    gpu = _thread_gpu()
    for _i in range(passes):
        if cancel and cancel():
            raise Cancelled()
        pack = np.dstack([height, rough, metal, ao])
        alb, normal, pack, cav_img = gpu.kuwahara_pass(
            alb, normal, pack, cav_img, r_px, sharpness, tile, classic, anisotropic, stretch,
            island_ids=ids)
        height, rough, metal, ao = pack[..., 0], pack[..., 1], pack[..., 2], pack[..., 3]
    if ids is not None:
        from qs.core.filter import dilate_islands
        r_dil = max(2, r_px // 2)
        alb = dilate_islands(alb, ids, r_dil)
        normal = dilate_islands(normal, ids, r_dil)
        height = dilate_islands(height, ids, r_dil)
        rough = dilate_islands(rough, ids, r_dil)
        metal = dilate_islands(metal, ids, r_dil)
        ao = dilate_islands(ao, ids, r_dil)
        if cav is not None:
            cav_img = dilate_islands(cav_img, ids, r_dil)
    return {
        "alb": alb.astype(np.float32),
        "normal": normal.astype(np.float32),
        "height": height.astype(np.float32),
        "rough": rough.astype(np.float32),
        "metal": metal.astype(np.float32),
        "ao": ao.astype(np.float32),
        "cavity": None if cav is None else cav_img.astype(np.float32),
        "r_px": r_px,
    }
