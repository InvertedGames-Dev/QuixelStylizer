"""Fused Kuwahara on the GPU.

One compute dispatch per paint pass. The sector loop stays in the shader, then the
filtered maps are read back once. Oklab, the lit view, and the mask run on the CPU
after that readback, so a light drag or a palette drag does not launch the sector
kernel again. The CPU paint in quixel_stylize.py is the fallback and the match check.

The OpenGL context is per thread. A context created on the Tk thread and used from
the preview thread (or destroyed at process exit from the wrong thread) faults on
this machine. release() must run on the thread that created the context.
"""
import threading
import numpy as np

_STATE = {"tried": False, "ok": False, "reason": "", "diff": None}


_TENSOR = r"""
#version 430
layout(local_size_x = 16, local_size_y = 16) in;
layout(std430, binding = 0) readonly buffer InA { vec4 A[]; };
layout(std430, binding = 1) writeonly buffer OutG { vec4 G[]; };
uniform int width;
uniform int height;
uniform int tile_wrap;

int wrap_index(int x, int n) {
    // Integer % on this driver does not follow the GLSL sign rule for a negative
    // dividend, so a tile's top and left edges sampled the wrong texels.
    if (tile_wrap == 1) {
        return int(floor(mod(float(x), float(n))));
    }
    if (n <= 1) return 0;
    int period = 2 * (n - 1);
    int m = int(floor(mod(float(x), float(period))));
    return m < n ? m : period - m;
}

vec3 at(int x, int y) {
    x = wrap_index(x, width);
    y = wrap_index(y, height);
    return A[y * width + x].rgb;
}

void main() {
    ivec2 pix = ivec2(gl_GlobalInvocationID.xy);
    if (pix.x >= width || pix.y >= height) return;
    float jxx = 0.0, jxy = 0.0, jyy = 0.0;
    float wsum = 0.0;
    // Small box on the structure tensor. The angle is the edge tangent (gradient + 90 degrees).
    for (int oy = -2; oy <= 2; oy++) {
        for (int ox = -2; ox <= 2; ox++) {
            int x = pix.x + ox;
            int y = pix.y + oy;
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
uniform int width;
uniform int height;
uniform int radius;
uniform float sharpness;
uniform int tile_wrap;
uniform int classic;
uniform int anisotropic;

int wrap_index(int x, int n) {
    // Integer % on this driver does not follow the GLSL sign rule for a negative
    // dividend, so a tile's top and left edges sampled the wrong texels.
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

void main() {
    ivec2 pix = ivec2(gl_GlobalInvocationID.xy);
    if (pix.x >= width || pix.y >= height) return;
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
                    d.y *= 2.0;
                }
                float w = sector_weight(d, s, nsec);
                if (w <= 0.0) continue;
                int id = idx(pix.x + ox, pix.y + oy);
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


class _Gpu:
    def __init__(self, ctx):
        self.ctx = ctx
        self.tensor = ctx.compute_shader(_TENSOR)
        self.kuwa = ctx.compute_shader(_KUWAHARA)

    def _buf(self, array):
        return self.ctx.buffer(np.ascontiguousarray(array, np.float32).tobytes())

    def _read4(self, buf, h, w):
        self.ctx.finish()
        raw = np.frombuffer(buf.read(), np.float32).reshape(h, w, 4).copy()
        return raw

    def kuwahara_pass(self, alb, normal, pack, cavity, radius, sharpness, tile, classic, anisotropic):
        h, w = alb.shape[:2]
        alb_i = self._buf(_pack4(alb, 3))
        alb_o = self.ctx.buffer(reserve=h * w * 16)
        nrm_i = self._buf(_pack4(normal, 3))
        nrm_o = self.ctx.buffer(reserve=h * w * 16)
        pak_i = self._buf(_pack4(pack, 4))
        pak_o = self.ctx.buffer(reserve=h * w * 16)
        cav_i = self._buf(_pack4(cavity, 1))
        cav_o = self.ctx.buffer(reserve=h * w * 16)
        ang = self.ctx.buffer(reserve=h * w * 16)
        gx = (w + 15) // 16
        gy = (h + 15) // 16
        if anisotropic:
            alb_i.bind_to_storage_buffer(0)
            ang.bind_to_storage_buffer(1)
            self.tensor["width"].value = w
            self.tensor["height"].value = h
            self.tensor["tile_wrap"].value = 1 if tile else 0
            self.tensor.run(group_x=gx, group_y=gy)
            self.ctx.finish()
        for binding, buf in (
            (0, alb_i), (1, alb_o), (2, nrm_i), (3, nrm_o),
            (4, pak_i), (5, pak_o), (6, cav_i), (7, cav_o), (8, ang),
        ):
            buf.bind_to_storage_buffer(binding)
        k = self.kuwa
        k["width"].value = w
        k["height"].value = h
        k["radius"].value = int(radius)
        k["sharpness"].value = float(sharpness)
        k["tile_wrap"].value = 1 if tile else 0
        k["classic"].value = 1 if classic else 0
        k["anisotropic"].value = 1 if anisotropic else 0
        k.run(group_x=gx, group_y=gy)
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
    """Drop this thread's context. Call it on that same thread, before the thread ends."""
    gpu = _thread_gpu()
    if gpu is None:
        return
    try:
        gpu.ctx.release()
    except Exception:
        pass
    _local.gpu = None


def _new_gpu():
    import moderngl
    ctx = moderngl.create_standalone_context(require=430)
    return _Gpu(ctx)


def _ensure():
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
            # The gather and filter2D agree to well under a 2/255 step on a tiled disc.
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
    """One generalized Kuwahara pass against quixel_stylize.kuwahara_pass. Returns max abs RGB error."""
    import quixel_stylize as qs
    rng = np.random.RandomState(0)
    h = w = 48
    alb = rng.rand(h, w, 3).astype(np.float32)
    guides = [rng.rand(h, w, 3).astype(np.float32), rng.rand(h, w).astype(np.float32)]
    r = 3
    try:
        cpu_alb, cpu_guides = qs.kuwahara_pass(alb, guides, r, 8.0, True, classic=False)
        gpu_alb, gpu_n, gpu_p, _cav = _thread_gpu().kuwahara_pass(
            alb, guides[0], np.dstack([guides[1], guides[1], guides[1], guides[1]]),
            np.zeros((h, w), np.float32), r, 8.0, True, False, False)
    except Exception as exc:
        _STATE["reason"] = f"match check failed ({exc})"
        return None
    return float(np.max(np.abs(cpu_alb - gpu_alb)))


def paint(src, st, cancel=None):
    """Same dict as quixel_stylize.paint_stage, for the Kuwahara family including anisotropic."""
    import quixel_stylize as qs
    if not _ensure() or _thread_gpu() is None:
        raise RuntimeError(_STATE["reason"] or "GPU paint unavailable")
    st = dict(qs.DEFAULTS, **st)
    tile = bool(st["tile"])
    W, H = src["W"], src["H"]
    r_px = max(1, int(round(qs.scaled(st["radius"], max(W, H)))))
    r_px = min(r_px, 48)
    passes = max(1, int(st["passes"]))
    method = st["filter"]
    classic = method == "kuwahara_classic"
    anisotropic = method == "anisotropic"
    alb = np.array(src["alb"], np.float32, copy=True)
    normal = np.array(src["normal"], np.float32, copy=True)
    height = np.array(src["height"], np.float32, copy=True)
    rough = np.array(src["rough"], np.float32, copy=True)
    metal = np.array(src["metal"], np.float32, copy=True)
    ao = np.array(src["ao"], np.float32, copy=True)
    cav = src["cavity"]
    cav_img = np.zeros((H, W), np.float32) if cav is None else np.array(cav, np.float32, copy=True)
    sharpness = float(st["sharpness"])
    for _i in range(passes):
        if cancel and cancel():
            raise qs.Cancelled()
        pack = np.dstack([height, rough, metal, ao])
        alb, normal, pack, cav_img = _thread_gpu().kuwahara_pass(
            alb, normal, pack, cav_img, r_px, sharpness, tile, classic, anisotropic)
        height, rough, metal, ao = pack[..., 0], pack[..., 1], pack[..., 2], pack[..., 3]
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
