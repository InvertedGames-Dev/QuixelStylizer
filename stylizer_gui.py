"""Tkinter GUI for QuixelStylizer with a live preview (launched by QuixelStylizer.bat with no arguments)."""
import os
import sys
import json
import math
import time
import queue
import threading
import traceback
from collections import OrderedDict
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, colorchooser

import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import quixel_stylize as qs  # noqa: E402

try:
    from PIL import Image, ImageTk
except ImportError:  # preview needs Pillow
    Image = ImageTk = None

SLIDERS = [  # key, label, min, max, step
    ("radius", "Paint radius (px @1024)", 1, 20, 0.5),
    ("passes", "Paint passes", 1, 4, 1),
    ("sharpness", "Paint edge sharpness", 2, 16, 0.5),
    ("ao_strength", "AO strength", 0, 1, 0.05),
    ("edge_strength", "Edge highlight strength", 0, 1, 0.05),
    ("saturation", "Saturation", 0.5, 2.0, 0.05),
    ("value_compression", "Value compression", 0, 0.6, 0.01),
    ("tint_strength", "Warm/cool tint", 0, 1, 0.05),
    ("posterize", "Posterize levels (0=off)", 0, 24, 1),
    ("roughness_flatten", "Roughness flatten", 0, 1, 0.05),
    ("normal_soften", "Normal softening", 0, 1, 0.05),
    ("normal_strength", "Normal strength", 0, 2, 0.05),
    ("height_soften", "Displacement softening", 0, 1, 0.05),
    ("mix_normal", "Normal paint mix", 0, 1, 0.05),
    ("mix_height", "Height paint mix", 0, 1, 0.05),
    ("mix_rough", "Roughness paint mix", 0, 1, 0.05),
    ("mix_metal", "Metallic paint mix", 0, 1, 0.05),
    ("mix_ao", "AO paint mix", 0, 1, 0.05),
    ("detail_restore", "Detail restore", 0, 1, 0.05),
    ("palette_strength", "Palette strength", 0, 1, 0.05),
    ("palette_hardness", "Palette hardness", 1, 32, 0.5),
    ("seam_fade", "Seam fade", 0, 1, 0.05),
    ("exposure", "Lit exposure (stops)", -2, 2, 0.05),
]

# Dark window. Tk's default theme is a light grey; clam is the one that actually takes these colours.
_BG = "#1c1c1c"
_PANEL = "#242424"
_FG = "#e6e6e6"
_MUTED = "#9a9a9a"
_FIELD = "#121212"
_ACCENT = "#4c8cbf"
_BUTTON = "#333333"


def apply_dark_theme(root):
    style = ttk.Style(root)
    try:
        style.theme_use("clam")
    except tk.TclError:
        pass
    root.configure(bg=_BG)
    root.option_add("*Background", _BG)
    root.option_add("*Foreground", _FG)
    root.option_add("*selectBackground", _ACCENT)
    root.option_add("*selectForeground", "#ffffff")
    root.option_add("*TCombobox*Listbox.background", _FIELD)
    root.option_add("*TCombobox*Listbox.foreground", _FG)
    root.option_add("*TCombobox*Listbox.selectBackground", _ACCENT)
    style.configure(".", background=_BG, foreground=_FG, fieldbackground=_FIELD,
                    troughcolor="#2a2a2a", bordercolor="#3a3a3a", lightcolor=_PANEL, darkcolor=_PANEL)
    style.configure("TFrame", background=_BG)
    style.configure("TLabel", background=_BG, foreground=_FG)
    style.configure("TButton", background=_BUTTON, foreground=_FG, padding=4, bordercolor="#3a3a3a")
    style.map("TButton", background=[("active", "#3e3e3e"), ("disabled", "#2a2a2a")],
              foreground=[("disabled", "#777777")])
    style.configure("TCheckbutton", background=_BG, foreground=_FG)
    style.map("TCheckbutton", background=[("active", _BG)])
    style.configure("TRadiobutton", background=_BG, foreground=_FG)
    style.map("TRadiobutton", background=[("active", _BG)])
    style.configure("TLabelframe", background=_BG, foreground=_FG, bordercolor="#3a3a3a")
    style.configure("TLabelframe.Label", background=_BG, foreground=_FG)
    style.configure("TCombobox", fieldbackground=_FIELD, background=_BUTTON, foreground=_FG, arrowcolor=_FG)
    style.map("TCombobox", fieldbackground=[("readonly", _FIELD)], foreground=[("readonly", _FG)])
    style.configure("TEntry", fieldbackground=_FIELD, foreground=_FG, insertcolor=_FG)
    style.configure("TScrollbar", background=_BUTTON, troughcolor=_BG, arrowcolor=_FG)
    style.configure("Horizontal.TProgressbar", background=_ACCENT, troughcolor="#2a2a2a")


def dark_scale(parent, **kw):
    sc = tk.Scale(parent, bg=_BG, fg=_FG, troughcolor="#2e2e2e", highlightthickness=0,
                  activebackground=_ACCENT, sliderrelief="flat", **kw)
    return sc


def _hex_rgb(rgb):
    r, g, b = [int(round(float(c) * 255)) for c in rgb[:3]]
    return f"#{r:02x}{g:02x}{b:02x}"
INT_KEYS = {"passes", "posterize", "size"}
VIEWS = ["Diffuse", "Lit", "Normal", "Roughness", "AO", "Displacement", "Metallic"]
PREVIEW_SIZES = [256, 512, 1024]
DEFAULT_PREVIEW_SIZE = 512


def open_in_explorer(path):
    try:
        if os.name == "nt":
            os.startfile(path)  # opens an Explorer window
    except OSError as e:
        qs.log_line(f"GUI: could not open {path}: {e}")


def enable_dpi_awareness():
    if os.name == "nt":
        try:
            import ctypes
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except Exception:
            pass


# ======================================================================================
# Preview engine: one worker thread, latest-request-wins, cached pipeline stages.
# Uses exactly the same load/paint/finish functions as Convert, just at a smaller size.
# ======================================================================================
def _grey3(a):
    return np.repeat(np.clip(a, 0, 1)[..., None], 3, axis=2)


def view_image(view, maps, light, exposure=0.0, shade_ao=False):
    """maps: dict with alb, normal, rough, metal, ao, height (stage output or source)."""
    if view == "Lit":
        return qs.shade_lit(maps["alb"], maps["normal"], maps["rough"], maps["metal"], *light,
                            ao=maps.get("ao"), exposure=exposure, shade_ao=shade_ao)
    if view == "Normal":
        return np.clip(maps["normal"] * 0.5 + 0.5, 0, 1)
    if view == "Roughness":
        return _grey3(maps["rough"])
    if view == "AO":
        return _grey3(maps["ao"])
    if view == "Displacement":
        return _grey3(maps["height"])
    if view == "Metallic":
        return _grey3(maps["metal"])
    return maps["alb"]


def _roll_half(img):
    h, w = img.shape[:2]
    return np.roll(np.roll(img, h // 2, axis=0), w // 2, axis=1)


def compose(after, before, job):
    """Split/before/after, half-tile offset, tile 2x2, zoom, fit to canvas. Returns (u8 image, mapping)."""
    mode, split = job["compare"], float(job["split"])
    base_h, base_w = after.shape[:2]
    offset = bool(job.get("offset"))
    if offset:
        after = _roll_half(after)
        before = _roll_half(before)
    if job["tile"]:
        after = np.tile(after, (2, 2, 1))
        before = np.tile(before, (2, 2, 1))
    H, W = after.shape[:2]
    if mode == "Before":
        img = before
    elif mode == "Split":
        x = int(round(split * W))
        img = after.copy()
        img[:, :x] = before[:, :x]
    else:
        img = after
    img = qs.to_u8(img)
    cw, ch = max(int(job["cw"]), 50), max(int(job["ch"]), 50)
    zoom = job["zoom"]
    if zoom == "Fit":
        z = min(cw / W, ch / H)
        x0 = y0 = 0
        dw, dh = max(int(W * z), 1), max(int(H * z), 1)
        img = cv2.resize(img, (dw, dh), interpolation=cv2.INTER_AREA if z < 1 else cv2.INTER_LINEAR)
    else:
        z = 1.0 if zoom == "1x" else 2.0
        vw, vh = min(W, int(cw / z)), min(H, int(ch / z))
        x0, y0 = (W - vw) // 2, (H - vh) // 2
        img = img[y0:y0 + vh, x0:x0 + vw]
        if z != 1.0:
            img = cv2.resize(img, (int(vw * z), int(vh * z)), interpolation=cv2.INTER_NEAREST)
        dw, dh = img.shape[1], img.shape[0]
    if mode == "Split":
        xd = int(round((split * W - x0) * z))
        if 0 <= xd < dw:
            img = np.ascontiguousarray(img)
            img[:, max(xd - 1, 0):xd + 1] = (255, 255, 255)
            cv2.putText(img, "before", (max(xd - 70, 2), 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(img, "after", (min(xd + 6, dw - 50), 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return img, {"z": z, "x0": x0, "y0": y0, "dw": dw, "dh": dh, "W": W, "H": H,
                 "base_w": base_w, "base_h": base_h, "offset": offset, "tiled": bool(job["tile"])}


class PreviewEngine:
    def __init__(self, on_result):
        self.on_result = on_result
        self.cond = threading.Condition()
        self.pending = None
        self.seq = 0
        self.load_cache = OrderedDict()
        self.paint_cache = OrderedDict()
        self.fin = (None, None)
        self.before = (None, None)
        self.last = None  # last finished stage outputs (for tests / match checks)
        self.stopping = False
        self.thread = threading.Thread(target=self._loop, daemon=False, name="preview")
        self.thread.start()

    def stop(self):
        """Ask the preview thread to drop its GL context and exit, then wait."""
        with self.cond:
            self.stopping = True
            self.cond.notify()
        self.thread.join(timeout=60)

    def submit(self, job):
        with self.cond:
            self.seq += 1
            job["id"] = self.seq
            job["t_submit"] = time.perf_counter()
            self.pending = job
            self.cond.notify()
        return job["id"]

    def clear(self):
        with self.cond:
            self.load_cache.clear()
            self.paint_cache.clear()
            self.fin = (None, None)
            self.before = (None, None)

    def _loop(self):
        try:
            while True:
                with self.cond:
                    while self.pending is None and not self.stopping:
                        self.cond.wait()
                    if self.stopping:
                        return
                    job, self.pending = self.pending, None
                try:
                    res = self.render(job)
                except qs.Cancelled:
                    continue
                except Exception:
                    tb = traceback.format_exc()
                    qs.log_line("Preview render failed\n" + tb)
                    self.on_result({"id": job["id"], "error": tb})
                    continue
                self.on_result(res)
        finally:
            try:
                import qs_gpu
                qs_gpu.release()
            except Exception:
                pass

    @staticmethod
    def _put(cache, key, val, keep):
        cache[key] = val
        cache.move_to_end(key)
        while len(cache) > keep:
            cache.popitem(last=False)

    def render(self, job):
        t0 = time.perf_counter()
        s, st = job["set"], job["st"]
        lk, pk = qs.stage_keys(s, st)
        timing = {}
        src = self.load_cache.get(lk)
        if src is None:
            t = time.perf_counter()
            src = qs.load_stage(s, st)
            self._put(self.load_cache, lk, src, 4)
            timing["load"] = (time.perf_counter() - t) * 1000
        painted = self.paint_cache.get(pk)
        if painted is None:
            t = time.perf_counter()

            # No mid-render cancel: finishing the in-flight render gives visible updates while a
            # filter slider is being dragged (cancelling made a continuous drag show nothing until
            # release). Stale intermediate requests are still skipped by the latest-wins queue.
            painted = qs.paint_dispatch(src, st)
            self._put(self.paint_cache, pk, painted, 6)
            timing["paint"] = (time.perf_counter() - t) * 1000
        fk = (pk, json.dumps({k: st[k] for k in sorted(st) if k != "_mask"},
                             sort_keys=True, default=str))
        if self.fin[0] == fk:
            fin = self.fin[1]
        else:
            t = time.perf_counter()
            fin = qs.finish_stage(src, painted, st)
            self.fin = (fk, fin)
            timing["finish"] = (time.perf_counter() - t) * 1000
        self.last = {"src": src, "fin": fin, "st": st}
        light = (float(job["light_az"]), float(job["light_el"]))
        exposure = float(st.get("exposure", 0) or 0)
        shade_ao = bool(st.get("shade_ao", False))
        t = time.perf_counter()
        after = view_image(job["view"], fin, light, exposure, shade_ao)
        bk = (lk, job["view"], light if job["view"] == "Lit" else None, exposure, shade_ao)
        if self.before[0] == bk:
            before = self.before[1]
        else:
            before = view_image(job["view"], src, light, exposure, shade_ao)
            self.before = (bk, before)
        seam = qs.seam_metrics(fin["alb"], fin["normal"])
        img, mapping = compose(after, before, job)
        timing["view"] = (time.perf_counter() - t) * 1000
        return {"id": job["id"], "img": img, "map": mapping, "timing": timing,
                "render_ms": (time.perf_counter() - t0) * 1000, "t_submit": job["t_submit"],
                "size": (src["W"], src["H"]), "r_px": painted["r_px"], "name": src["name"],
                "report": src["report"], "seam": seam}


# ======================================================================================
class App:
    def __init__(self, root, settings, folders):
        self.root = root
        self.q = queue.Queue()
        self.busy = False
        self._photo = None
        self._sched = None
        self._shown_id = 0
        self._last_err = None
        self.last_out_dirs = []
        self.sets = []
        self.last_result = None
        self.result_count = 0
        self.engine = PreviewEngine(lambda r: self.q.put(("preview", r)))
        self.mask = None
        self.mask_rev = 0
        apply_dark_theme(root)
        root.title(f"Quixel Stylizer v{qs.TOOL_VERSION}  -  LIVE PREVIEW")
        root.protocol("WM_DELETE_WINDOW", self.on_close)
        f = self.f = max(1.0, root.winfo_fpixels("1i") / 96.0)
        sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
        w, h = min(int(1600 * f), int(sw * 0.92)), min(int(1000 * f), int(sh * 0.86))
        root.geometry(f"{w}x{h}+{max((sw - w) // 2, 0)}+{max((sh - h) // 3, 0)}")
        root.minsize(int(1000 * f), int(560 * f))

        # ---------------- top bar ----------------
        top = ttk.Frame(root, padding=(8, 6, 8, 2))
        top.pack(side="top", fill="x")
        self.btn_convert = ttk.Button(top, text="CONVERT", command=self.convert)
        self.btn_convert.pack(side="left")
        ttk.Button(top, text="Open output folder", command=self.open_output).pack(side="left", padx=4)
        ttk.Separator(top, orient="vertical").pack(side="left", fill="y", padx=6)
        ttk.Button(top, text="Save preset...", command=self.save_preset).pack(side="left")
        ttk.Button(top, text="Load preset...", command=self.load_preset).pack(side="left", padx=4)
        ttk.Button(top, text="Defaults", command=lambda: self.apply(qs.DEFAULTS)).pack(side="left")
        self.progress = ttk.Progressbar(top, mode="determinate", length=int(180 * f))
        self.progress.pack(side="right")
        ttk.Label(top, text=f"v{qs.TOOL_VERSION}", foreground="#888").pack(side="right", padx=8)
        self.status = tk.StringVar(value="Add a material folder - the preview loads automatically.")
        ttk.Label(root, textvariable=self.status, foreground="#8ec1ff", background=_BG,
                  padding=(10, 0)).pack(side="top", anchor="w")

        body = ttk.Frame(root)
        body.pack(side="top", fill="both", expand=True)

        # ---------------- left: scrollable side panel ----------------
        side = ttk.Frame(body, padding=(8, 4))
        side.pack(side="left", fill="y")
        canvas = tk.Canvas(side, highlightthickness=0, width=int(440 * f), bg=_BG, highlightbackground=_BG)
        sb = ttk.Scrollbar(side, orient="vertical", command=canvas.yview)
        panel = ttk.Frame(canvas)
        panel.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=panel, anchor="nw")
        canvas.configure(yscrollcommand=sb.set)
        canvas.pack(side="left", fill="y", expand=True)
        sb.pack(side="right", fill="y")
        self._side_canvas = canvas
        canvas.bind("<Enter>", lambda e: canvas.bind_all(
            "<MouseWheel>", lambda ev: canvas.yview_scroll(int(-ev.delta / 120), "units")))
        canvas.bind("<Leave>", lambda e: canvas.unbind_all("<MouseWheel>"))
        # Tk's default TCombobox <MouseWheel> binding cycles the value (readonly too). Scrolling the settings
        # panel over "Output size" etc. silently changed settings. Replace it: wheel over a combobox only
        # scrolls the panel (if the combobox is in it), never changes the value.
        def combo_wheel(ev, canvas=canvas, panel_path=str(panel)):
            if str(ev.widget).startswith(panel_path):
                canvas.yview_scroll(int(-ev.delta / 120), "units")
            return "break"
        root.bind_class("TCombobox", "<MouseWheel>", combo_wheel)

        ttk.Label(panel, text="Material folders (set or parent folders):").pack(anchor="w")
        self.lb = tk.Listbox(panel, height=4, width=56, selectmode="browse", exportselection=False,
                             bg=_FIELD, fg=_FG, selectbackground=_ACCENT, selectforeground="#ffffff",
                             highlightthickness=0, relief="flat", activestyle="none")
        self.lb.pack(fill="x")
        self.lb.bind("<<ListboxSelect>>", self.on_folder_select)
        bf = ttk.Frame(panel)
        bf.pack(fill="x", pady=2)
        ttk.Button(bf, text="Add folder...", command=self.add_folder).pack(side="left")
        ttk.Button(bf, text="Remove", command=self.remove_sel).pack(side="left", padx=4)
        ttk.Button(bf, text="Clear", command=self.clear_folders).pack(side="left")
        self.recursive = tk.BooleanVar(value=False)
        ttk.Checkbutton(bf, text="Recursive", variable=self.recursive,
                        command=self.refresh_sets).pack(side="left", padx=8)
        sf = ttk.Frame(panel)
        sf.pack(fill="x", pady=2)
        ttk.Label(sf, text="Preview set").pack(side="left")
        self.set_var = tk.StringVar()
        self.set_combo = ttk.Combobox(sf, textvariable=self.set_var, state="readonly", width=40)
        self.set_combo.pack(side="left", padx=4, fill="x", expand=True)
        self.set_combo.bind("<<ComboboxSelected>>", lambda e: self.request(clear_view=True))

        of = ttk.Frame(panel)
        of.pack(fill="x", pady=2)
        ttk.Label(of, text="Output folder (blank = <set folder>\\Stylized):").pack(anchor="w")
        self.out_var = tk.StringVar()
        ttk.Entry(of, textvariable=self.out_var, width=46).pack(side="left", fill="x", expand=True)
        ttk.Button(of, text="...", width=3, command=self.pick_out).pack(side="left")
        ttk.Button(of, text="x", width=2, command=lambda: self.out_var.set("")).pack(side="left")

        opt = ttk.LabelFrame(panel, text="Settings (preview updates live)", padding=4)
        opt.pack(fill="x", pady=6)
        self.vars = {}
        row = ttk.Frame(opt)
        row.pack(fill="x")
        ttk.Label(row, text="Output size").pack(side="left")
        self.vars["size"] = tk.StringVar(value=str(settings["size"]))
        ttk.Combobox(row, textvariable=self.vars["size"], values=[str(s) for s in qs.SIZES],
                     width=6, state="readonly").pack(side="left", padx=4)
        ttk.Label(row, text="Filter").pack(side="left", padx=(8, 0))
        self.vars["filter"] = tk.StringVar(value=settings["filter"])
        ttk.Combobox(row, textvariable=self.vars["filter"], values=qs.FILTERS, width=16,
                     state="readonly").pack(side="left", padx=4)
        ttk.Label(row, text="Format").pack(side="left", padx=(8, 0))
        self.vars["format"] = tk.StringVar(value=settings.get("format", "png"))
        ttk.Combobox(row, textvariable=self.vars["format"], values=["png", "tga"], width=4,
                     state="readonly").pack(side="left", padx=4)
        row2 = ttk.Frame(opt)
        row2.pack(fill="x", pady=2)
        for key, text in (("flip_green", "Flip source green"), ("tile", "Tileable (wrap)"),
                          ("preview", "Write preview PNG"), ("shade_ao", "AO again in Lit view")):
            self.vars[key] = tk.BooleanVar(value=bool(settings[key]))
            ttk.Checkbutton(row2, text=text, variable=self.vars[key]).pack(side="left", padx=(0, 8))
        row3 = ttk.Frame(opt)
        row3.pack(fill="x", pady=2)
        for key, text, vals, wd in (("normal_out", "Normal out", ["dx", "gl"], 4),
                                    ("normal_in", "Src normal", ["auto", "dx", "gl"], 5),
                                    ("curvature_source", "Curvature", ["auto", "cavity", "normal", "height"], 7)):
            ttk.Label(row3, text=text).pack(side="left", padx=(4, 0))
            self.vars[key] = tk.StringVar(value=settings[key])
            ttk.Combobox(row3, textvariable=self.vars[key], values=vals, width=wd,
                         state="readonly").pack(side="left", padx=2)
        self.scales = {}
        for key, text, lo, hi, step in SLIDERS:
            fr = ttk.Frame(opt)
            fr.pack(fill="x")
            ttk.Label(fr, text=text, width=23).pack(side="left")
            v = tk.DoubleVar(value=float(settings[key]))
            self.vars[key] = v
            sc = dark_scale(fr, variable=v, from_=lo, to=hi, resolution=step, orient="horizontal",
                            length=int(180 * f), showvalue=False)
            sc.pack(side="left", fill="x", expand=True)
            ent = ttk.Entry(fr, width=6, textvariable=v)
            ent.pack(side="left", padx=(4, 0))
            self.scales[key] = sc
        self.extra = {}
        for k in qs.DEFAULTS:
            if k in self.vars:
                continue
            v = settings[k]
            if k == "palette":
                self.extra[k] = [list(map(float, c)) for c in (v or [])]
            elif isinstance(v, list):
                self.extra[k] = [float(x) for x in v]
            else:
                self.extra[k] = v
        pal = ttk.Frame(opt)
        pal.pack(fill="x", pady=(4, 0))
        ttk.Label(pal, text="Palette").pack(side="left")
        self.swatch_row = ttk.Frame(pal)
        self.swatch_row.pack(side="left", padx=4)
        ttk.Button(pal, text="+", width=2, command=self.add_swatch).pack(side="left")
        for name in ("plaster", "stone", "moss"):
            ttk.Button(pal, text=name, command=lambda n=name: self.use_palette(n)).pack(side="left", padx=2)
        tint = ttk.Frame(opt)
        tint.pack(fill="x", pady=2)
        ttk.Label(tint, text="Tint").pack(side="left")
        self.warm_btn = tk.Button(tint, text="warm", command=lambda: self.pick_tint("tint_warm"),
                                  relief="flat", bd=0)
        self.warm_btn.pack(side="left", padx=4)
        self.cool_btn = tk.Button(tint, text="cool", command=lambda: self.pick_tint("tint_cool"),
                                  relief="flat", bd=0)
        self.cool_btn.pack(side="left")
        self.sync_palette_ui()
        ttk.Label(panel, text=f"Log (errors also go to {qs.LOG_PATH}):", wraplength=int(420 * f)).pack(anchor="w")
        self.log = tk.Text(panel, height=8, width=56, bg=_FIELD, fg=_FG, insertbackground=_FG,
                           relief="flat", highlightthickness=0)
        self.log.pack(fill="x")
        panel.update_idletasks()
        canvas.configure(width=max(panel.winfo_reqwidth(), int(400 * f)))

        # ---------------- right: preview ----------------
        right = ttk.Frame(body, padding=(4, 4, 8, 8))
        right.pack(side="left", fill="both", expand=True)
        tb = ttk.Frame(right)
        tb.pack(fill="x")
        self.pv = {}
        ttk.Label(tb, text="View").pack(side="left")
        self.pv["view"] = tk.StringVar(value="Diffuse")
        ttk.Combobox(tb, textvariable=self.pv["view"], values=VIEWS, width=10, state="readonly").pack(side="left", padx=4)
        self.pv["compare"] = tk.StringVar(value="Split")
        for m in ("Split", "After", "Before"):
            ttk.Radiobutton(tb, text=m, value=m, variable=self.pv["compare"]).pack(side="left")
        self.pv["tile"] = tk.BooleanVar(value=False)
        ttk.Checkbutton(tb, text="Tile 2x2", variable=self.pv["tile"]).pack(side="left", padx=8)
        self.pv["offset"] = tk.BooleanVar(value=False)
        ttk.Checkbutton(tb, text="Offset", variable=self.pv["offset"]).pack(side="left")
        ttk.Label(tb, text="Zoom").pack(side="left")
        self.pv["zoom"] = tk.StringVar(value="Fit")
        ttk.Combobox(tb, textvariable=self.pv["zoom"], values=["Fit", "1x", "2x"], width=4, state="readonly").pack(side="left", padx=4)
        ttk.Label(tb, text="Preview size").pack(side="left", padx=(8, 0))
        self.pv["psize"] = tk.StringVar(value=str(DEFAULT_PREVIEW_SIZE))
        ttk.Combobox(tb, textvariable=self.pv["psize"], values=[str(s) for s in PREVIEW_SIZES], width=5,
                     state="readonly").pack(side="left", padx=4)
        ttk.Button(tb, text="Refresh", command=self.refresh_preview).pack(side="left", padx=6)
        self.indicator = tk.StringVar(value="")
        ttk.Label(tb, textvariable=self.indicator, width=46, anchor="e").pack(side="right")
        tb2 = ttk.Frame(right)
        tb2.pack(fill="x", pady=(2, 2))
        self.pv["split"] = tk.DoubleVar(value=0.5)
        self.pv["light_az"] = tk.DoubleVar(value=135.0)
        self.pv["light_el"] = tk.DoubleVar(value=45.0)
        for key, text, lo, hi, res in (("split", "Split", 0, 1, 0.01), ("light_az", "Light angle", 0, 360, 1),
                                       ("light_el", "Light height", 5, 90, 1)):
            ttk.Label(tb2, text=text).pack(side="left", padx=(6, 0))
            dark_scale(tb2, variable=self.pv[key], from_=lo, to=hi, resolution=res, orient="horizontal",
                       length=int(120 * f), showvalue=False).pack(side="left")
        ttk.Label(tb2, text="  Right-drag: light.", foreground=_MUTED, background=_BG).pack(side="left")
        tb3 = ttk.Frame(right)
        tb3.pack(fill="x", pady=(0, 2))
        self.paint_mask = tk.BooleanVar(value=False)
        ttk.Checkbutton(tb3, text="Paint mask", variable=self.paint_mask).pack(side="left")
        self.mask_tool = tk.StringVar(value="amount")
        for name in ("amount", "smooth", "erase"):
            ttk.Radiobutton(tb3, text=name, value=name, variable=self.mask_tool).pack(side="left", padx=2)
        ttk.Label(tb3, text="Brush").pack(side="left", padx=(8, 0))
        self.brush = tk.DoubleVar(value=24)
        dark_scale(tb3, variable=self.brush, from_=2, to=80, resolution=1, orient="horizontal",
                   length=int(90 * f), showvalue=False).pack(side="left")
        ttk.Label(tb3, text="Amount").pack(side="left", padx=(6, 0))
        self.brush_amount = tk.DoubleVar(value=0.0)
        dark_scale(tb3, variable=self.brush_amount, from_=0, to=1, resolution=0.05, orient="horizontal",
                   length=int(80 * f), showvalue=False).pack(side="left")
        ttk.Button(tb3, text="Clear mask", command=self.clear_mask).pack(side="left", padx=6)
        self.seam_var = tk.StringVar(value="")
        ttk.Label(tb3, textvariable=self.seam_var, foreground=_MUTED).pack(side="right")
        self.pcanvas = tk.Canvas(right, background="#141414", highlightthickness=0)
        self.pcanvas.pack(fill="both", expand=True)
        self.pc_img = self.pcanvas.create_image(0, 0, anchor="nw")
        self.pc_text = self.pcanvas.create_text(20, 20, anchor="nw", fill="#bbbbbb",
                                                text="Add a material folder on the left - preview appears here.")
        self.pcanvas.bind("<Configure>", lambda e: self.request())
        self.pcanvas.bind("<Button-1>", self.on_left)
        self.pcanvas.bind("<B1-Motion>", self.on_left)
        self.pcanvas.bind("<Button-3>", self.on_right)
        self.pcanvas.bind("<B3-Motion>", self.on_right)

        for v in list(self.vars.values()) + list(self.pv.values()):
            v.trace_add("write", lambda *a: self.request())
        root.report_callback_exception = self.tk_error
        for fo in folders:
            self.add_path(fo)
        root.after(50, self.poll)
        qs.log_line(f"GUI start v{qs.TOOL_VERSION} code={qs.MODULE_PATH} gui={os.path.abspath(__file__)} "
                    f"pid={os.getpid()}")
        self.say(qs.version_banner())
        try:
            import qs_gpu
            gpu = qs_gpu.status()
            line = "GPU paint: " + gpu.get("reason", "")
            qs.log_line(line)
            self.say(line)
            # The check ran on this thread. Drop the context here so Tk does not
            # share it, and so process exit does not destroy it from the wrong place.
            qs_gpu.release()
        except Exception as exc:
            qs.log_line(f"GPU paint: {exc}")
            self.say(f"GPU paint: {exc}")
        warn = qs.old_window_warning()
        if warn:
            qs.log_line("WARNING " + warn)
            self.say("!! " + warn)
            root.after(400, lambda: messagebox.showwarning("Quixel Stylizer - old window open", warn))

    # ---------------- helpers ----------------
    def tk_error(self, exc, val, tb):
        text = "".join(traceback.format_exception(exc, val, tb))
        qs.log_line("GUI callback error\n" + text)
        messagebox.showerror("Quixel Stylizer - error", text[-3000:])

    def settings(self):
        st = dict(qs.DEFAULTS)
        st.update(self.extra)
        # Copy the lists so a render in flight does not share the swatch row's storage.
        pal = self.extra.get("palette") or []
        st["palette"] = [list(map(float, c)) for c in pal]
        for key in ("tint_warm", "tint_cool"):
            st[key] = [float(x) for x in self.extra.get(key, qs.DEFAULTS[key])]
        for k, v in self.vars.items():
            try:
                val = v.get()
            except tk.TclError:
                val = qs.DEFAULTS[k]
            if k in INT_KEYS:
                val = int(float(val))
            st[k] = val
        return st

    def apply(self, st):
        for k, v in self.vars.items():
            if k in st:
                v.set(str(st[k]) if isinstance(v, tk.StringVar) else st[k])
        for k in self.extra:
            if k not in st:
                continue
            v = st[k]
            if k == "palette":
                self.extra[k] = [list(map(float, c)) for c in (v or [])]
            elif isinstance(v, list):
                self.extra[k] = [float(x) for x in v]
            else:
                self.extra[k] = v
        self.sync_palette_ui()
        self.request()

    def folders(self):
        return [f for f in self.lb.get(0, "end") if f.strip()]

    def refresh_sets(self, select_folder=None):
        self.sets = qs.find_sets(self.folders(), self.recursive.get())
        labels = [f"{s['name']}   ({s['folder']})" for s in self.sets]
        self.set_combo.configure(values=labels)
        idx = None
        if select_folder:
            sf = os.path.normcase(select_folder)
            for i, s in enumerate(self.sets):
                if os.path.normcase(s["folder"]).startswith(sf):
                    idx = i
                    break
        if idx is None and labels:
            cur = self.set_var.get()
            idx = labels.index(cur) if cur in labels else 0
        if idx is not None:
            self.set_var.set(labels[idx])
        else:
            self.set_var.set("")
        self.request()

    def current_set(self):
        labels = list(self.set_combo.cget("values") or [])
        cur = self.set_var.get()
        if cur in labels:
            return self.sets[labels.index(cur)]
        return self.sets[0] if self.sets else None

    def add_path(self, d):
        """Used by the Add folder button (askdirectory returns forward slashes)."""
        if d:
            d = os.path.normpath(os.path.abspath(d))
            if d not in self.folders():
                self.lb.insert("end", d)
            self.lb.selection_clear(0, "end")
            self.lb.selection_set(self.folders().index(d))
            self.refresh_sets(select_folder=d)
            n = len([s for s in self.sets if os.path.normcase(s["folder"]).startswith(os.path.normcase(d))])
            self.status.set(f"Folder added: {d}  ({n} set(s) found)" if n else
                            f"Folder added: {d}  - no material set found in it")

    def add_folder(self):
        self.add_path(filedialog.askdirectory(title="Pick a material folder or a parent folder"))

    def on_folder_select(self, _e=None):
        sel = self.lb.curselection()
        if sel:
            self.refresh_sets(select_folder=self.lb.get(sel[0]))

    def remove_sel(self):
        for i in reversed(self.lb.curselection()):
            self.lb.delete(i)
        self.refresh_sets()

    def clear_folders(self):
        self.lb.delete(0, "end")
        self.refresh_sets()

    def pick_out(self):
        d = filedialog.askdirectory(title="Output folder (Cancel = default <set>\\Stylized)")
        if d:
            self.out_var.set(os.path.normpath(d))

    def open_output(self):
        if self.last_out_dirs:
            for d in self.last_out_dirs[:5]:
                open_in_explorer(d)
            return
        for s in self.sets[:5]:
            d = qs.resolve_out_dir(s["folder"], self.out_var.get().strip() or None)
            if os.path.isdir(d):
                open_in_explorer(d)
                return
        messagebox.showinfo("Quixel Stylizer", "No output folder yet - run CONVERT first.")

    def sync_palette_ui(self):
        for child in self.swatch_row.winfo_children():
            child.destroy()
        palette = self.extra.get("palette") or []
        for i, rgb in enumerate(list(palette)[:8]):
            btn = tk.Button(self.swatch_row, text=" ", width=2, bg=_hex_rgb(rgb),
                            activebackground=_hex_rgb(rgb), relief="flat", bd=0,
                            command=lambda i=i: self.edit_swatch(i))
            btn.pack(side="left", padx=1)
            btn.bind("<Button-3>", lambda e, i=i: self.remove_swatch(i))
        self.warm_btn.configure(bg=_hex_rgb(self.extra.get("tint_warm", qs.DEFAULTS["tint_warm"])))
        self.cool_btn.configure(bg=_hex_rgb(self.extra.get("tint_cool", qs.DEFAULTS["tint_cool"])))

    def _ask_colour(self, initial):
        picked = colorchooser.askcolor(color=_hex_rgb(initial), title="Colour")
        if not picked or not picked[0]:
            return None
        r, g, b = picked[0]
        return [r / 255.0, g / 255.0, b / 255.0]

    def add_swatch(self):
        palette = list(self.extra.get("palette") or [])
        if len(palette) >= 8:
            self.status.set("Palette holds 8 swatches. Right-click one to remove it.")
            return
        rgb = self._ask_colour(palette[-1] if palette else [0.7, 0.7, 0.7])
        if rgb is None:
            return
        palette.append(rgb)
        self.extra["palette"] = palette
        self.sync_palette_ui()
        self.request()

    def edit_swatch(self, index):
        palette = list(self.extra.get("palette") or [])
        if index >= len(palette):
            return
        rgb = self._ask_colour(palette[index])
        if rgb is None:
            return
        palette[index] = rgb
        self.extra["palette"] = palette
        self.sync_palette_ui()
        self.request()

    def remove_swatch(self, index):
        palette = list(self.extra.get("palette") or [])
        if index >= len(palette):
            return
        del palette[index]
        self.extra["palette"] = palette
        self.sync_palette_ui()
        self.request()

    def use_palette(self, name):
        self.extra["palette"] = [list(c) for c in qs.PALETTES[name]]
        if "palette_strength" in self.vars and float(self.vars["palette_strength"].get() or 0) <= 0:
            self.vars["palette_strength"].set(0.7)
        self.sync_palette_ui()
        self.request()

    def pick_tint(self, key):
        rgb = self._ask_colour(self.extra.get(key, qs.DEFAULTS[key]))
        if rgb is None:
            return
        self.extra[key] = rgb
        self.sync_palette_ui()
        self.request()

    def clear_mask(self):
        self.mask = None
        self.mask_rev += 1
        self.request()

    def _paint_mask_at(self, e):
        m = getattr(self, "_map", None)
        if not m:
            return
        u = (e.x - self._offset[0]) / m["z"] + m["x0"]
        v = (e.y - self._offset[1]) / m["z"] + m["y0"]
        bw, bh = int(m["base_w"]), int(m["base_h"])
        if m.get("tiled"):
            u %= bw
            v %= bh
        if m.get("offset"):
            u = (u - bw / 2.0) % bw
            v = (v - bh / 2.0) % bh
        if self.mask is None or self.mask.shape != (bh, bw):
            prev = self.mask
            self.mask = np.ones((bh, bw), np.float32)
            if prev is not None:
                self.mask = np.clip(qs.resize(prev, bw, bh), 0, 1).astype(np.float32)
        radius = max(1, int(round(float(self.brush.get()))))
        cx, cy = int(round(u)), int(round(v))
        y0, y1 = cy - radius, cy + radius + 1
        x0, x1 = cx - radius, cx + radius + 1
        yy, xx = np.mgrid[y0:y1, x0:x1]
        dist = np.hypot(xx - cx, yy - cy)
        weight = np.clip(1.0 - dist / float(radius), 0, 1).astype(np.float32)
        yy_w = np.mod(yy, bh).astype(int)
        xx_w = np.mod(xx, bw).astype(int)
        tool = self.mask_tool.get()
        if tool == "erase":
            target = np.ones_like(weight)
        elif tool == "smooth":
            blurred = qs.gblur(self.mask, max(radius / 3.0, 0.6), True)
            target = blurred[yy_w, xx_w]
        else:
            target = np.full_like(weight, float(self.brush_amount.get()))
        current = self.mask[yy_w, xx_w]
        self.mask[yy_w, xx_w] = current * (1.0 - weight) + target * weight
        self.mask_rev += 1
        self.request()

    def save_preset(self):
        p = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON", "*.json")],
                                         initialdir=os.path.join(qs.TOOL_DIR, "presets"))
        if p:
            qs.save_preset(p, self.settings())
            mp = qs.mask_path_for(p)
            if self.mask is not None and float(np.min(self.mask)) < 0.999:
                qs.save_mask(mp, self.mask)
                self.say(f"preset saved: {p}  mask: {mp}")
            else:
                if os.path.isfile(mp):
                    os.remove(mp)
                self.say(f"preset saved: {p}")

    def load_preset(self):
        p = filedialog.askopenfilename(filetypes=[("JSON", "*.json")],
                                       initialdir=os.path.join(qs.TOOL_DIR, "presets"))
        if p:
            try:
                self.apply(dict(qs.DEFAULTS, **qs.load_preset(p)))
                loaded = qs.load_mask(qs.mask_path_for(p))
                self.mask = loaded
                self.mask_rev += 1
                self.say(f"preset loaded: {p}" + (" with mask" if loaded is not None else ""))
                self.request()
            except Exception as e:
                qs.log_line(f"GUI preset load failed {p}\n{traceback.format_exc()}")
                messagebox.showerror("Preset", str(e))

    def say(self, msg):
        self.q.put(("log", msg))

    # ---------------- live preview ----------------
    def preview_job(self):
        s = self.current_set()
        if s is None:
            return None
        st = self.settings()
        try:
            psize = int(self.pv["psize"].get())
        except (tk.TclError, ValueError):
            psize = DEFAULT_PREVIEW_SIZE
        st["size"] = min(psize, int(st["size"]))  # never show more detail than the final output
        try:
            job = {k: v.get() for k, v in self.pv.items() if k != "psize"}
        except tk.TclError:
            return None
        if self.mask is not None:
            st["_mask"] = np.array(self.mask, np.float32, copy=True)
            st["_mask_rev"] = self.mask_rev
        job.update(set=s, st=st, cw=self.pcanvas.winfo_width(), ch=self.pcanvas.winfo_height())
        job["pk"] = qs.stage_keys(s, st)[1]
        return job

    def request(self, clear_view=False):
        """Throttled: at most one submit per 120 ms while dragging; the worker drops stale jobs."""
        if self._sched is None:
            self._sched = self.root.after(120, self._fire)

    def _fire(self):
        self._sched = None
        job = self.preview_job()
        if job is None:
            return
        self.engine.submit(job)
        self.indicator.set("rendering...")

    def refresh_preview(self):
        self.engine.clear()
        self.request()

    def show_preview(self, r):
        if "error" in r:
            first = r["error"].strip().splitlines()[-1]
            self.indicator.set("preview error (see log)")
            self.say(r["error"])
            if first != self._last_err:
                self._last_err = first
                messagebox.showerror("Quixel Stylizer - preview failed", r["error"][-3000:] +
                                     f"\nLogged to {qs.LOG_PATH}")
            return
        if r["id"] < self._shown_id or Image is None:
            return
        self._shown_id = r["id"]
        self._photo = ImageTk.PhotoImage(Image.fromarray(r["img"]))
        cw, ch = self.pcanvas.winfo_width(), self.pcanvas.winfo_height()
        m = r["map"]
        self._offset = ((cw - m["dw"]) // 2, (ch - m["dh"]) // 2)
        self.pcanvas.coords(self.pc_img, *self._offset)
        self.pcanvas.itemconfigure(self.pc_img, image=self._photo)
        self.pcanvas.itemconfigure(self.pc_text, text="")
        self._map = m
        self.last_result = r
        self.result_count += 1
        if self.result_count == 1 or r["name"] != getattr(self, "_logged_name", None):
            self._logged_name = r["name"]
            qs.log_line(f"GUI preview ready: {r['name']} {r['size'][0]}x{r['size'][1]} in {r['render_ms']:.0f} ms")
        self._changes_since_log = getattr(self, "_changes_since_log", 0) + 1
        if self._changes_since_log >= 25:
            self._changes_since_log = 0
            qs.log_line(f"GUI preview: {self.result_count} renders so far, last {r['render_ms']:.0f} ms")
        total = (time.perf_counter() - r["t_submit"]) * 1000
        tm = r["timing"]
        parts = " ".join(f"{k} {v:.0f}" for k, v in tm.items() if k != "view")
        pending = self.engine.pending is not None
        self.indicator.set(("rendering...  " if pending else "") +
                           f"updated in {total:.0f} ms  [{parts} view {tm.get('view', 0):.0f}]  "
                           f"{r['size'][0]}x{r['size'][1]} r={r['r_px']}px")
        seam = r.get("seam")
        if seam:
            self.seam_var.set(
                f"seam  D {seam['diffuse_lr']:.3f}/{seam['diffuse_tb']:.3f}"
                f"   N {seam['normal_lr']:.3f}/{seam['normal_tb']:.3f}")
        if self.status.get().startswith(("Add a material", "Folder added")):
            notes = "; ".join(r["report"]["notes"][-1:])
            self.status.set(f"Previewing {r['name']}" + (f" - {notes}" if notes else ""))

    def _canvas_to_img_x(self, x):
        m = getattr(self, "_map", None)
        if not m:
            return None
        return ((x - self._offset[0]) / m["z"] + m["x0"]) / m["W"]

    def on_left(self, e):
        if self.paint_mask.get():
            self._paint_mask_at(e)
            return
        if self.pv["compare"].get() == "Split":
            v = self._canvas_to_img_x(e.x)
            if v is not None:
                self.pv["split"].set(round(min(max(v, 0.0), 1.0), 3))
        elif self.pv["view"].get() == "Lit":
            self.on_right(e)

    def on_right(self, e):
        cw, ch = self.pcanvas.winfo_width(), self.pcanvas.winfo_height()
        dx, dy = e.x - cw / 2, e.y - ch / 2
        self.pv["light_az"].set(round(math.degrees(math.atan2(-dy, dx)) % 360, 0))
        d = min(math.hypot(dx, dy) / (0.5 * min(cw, ch)), 1.0)
        self.pv["light_el"].set(round(max(5.0, 90.0 * (1 - d)), 0))
        if self.pv["view"].get() != "Lit":
            self.pv["view"].set("Lit")

    # ---------------- event pump ----------------
    def poll(self):
        try:
            while True:
                kind, payload = self.q.get_nowait()
                if kind == "log":
                    self.log.insert("end", payload + "\n")
                    self.log.see("end")
                elif kind == "status":
                    self.status.set(payload)
                elif kind == "progress":
                    self.progress.configure(maximum=payload[1], value=payload[0])
                elif kind == "preview":
                    self.show_preview(payload)
                elif kind == "call":
                    payload()
                elif kind == "done":
                    self.busy = False
                    self.btn_convert.state(["!disabled"])
        except queue.Empty:
            pass
        self.root.after(30, self.poll)

    def on_close(self):
        self.engine.stop()
        try:
            import qs_gpu
            qs_gpu.release()
        except Exception:
            pass
        self.root.destroy()

    def error_box(self, title, text):
        self.q.put(("call", lambda: messagebox.showerror(title, text[-3000:])))

    def run_bg(self, name, fn):
        if self.busy:
            self.status.set("Busy - wait for the current job to finish.")
            return False
        self.busy = True
        self.btn_convert.state(["disabled"])

        def wrap():
            try:
                fn()
            except Exception as e:
                tb = traceback.format_exc()
                qs.log_line(f"GUI {name} failed\n{tb}")
                self.say(tb)
                self.q.put(("status", f"{name} failed: {e}  (see {qs.LOG_PATH})"))
                self.error_box(f"Quixel Stylizer - {name} failed", f"{tb}\nLogged to {qs.LOG_PATH}")
            finally:
                try:
                    import qs_gpu
                    qs_gpu.release()
                except Exception:
                    pass
                self.q.put(("done", None))
        threading.Thread(target=wrap, daemon=True).start()
        return True

    def convert(self, ask_open=True):
        folders = self.folders()
        if not folders:
            messagebox.showwarning("Quixel Stylizer",
                                   "No folder in the list.\nClick 'Add folder...' and pick a Megascans set "
                                   "folder (or a parent folder), then CONVERT.")
            return False
        st, rec = self.settings(), self.recursive.get()
        if self.mask is not None:
            st["_mask"] = np.array(self.mask, np.float32, copy=True)
            st["_mask_rev"] = self.mask_rev
        out = self.out_var.get().strip() or None
        qs.log_line(f"GUI convert v{qs.TOOL_VERSION} code={qs.MODULE_PATH}: folders={folders} out={out!r} "
                    f"recursive={rec} size={st['size']} format={st['format']}")
        self.say(f"Convert v{qs.TOOL_VERSION}: output size {st['size']} px, format {st['format']}")
        warn = qs.old_window_warning()
        if warn:
            qs.log_line("WARNING " + warn)
            self.say("!! " + warn)

        def job():
            sets = qs.find_sets(folders, rec)
            if not sets:
                raise RuntimeError("No material set found in:\n" + "\n".join(folders) +
                                   "\n(need an image ending in _Albedo/_BaseColor/_Diffuse...)")
            ok, failed = [], []
            for i, s in enumerate(sets, 1):
                self.q.put(("progress", (i - 1, len(sets))))
                self.q.put(("status", f"Converting {i}/{len(sets)}: {s['name']} at {st['size']} px ..."))
                try:
                    r = qs.process_set(s, st, write=True, out_dir=out)
                    d = os.path.dirname(r["files"]["D"])
                    ok.append((s["name"], d))
                    self.say(f"[ok] {i}/{len(sets)} {s['name']} -> {d} ({r['seconds']:.1f}s)")
                    for k, v in r["files"].items():
                        self.say(f"      {k:8s} {v}" + (f"   [{r['checked'][k]}]" if k in r.get("checked", {}) else ""))
                    if r["report"]["defaulted"]:
                        self.say("      defaulted: " + ", ".join(r["report"]["defaulted"]))
                    qs.log_line(f"GUI ok: {s['name']} -> {d}")
                except Exception:
                    tb = traceback.format_exc()
                    failed.append((s["folder"], tb))
                    self.say(f"[FAILED] {s['folder']}\n{tb}")
                    qs.log_line(f"GUI FAILED: {s['folder']}\n{tb}")
            self.q.put(("progress", (len(sets), len(sets))))
            dirs = sorted({d for _, d in ok})
            self.last_out_dirs = dirs
            self.q.put(("status", f"Done: {len(ok)} converted, {len(failed)} failed. Output: "
                                  + ("; ".join(dirs) if dirs else "-")))
            if failed:
                self.error_box("Quixel Stylizer - some sets failed",
                               "\n\n".join(f"{f}\n{tb}" for f, tb in failed) + f"\nLogged to {qs.LOG_PATH}")
            if ok and ask_open:
                msg = (f"Converted {len(ok)} set(s).\n\nOutput folder(s):\n" + "\n".join(dirs) +
                       "\n\nOpen the output folder in Explorer?")

                def ask():
                    if messagebox.askyesno("Quixel Stylizer - done", msg):
                        for d in dirs[:5]:
                            open_in_explorer(d)
                self.q.put(("call", ask))
        return self.run_bg("Convert", job)


def run_gui(settings=None, folders=None, selftest=False):
    settings = dict(qs.DEFAULTS, **(settings or {}))
    enable_dpi_awareness()
    root = tk.Tk()
    if selftest:
        root.withdraw()
    try:
        app = App(root, settings, folders or [])
    except Exception:
        qs.log_line("GUI init failed\n" + traceback.format_exc())
        raise
    if selftest:
        root.update()
        st = app.settings()
        app.engine.stop()
        root.destroy()
        print("GUI selftest ok; settings round-trip:", json.dumps(st))
        return 0
    root.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(run_gui(selftest="--selftest" in sys.argv))
