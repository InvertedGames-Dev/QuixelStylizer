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
from tkinter import ttk, filedialog, messagebox

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
]
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


def view_image(view, maps, light):
    """maps: dict with alb, normal, rough, metal, ao, height (stage output or source)."""
    if view == "Lit":
        return qs.shade_lit(maps["alb"], maps["normal"], maps["rough"], maps["metal"], *light)
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


def compose(after, before, job):
    """Split/before/after, tile 2x2, zoom, fit to canvas. Returns (u8 image, mapping)."""
    mode, split = job["compare"], float(job["split"])
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
    return img, {"z": z, "x0": x0, "y0": y0, "dw": dw, "dh": dh, "W": W, "H": H}


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
        threading.Thread(target=self._loop, daemon=True, name="preview").start()

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
        while True:
            with self.cond:
                while self.pending is None:
                    self.cond.wait()
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
            painted = qs.paint_stage(src, st)
            self._put(self.paint_cache, pk, painted, 6)
            timing["paint"] = (time.perf_counter() - t) * 1000
        fk = (pk, json.dumps({k: st[k] for k in sorted(st)}, sort_keys=True, default=str))
        if self.fin[0] == fk:
            fin = self.fin[1]
        else:
            t = time.perf_counter()
            fin = qs.finish_stage(src, painted, st)
            self.fin = (fk, fin)
            timing["finish"] = (time.perf_counter() - t) * 1000
        self.last = {"src": src, "fin": fin, "st": st}
        light = (float(job["light_az"]), float(job["light_el"]))
        t = time.perf_counter()
        after = view_image(job["view"], fin, light)
        bk = (lk, job["view"], light if job["view"] == "Lit" else None)
        if self.before[0] == bk:
            before = self.before[1]
        else:
            before = view_image(job["view"], src, light)
            self.before = (bk, before)
        img, mapping = compose(after, before, job)
        timing["view"] = (time.perf_counter() - t) * 1000
        return {"id": job["id"], "img": img, "map": mapping, "timing": timing,
                "render_ms": (time.perf_counter() - t0) * 1000, "t_submit": job["t_submit"],
                "size": (src["W"], src["H"]), "r_px": painted["r_px"], "name": src["name"],
                "report": src["report"]}


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
        root.title(f"Quixel Stylizer v{qs.TOOL_VERSION}  -  LIVE PREVIEW")
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
        ttk.Label(root, textvariable=self.status, foreground="#0050a0", padding=(10, 0)).pack(side="top", anchor="w")

        body = ttk.Frame(root)
        body.pack(side="top", fill="both", expand=True)

        # ---------------- left: scrollable side panel ----------------
        side = ttk.Frame(body, padding=(8, 4))
        side.pack(side="left", fill="y")
        canvas = tk.Canvas(side, highlightthickness=0, width=int(440 * f))
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
        self.lb = tk.Listbox(panel, height=4, width=56, selectmode="browse", exportselection=False)
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
                          ("preview", "Write preview PNG")):
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
            sc = tk.Scale(fr, variable=v, from_=lo, to=hi, resolution=step, orient="horizontal",
                          length=int(210 * f), showvalue=True)
            sc.pack(side="left", fill="x", expand=True)
            self.scales[key] = sc
        self.extra = {k: settings[k] for k in qs.DEFAULTS if k not in self.vars}
        ttk.Label(panel, text=f"Log (errors also go to {qs.LOG_PATH}):", wraplength=int(420 * f)).pack(anchor="w")
        self.log = tk.Text(panel, height=8, width=56)
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
            tk.Scale(tb2, variable=self.pv[key], from_=lo, to=hi, resolution=res, orient="horizontal",
                     length=int(150 * f), showvalue=False).pack(side="left")
        ttk.Label(tb2, text="  Left-drag: move split (or light in Lit view w/o split).  Right-drag: light.",
                  foreground="#666").pack(side="left")
        self.pcanvas = tk.Canvas(right, background="#202020", highlightthickness=0)
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
            if k in st:
                self.extra[k] = st[k]
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

    def save_preset(self):
        p = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON", "*.json")],
                                         initialdir=os.path.join(qs.TOOL_DIR, "presets"))
        if p:
            qs.save_preset(p, self.settings())
            self.say(f"preset saved: {p}")

    def load_preset(self):
        p = filedialog.askopenfilename(filetypes=[("JSON", "*.json")],
                                       initialdir=os.path.join(qs.TOOL_DIR, "presets"))
        if p:
            try:
                self.apply(dict(qs.DEFAULTS, **qs.load_preset(p)))
                self.say(f"preset loaded: {p}")
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
        if self.status.get().startswith(("Add a material", "Folder added")):
            notes = "; ".join(r["report"]["notes"][-1:])
            self.status.set(f"Previewing {r['name']}" + (f" - {notes}" if notes else ""))

    def _canvas_to_img_x(self, x):
        m = getattr(self, "_map", None)
        if not m:
            return None
        return ((x - self._offset[0]) / m["z"] + m["x0"]) / m["W"]

    def on_left(self, e):
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
        root.destroy()
        print("GUI selftest ok; settings round-trip:", json.dumps(st))
        return 0
    root.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(run_gui(selftest="--selftest" in sys.argv))
