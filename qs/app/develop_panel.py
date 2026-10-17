from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QCheckBox, QColorDialog, QComboBox, QDoubleSpinBox, QFormLayout, QGroupBox,
    QHBoxLayout, QLabel, QPushButton, QScrollArea, QSlider, QVBoxLayout, QWidget,
)

from qs.core.packing import PACKINGS
from qs.core.recipe import FILTERS, SIZES, Recipe

_MAX_SWATCHES = 8


def _slider_row(label, lo, hi, step, value, decimals=2, tip=""):
    box = QWidget()
    lay = QHBoxLayout(box)
    lay.setContentsMargins(0, 0, 0, 0)
    lab = QLabel(label)
    lab.setMinimumWidth(110)
    s = QSlider(Qt.Orientation.Horizontal)
    steps = int(round((hi - lo) / step))
    s.setRange(0, steps)
    spin = QDoubleSpinBox()
    spin.setRange(lo, hi)
    spin.setSingleStep(step)
    spin.setDecimals(decimals)
    spin.setValue(value)
    if tip:
        lab.setToolTip(tip)
        s.setToolTip(tip)
        spin.setToolTip(tip)
        box.setToolTip(tip)

    def from_slider(v):
        spin.blockSignals(True)
        spin.setValue(lo + v * step)
        spin.blockSignals(False)
        if not s.isSliderDown():
            spin.valueChanged.emit(spin.value())

    def from_spin(v):
        s.blockSignals(True)
        s.setValue(int(round((v - lo) / step)))
        s.blockSignals(False)

    s.valueChanged.connect(from_slider)
    s.sliderReleased.connect(lambda: spin.valueChanged.emit(spin.value()))
    spin.valueChanged.connect(from_spin)
    from_spin(value)
    lay.addWidget(lab)
    lay.addWidget(s, 1)
    lay.addWidget(spin)
    return box, spin


def _swatch_qcolor(rgb) -> QColor:
    r, g, b = (max(0.0, min(1.0, float(c))) for c in rgb[:3])
    return QColor(int(round(r * 255)), int(round(g * 255)), int(round(b * 255)))


class PaletteStrip(QWidget):
    """Add/remove RGB swatches matching Recipe.palette (up to eight)."""
    changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._colors: list[list[float]] = []
        self._row = QHBoxLayout(self)
        self._row.setContentsMargins(0, 0, 0, 0)
        self._row.setSpacing(4)
        self._swatches = QHBoxLayout()
        self._swatches.setSpacing(4)
        self._row.addLayout(self._swatches)
        self.add_btn = QPushButton("+")
        self.add_btn.setFixedSize(28, 28)
        self.add_btn.setToolTip("Add a palette colour")
        self.add_btn.clicked.connect(self._add)
        self._row.addWidget(self.add_btn)
        self._row.addStretch(1)
        self._rebuild()

    def colors(self) -> list[list[float]]:
        return [list(c) for c in self._colors]

    def set_colors(self, colors):
        out = []
        for item in list(colors or [])[:_MAX_SWATCHES]:
            try:
                out.append([
                    float(np_clip(item[0])),
                    float(np_clip(item[1])),
                    float(np_clip(item[2])),
                ])
            except (TypeError, ValueError, IndexError):
                continue
        self._colors = out
        self._rebuild()

    def _add(self):
        if len(self._colors) >= _MAX_SWATCHES:
            return
        start = _swatch_qcolor(self._colors[-1] if self._colors else (0.55, 0.55, 0.55))
        picked = QColorDialog.getColor(start, self, "Palette colour")
        if not picked.isValid():
            return
        self._colors.append([picked.redF(), picked.greenF(), picked.blueF()])
        self._rebuild()
        self.changed.emit()

    def _edit(self, index: int):
        if index < 0 or index >= len(self._colors):
            return
        picked = QColorDialog.getColor(_swatch_qcolor(self._colors[index]), self, "Palette colour")
        if not picked.isValid():
            return
        self._colors[index] = [picked.redF(), picked.greenF(), picked.blueF()]
        self._rebuild()
        self.changed.emit()

    def _remove(self, index: int):
        if index < 0 or index >= len(self._colors):
            return
        del self._colors[index]
        self._rebuild()
        self.changed.emit()

    def _rebuild(self):
        while self._swatches.count():
            item = self._swatches.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
        for i, rgb in enumerate(self._colors):
            cell = QWidget()
            cl = QVBoxLayout(cell)
            cl.setContentsMargins(0, 0, 0, 0)
            cl.setSpacing(2)
            btn = QPushButton()
            btn.setFixedSize(28, 28)
            q = _swatch_qcolor(rgb)
            btn.setStyleSheet(
                f"QPushButton {{ background: {q.name()}; border: 1px solid #888; }}"
            )
            btn.setToolTip("Click to edit, × to remove")
            btn.clicked.connect(lambda _=False, idx=i: self._edit(idx))
            x = QPushButton("×")
            x.setFixedSize(28, 16)
            x.clicked.connect(lambda _=False, idx=i: self._remove(idx))
            cl.addWidget(btn)
            cl.addWidget(x)
            self._swatches.addWidget(cell)
        self.add_btn.setEnabled(len(self._colors) < _MAX_SWATCHES)


def np_clip(v):
    return 0.0 if v < 0 else 1.0 if v > 1 else v


class DevelopPanel(QWidget):
    recipe_changed = Signal(dict)
    export_clicked = Signal()
    fit_budget_clicked = Signal()
    decimate_changed = Signal(float)
    shape_changed = Signal(str)
    channel_changed = Signal(str)
    compare_changed = Signal(str)
    view_ev_changed = Signal(float)
    displace_changed = Signal(bool)
    rewrap_bake_clicked = Signal()
    reset_clicked = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._spins = {}
        self._loading = False
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        inner = QWidget()
        form = QVBoxLayout(inner)

        view = QGroupBox("View")
        vf = QFormLayout(view)
        self.shape = QComboBox()
        self.shape.addItems(["sphere", "cube", "plane", "mesh"])
        self.shape.setToolTip("Preview mesh. Sphere, cube, and plane UV the maps 0–1; Mesh uses the asset.")
        self.shape.currentTextChanged.connect(self.shape_changed.emit)
        vf.addRow("Shape", self.shape)
        self.channel = QComboBox()
        self.channel.addItems(["Lit", "Diffuse", "Normal", "Roughness", "AO", "Displacement", "Metallic"])
        self.channel.setToolTip("Lit is the Exodus BRDF. Other modes show that map on the mesh.")
        self.channel.currentTextChanged.connect(self.channel_changed.emit)
        vf.addRow("Channel", self.channel)
        self.compare = QComboBox()
        self.compare.addItems(["After", "Before", "Split"])
        self.compare.setToolTip("After is stylized. Before is the source maps. Split is source left, stylized right.")
        self.compare.currentTextChanged.connect(self.compare_changed.emit)
        vf.addRow("Compare", self.compare)
        row, self.view_ev = _slider_row(
            "View EV", -4, 4, 0.05, 0.0, 2,
            "Viewport exposure only. Does not write into the exported maps.")
        self.view_ev.valueChanged.connect(self.view_ev_changed.emit)
        vf.addRow(row)
        self.displace = QCheckBox("Displace")
        self.displace.setChecked(True)
        self.displace.setToolTip(
            "Tessellate the preview mesh and push it with the height map. "
            "Off is a flat mesh. Viewport only; not written on export.")
        self.displace.toggled.connect(self.displace_changed.emit)
        vf.addRow(self.displace)
        form.addWidget(view)

        basic = QGroupBox("Basic")
        bf = QFormLayout(basic)
        self.size = QComboBox()
        for s in SIZES:
            self.size.addItem(str(s), s)
        self.size.setToolTip("Export resolution. Live preview caps at 1024.")
        self.size.currentIndexChanged.connect(self._emit)
        bf.addRow("Output size", self.size)
        self.filt = QComboBox()
        self.filt.addItems(FILTERS)
        self.filt.setToolTip("Kuwahara is the default painterly filter. Classic is the 4-sector original.")
        self.filt.currentIndexChanged.connect(self._emit)
        bf.addRow("Filter", self.filt)
        self.packing = QComboBox()
        for key in PACKINGS:
            self.packing.addItem(key.upper(), key)
        self.packing.setToolTip("MROD is metal, roughness, AO, displacement. Used by Exodus.")
        self.packing.currentIndexChanged.connect(self._emit)
        bf.addRow("Packing", self.packing)
        form.addWidget(basic)

        paint = QGroupBox("Paint")
        pv = QVBoxLayout(paint)
        for key, lab, lo, hi, step, dec, tip in (
            ("radius", "Radius", 0, 32, 0.1, 2,
             "Kuwahara neighbourhood in pixels at 1024. 0 leaves the source maps unfiltered."),
            ("passes", "Passes", 1, 4, 1, 0,
             "How many times the filter runs. More passes flatten small detail."),
            ("sharpness", "Sharpness", 1, 24, 0.5, 1,
             "How strongly the filter prefers a low-variance sector. Higher keeps edges."),
            ("stroke", "Stroke", 0, 1, 0.01, 2,
             "Anisotropic stretch along image flow. Needs Radius above 0. 0 is circular Kuwahara."),
        ):
            row, spin = _slider_row(lab, lo, hi, step, 0, dec, tip)
            spin.valueChanged.connect(self._emit)
            self._spins[key] = spin
            pv.addWidget(row)
        form.addWidget(paint)

        grade = QGroupBox("Grade")
        gv = QVBoxLayout(grade)
        for key, lab, lo, hi, step, dec, tip in (
            ("grade_exposure", "Exposure", -2, 2, 0.05, 2, "Stops of exposure baked into the albedo."),
            ("temp", "Temperature", -1, 1, 0.01, 2, "Cool left, warm right. Baked into albedo."),
            ("grade_tint", "Tint", -1, 1, 0.01, 2, "Green left, magenta right."),
            ("contrast", "Contrast", -1, 1, 0.01, 2, "Oklab lightness contrast."),
            ("saturation", "Saturation", 0, 2, 0.01, 2, "1 is unchanged. 0 is grey. Baked into albedo."),
            ("value_compression", "Value compress", 0, 1, 0.01, 2, "Pulls lightness toward mid-grey."),
            ("posterize", "Posterize", 0, 16, 1, 0, "Steps of lightness only. Chroma is kept. 0 is off."),
            ("ao_strength", "AO into albedo", 0, 1, 0.01, 2, "Multiplies albedo by ambient occlusion."),
            ("edge_strength", "Edge", 0, 1, 0.01, 2, "Bakes curvature into the albedo as edge darkening and highlights."),
            ("palette_strength", "Palette strength", 0, 1, 0.01, 2, "How far albedo is pulled toward the swatches below."),
            ("palette_hardness", "Palette hardness", 0.5, 32, 0.5, 1, "How strictly each pixel snaps to the nearest swatch."),
        ):
            row, spin = _slider_row(lab, lo, hi, step, 0, dec, tip)
            spin.valueChanged.connect(self._emit)
            self._spins[key] = spin
            gv.addWidget(row)
        pal_lab = QLabel("Palette")
        pal_lab.setToolTip("Click + to add a colour. The grade snaps albedo toward these swatches.")
        gv.addWidget(pal_lab)
        self.palette = PaletteStrip()
        self.palette.setToolTip("Click a swatch to edit. × removes it.")
        self.palette.changed.connect(self._emit)
        gv.addWidget(self.palette)
        form.addWidget(grade)

        maps = QGroupBox("Maps")
        mv = QVBoxLayout(maps)
        for key, lab, lo, hi, step, val, dec, tip in (
            ("normal_strength", "Normal strength", 0.0, 2.0, 0.05, 1.0, 2,
             "0 is a flat normal. 1 is the source slope. 2 doubles XY. Lit uses this as the bump."),
            ("normal_soften", "Normal soften", 0.0, 1.0, 0.05, 0.0, 2,
             "Blur the normal. Independent of paint radius."),
            ("roughness_flatten", "Rough flatten", 0.0, 1.0, 0.05, 0.0, 2,
             "0 keeps roughness variation. 1 is a flat average."),
            ("roughness_bias", "Rough bias", -0.5, 0.5, 0.02, 0.0, 2,
             "Adds or subtracts roughness after flatten."),
            ("height_soften", "Height soften", 0.0, 1.0, 0.05, 0.0, 2,
             "Blur displacement. Lit tessellates and displaces the mesh; also the Displacement channel."),
            ("detail_restore", "Detail restore", 0.0, 1.0, 0.05, 0.0, 2,
             "Adds a high-pass of the source albedo back after paint."),
            ("mix_normal", "Normal mix", 0.0, 1.0, 0.01, 1.0, 2,
             "1 is the painted normal. 0 is the source normal. Needs Radius above 0 to differ."),
            ("mix_height", "Height mix", 0.0, 1.0, 0.01, 1.0, 2,
             "1 is painted displacement. 0 is the source height map."),
            ("mix_rough", "Rough mix", 0.0, 1.0, 0.01, 1.0, 2,
             "1 is painted roughness. 0 is the source roughness."),
            ("mix_metal", "Metal mix", 0.0, 1.0, 0.01, 1.0, 2,
             "1 is painted metallic. 0 is the source metallic."),
            ("mix_ao", "AO mix", 0.0, 1.0, 0.01, 1.0, 2,
             "1 is painted AO. 0 is the source AO. Lit uses this as occlusion."),
            ("seam_fade", "Seam fade", 0.0, 1.0, 0.01, 0.0, 2,
             "Blurs across UV tile edges so the filter does not leave a seam."),
        ):
            row, spin = _slider_row(lab, lo, hi, step, val, dec, tip)
            spin.valueChanged.connect(self._emit)
            self._spins[key] = spin
            mv.addWidget(row)
        form.addWidget(maps)

        mesh = QGroupBox("Mesh")
        me = QVBoxLayout(mesh)
        self.poly = QLabel("Triangles: —")
        self.poly.setToolTip("Triangle and vertex counts after the last load or decimate.")
        me.addWidget(self.poly)
        self.texel = QLabel("Texel density: —")
        self.texel.setToolTip("Median texels per metre from UVs and centimetre-scale positions.")
        me.addWidget(self.texel)
        row, self.decimate = _slider_row(
            "Decimate", 0.05, 1.0, 0.01, 1.0, 2,
            "Keep this fraction of triangles. 1 is the source mesh.")
        self.decimate.valueChanged.connect(self._decimate_emit)
        me.addWidget(row)
        self.fit = QPushButton("Fit to budget")
        self.fit.setToolTip("Decimate to the triangle ceiling of the selected class.")
        self.fit.clicked.connect(self.fit_budget_clicked.emit)
        me.addWidget(self.fit)
        self.rewrap_bake = QPushButton("Rewrap && bake from hi-poly")
        self.rewrap_bake.setToolTip(
            "Unwrap the mesh at its current decimated triangle count in "
            "Blender, then bake the hi-poly original's maps and surface "
            "normals onto the new UVs. The result replaces what you see "
            "here; Reset to default goes back to the untouched original.")
        self.rewrap_bake.clicked.connect(self.rewrap_bake_clicked.emit)
        me.addWidget(self.rewrap_bake)
        form.addWidget(mesh)

        actions = QHBoxLayout()
        self.reset = QPushButton("Reset to default")
        self.reset.setToolTip(
            "Full factory state for this material: default recipe, decimate "
            "back to 1.0, the original mesh, and the original maps. Any "
            "rewrap-and-bake result is cleared. Ctrl+Z walks recipe edits back.")
        self.reset.clicked.connect(self.reset_clicked.emit)
        self.export = QPushButton("Export")
        self.export.setToolTip("Write D / MROD / N into the export folder at the recipe size.")
        self.export.clicked.connect(self.export_clicked.emit)
        actions.addWidget(self.reset)
        actions.addWidget(self.export)
        form.addLayout(actions)
        form.addStretch(1)
        scroll.setWidget(inner)
        wrap = QVBoxLayout(self)
        wrap.addWidget(scroll)
        self.set_recipe(Recipe().to_dict())

    def set_recipe(self, st: dict):
        self._loading = True
        rec = Recipe.from_dict(st)
        idx = self.size.findData(rec.size)
        self.size.setCurrentIndex(max(idx, 0))
        i = self.filt.findText(rec.filter)
        self.filt.setCurrentIndex(max(i, 0))
        pi = self.packing.findData(rec.packing)
        self.packing.setCurrentIndex(max(pi, 0))
        d = rec.to_dict()
        for key, spin in self._spins.items():
            if key in d:
                spin.setValue(float(d[key]))
        self.palette.set_colors(rec.palette)
        self._loading = False

    def recipe(self) -> dict:
        rec = Recipe()
        rec.size = int(self.size.currentData() or 1024)
        rec.filter = self.filt.currentText()
        rec.packing = self.packing.currentData() or "mrod"
        d = rec.to_dict()
        for key, spin in self._spins.items():
            d[key] = spin.value()
        d["packing"] = rec.packing
        d["palette"] = self.palette.colors()
        return Recipe.from_dict(d).to_dict()

    def set_poly(self, tris, verts, cap_tris=None, islands=None, density=None):
        if tris is None:
            self.poly.setText("Triangles: —")
            self.texel.setText("Texel density: —")
            return
        extra = f" / {cap_tris} budget" if cap_tris else ""
        isl = f"   Islands: {islands}" if islands is not None else ""
        self.poly.setText(f"Triangles: {tris}   Vertices: {verts}{extra}{isl}")
        if density is not None:
            self.texel.setText(f"Texel density: {density:.0f} px/m")
        else:
            self.texel.setText("Texel density: —")

    def _emit(self, *_):
        if self._loading:
            return
        self.recipe_changed.emit(self.recipe())

    def _decimate_emit(self, *_):
        if self._loading:
            return
        self.decimate_changed.emit(float(self.decimate.value()))
