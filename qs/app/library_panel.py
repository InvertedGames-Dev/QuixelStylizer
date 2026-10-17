from __future__ import annotations

from PySide6.QtCore import QRectF, Qt, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import (
    QComboBox, QFileDialog, QHBoxLayout, QLabel, QLineEdit, QListWidget,
    QListWidgetItem, QMenu, QMessageBox, QPushButton, QSizePolicy, QVBoxLayout,
    QWidget,
)

from qs import budgets, library

# Dark Fusion: light ticks, green/amber/red fill on a charcoal track.
_TRACK = QColor("#2c2c2c")
_TRACK_EDGE = QColor("#5a5a5a")
_FILL_OK = QColor("#3d9a6a")
_FILL_NEAR = QColor("#c9a227")
_FILL_OVER = QColor("#c44c4c")
_TICK = QColor("#e8e8e8")
_TICK_SEL = QColor("#f0c040")
_TEXT = QColor("#ececec")
_BG = QColor("#1a1a1a")


def _fmt_cap(n: int) -> str:
    if n >= 10000:
        return f"{n // 1000}k"
    return f"{n:,}"


class BudgetTracks(QWidget):
    """Two ruler tracks: fill is the asset, class names sit under each tick."""

    def __init__(self, meter: "BudgetMeter", parent=None):
        super().__init__(parent)
        self._meter = meter
        self.setMinimumHeight(118)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def paintEvent(self, _event):
        m = self._meter
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        p.fillRect(self.rect(), _BG)
        font = QFont(self.font())
        font.setPixelSize(11)
        p.setFont(font)
        fm = p.fontMetrics()
        left = 6
        title_w = fm.horizontalAdvance("Tris") + 8
        value_w = max(fm.horizontalAdvance("20,000"), fm.horizontalAdvance("2048 px")) + 8
        x = left + title_w
        w = max(self.width() - left - title_w - value_w - 6, 20)
        track_h = 12
        row_h = 56
        self._paint_track(p, "Tris", m.tris, "tris", "", x, 6, w, track_h, title_w, value_w)
        self._paint_track(p, "Tex", m.tex, "tex", " px", x, 6 + row_h, w, track_h, title_w, value_w)
        p.end()

    def _paint_track(self, p, title, value, kind, unit, x, y, w, h, title_w, value_w):
        m = self._meter
        p.setPen(_TEXT)
        p.drawText(QRectF(6, y, title_w, h), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, title)
        track = QRectF(x, y, w, h)
        p.fillRect(track, _TRACK)
        p.setPen(QPen(_TRACK_EDGE, 1))
        p.drawRect(track.adjusted(0, 0, -1, -1))
        frac = m.fill_pos(kind)
        if frac > 0:
            fw = max(track.width() * frac, 2.0)
            p.fillRect(QRectF(track.x() + 1, track.y() + 1, fw - 1, track.height() - 2), m._fill_color())
        inset = 8.0
        span = max(track.width() - 2 * inset, 1.0)
        rows = m.collapsed_ticks(kind)
        font = QFont(self.font())
        font.setPixelSize(10)
        p.setFont(font)
        fm = p.fontMetrics()
        n = max(len(rows), 1)
        for i, row in enumerate(rows):
            pos = min(max(row["pos"], 0.0), 1.0)
            tx = int(round(track.x() + inset + span * pos))
            selected = m.asset_class in row["names"]
            color = _TICK_SEL if selected else _TICK
            p.setPen(QPen(QColor(0, 0, 0, 180), 3))
            p.drawLine(tx, int(track.y() - 1), tx, int(track.bottom() + 4))
            p.setPen(QPen(color, 1.5))
            p.drawLine(tx, int(track.y() - 1), tx, int(track.bottom() + 4))
            preferred = [n for n in ("prop", "hero", "modular", "terrain", "foliage", "surface")
                         if n in row["names"]]
            label = preferred[0] if preferred else row["names"][0]
            tw = fm.horizontalAdvance(label)
            lx = tx - tw // 2
            # Keep labels inside the track so edge classes (surface, hero) stay readable.
            lx = max(int(track.x()), min(lx, int(track.right()) - tw))
            p.setPen(_TICK_SEL if selected else _TEXT)
            p.drawText(lx, int(track.bottom()) + 4 + fm.ascent(), label)
        p.setFont(self.font())
        p.setPen(_TEXT)
        p.drawText(
            QRectF(track.right() + 4, y, value_w, h),
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            f"{value:,}{unit}",
        )


class BudgetMeter(QWidget):
    """Asset fill on tris/tex tracks, class-ceiling ticks, wrapping legend of full names."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.tris = 0
        self.tex = 0
        self.asset_class = "prop"
        self._classes = budgets.class_table()
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)
        self.tracks = BudgetTracks(self)
        lay.addWidget(self.tracks)
        self.set_values(0, 0, "prop")

    def class_names(self) -> list[str]:
        names = [k for k in budgets.CLASS_ORDER if k in self._classes]
        for k in self._classes:
            if k not in names:
                names.append(k)
        return names

    def scale(self, kind: str) -> float:
        caps = [max(int(self._classes[k].get(kind) or 0), 1) for k in self.class_names()]
        actual = self.tris if kind == "tris" else self.tex
        return float(max(max(caps), int(actual or 0), 1))

    def _caps(self, kind: str) -> list[int]:
        seen = []
        for name in self.class_names():
            cap = int(self._classes[name].get(kind) or 0)
            if cap not in seen:
                seen.append(cap)
        return sorted(seen)

    def normalized_pos(self, kind: str, value: float) -> float:
        """Map a raw count onto equal-spaced class marks. Each distinct ceiling gets the same width."""
        caps = self._caps(kind)
        if not caps:
            return 0.0
        n = len(caps)
        v = float(max(value, 0.0))
        if n == 1:
            return 0.0 if v <= caps[0] else 1.0
        if v <= caps[0]:
            return 0.0
        if v >= caps[-1]:
            return 1.0
        for i in range(len(caps) - 1):
            lo, hi = float(caps[i]), float(caps[i + 1])
            if v <= hi:
                t = 0.0 if hi <= lo else (v - lo) / (hi - lo)
                return (i + t) / (n - 1)
        return 1.0

    def fill_pos(self, kind: str) -> float:
        value = self.tris if kind == "tris" else self.tex
        return self.normalized_pos(kind, value)

    def tick_positions(self, kind: str) -> dict[str, float]:
        out = {}
        for row in self.collapsed_ticks(kind):
            for name in row["names"]:
                out[name] = row["pos"]
        return out

    def collapsed_ticks(self, kind: str) -> list[dict]:
        """One row per distinct ceiling. Positions are equally spaced, not linear in count."""
        buckets: dict[int, list[str]] = {}
        for name in self.class_names():
            cap = int(self._classes[name].get(kind) or 0)
            buckets.setdefault(cap, []).append(name)
        caps = sorted(buckets)
        n = max(len(caps), 1)
        rows = []
        for i, cap in enumerate(caps):
            pos = 0.0 if n == 1 else i / (n - 1)
            rows.append({"pos": pos, "names": buckets[cap], "cap": cap})
        return rows

    def fits_classes(self) -> list[str]:
        fits = []
        for name in self.class_names():
            cap = self._classes[name]
            if self.tris <= int(cap.get("tris") or 0) and self.tex <= int(cap.get("tex") or 0):
                fits.append(name)
        return fits

    def legend_text(self) -> str:
        lines = []
        for kind, title, unit in (("tris", "Tris", ""), ("tex", "Tex", " px")):
            bits = []
            for row in self.collapsed_ticks(kind):
                names = " / ".join(row["names"])
                bits.append(f"{names} {_fmt_cap(row['cap'])}{unit}")
            lines.append(f"{title}:  " + "  ·  ".join(bits))
        fits = self.fits_classes()
        lines.append("Fits:  " + (", ".join(fits) if fits else "none"))
        return "\n".join(lines)

    def set_values(self, tris, tex, asset_class: str):
        self.tris = 0 if tris is None else int(tris)
        self.tex = 0 if tex is None else int(tex)
        self.asset_class = asset_class or "prop"
        self._classes = budgets.class_table()
        self.setToolTip(self.legend_text())
        self.tracks.update()
        self.tracks.setToolTip(self.legend_text())

    def _fill_color(self) -> QColor:
        st = budgets.status(self.tris, self.tex, self.asset_class)
        if st == "over":
            return _FILL_OVER
        if st == "near":
            return _FILL_NEAR
        return _FILL_OK


class LibraryPanel(QWidget):
    material_chosen = Signal(str)
    import_requested = Signal(str)
    export_root_changed = Signal(str)
    bulk_apply_requested = Signal()
    bulk_export_requested = Signal()
    bulk_fit_requested = Signal()
    delete_requested = Signal(str)
    class_changed = Signal(str, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.filter_over = False
        lay = QVBoxLayout(self)
        lib_lab = QLabel("Library")
        lib_lab.setToolTip("Imported Quixel sets. Click a row to load it in the viewport.")
        lay.addWidget(lib_lab)
        self.search = QLineEdit()
        self.search.setPlaceholderText("Filter…")
        self.search.setToolTip("Filter the list by display name.")
        self.search.textChanged.connect(self.refresh)
        lay.addWidget(self.search)
        row = QHBoxLayout()
        self.cls = QComboBox()
        self.cls.addItems(["all", "hero", "prop", "foliage", "modular", "terrain", "surface"])
        self.cls.setToolTip("Show only this asset class.")
        self.cls.currentTextChanged.connect(self.refresh)
        row.addWidget(self.cls)
        self.over = QPushButton("Over budget")
        self.over.setCheckable(True)
        self.over.setToolTip("Show only assets over their class triangle or texture ceiling.")
        self.over.toggled.connect(lambda v: setattr(self, "filter_over", v) or self.refresh())
        row.addWidget(self.over)
        lay.addLayout(row)
        self.list = QListWidget()
        self.list.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        self.list.setToolTip("Select one material to edit. Shift or Ctrl select for bulk actions.")
        self.list.currentItemChanged.connect(self._picked)
        self.list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.list.customContextMenuRequested.connect(self._context_menu)
        lay.addWidget(self.list, 1)
        self.meter = BudgetMeter()
        self.bar = self.meter  # alias: tests and older callers
        self.meter.setToolTip(
            "Green fill is this asset. Marks are equally spaced per class, not linear in count. "
            "Pick the class whose mark sits past the fill.")
        lay.addWidget(self.meter)
        self.name = QLineEdit()
        self.name.setPlaceholderText("Display name")
        self.name.setToolTip("Folder name used on export. Must be unique and legal for Windows.")
        self.name.editingFinished.connect(self._rename)
        lay.addWidget(self.name)
        cls_row = QHBoxLayout()
        cls_lab = QLabel("Class")
        cls_lab.setToolTip("Triangle and texture ceiling for this asset. Used by Fit to budget.")
        cls_row.addWidget(cls_lab)
        self.item_cls = QComboBox()
        self.item_cls.addItems(["hero", "prop", "foliage", "modular", "terrain", "surface"])
        self.item_cls.setToolTip("Assign the PBR 360 class. The meter highlights this class’s mark.")
        self.item_cls.currentTextChanged.connect(self._set_class)
        cls_row.addWidget(self.item_cls)
        lay.addLayout(cls_row)
        btns = QHBoxLayout()
        for label, slot, tip in (
            ("Import folder", self._import_folder, "Add a folder of Quixel maps to the library."),
            ("Import zip", self._import_zip, "Extract a Megascans zip into the library."),
            ("Export folder", self._export_root, "Root folder for D / MROD / N writes."),
        ):
            b = QPushButton(label)
            b.setToolTip(tip)
            b.clicked.connect(slot)
            btns.addWidget(b)
        lay.addLayout(btns)
        bulk = QHBoxLayout()
        for label, sig, tip in (
            ("Apply recipe", self.bulk_apply_requested, "Copy this recipe onto every selected material."),
            ("Fit selected", self.bulk_fit_requested, "Decimate each selected mesh to its class ceiling."),
            ("Export selected", self.bulk_export_requested, "Write selected materials into the export folder."),
        ):
            b = QPushButton(label)
            b.setToolTip(tip)
            b.clicked.connect(sig.emit)
            bulk.addWidget(b)
        lay.addLayout(bulk)
        self.remove_btn = QPushButton("Remove from library")
        self.remove_btn.setToolTip(
            "Remove the selected material from the library catalog. "
            "Your source texture and mesh files on disk are not deleted.")
        self.remove_btn.clicked.connect(lambda: self._request_delete())
        lay.addWidget(self.remove_btn)
        self.status = QLabel("")
        self.status.setWordWrap(True)
        lay.addWidget(self.status)
        self._setting_class = False
        self.refresh()

    def refresh(self):
        q = self.search.text().lower().strip()
        cls = self.cls.currentText()
        current = self.current_id()
        selected = set(self.selected_ids())
        self.list.blockSignals(True)
        try:
            self._fill_list(q, cls, current, selected)
        finally:
            self.list.blockSignals(False)
        self._update_bar(current)

    def _fill_list(self, q, cls, current, selected):
        self.list.clear()
        restore = None
        for mat in library.list_materials():
            if cls != "all" and mat["asset_class"] != cls:
                continue
            if q and q not in mat["display_name"].lower():
                continue
            tex = max(mat.get("tex_w") or 0, mat.get("tex_h") or 0)
            st = budgets.status(mat.get("tris"), tex, mat["asset_class"])
            if self.filter_over and st != "over":
                continue
            label = self._material_label(mat)
            item = QListWidgetItem(label)
            item.setData(Qt.ItemDataRole.UserRole, mat["id"])
            if st == "over":
                item.setForeground(Qt.GlobalColor.red)
            elif st == "near":
                item.setForeground(Qt.GlobalColor.yellow)
            if mat["missing"]:
                item.setForeground(Qt.GlobalColor.gray)
            self.list.addItem(item)
            if mat["id"] == current:
                restore = item
            if mat["id"] in selected:
                item.setSelected(True)
        if restore:
            self.list.setCurrentItem(restore)

    @staticmethod
    def _material_label(mat: dict) -> str:
        label = mat["display_name"]
        if mat.get("missing"):
            label += "  [missing]"
        tris = mat.get("tris")
        cap = budgets.ceiling(mat["asset_class"])
        if tris is not None:
            label += f"  {tris}/{cap['tris']} tris"
        tex = max(mat.get("tex_w") or 0, mat.get("tex_h") or 0)
        if tex:
            label += f"  {tex}px"
        dens = mat.get("texel_density")
        if dens:
            label += f"  {dens:.0f}px/m"
        return label

    def refresh_current(self):
        """Update the selected row and meter without rebuilding the list (avoids a reload)."""
        mid = self.current_id()
        self._update_bar(mid)
        it = self.list.currentItem()
        if it is None or not mid:
            return
        mats = {m["id"]: m for m in library.list_materials()}
        mat = mats.get(mid)
        if not mat:
            return
        it.setText(self._material_label(mat))

    def current_id(self) -> str | None:
        it = self.list.currentItem()
        return it.data(Qt.ItemDataRole.UserRole) if it else None

    def selected_ids(self) -> list[str]:
        return [it.data(Qt.ItemDataRole.UserRole) for it in self.list.selectedItems()]

    def _picked(self, cur, _prev):
        if cur is None:
            return
        mid = cur.data(Qt.ItemDataRole.UserRole)
        mats = {m["id"]: m for m in library.list_materials()}
        mat = mats.get(mid)
        if mat:
            self.name.setText(mat["display_name"])
            self._setting_class = True
            i = self.item_cls.findText(mat["asset_class"])
            self.item_cls.setCurrentIndex(max(i, 0))
            self._setting_class = False
            if mat["missing"]:
                self.status.setText(f"Source folder is missing:\n{mat['source_folder']}")
            else:
                self.status.setText(mat["source_folder"])
            self._update_bar(mid)
        self.material_chosen.emit(mid)

    def _update_bar(self, mid):
        mats = {m["id"]: m for m in library.list_materials()}
        mat = mats.get(mid)
        if not mat:
            self.meter.set_values(0, 0, "prop")
            return
        tris = mat.get("tris") or 0
        tex = max(mat.get("tex_w") or 0, mat.get("tex_h") or 0)
        self.meter.set_values(tris, tex, mat["asset_class"])

    def _context_menu(self, pos):
        item = self.list.itemAt(pos)
        if item is None:
            return
        self.list.setCurrentItem(item)
        menu = QMenu(self)
        act = menu.addAction("Remove from library")
        act.triggered.connect(lambda: self._request_delete(item.data(Qt.ItemDataRole.UserRole)))
        menu.exec(self.list.viewport().mapToGlobal(pos))

    def _request_delete(self, mid=None):
        mid = mid or self.current_id()
        if not mid:
            return
        mats = {m["id"]: m for m in library.list_materials()}
        mat = mats.get(mid)
        name = mat["display_name"] if mat else mid
        answer = QMessageBox.question(
            self, "Remove from library",
            f"Remove “{name}” from the library?\n\n"
            "This removes the catalog entry, its recipe, and its undo "
            "history. Your source texture and mesh files on disk will "
            "not be deleted.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer == QMessageBox.StandardButton.Yes:
            self.delete_requested.emit(mid)

    def _rename(self):
        mid = self.current_id()
        if not mid:
            return
        try:
            library.rename_material(mid, self.name.text())
            self.refresh()
        except ValueError as exc:
            self.status.setText(str(exc))

    def _set_class(self, name: str):
        if self._setting_class:
            return
        mid = self.current_id()
        if not mid:
            return
        library.set_class(mid, name)
        self.class_changed.emit(mid, name)
        self.refresh()

    def _import_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Import material folder")
        if folder:
            self.import_requested.emit(folder)

    def _import_zip(self):
        path, _ = QFileDialog.getOpenFileName(self, "Import zip", "", "Zip (*.zip)")
        if path:
            self.import_requested.emit(path)

    def _export_root(self):
        folder = QFileDialog.getExistingDirectory(
            self, "Export folder", library.get_setting("export_root", r"C:\Users\BigMoney\Projects\Exodus\Content"))
        if folder:
            library.set_setting("export_root", folder)
            self.export_root_changed.emit(folder)
            self.status.setText(f"Export: {folder}")
