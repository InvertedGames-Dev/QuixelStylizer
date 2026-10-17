"""Clicking a meshed asset must load its maps, not the previous material's."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QSurfaceFormat
from PySide6.QtWidgets import QApplication

from qs import library
from qs.app.jobs import _ACTIVE_JOBS
from qs.app.viewport import default_gl_format
from qs.app.window import MainWindow

from tests.test_gui_e2e import _make_surface_set, _pump, _write_dense_grid_obj


def _isolate_library(tmp_path, monkeypatch):
    lib_root = tmp_path / "library"
    monkeypatch.setattr(library, "LIBRARY", lib_root)
    monkeypatch.setattr(library, "DB_PATH", lib_root / "library.sqlite")
    monkeypatch.setattr(library, "SOURCES", lib_root / "sources")
    monkeypatch.setattr(library, "INBOX", lib_root / "inbox")
    monkeypatch.setattr(library, "MASKS", lib_root / "masks")
    monkeypatch.setattr(library, "RECIPES", lib_root / "recipes")
    monkeypatch.setattr(library, "THUMBS", lib_root / "thumbs")
    monkeypatch.setattr(library, "UNDO", lib_root / "undo")
    library._SCHEMA_READY.clear()


def _select_id(win, mid):
    for i in range(win.library.list.count()):
        if win.library.list.item(i).data(Qt.ItemDataRole.UserRole) == mid:
            win.library.list.setCurrentRow(i)
            return
    raise AssertionError(f"material {mid} not in list")


def test_mesh_click_does_not_keep_previous_maps(tmp_path, monkeypatch):
    _isolate_library(tmp_path, monkeypatch)
    moss_dir = tmp_path / "src" / "Nordic_Moss"
    stump_dir = tmp_path / "src" / "Old_Tree_Stump"
    _make_surface_set(moss_dir, "Nordic_Moss", (55, 175, 70), size=64)
    _make_surface_set(stump_dir, "Old_Tree_Stump", (160, 90, 40), size=64)
    _write_dense_grid_obj(stump_dir / "Old_Tree_Stump.obj", n=8)

    mid_moss = library.upsert_material(str(moss_dir), "Nordic_Moss")
    mid_stump = library.upsert_material(str(stump_dir), "Old_Tree_Stump")

    QSurfaceFormat.setDefaultFormat(default_gl_format())
    app = QApplication.instance() or QApplication([])
    win = MainWindow()
    win.show()
    _pump(app, 0.3)

    _select_id(win, mid_moss)
    _pump(app, 0.8)
    assert win._mid == mid_moss
    assert win._src is not None
    assert "Moss" in win._src["name"]
    moss_alb = np.asarray(win._src["alb"])
    assert float(moss_alb[..., 1].mean()) > float(moss_alb[..., 0].mean())

    real_load = win._load_src

    def delayed_load(folder, rec):
        # Longer than the 40 ms preview debounce so the mesh arrives first.
        time.sleep(0.12)
        return real_load(folder, rec)

    win._load_src = delayed_load
    _select_id(win, mid_stump)
    _pump(app, 1.5)

    assert win._mid == mid_stump
    assert win._src is not None, "stump maps never landed"
    assert "Stump" in win._src["name"], f"stump kept the moss maps: {win._src.get('name')}"
    assert Path(win._src["_set"]["folder"]).resolve() == stump_dir.resolve()
    alb = np.asarray(win._src["alb"])
    assert float(alb[..., 0].mean()) > float(alb[..., 1].mean()), "albedo is still the moss green"
    assert win._mesh is not None
    _pump(app, 0.2)
    win.close()
    deadline = time.time() + 2.0
    while _ACTIVE_JOBS and time.time() < deadline:
        app.processEvents()
        time.sleep(0.02)
    app.processEvents()


def test_live_stump_not_moss_if_present():
    root = Path(__file__).resolve().parents[1] / "library" / "sources"
    stump = next(root.glob("old_tree_stump_*"), None)
    moss = next(root.glob("nordic_moss_*"), None)
    if stump is None or moss is None:
        return
    from qs.core.maps import detect_maps

    s_stump = detect_maps(str(stump))
    s_moss = detect_maps(str(moss))
    assert s_stump is not None and s_moss is not None
    assert "Stump" in s_stump["name"]
    assert "Moss" in s_moss["name"]
    assert Path(s_stump["maps"]["albedo"]).parent == stump
    assert Path(s_moss["maps"]["albedo"]).parent == moss
