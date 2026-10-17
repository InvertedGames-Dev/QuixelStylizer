"""Decimate through the live viewport with mixed UVs — the path that heap-smashed."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from PySide6.QtGui import QSurfaceFormat
from PySide6.QtWidgets import QApplication

from qs import library
from qs.app.jobs import _ACTIVE_JOBS, pool
from qs.app.viewport import default_gl_format
from qs.app.window import MainWindow
from tests.test_gui_e2e import _grab_rgb, _make_surface_set
from tests.test_mesh import _write_mixed_uv_obj


def _pump(app: QApplication, seconds: float = 0.4):
    deadline = time.time() + seconds
    while time.time() < deadline or len(_ACTIVE_JOBS) > 0 or not pool().waitForDone(10):
        app.processEvents()
        if time.time() >= deadline and len(_ACTIVE_JOBS) == 0:
            break
        time.sleep(0.015)
    app.processEvents()


def test_gui_decimate_mixed_uv_mesh(tmp_path, monkeypatch):
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

    folder = tmp_path / "src" / "Brick_Prop"
    _make_surface_set(folder, "Brick_Prop", (140, 90, 70), size=64)
    _write_mixed_uv_obj(folder / "Brick_Prop.obj", n=24)
    mid = library.upsert_material(str(folder), "Brick_Prop")
    library.save_recipe(mid, {"size": 256, "radius": 4.0})

    QSurfaceFormat.setDefaultFormat(default_gl_format())
    app = QApplication.instance() or QApplication([])
    win = MainWindow()
    win.show()
    _pump(app, 0.6)
    assert win.view._gl_ok
    assert win._mesh is not None
    src_tris = win._mesh["tris"]
    assert src_tris > 200

    win.develop.decimate.setValue(0.45)
    _pump(app, 0.8)
    assert win._mesh["tris"] < src_tris * 0.7
    frame = _grab_rgb(win.view)
    assert frame.mean() > 8.0

    win.develop.decimate.setValue(0.2)
    _pump(app, 0.8)
    low = win._mesh["tris"]
    assert low < src_tris * 0.4

    win.develop.decimate.setValue(0.6)
    _pump(app, 0.8)
    assert win._mesh["tris"] > low
    assert win.view._gl_ok

    skull = Path(
        r"C:\Users\BigMoney\Projects\QuixelStylizer\library\sources"
        r"\ram_skull_tiiief2fa_low_345e19b6\mesh\Ram_Skull_tiiief2fa_Low.glb"
    )
    if skull.is_file():
        from qs.core.mesh import load_mesh, decimate
        m = load_mesh(str(skull))
        win._mesh_src = m
        win._mesh = m
        win.view.set_mesh(m)
        win.develop.decimate.setValue(0.35)
        _pump(app, 1.0)
        assert win._mesh["tris"] < m["tris"]
        d = decimate(m, 0.35)
        assert d["tris"] < m["tris"]

    win.view.shutdown_gl()
    win.close()
    app.processEvents()


def test_mesh_paint_is_gpu_and_stroke_changes(tmp_path, monkeypatch):
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

    folder = tmp_path / "src" / "Moss_On_Mesh"
    _make_surface_set(folder, "Moss_On_Mesh", (55, 175, 70), size=64)
    _write_mixed_uv_obj(folder / "Moss_On_Mesh.obj", n=16)
    mid = library.upsert_material(str(folder), "Moss_On_Mesh")
    library.save_recipe(mid, {"size": 256, "radius": 8.0, "stroke": 0.0})

    QSurfaceFormat.setDefaultFormat(default_gl_format())
    app = QApplication.instance() or QApplication([])
    win = MainWindow()
    win.show()
    _pump(app, 0.8)
    assert win._mesh is not None
    assert win._src is not None
    notes = (win._src.get("report") or {}).get("notes") or []
    assert any("GPU" in n for n in notes), f"mesh paint stayed on CPU: {notes}"
    alb0 = np.array(win._painted["alb"], copy=True)
    t0 = time.perf_counter()
    win.develop._spins["stroke"].setValue(1.0)
    _pump(app, 0.8)
    elapsed = time.perf_counter() - t0
    assert elapsed < 2.0, f"mesh stroke preview took {elapsed:.2f}s"
    alb1 = np.array(win._painted["alb"], copy=True)
    assert float(np.abs(alb1 - alb0).mean()) > 0.002, "Stroke did not change the painted maps"
    win.view.shutdown_gl()
    win.close()
    app.processEvents()
