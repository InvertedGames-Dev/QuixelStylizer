"""One-button rewrap-and-bake, full Reset, and library delete."""
from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
from PIL import Image
from PySide6.QtGui import QSurfaceFormat
from PySide6.QtWidgets import QApplication, QMessageBox

import qs.blender as blender_client
from qs import library
from qs.app.viewport import default_gl_format
from qs.app.window import MainWindow
from qs.core.mesh import load_mesh, write_obj
from tests.test_gui_e2e import _make_surface_set, _pump
from tests.test_mesh import _write_mixed_uv_obj


def _setup_library(tmp_path, monkeypatch):
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


def _count_obj_faces(path: Path) -> int:
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines()
               if line.startswith("f "))


def test_one_button_bake_keeps_material_reset_and_delete(tmp_path, monkeypatch):
    _setup_library(tmp_path, monkeypatch)
    folder = tmp_path / "src" / "Brick_Prop"
    _make_surface_set(folder, "Brick_Prop", (150, 80, 60), size=64)
    original = folder / "Brick_Prop.obj"
    _write_mixed_uv_obj(original, n=12)
    mid = library.upsert_material(str(folder), "Brick_Prop")
    library.save_recipe(mid, {"size": 256, "radius": 2.0})
    rows_before = len(library.list_materials())

    calls = {}

    def fake_rewrap_bake(high, low, dest_dir, dest_mesh, cage_offset=0.02,
                         maps=None, size=1024, base_name="Baked"):
        calls.update(high=high, low=low, maps=dict(maps or {}), size=size,
                     low_faces=_count_obj_faces(Path(low)))
        dest = Path(dest_dir)
        dest.mkdir(parents=True, exist_ok=True)
        files = {}
        for kind, suffix in (("albedo", "BaseColor"), ("roughness", "Roughness")):
            path = dest / f"{base_name}_1K_{suffix}.png"
            Image.open(maps[kind]).save(path)
            files[kind] = str(path)
        # A tangent-space normal map: predominantly blue/violet.
        normal_path = dest / f"{base_name}_1K_Normal.png"
        nrm = np.zeros((32, 32, 3), np.uint8)
        nrm[..., 0] = 128
        nrm[..., 1] = 128
        nrm[..., 2] = 255
        Image.fromarray(nrm, "RGB").save(normal_path)
        files["normal"] = str(normal_path)
        # The fake unwrapped mesh is a loadable mesh beside the baked maps.
        mesh_out = dest / "low.obj"
        shutil.copyfile(low, mesh_out)
        return {"files": files, "dest": str(dest), "mesh": str(mesh_out)}

    monkeypatch.setattr(blender_client, "rewrap_bake", fake_rewrap_bake)

    QSurfaceFormat.setDefaultFormat(default_gl_format())
    app = QApplication.instance() or QApplication([])
    win = MainWindow()
    win.show()
    _pump(app, 0.8)
    try:
        assert win._mesh is not None
        assert Path(win._mesh["path"]) == original
        # Exactly one button for this workflow.
        assert win.develop.rewrap_bake.text().replace("&&", "&") == "Rewrap & bake from hi-poly"
        assert not hasattr(win.develop, "bake")
        assert not hasattr(win.develop, "reuv")
        assert not hasattr(win.develop, "restore_uv")

        # Decimate first: the button must work at the current count.
        source_tris = win._mesh["tris"]
        win.develop.decimate.setValue(0.5)
        _pump(app, 0.8)
        low_tris = win._mesh["tris"]
        assert low_tris < source_tris

        win._rewrap_bake()
        _pump(app, 1.0)

        # Blender was handed the hi-poly original and a serialized copy of
        # the current decimated mesh (not the source file path).
        assert Path(calls["high"]) == original
        assert Path(calls["low"]).name == "low_input.obj"
        assert calls["low_faces"] == low_tris
        assert Path(calls["maps"]["albedo"]).name.endswith("Albedo.png")

        # Same material, same row count, baked state stored in place.
        assert win._mid == mid
        assert len(library.list_materials()) == rows_before
        row = win._mat(mid)
        assert row["display_name"] == "Brick_Prop"
        assert Path(row["baked_maps_folder"]).is_dir()
        assert Path(row["baked_mesh_path"]).is_file()
        assert Path(row["original_mesh_path"]) == original
        assert win._mesh is not None
        assert win._mesh["tris"] == low_tris
        assert Path(win._mesh["path"]) == Path(row["baked_mesh_path"])
        assert win._src is not None
        assert Path(win._src["folder"]) == Path(row["baked_maps_folder"])
        # Original source maps were not overwritten.
        assert Path(calls["maps"]["albedo"]).is_file()

        # Reset: default recipe, decimate 1.0, original mesh and maps.
        win.develop._spins["radius"].setValue(9.0)
        _pump(app, 0.3)
        win._reset_recipe()
        _pump(app, 0.8)
        assert abs(win.develop.recipe()["radius"]) < 1e-4
        assert abs(win.develop.decimate.value() - 1.0) < 1e-9
        row = win._mat(mid)
        assert not row.get("baked_mesh_path")
        assert not row.get("baked_maps_folder")
        assert Path(row["mesh_path"]) == original
        assert win._mesh is not None
        assert Path(win._mesh["path"]) == original
        assert win._mesh["tris"] == source_tris
        assert win._src is not None
        assert Path(win._src["folder"]) == folder

        # Delete removes the entry, its recipe and undo history, but the
        # user's source files stay on disk.
        assert library.recipe_path(mid).is_file()
        assert (library.UNDO / mid).is_dir()
        win._delete_material(mid)
        _pump(app, 0.5)
        assert win._mat(mid) is None
        assert len(library.list_materials()) == rows_before - 1
        assert not library.recipe_path(mid).exists()
        assert not (library.UNDO / mid).exists()
        assert original.is_file()
        assert Path(calls["maps"]["albedo"]).is_file()
        # With no entries left the window is in a clean state.
        if win.library.list.count() == 0:
            assert win._mid is None
            assert win._mesh is None
            assert win._src is None
    finally:
        win.view.shutdown_gl()
        win.close()
        app.processEvents()


def test_library_panel_delete_button_confirms(tmp_path, monkeypatch):
    _setup_library(tmp_path, monkeypatch)
    folder = tmp_path / "src" / "Stone"
    _make_surface_set(folder, "Stone", (100, 100, 100), size=32)
    mid = library.upsert_material(str(folder), "Stone")

    QSurfaceFormat.setDefaultFormat(default_gl_format())
    app = QApplication.instance() or QApplication([])
    win = MainWindow()
    win.show()
    _pump(app, 0.6)
    try:
        assert win.library.list.count() == 1
        monkeypatch.setattr(
            QMessageBox, "question",
            staticmethod(lambda *a, **k: QMessageBox.StandardButton.Yes))
        win.library.remove_btn.click()
        _pump(app, 0.5)
        assert library.list_materials() == []
        assert win._mid is None
        assert win._mesh is None
        assert win.library.list.count() == 0
    finally:
        win.view.shutdown_gl()
        win.close()
        app.processEvents()


def test_write_obj_serializes_current_mesh(tmp_path):
    mesh_path = tmp_path / "grid.obj"
    _write_mixed_uv_obj(mesh_path, n=4)
    mesh = load_mesh(str(mesh_path))
    out = tmp_path / "low.obj"
    write_obj(mesh, str(out))
    faces = _count_obj_faces(out)
    assert faces == mesh["tris"]
    reloaded = load_mesh(str(out))
    assert reloaded["tris"] == mesh["tris"]
