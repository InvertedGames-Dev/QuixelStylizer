"""End-to-end GUI, OpenGL viewport, and feature audit tests."""
from __future__ import annotations

import time
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image
from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QMouseEvent, QSurfaceFormat, QWheelEvent
from PySide6.QtWidgets import QApplication

from qs import budgets, library
from qs.app.jobs import _ACTIVE_JOBS, pool
from qs.app.viewport import default_gl_format
from qs.app.window import MainWindow


def _pump(app: QApplication, seconds: float = 0.6):
    deadline = time.time() + seconds
    while time.time() < deadline or len(_ACTIVE_JOBS) > 0 or not pool().waitForDone(10):
        app.processEvents()
        if time.time() >= deadline and len(_ACTIVE_JOBS) == 0:
            break
        time.sleep(0.015)
    app.processEvents()


def _grab_rgb(view) -> np.ndarray:
    qimg = view.grabFramebuffer().convertToFormat(
        view.grabFramebuffer().Format.Format_RGB888
    )
    w, h = qimg.width(), qimg.height()
    ptr = qimg.bits()
    arr = np.frombuffer(ptr, np.uint8).reshape((h, qimg.bytesPerLine()))[:, : w * 3]
    return arr.reshape((h, w, 3)).copy()


def _make_surface_set(folder: Path, name: str, color: tuple[int, int, int], size: int = 64):
    folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(sum(color))
    base = np.full((size, size, 3), color, np.float32) + rng.randint(-35, 35, (size, size, 3))
    alb = np.clip(base, 0, 255).astype(np.uint8)
    Image.fromarray(alb, "RGB").save(folder / f"{name}_1K_Albedo.png")
    rough = np.clip(120 + rng.randint(-30, 30, (size, size)), 0, 255).astype(np.uint8)
    Image.fromarray(rough, "L").save(folder / f"{name}_1K_Roughness.png")
    Image.fromarray(np.full((size, size), 12, np.uint8), "L").save(folder / f"{name}_1K_Metallic.png")
    Image.fromarray(np.full((size, size), 220, np.uint8), "L").save(folder / f"{name}_1K_AO.png")
    u = np.linspace(-1.0, 1.0, size)
    uu, vv = np.meshgrid(u, u)
    nrm = np.dstack([
        np.clip(128 + uu * 90 + rng.randint(-8, 8, (size, size)), 0, 255).astype(np.uint8),
        np.clip(128 + vv * 90 + rng.randint(-8, 8, (size, size)), 0, 255).astype(np.uint8),
        np.full((size, size), 200, np.uint8),
    ])
    Image.fromarray(nrm, "RGB").save(folder / f"{name}_1K_NormalDX.png")
    Image.fromarray(np.full((size, size), 128, np.uint8), "L").save(folder / f"{name}_1K_Displacement.png")


def _write_dense_grid_obj(path: Path, n: int = 24):
    """Write a UV-mapped grid mesh in cm with 2 * n * n triangles (e.g. n=24 -> 1152 tris)."""
    lines = []
    for j in range(n + 1):
        for i in range(n + 1):
            u, v = i / n, j / n
            x = (u - 0.5) * 100.0
            y = (v - 0.5) * 100.0
            z = 8.0 * np.sin(u * np.pi * 2) * np.cos(v * np.pi * 2)
            lines.append(f"v {x:.4f} {y:.4f} {z:.4f}")
            lines.append(f"vt {u:.4f} {v:.4f}")
    for j in range(n):
        for i in range(n):
            a = j * (n + 1) + i + 1
            b = a + 1
            c = a + (n + 1) + 1
            d = a + (n + 1)
            lines.append(f"f {a}/{a} {b}/{b} {c}/{c}")
            lines.append(f"f {a}/{a} {c}/{c} {d}/{d}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_full_gui_e2e_audit(tmp_path, monkeypatch):
    # Isolate library to tmp_path so tests never pollute user data.
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

    mat_a_dir = tmp_path / "src" / "Red_Cliff_Rock"
    mat_b_dir = tmp_path / "src" / "Moss_Forest_Ground"
    _make_surface_set(mat_a_dir, "Red_Cliff_Rock", (200, 75, 50), size=64)
    _make_surface_set(mat_b_dir, "Moss_Forest_Ground", (55, 175, 70), size=64)
    _write_dense_grid_obj(mat_b_dir / "Moss_Forest_Ground.obj", n=24)  # 1152 tris

    mid_a = library.upsert_material(str(mat_a_dir), "Red_Cliff_Rock")
    mid_b = library.upsert_material(str(mat_b_dir), "Moss_Forest_Ground")
    library.set_class(mid_b, "foliage")  # foliage cap is 800 tris, 512 px -> 1152 tris is over budget!
    library.save_recipe(mid_a, {"size": 256, "radius": 20.0, "passes": 2, "saturation": 1.15})
    library.save_recipe(mid_b, {"size": 256, "radius": 6.0, "passes": 1, "saturation": 1.0})

    QSurfaceFormat.setDefaultFormat(default_gl_format())
    app = QApplication.instance() or QApplication([])
    win = MainWindow()
    win.show()
    _pump(app, 0.3)

    assert win.view._gl_ok, "OpenGL 4.3 viewport context failed to initialize"
    assert win.library.list.count() == 2

    # 1. Select first material (Red_Cliff_Rock, radius=20.0 > 16)
    win.library.list.setCurrentRow(1)  # Red_Cliff_Rock (sorted alphabetically after Moss)
    _pump(app, 0.5)
    assert win._mid == mid_a
    assert abs(win.develop.recipe()["radius"] - 20.0) < 1e-4, "Radius > 16 was clamped!"
    frame_a = _grab_rgb(win.view)
    assert frame_a.mean() > 15.0 and frame_a.std() > 15.0, "Viewport rendered blank/flat frame!"
    assert win.view._tess, "Lit should tessellate and displace from height, not parallax"

    # 2. Switch shape (sphere -> cube -> plane -> sphere) and Channel view modes
    win.develop.shape.setCurrentText("cube")
    _pump(app, 0.2)
    frame_cube = _grab_rgb(win.view)
    assert np.abs(frame_cube.astype(np.float32) - frame_a.astype(np.float32)).mean() > 2.0
    win.develop.shape.setCurrentText("plane")
    _pump(app, 0.2)
    frame_plane = _grab_rgb(win.view)
    assert np.abs(frame_plane.astype(np.float32) - frame_cube.astype(np.float32)).mean() > 2.0
    h, w = win._fin["height"].shape[:2]
    ramp = np.linspace(0.05, 0.95, w, dtype=np.float32)
    high = dict(win._fin)
    high["height"] = np.broadcast_to(ramp, (h, w)).copy()
    low = dict(win._fin)
    low["height"] = np.full((h, w), 0.5, np.float32)
    win.view.set_maps(low, win._src)
    _pump(app, 0.1)
    flat_h = _grab_rgb(win.view)
    win.view.set_maps(high, win._src)
    _pump(app, 0.1)
    disp_h = _grab_rgb(win.view)
    win.view.set_maps(win._fin, win._src)
    _pump(app, 0.1)
    assert float(np.abs(disp_h.astype(np.float32) - flat_h.astype(np.float32)).mean()) > 1.5, (
        "Height ramp did not displace the tessellated mesh")
    win.develop.displace.setChecked(False)
    _pump(app, 0.1)
    assert win.view.displace is False
    win.view.set_maps(high, win._src)
    _pump(app, 0.1)
    off_high = _grab_rgb(win.view)
    win.view.set_maps(low, win._src)
    _pump(app, 0.1)
    off_flat = _grab_rgb(win.view)
    assert float(np.abs(off_high.astype(np.float32) - off_flat.astype(np.float32)).mean()) < 1.0, (
        "Displace off still moved the mesh with height")
    win.develop.displace.setChecked(True)
    win.view.set_maps(win._fin, win._src)
    _pump(app, 0.1)
    for ch in ("Normal", "Roughness", "AO", "Displacement", "Metallic", "Diffuse", "Lit"):
        win.develop.channel.setCurrentText(ch)
        _pump(app, 0.1)
        assert win.view.channel == ch
    win.develop.shape.setCurrentText("sphere")
    _pump(app, 0.2)

    # 3. View EV exposure slider via QSlider directly (-2.0 vs +1.5)
    from PySide6.QtWidgets import QSlider
    ev_slider = win.develop.view_ev.parentWidget().findChild(QSlider)
    ev_slider.setValue(int(round((-2.0 - (-4.0)) / 0.05)))
    _pump(app, 0.15)
    assert abs(win.develop.view_ev.value() - (-2.0)) < 1e-4
    assert abs(win.view.exposure - (-2.0)) < 1e-4
    frame_dark = _grab_rgb(win.view)
    ev_slider.setValue(int(round((1.5 - (-4.0)) / 0.05)))
    _pump(app, 0.15)
    assert abs(win.view.exposure - 1.5) < 1e-4
    frame_bright = _grab_rgb(win.view)
    assert frame_bright.mean() > frame_dark.mean() + 15.0
    ev_slider.setValue(int(round((0.0 - (-4.0)) / 0.05)))
    win.develop.shape.setCurrentText("cube")
    _pump(app, 0.15)

    # 4. Orbit camera (turntable: dragging right decreases yaw, dragging down increases pitch)
    before_orbit = _grab_rgb(win.view)
    yaw0, pitch0 = win.view.yaw, win.view.pitch
    press = QMouseEvent(
        QMouseEvent.Type.MouseButtonPress, QPointF(200, 200), QPointF(200, 200),
        Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier)
    move = QMouseEvent(
        QMouseEvent.Type.MouseMove, QPointF(290, 240), QPointF(290, 240),
        Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier)
    rel = QMouseEvent(
        QMouseEvent.Type.MouseButtonRelease, QPointF(290, 240), QPointF(290, 240),
        Qt.MouseButton.LeftButton, Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier)
    win.view.mousePressEvent(press)
    win.view.mouseMoveEvent(move)
    win.view.mouseReleaseEvent(rel)
    assert win.view.yaw < yaw0 and win.view.pitch > pitch0
    _pump(app, 0.15)
    after_orbit = _grab_rgb(win.view)
    assert np.abs(after_orbit.astype(np.float32) - before_orbit.astype(np.float32)).mean() > 1.0

    # 5. All filter modes & grade controls & output size change
    for flt in ("kuwahara_classic", "anisotropic", "median", "bilateral", "kuwahara"):
        win.develop.filt.setCurrentText(flt)
        _pump(app, 0.25)
        assert win._fin is not None

    exp_slider = win.develop._spins["grade_exposure"].parentWidget().findChild(QSlider)
    exp_slider.setValue(int(round((0.5 - (-2.0)) / 0.05)))
    win.develop._spins["contrast"].setValue(0.25)
    win.develop._spins["posterize"].setValue(6)
    win.develop._spins["ao_strength"].setValue(0.4)
    win.develop._spins["edge_strength"].setValue(0.3)
    win.develop.channel.setCurrentText("Lit")
    win.develop.compare.setCurrentText("After")
    win.develop.shape.setCurrentText("sphere")
    win.develop._spins["normal_strength"].setValue(0.0)
    _pump(app, 0.35)
    xy0 = float(np.abs(win._fin["normal"][..., :2]).mean())
    lit0 = _grab_rgb(win.view)
    win.develop._spins["normal_strength"].setValue(2.0)
    _pump(app, 0.35)
    xy2 = float(np.abs(win._fin["normal"][..., :2]).mean())
    lit2 = _grab_rgb(win.view)
    assert xy2 > xy0 + 0.02, "Normal strength did not change the finished normal"
    assert float(np.abs(lit2.astype(np.float32) - lit0.astype(np.float32)).mean()) > 1.5, (
        "Lit ignored the normal map; strength slider is a no-op in the viewport")
    win.develop._spins["normal_strength"].setValue(1.6)
    win.develop._spins["normal_soften"].setValue(0.3)
    win.develop._spins["roughness_flatten"].setValue(0.5)
    win.develop._spins["height_soften"].setValue(0.4)
    win.develop._spins["detail_restore"].setValue(0.25)
    win.develop._spins["seam_fade"].setValue(0.5)
    _pump(app, 0.3)
    assert abs(library.load_recipe(mid_a)["grade_exposure"] - 0.5) < 1e-4
    assert abs(library.load_recipe(mid_a)["normal_strength"] - 1.6) < 1e-4

    # Posterize must keep chroma (Diffuse view of the red cliff).
    assert "palette_strength" in win.develop._spins
    assert "palette_hardness" in win.develop._spins
    win.develop.channel.setCurrentText("Diffuse")
    _pump(app, 0.15)
    frame_post = _grab_rgb(win.view)
    chroma = frame_post.max(axis=-1).astype(np.float32) - frame_post.min(axis=-1).astype(np.float32)
    assert float(chroma.mean()) > 12.0, "Posterize collapsed saturation to grey"
    win.develop.palette.set_colors([[0.90, 0.22, 0.12], [0.40, 0.16, 0.10]])
    win.develop._spins["palette_strength"].setValue(0.55)
    win.develop._spins["palette_hardness"].setValue(10.0)
    _pump(app, 0.25)
    rec_pal = win.develop.recipe()
    assert len(rec_pal["palette"]) == 2
    assert abs(rec_pal["palette_strength"] - 0.55) < 1e-4
    assert abs(library.load_recipe(mid_a)["palette_strength"] - 0.55) < 1e-4
    win.develop.channel.setCurrentText("Lit")
    _pump(app, 0.1)

    # Test output size change reloads _src at new resolution
    idx_512 = win.develop.size.findData(512)
    win.develop.size.setCurrentIndex(idx_512)
    _pump(app, 0.4)
    assert win._src["W"] == 512 and win._src["H"] == 512
    assert win._mat(mid_a)["tex_w"] == 512

    # Grade-only preview must reuse GL textures (write, not recreate) including alb0.
    creates_at_512 = win.view.tex_creates
    writes_at_512 = win.view.tex_writes
    alb0 = win.view._tex["alb0"]
    n0 = win.view._tex["n0"]
    pack0 = win.view._tex["pack0"]
    win.develop._spins["saturation"].setValue(1.25)
    _pump(app, 0.3)
    assert win.view.tex_creates == creates_at_512, "preview leaked a new GL texture instead of tex.write"
    assert win.view.tex_writes > writes_at_512
    assert win.view._tex["alb0"] is alb0
    assert win.view._tex["n0"] is n0
    assert win.view._tex["pack0"] is pack0

    # 6. Test Undo (Ctrl+Z) walking backward monotonically
    win._spins_before_undo = win.develop._spins["contrast"].value()
    win.develop._spins["contrast"].setValue(-0.4)
    _pump(app, 0.2)
    win.develop._spins["contrast"].setValue(0.8)
    _pump(app, 0.2)
    assert abs(win.develop.recipe()["contrast"] - 0.8) < 1e-4
    win._undo()
    _pump(app, 0.2)
    assert abs(win.develop.recipe()["contrast"] - (-0.4)) < 1e-4
    win._undo()
    _pump(app, 0.2)
    assert abs(win.develop.recipe()["contrast"] - win._spins_before_undo) < 1e-4

    # 6b. Reset to default and before/after compare
    win.develop.reset.click()
    _pump(app, 0.4)
    rec_reset = win.develop.recipe()
    assert abs(rec_reset["radius"]) < 1e-4
    assert abs(rec_reset["grade_exposure"]) < 1e-4
    assert abs(rec_reset["normal_strength"] - 1.0) < 1e-4
    assert abs(library.load_recipe(mid_a)["radius"]) < 1e-4
    frame_after = _grab_rgb(win.view)
    win.develop.compare.setCurrentText("Before")
    _pump(app, 0.15)
    assert win.view.compare == "Before"
    frame_before = _grab_rgb(win.view)
    win.develop.compare.setCurrentText("Split")
    _pump(app, 0.15)
    assert win.view.compare == "Split"
    frame_split = _grab_rgb(win.view)
    assert frame_split.std() > 3.0
    win.develop.compare.setCurrentText("After")
    _pump(app, 0.1)
    assert np.abs(frame_before.astype(np.float32) - frame_after.astype(np.float32)).mean() >= 0.0

    # 7. Select second material with a 3D mesh (Moss_Forest_Ground, 1152 tris, foliage cap 800)
    win.library.list.setCurrentRow(0)
    _pump(app, 0.6)
    assert win._mid == mid_b
    assert win._src is not None
    assert "Moss" in win._src["name"], f"mesh kept the previous material's maps: {win._src.get('name')}"
    assert win._mesh is not None
    assert win._mesh["tris"] == 1152
    assert win.view.shape == "mesh"
    assert "1152" in win.develop.poly.text()
    frame_mesh = _grab_rgb(win.view)
    assert frame_mesh.mean() > 15.0 and frame_mesh.std() > 10.0

    # Budget meter: fill is this mesh; ticks mark every class ceiling.
    meter = win.library.meter
    for key in ("hero", "prop", "foliage", "modular", "terrain", "surface"):
        assert key in meter.class_names()
    assert meter.tris == 1152
    tris_ticks = meter.tick_positions("tris")
    fill = meter.fill_pos("tris")
    assert tris_ticks["foliage"] < fill < tris_ticks["modular"]
    tex_ticks = meter.tick_positions("tex")
    assert meter.tex > 0
    assert tex_ticks["hero"] >= tex_ticks["foliage"]
    assert win.library.item_cls.currentText() == "foliage"
    legend = meter.legend_text()
    assert "hero" in legend and "modular" in legend
    assert win.develop._spins["normal_strength"].toolTip()
    assert win.library.list.toolTip()
    tex_groups = {row["cap"]: row["names"] for row in meter.collapsed_ticks("tex")}
    assert len(tex_groups[1024]) >= 2
    assert "hero" in meter.fits_classes()

    # Test Decimate slider
    win.develop.decimate.setValue(0.5)
    _pump(app, 0.5)
    assert win._mesh["tris"] < 800

    # Test Fit to budget button
    win._mesh = dict(win._mesh_src)
    win._fit_budget()
    _pump(app, 0.5)
    assert win._mesh["tris"] <= budgets.ceiling("foliage")["tris"]

    # 8. Switch back to surface material (Red_Cliff_Rock) and verify stale mesh is cleared
    win.library.list.setCurrentRow(1)
    _pump(app, 0.4)
    assert win._mesh is None
    assert win.view.shape == "sphere"
    assert "—" in win.develop.poly.text()

    # 9. Test zip import, rename, class change, bulk apply, bulk fit, and export
    zip_src_dir = tmp_path / "zip_src" / "Cobble_Path"
    _make_surface_set(zip_src_dir, "Cobble_Path", (130, 130, 140), size=64)
    zip_file = tmp_path / "Cobble_Path.zip"
    with zipfile.ZipFile(zip_file, "w") as zf:
        for p in zip_src_dir.iterdir():
            zf.write(p, arcname=f"Cobble_Path/{p.name}")
    win._import(str(zip_file))
    _pump(app, 0.5)
    assert win.library.list.count() == 3

    # Rename current material
    win.library.name.setText("Red_Cliff_Renamed")
    win.library._rename()
    _pump(app, 0.1)
    assert win._mat(mid_a)["display_name"] == "Red_Cliff_Renamed"

    # Bulk apply & Bulk export & Single export
    export_dir = tmp_path / "exported"
    library.set_setting("export_root", str(export_dir))
    win.library.list.selectAll()
    win._bulk_apply()
    win._bulk_fit()
    _pump(app, 0.5)

    for packing_mode, packed_suffix in (("mrod", "MROD"), ("orm", "ORM"), ("ormd", "ORMD")):
        pi = win.develop.packing.findData(packing_mode)
        win.develop.packing.setCurrentIndex(pi)
        _pump(app, 0.15)
        win._export()
        _pump(app, 0.6)
        out_folder = export_dir / "Red_Cliff_Renamed"
        assert (out_folder / "Red_Cliff_Renamed_D.png").is_file()
        assert (out_folder / f"Red_Cliff_Renamed_{packed_suffix}.png").is_file()
        assert (out_folder / "Red_Cliff_Renamed_N.png").is_file()
        assert (out_folder / "Red_Cliff_Renamed_stylize.json").is_file()

    win._bulk_export()
    _pump(app, 1.0)
    assert (export_dir / "Moss_Forest_Ground" / "Moss_Forest_Ground_D.png").is_file()
    assert (export_dir / "Cobble_Path" / "Cobble_Path_D.png").is_file()

    win.close()
    app.processEvents()
