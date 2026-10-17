"""PySide6 Lightroom-style shell."""
from __future__ import annotations

import faulthandler
import json
import os
import sys
import traceback
from pathlib import Path

from PySide6.QtCore import QEventLoop, Qt, QTimer
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QMessageBox, QSplitter, QStatusBar,
)

from qs import budgets, library
from qs.app.develop_panel import DevelopPanel
from qs.app.jobs import submit
from qs.app.library_panel import LibraryPanel
from qs.app.viewport import Viewport
from qs.core.maps import detect_maps, find_sets
from qs.core.mesh import (
    find_mesh_file, find_original_mesh_file, is_reuv_path, load_mesh,
    texel_density, write_obj,
)
from qs.core.recipe import Recipe, SIZES
from qs.core.stylize import finish_stage, load_stage, paint_dispatch, process_set
from qs.logging import LOG_PATH, log_line, log_start, phase
from qs.version import TOOL_VERSION, WINDOW_TITLE

# Live preview never loads 2048² into the UI thread. Export still uses the recipe size.
PREVIEW_MAX = 1024


def _src_as_maps(src: dict) -> dict:
    return {
        "alb": src["alb"],
        "normal": src["normal"],
        "height": src["height"],
        "rough": src["rough"],
        "metal": src["metal"],
        "ao": src["ao"],
    }


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(WINDOW_TITLE)
        self.resize(1400, 860)
        phase("window")
        library.migrate_catalog_json()
        library.scan_inbox()
        self._mid = None
        self._src = None
        self._painted = None
        self._paint_key = None
        self._fin = None
        self._mesh = None
        self._mesh_src = None
        self._gen = 0
        self._preview_gen = 0

        split = QSplitter(Qt.Orientation.Horizontal)
        self.library = LibraryPanel()
        self.view = Viewport()
        self.develop = DevelopPanel()
        split.addWidget(self.library)
        split.addWidget(self.view)
        split.addWidget(self.develop)
        split.setStretchFactor(0, 0)
        split.setStretchFactor(1, 1)
        split.setStretchFactor(2, 0)
        split.setSizes([280, 820, 320])
        self.setCentralWidget(split)
        self.setStatusBar(QStatusBar())
        self.statusBar().showMessage(f"v{TOOL_VERSION}")

        self.library.material_chosen.connect(self._load_material)
        self.library.import_requested.connect(self._import)
        self.library.bulk_apply_requested.connect(self._bulk_apply)
        self.library.bulk_export_requested.connect(self._bulk_export)
        self.library.bulk_fit_requested.connect(self._bulk_fit)
        self.develop.recipe_changed.connect(self._on_recipe)
        self.develop.export_clicked.connect(self._export)
        self.develop.fit_budget_clicked.connect(self._fit_budget)
        self.develop.decimate_changed.connect(self._on_decimate)
        self.develop.shape_changed.connect(self._on_shape)
        self.develop.channel_changed.connect(self._on_channel)
        self.develop.compare_changed.connect(self._on_compare)
        self.develop.view_ev_changed.connect(self._on_view_ev)
        self.develop.displace_changed.connect(self._on_displace)
        self.develop.rewrap_bake_clicked.connect(self._rewrap_bake)
        self.develop.reset_clicked.connect(self._reset_recipe)
        self.library.delete_requested.connect(self._delete_material)

        undo = QAction("Undo", self)
        undo.setShortcut(QKeySequence.StandardKey.Undo)
        undo.triggered.connect(self._undo)
        self.addAction(undo)

        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(40)
        self._debounce.timeout.connect(self._run_preview)
        self._dec_debounce = QTimer(self)
        self._dec_debounce.setSingleShot(True)
        self._dec_debounce.setInterval(80)
        self._dec_debounce.timeout.connect(self._run_decimate)
        self._pending_ratio = 1.0
        QTimer.singleShot(0, self._select_initial_material)
        phase("ui-ready")

    def _select_initial_material(self):
        if self.library.list.count() > 0 and self.library.list.currentRow() < 0:
            self.library.list.setCurrentRow(0)

    def _mat(self, mid=None):
        mid = mid or self._mid
        mats = {m["id"]: m for m in library.list_materials()}
        return mats.get(mid)

    def _original_mesh(self, mat=None, current=None):
        """The untouched hi-poly mesh: pre-decimation, pre-rewrap, original UVs."""
        mat = mat or self._mat()
        if not mat:
            return None
        current = current or (self._mesh or {}).get("path") or mat.get("mesh_path")
        baked = mat.get("baked_mesh_path")
        if baked and current and os.path.normcase(str(current)) == os.path.normcase(str(baked)):
            stored = mat.get("original_mesh_path")
            if stored and Path(stored).is_file():
                return stored
            return find_original_mesh_file(mat["source_folder"], None)
        stored = mat.get("original_mesh_path")
        if stored and not is_reuv_path(stored) and Path(stored).is_file():
            if is_reuv_path(current) or (baked and current and
                                         os.path.normcase(str(current)) == os.path.normcase(str(baked))):
                return stored
            if current and Path(current).is_file():
                # The stored original is authoritative once it is known; a
                # decimated mesh dict still points at this same source file.
                return stored
            return stored
        return find_original_mesh_file(mat["source_folder"], current)

    @staticmethod
    def _maps_folder(mat) -> str:
        """Baked maps win when present; Reset clears them back to the source folder."""
        baked = (mat or {}).get("baked_maps_folder")
        if baked and Path(baked).is_dir():
            return str(baked)
        return (mat or {}).get("source_folder") or ""

    @staticmethod
    def _effective_mesh_path(mat) -> str | None:
        baked = (mat or {}).get("baked_mesh_path")
        if baked and Path(baked).is_file():
            return str(baked)
        return (mat or {}).get("mesh_path") or find_mesh_file((mat or {})["source_folder"])

    def _load_material(self, mid: str):
        gen = self._bump()
        self._mid = mid
        self._src = None
        self._painted = None
        self._paint_key = None
        self._fin = None
        self._mesh = None
        self._mesh_src = None
        self.view.set_maps(None)
        self.view.set_mesh(None)
        if self.develop.shape.currentText() == "mesh":
            self.develop.shape.blockSignals(True)
            self.develop.shape.setCurrentText(self.view.shape)
            self.develop.shape.blockSignals(False)
        self.develop.set_poly(None, None)
        mat = self._mat(mid)
        if not mat:
            return
        if not mat["source_ok"]:
            self.statusBar().showMessage("Source folder is missing")
            return
        rec = library.load_recipe(mid)
        self.develop.set_recipe(rec)
        self.develop.decimate.blockSignals(True)
        self.develop.decimate.setValue(1.0)
        self.develop.decimate.blockSignals(False)
        folder = self._maps_folder(mat)
        self.statusBar().showMessage(f"Loading {mat['display_name']}…")
        mesh_path = self._effective_mesh_path(mat)
        submit(self._load_src, folder, self._preview_rec(rec),
               on_ok=lambda r: self._src_ready(gen, mid, r),
               on_err=self._job_err)
        if mesh_path:
            self._meshlab(load_mesh, mesh_path, on_ok=lambda m: self._mesh_ready(m, gen=gen, mid=mid))
        else:
            self._maybe_convert(folder, mid, gen=gen)

    def _load_src(self, folder, rec):
        s = detect_maps(folder)
        if s is None:
            found = find_sets([folder], True)
            if not found:
                raise FileNotFoundError(f"No material maps in {folder}")
            s = found[0]
        src = load_stage(s, rec)
        src["_set"] = s
        return src

    def _src_ready(self, gen, mid, src):
        if gen != self._gen or mid != self._mid:
            return
        src["_mid"] = mid
        self._src = src
        self._painted = None
        self._paint_key = None
        library.update_mesh_stats(mid, tex_w=src["W"], tex_h=src["H"])
        self.library.refresh_current()
        try:
            self.view.set_maps(_src_as_maps(src), src)
        except Exception as exc:
            log_line(f"source preview failed: {exc}")
        self._queue_preview()

    @staticmethod
    def _preview_size(rec: dict) -> int:
        return min(int(rec.get("size") or 1024), PREVIEW_MAX)

    def _preview_rec(self, rec: dict) -> dict:
        rec = dict(rec)
        rec["size"] = self._preview_size(rec)
        rec["passes"] = 1
        return rec

    def _on_recipe(self, rec):
        if not self._mid:
            return
        library.save_recipe(self._mid, rec)
        want = self._preview_size(rec)
        if self._src is not None and want != max(self._src["W"], self._src["H"]):
            mat = self._mat(self._mid)
            if mat and mat["source_ok"]:
                mid = self._mid
                gen = self._bump()
                submit(self._load_src, self._maps_folder(mat), self._preview_rec(rec),
                       on_ok=lambda r: self._src_ready(gen, mid, r),
                       on_err=self._job_err)
                return
        self._queue_preview()

    def _queue_preview(self):
        self._debounce.start()

    @staticmethod
    def _make_paint_key(src, rec):
        return (
            id(src),
            rec.get("filter"),
            round(float(rec.get("radius") or 0.0), 4),
            int(rec.get("passes") or 1),
            round(float(rec.get("sharpness") or 8.0), 4),
            round(float(rec.get("stroke") or 0.0), 4),
            bool(rec.get("tile", True)),
            rec.get("island_mask"),
            id(rec.get("_island_ids")),
        )

    def _run_preview(self):
        if self._src is None or self._mid is None:
            return
        rec = self.develop.recipe()
        src = self._src
        mid = self._mid
        pgen = self._bump_preview()
        pkey = self._make_paint_key(src, rec)
        if self._painted is not None and self._paint_key == pkey:
            try:
                fin = finish_stage(src, self._painted, rec)
                self._preview_ready(pgen, (self._painted, fin, rec), src=src, mid=mid)
            except Exception as exc:
                self._job_err(str(exc))
            return
        from qs.core import gpu as qs_gpu
        if qs_gpu.attached() and self.view._gl_ok and self.view.context() is not None:
            qs_gpu.set_yield(self._gpu_yield)
            self.view.makeCurrent()
            try:
                painted = paint_dispatch(src, rec)
            except Exception as exc:
                self._job_err(str(exc))
                return
            finally:
                qs_gpu.set_yield(None)
                self.view.doneCurrent()
            try:
                self._paint_key = pkey
                fin = finish_stage(src, painted, rec)
                self._preview_ready(pgen, (painted, fin, rec), src=src, mid=mid)
            except Exception as exc:
                self._job_err(str(exc))
            return
        submit(self._stylize, src, rec,
               on_ok=lambda r: self._preview_ready(pgen, r, pkey, src=src, mid=mid),
               on_err=self._job_err)

    def _gpu_yield(self):
        """Keep the window alive during compute without leaving the viewport context."""
        QApplication.processEvents(
            QEventLoop.ProcessEventsFlag.ExcludeUserInputEvents, 8)
        if self.view.context() is not None:
            self.view.makeCurrent()

    def _stylize(self, src, rec):
        painted = paint_dispatch(src, rec)
        fin = finish_stage(src, painted, rec)
        return painted, fin, rec

    def _preview_ready(self, pgen, result, pkey=None, src=None, mid=None):
        if pgen != self._preview_gen:
            return
        if mid is not None and mid != self._mid:
            return
        if src is not None and src is not self._src:
            return
        if self._src is None:
            return
        self._painted, self._fin, rec = result
        if pkey is not None:
            self._paint_key = pkey
        self.view.set_maps(self._fin, self._src)
        w, h = self._src["W"], self._src["H"]
        export = int(rec.get("size") or w)
        extra = f"  export {export}" if export > w else ""
        self.statusBar().showMessage(
            f"{self._src['name']}  {w}×{h}{extra}  radius {rec.get('radius')}")

    def _export_one(self, mid: str, rec: dict):
        root = library.get_setting("export_root")
        mat = self._mat(mid)
        if not mat:
            raise FileNotFoundError(mid)
        err = library.name_error(mat["display_name"])
        if err:
            raise ValueError(err)
        if not root:
            raise ValueError("Set an export folder first.")
        dest = os.path.join(root, mat["display_name"])
        maps_folder = self._maps_folder(mat)
        s = detect_maps(maps_folder)
        if s is None:
            found = find_sets([maps_folder], True)
            if not found:
                raise FileNotFoundError(maps_folder)
            s = found[0]
        s = dict(s, name=mat["display_name"])
        cap = budgets.ceiling(mat["asset_class"])
        size = max(s for s in SIZES if s <= cap["tex"]) if cap["tex"] else rec.get("size")
        rec = dict(rec, size=size)
        return process_set(s, rec, True, dest)

    def _export(self):
        if not self._mid:
            return
        rec = self.develop.recipe()
        mat = self._mat()
        err = library.name_error(mat["display_name"]) if mat else "Nothing selected"
        if err:
            QMessageBox.warning(self, "Export", err)
            return
        if not library.get_setting("export_root"):
            QMessageBox.warning(self, "Export", "Set an export folder first.")
            return
        submit(self._export_one, self._mid, rec,
               on_ok=lambda r: self.statusBar().showMessage(f"Exported {mat['display_name']}"),
               on_err=self._job_err)

    def _import(self, path: str):
        def work():
            if path.lower().endswith(".zip"):
                return library.import_zip(path)
            return library.import_folder(path)
        def done(ids):
            self.library.refresh()
            for mid in ids or []:
                mat = self._mat(mid)
                if mat:
                    self._maybe_convert(mat["source_folder"], mid)
        submit(work, on_ok=done, on_err=self._job_err)

    def _maybe_convert(self, folder: str, mid: str | None = None, gen: int | None = None):
        from qs.core.mesh import MESH_SUFFIXES
        existing = find_mesh_file(folder)
        if existing and Path(existing).suffix.lower() in (".gltf", ".glb"):
            self._meshlab(load_mesh, existing, on_ok=lambda m: self._mesh_ready(m, gen=gen, mid=mid))
            return
        fbx = None
        for suf in MESH_SUFFIXES:
            if suf in (".fbx", ".obj"):
                hits = list(Path(folder).rglob(f"*{suf}"))
                if hits:
                    fbx = str(hits[0])
                    break
        if not fbx:
            return
        dest_dir = Path(folder) / "mesh"
        dest = str(dest_dir / (Path(fbx).stem + ".glb"))

        def work():
            from qs.blender import convert_to_gltf
            return convert_to_gltf(fbx, dest)

        def ok(path):
            if mid:
                library.update_mesh_stats(mid, mesh_path=path)
            self._meshlab(load_mesh, path, on_ok=lambda m: self._mesh_ready(m, gen=gen, mid=mid))

        self.statusBar().showMessage("Converting mesh in Blender…")
        submit(work, on_ok=ok, on_err=self._job_err)

    def _on_decimate(self, ratio: float):
        self._pending_ratio = ratio
        self._dec_debounce.start()

    def _run_decimate(self):
        if self._mesh_src is None:
            self.statusBar().showMessage("No mesh on this material")
            return
        ratio = self._pending_ratio
        src = self._mesh_src
        mid = self._mid
        self.statusBar().showMessage(f"Decimating to {ratio:.0%}…")

        from qs.core.mesh import decimate
        gen = self._gen
        self._meshlab(decimate, src, ratio, on_ok=lambda m: self._mesh_ready(m, gen=gen, mid=mid))

    def _fit_budget(self):
        from qs.core.mesh import fit_to_budget
        if not self._mid:
            return
        mat = self._mat()
        if not mat:
            return
        mid = self._mid
        cap = budgets.ceiling(mat["asset_class"])
        rec = self.develop.recipe()
        size = max((s for s in SIZES if s <= cap["tex"]), default=rec["size"])
        rec = dict(rec, size=size)
        library.save_recipe(self._mid, rec)
        self.develop.set_recipe(rec)
        gen = self._gen
        self._meshlab(fit_to_budget, mat, cap, on_ok=lambda m: self._mesh_ready(m, gen=gen, mid=mid))

    def _mesh_ready(self, mesh, gen: int | None = None, mid: str | None = None):
        if gen is not None and gen != self._gen:
            return
        if mid is not None and mid != self._mid:
            return
        if not mesh:
            self.statusBar().showMessage("No mesh on this material")
            return
        if mesh.get("source_path") and (self._mesh_src is None or mesh.get("ratio") in (None, 1.0)):
            self._mesh_src = dict(mesh)
        self._mesh = mesh
        tex = (self._src or {}).get("W") or 1024
        dens = mesh.get("texel_density")
        if dens is None and mesh.get("uvs") is not None:
            dens = texel_density(mesh["uvs"], mesh["positions"], mesh["indices"], tex)
            mesh["texel_density"] = dens
        try:
            self.view.set_mesh(mesh)
        except Exception as exc:
            log_line(f"mesh upload failed: {exc}")
            self.statusBar().showMessage(f"Mesh upload failed: {exc}")
        self.develop.shape.blockSignals(True)
        self.develop.shape.setCurrentText("mesh")
        self.develop.shape.blockSignals(False)
        self.develop.set_poly(mesh["tris"], mesh["verts"], mesh.get("cap_tris"),
                              mesh.get("islands"), dens)
        if self._mid:
            mat = self._mat()
            original = self._original_mesh(mat, mesh.get("path"))
            path = mesh.get("path")
            baked = (mat or {}).get("baked_mesh_path")
            is_baked = bool(baked and path and
                            os.path.normcase(str(path)) == os.path.normcase(str(baked)))
            if path and not is_reuv_path(path) and not is_baked:
                # A plain source mesh is its own original. A baked low mesh
                # or an old _reuv file must never overwrite that record.
                original = path
            library.update_mesh_stats(
                self._mid, tris=mesh["tris"], verts=mesh["verts"],
                islands=mesh.get("islands"), mesh_path=path,
                texel_density=dens, original_mesh_path=original)
        self.library.refresh_current()
        if mesh.get("ratio") not in (None, 1.0):
            self.statusBar().showMessage(
                f"Mesh {mesh['tris']} tris ({mesh.get('ratio', 1):.0%} of source)")
        elif self._src is not None:
            self._queue_preview()

    def _attach_islands(self, mesh: dict):
        """Raster UV islands at the live preview size so Kuwahara can match the maps."""
        src = self._src
        if mesh is None:
            return
        if src is None:
            mesh.pop("island_ids", None)
            return
        try:
            from qs.core.mesh import rasterize_islands
            mesh["island_ids"] = rasterize_islands(
                mesh["uvs"], mesh["indices"], int(src["W"]), int(src["H"]))
        except Exception as exc:
            log_line(f"island raster failed: {exc}")
            mesh.pop("island_ids", None)

    def _on_shape(self, name: str):
        self.view.shape = name
        self.view.update()

    def _on_channel(self, name: str):
        self.view.channel = name
        self.view.update()

    def _on_compare(self, name: str):
        self.view.compare = name
        self.view.update()

    def _reset_recipe(self):
        """Full factory state: default recipe, decimate 1.0, original mesh and maps."""
        if not self._mid:
            return
        mid = self._mid
        mat = self._mat(mid)
        if not mat:
            return
        original = self._original_mesh(mat, (self._mesh or {}).get("path"))
        rec = Recipe().to_dict()
        library.save_recipe(mid, rec, record_undo=False)
        library.clear_baked_state(mid)
        if original and Path(original).is_file():
            library.update_mesh_stats(mid, mesh_path=original, original_mesh_path=original)
        self.develop.set_recipe(rec)
        self.develop.decimate.blockSignals(True)
        self.develop.decimate.setValue(1.0)
        self.develop.decimate.blockSignals(False)
        name = mat["display_name"]
        # Reloading the material now picks up the original maps and mesh,
        # because the baked state that pointed elsewhere has been cleared.
        self._load_material(mid)
        self.statusBar().showMessage(
            f"Reset {name} to default: original mesh, original maps, decimate 1.0")

    def _delete_material(self, mid: str):
        """Remove one library entry and land cleanly if it was on screen."""
        if not mid:
            return
        mat = self._mat(mid)
        name = mat["display_name"] if mat else mid
        was_current = mid == self._mid
        library.delete_material(mid)
        if was_current:
            self._bump()
            self._mid = None
            self._src = None
            self._painted = None
            self._paint_key = None
            self._fin = None
            self._mesh = None
            self._mesh_src = None
            self.view.set_maps(None)
            self.view.set_mesh(None)
            self.develop.set_poly(None, None)
            self.develop.decimate.blockSignals(True)
            self.develop.decimate.setValue(1.0)
            self.develop.decimate.blockSignals(False)
            self.library.name.clear()
        self.library.refresh()
        self.statusBar().showMessage(f"Removed {name} from the library")
        if was_current and self.library.list.count() > 0:
            self.library.list.setCurrentRow(0)

    def _on_view_ev(self, value: float):
        self.view.exposure = float(value)
        self.view.update()

    def _on_displace(self, on: bool):
        self.view.displace = bool(on)
        self.view.update()

    def _rewrap_bake(self):
        """One button: unwrap the current decimated mesh and bake hi-poly onto it.

        The low mesh exists only in memory after PyMeshLab decimation (its
        dict path still points at the source file), so it is serialized to
        an OBJ that Blender can import. Blender smart-projects that mesh,
        transfers the hi-poly maps by EMIT, bakes true tangent-space
        geometry normals, and exports the unwrapped low mesh. Only then is
        anything shown: the low mesh together with maps made for its UVs,
        on this same material. No new library entry is created.
        """
        mat = self._mat()
        if not mat or not self._mesh:
            self.statusBar().showMessage("Load a mesh first")
            return
        current = self._mesh
        high = self._original_mesh(mat, current.get("path"))
        if not high or not Path(high).is_file():
            self.statusBar().showMessage("No original hi-poly mesh to bake from")
            return
        # Source maps always come from the untouched original folder, even
        # when this material is already showing an earlier bake.
        srcset = detect_maps(mat["source_folder"])
        if srcset is None:
            found = find_sets([mat["source_folder"]], True)
            srcset = found[0] if found else None
        maps = dict((srcset or {}).get("maps") or {})
        if not maps:
            self.statusBar().showMessage("No source maps to bake from")
            return
        mid, gen = self._mid, self._gen
        dest_dir = library.baked_dir(mid)
        dest_mesh = str(dest_dir / "low.glb")
        low_obj = str(dest_dir / "low_input.obj")
        size = max((self._src or {}).get("W") or 1024, (self._src or {}).get("H") or 1024)
        size = min(int(size), 2048)
        low_snapshot = dict(current)
        low_tris = int(current.get("tris") or 0)

        def work():
            from qs.blender import rewrap_bake
            write_obj(low_snapshot, low_obj)
            return rewrap_bake(high, low_obj, str(dest_dir), dest_mesh,
                               maps=maps, size=size,
                               base_name=srcset.get("name") or "Baked")

        def done(result):
            if gen != self._gen or mid != self._mid:
                return
            files = (result or {}).get("files") or {}
            mesh_out = (result or {}).get("mesh") or dest_mesh
            if not files or not Path(mesh_out).is_file():
                self.statusBar().showMessage("Rewrap bake finished but wrote no maps")
                return
            library.set_baked_state(mid, mesh_out, str(dest_dir))
            library.update_mesh_stats(mid, original_mesh_path=high)
            self.library.refresh_current()
            # Display only the finished pair: unwrapped low mesh + baked maps.
            # The mesh is applied in the same main-thread turn as the baked
            # maps, so it is never painted wearing the old maps on new UVs.
            rec = self.develop.recipe()

            def apply_pair(src_result):
                self._src_ready(gen, mid, src_result)
                self._meshlab(load_mesh, mesh_out,
                              on_ok=lambda m: self._mesh_ready(m, gen=gen, mid=mid))

            submit(self._load_src, str(dest_dir), self._preview_rec(rec),
                   on_ok=apply_pair, on_err=self._job_err)
            self.statusBar().showMessage(
                f"Rewrapped {low_tris} tris and baked {len(files)} map(s) from the hi-poly mesh")

        self.statusBar().showMessage("Rewrapping and baking from the hi-poly mesh in Blender\u2026")
        submit(work, on_ok=done, on_err=self._job_err)

    def _bulk_apply(self):
        if not self._mid:
            return
        rec = self.develop.recipe()
        for mid in self.library.selected_ids():
            library.save_recipe(mid, rec)
        self.statusBar().showMessage(f"Applied recipe to {len(self.library.selected_ids())} materials")

    def _bulk_export(self):
        ids = self.library.selected_ids()
        rec = self.develop.recipe()
        if not ids:
            return
        if not library.get_setting("export_root"):
            QMessageBox.warning(self, "Export", "Set an export folder first.")
            return

        def work():
            n = 0
            for mid in ids:
                self._export_one(mid, rec)
                n += 1
            return n

        submit(work, on_ok=lambda n: self.statusBar().showMessage(f"Exported {n} materials"),
               on_err=self._job_err)

    def _bulk_fit(self):
        from qs.core.mesh import fit_to_budget
        ids = self.library.selected_ids()

        def work():
            n = 0
            for mid in ids:
                mat = self._mat(mid)
                if not mat:
                    continue
                cap = budgets.ceiling(mat["asset_class"])
                mesh = fit_to_budget(mat, cap)
                if mesh:
                    library.update_mesh_stats(
                        mid, tris=mesh["tris"], verts=mesh["verts"],
                        islands=mesh.get("islands"), texel_density=mesh.get("texel_density"))
                    n += 1
            return n

        def done(n):
            self.library.refresh()
            self.statusBar().showMessage(f"Fit {n} meshes to budget")

        submit(work, on_ok=done, on_err=self._job_err)

    def _undo(self):
        if not self._mid:
            return
        folder = library.UNDO / self._mid
        snaps = sorted(folder.glob("*.json")) if folder.is_dir() else []
        if len(snaps) < 2:
            return
        snaps[-1].unlink(missing_ok=True)
        rec = Recipe.from_dict(json.loads(snaps[-2].read_text(encoding="utf-8"))).to_dict()
        library.save_recipe(self._mid, rec, record_undo=False)
        self.develop.set_recipe(rec)
        self._queue_preview()

    def _meshlab(self, fn, *args, on_ok=None):
        """PyMeshLab native code is not safe on the Qt worker pool (heap 0xc0000374)."""
        try:
            result = fn(*args)
        except Exception as exc:
            self._job_err(str(exc))
            return
        if on_ok is not None:
            on_ok(result)

    def _job_err(self, msg):
        log_line(f"job failed: {msg}")
        self.statusBar().showMessage(str(msg).split("\n")[0][:200])

    def _bump(self):
        """Cancel in-flight source loads and previews (material switch / close)."""
        self._gen += 1
        self._preview_gen += 1
        self._debounce.stop()
        self._dec_debounce.stop()
        return self._gen

    def _bump_preview(self):
        """Cancel a stale stylize without dropping the maps still loading."""
        self._preview_gen += 1
        return self._preview_gen

    def closeEvent(self, event):
        self._debounce.stop()
        self._dec_debounce.stop()
        self._bump()
        try:
            self.view.shutdown_gl()
        except Exception:
            pass
        super().closeEvent(event)


def _missing_packages() -> list[str]:
    needed = [
        ("numpy", "numpy"),
        ("cv2", "opencv-python"),
        ("PIL", "pillow"),
        ("moderngl", "moderngl"),
        ("PySide6", "PySide6"),
    ]
    missing = []
    for mod, pip_name in needed:
        try:
            __import__(mod)
        except ImportError:
            missing.append(pip_name)
    return missing


def run_app(argv=None) -> int:
    try:
        log_dir = os.path.dirname(LOG_PATH) or "."
        os.makedirs(log_dir, exist_ok=True)
        faulthandler.enable(open(LOG_PATH, "a", encoding="utf-8"))
    except Exception:
        faulthandler.enable()
    log_start("gui")
    missing = _missing_packages()
    if missing:
        msg = (
            "Quixel Stylizer is missing packages:\n\n  "
            + ", ".join(missing)
            + "\n\nRun bootstrap.ps1 next to this program, or:\n"
            "  python -m pip install -r requirements.txt"
        )
        try:
            app = QApplication.instance() or QApplication(argv or sys.argv)
            QMessageBox.critical(None, "Quixel Stylizer", msg)
        except Exception:
            if sys.platform == "win32":
                import ctypes
                ctypes.windll.user32.MessageBoxW(0, msg, "Quixel Stylizer", 0x10)
            else:
                print(msg, file=sys.stderr)
        return 1
    phase("qapp")
    from PySide6.QtGui import QSurfaceFormat
    from qs.app.viewport import default_gl_format
    QSurfaceFormat.setDefaultFormat(default_gl_format())
    app = QApplication.instance() or QApplication(argv or sys.argv)
    app.setStyle("Fusion")
    app.setApplicationName("Quixel Stylizer")
    try:
        win = MainWindow()
        win.show()
        phase("shown")
        return app.exec()
    except Exception:
        log_line(traceback.format_exc())
        QMessageBox.critical(None, "Quixel Stylizer",
                             "The window failed to start.\nSee stylizer.log next to the program.")
        return 1
