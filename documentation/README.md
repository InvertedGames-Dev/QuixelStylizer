# QuixelStylizer system

v2.0 rebuild: a headless `qs.core` package, a PySide6 shell, PyMeshLab decimate, and a warm Blender process. The v1.5 Tk shell is deleted.

## Layout

| Path | Role |
|---|---|
| `qs/core/` | Maps, Kuwahara, Oklab grade, packing, Exodus look, mesh counts |
| `qs/app/` | PySide6 library (BudgetMeter class ticks) / develop (palette swatches) / `QOpenGLWidget` viewport / job queue |
| `qs/library.py` | SQLite catalog (WAL mode), recipes, undo |
| `qs/budgets.py` | PBR 360 class ceilings |
| `qs/blender.py` | Client for the warm Blender process |
| `qs_blender_server.py` | `blender -b --python` JSON-lines server |
| `run.py` | Entry next to `QuixelStylizer.exe` |
| `tools/groundtruth/` | Editor capture + PNG diff |
| `tools/unreal/import_stylized.py` | Editor import helper |

## Threading and OpenGL context invariants

- One ModernGL context, owned by the `Viewport` (`QOpenGLWidget`) on the Qt main thread.
- `QOpenGLWidget` renders into an internal offscreen framebuffer (`self.defaultFramebufferObject()`), not FBO `0`. `_paint_gl` binds it with `self.ctx.detect_framebuffer(int(self.defaultFramebufferObject())).use()`.
- Any ModernGL call outside `initializeGL`/`paintGL` (such as `set_maps`, `set_mesh`, or `qs_gpu.paint` in `_run_preview`) must wrap the call in `self.view.makeCurrent()` / `self.view.doneCurrent()`, and must never pump `QApplication.processEvents()` while a compute dispatch or fence wait is active, because Qt window compositing switches the thread's current `QOpenGLContext`.
- Background worker jobs (`FnJob` in `qs/app/jobs.py`) run with `setAutoDelete(False)` and are retained in `_ACTIVE_JOBS` until queued `finished`/`failed` slots complete on the main thread.
- Viewport `_upload` reuses existing GL textures with `Texture.write` when size matches, including before-compare `alb0`/`n0`/`pack0`. GPU Kuwahara SSBOs are reused per size on the attached context; host pack scratch is reused so a 2048 pass does not allocate a new float4 plane each bind.

## Look

ShaderShift Burley (`CUSTOM_DIFFUSE_MODE=2`) + dual-lobe GGX from `DefaultShaderShift.ini` (primary weight 0.7, primary roughness scale 0.6, diffuse scale 0.96). Tonemapper is AgX Punchy with linear-to-sRGB encoding. Lit uses vertex UV tangents (not screen-space dFdx) so the normal map, strength, and soften show up under 4x MSAA. Height is hardware tessellation plus displacement along the normal (mid-grey rest pose), with a View checkbox to turn it off. There is no parallax / POM. Studio hemisphere + key/fill/rim lighting. SMAA 1x PRESET_HIGH runs after tonemap on `qs render`.

## Develop colour and library budget

Posterize steps Oklab lightness only and keeps chroma. Palette strength / hardness and an add/remove swatch strip live in Grade and write `Recipe.palette`. The library Budget meter fills with the selected asset's tris and tex, draws a tick per distinct class ceiling, and lists full class names in a wrapping legend (shared ceilings collapsed, e.g. prop / modular / terrain / surface at 1024 px). The class dropdown stays.

Viewport map upload reuses GL textures with `tex.write` when size matches, including the before-compare `alb0` / `n0` / `pack0` keys. GPU Kuwahara SSBOs are sized-reused; compute fences are released after wait.

## Limits

**Rewrap & bake from hi-poly** is the one button for fixing a mesh that has been decimated far enough for its textures to stretch. It takes the mesh at its current decimated triangle count (decimation is in-memory, so the current geometry is serialized to an OBJ for Blender), smart-projects fresh UVs onto that low mesh in Blender, then bakes from the untouched hi-poly original: colour, roughness, metallic, AO and height are transferred by EMIT selected-to-active, and the normal map is a true tangent-space NORMAL bake of the hi-poly geometry onto the low mesh. Overlaying the two meshes inside Blender needs care: the glTF and OBJ importers disagree about world orientation by a quarter turn about X, so a bounding-box alignment can centre two meshes that are not actually overlaid, and the bake then samples a rotated copy (this shipped once: the exported mesh came back rotated with a fifth of its texels black). The server therefore tries the quarter turns, measures the true surface gap to the hi-poly for each, and keeps the orientation that sits on it, then bakes through an inflated cage copy of the low mesh so decimation chords still land rays. The exported GLB is put back under the exact transform the OBJ arrived with, because PyMeshLab reads raw vertex data and ignores node transforms; `tests/test_rewrap_blender.py` runs the real decimate → bake → reload path on the ram skull and asserts the frame and texel coverage. The unwrapped low mesh and its baked maps are shown together only when the bake finishes — the low mesh is never left wearing the old maps on new UVs. The result is stored against the same library entry (``baked_mesh_path`` / ``baked_maps_folder``, in ``library/baked/<id>/``); no new ``<name> baked`` entry is created, and the original source maps and mesh are never overwritten. **Reset to default** is the full factory reset for a material: default recipe, decimate slider back to 1.0, the original pre-decimation mesh, and the original maps, with any baked state cleared. Older rows still recover their original mesh from ``original_mesh_path`` or by stripping a ``_reuv`` suffix. The library panel's **Remove from library** button (also on the list's right-click menu, with a confirmation) deletes the catalog row, its recipe, and its undo snapshots; the source texture and mesh files on disk are not deleted, and removing the entry on screen loads the next entry or leaves a clean empty view. Live viewport lighting is a studio hemisphere + directional key/fill/rim with the Exodus BRDF, not UE reflection captures. PyMeshLab (`pymeshlab==2025.7.post1`) is used for in-process quadric decimation. Missing face UVs are filled (wedge → vertex → wedge) so the texture-preserving collapse can run; a throw still never reuses that MeshSet. Live preview paints on the GPU without per-island work, uses one Kuwahara pass, and only rebuilds when a slider is released so dragging does not freeze the window. Asset meshes are not tessellated (UV-split verts plus height would tear them open); plane/cube/sphere still tessellate. Decimate welds coincident positions into one surface before quadric collapse, then splits face corners back out by wedge UV; coincident points also share one displacement height so UV seams cannot open.
