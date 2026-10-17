# QuixelStylizer 2

Headless stylize core plus a PySide6 Lightroom-style shell. Textures get a painted Kuwahara look; meshes can be decimated in-process; the viewport shades with the Exodus look (ShaderShift Burley + dual-lobe GGX, AgX Punchy, SMAA 1x on `qs render` and the CPU blit).

Old Tk files (`stylizer_gui.py`, `quixel_stylize.py`, and the rest of the v1.5 shell) are gone. Kernels live in `qs/core`.

## Setup

Python 3.12. From this folder:

```
powershell -File bootstrap.ps1
```

That creates `.venv` and installs `requirements.txt`. Then:

| How | What happens |
|---|---|
| `QuixelStylizer.bat` | GUI (`python run.py --gui`) |
| `python run.py --gui` / `python -m qs` | same window |
| `python run.py stylize <folder>` | one material, no window |
| `python run.py batch <folders> --out <dir>` | many materials |
| `python run.py render --out sphere.png` | Exodus-look PNG for the calibration rig |
| `python run.py blender` | warm headless Blender on `127.0.0.1:18765` |

`QuixelStylizer.exe` is a windowed host so the taskbar can pin this program. It runs `run.py` next to the exe. Rebuild it from `launcher.c` if the exe is stale. Missing packages open a dialog that points at `bootstrap.ps1`.

GPU paint uses the Qt widget's OpenGL context on the UI thread (fence-polled compute). Image decode, zip extract, and PyMeshLab run in worker threads. No second GL context, no Win32 window enumeration.

## Library

`library/library.sqlite` plus one `recipe.json` per material, undo snaps, and a thumbnail cache. Drop zips in `library/inbox` or use Import. Missing source folders show as missing; they do not crash the window.

Recipes are a non-destructive stack (paint, grade, maps, packing, optional UV-island mask). Apply the open recipe to the selected rows, or export them in bulk.

## Mesh and budgets

`budgets.json` (created on first run) has class ceilings: hero, prop, foliage, modular, terrain, surface. Library rows colour amber near the cap and red over it. **Fit to budget** decimates to the class triangle cap and writes textures at the class size.

Live decimate is PyMeshLab `meshing_decimation_quadric_edge_collapse_with_texture` on a cached source. Quixel LODs share one UV set, so stylized textures still fit.

Blender (5.2 headless, warm process) is for convert FBX→glTF, re-UV, cage bake, FBX+LOD export, and an EEVEE thumbnail. It never runs on a slider tick. Point `BLENDER_EXE` at `blender.exe` if it is not in `Program Files\Blender Foundation`.

## Outputs

Written into the export folder you pick (picker starts at Exodus `Content`):

| File | Channels |
|---|---|
| `<Name>_D.png` | stylised diffuse, sRGB |
| `<Name>_MROD.png` | R metallic, G roughness, B AO, A displacement (min 1/255). ORM / ORMD are selectable. |
| `<Name>_N.png` | tangent-space normal, DirectX |
| `<Name>_Opacity.png` | only if the set has opacity |

Unreal import helper: `tools/unreal/import_stylized.py` (run inside the editor). Sets sRGB / compression per texture and can spawn a material instance from a parent you name.

## Exodus look

`qs/core/exodus_look.py` reads `Exodus/Config/DefaultShaderShift.ini` and `DefaultEngine.ini` when that project is present. Ground-truth capture and a diff live in `tools/groundtruth/`.

## Tests

```
.venv\Scripts\python.exe -m pytest
```
