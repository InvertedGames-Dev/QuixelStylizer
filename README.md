# QuixelStylizer 2

## ⚠️ Read this first: this is vibe coded

**This program was made entirely with AI coding agents.** Every line here was written by coding agents (in Cursor) working under my direction — I described what I wanted, tested the results against my own library of assets, and sent back screenshots when things were wrong. I did not hand-write this codebase.

It exists for exactly one purpose: **stylizing Quixel/Fab assets for Exodus, my own game.** It is fitted to my machine, my asset library, my texture budgets, and my taste, in roughly that order. Concretely, that means:

- Paths, Blender versions, and pipeline assumptions reflect one specific local setup (Windows, a locally installed Blender, and a warm local Blender process).
- There is no support, no roadmap, and no promise that any of this works on your machine or your assets. Issues may be read; they may also sit there forever.
- The code has been tested heavily against *my* library of Quixel meshes and against its own test suite, but large parts carry agent-grade confidence rather than battle-tested production mileage. Two separate bake implementations shipped broken before the current one was verified properly — that is the honest calibre of "vibe coded," and this disclaimer is here so nobody mistakes it for anything else.

I'm making it public because it might be useful or interesting to somebody, and because hiding agent-made code behind a straight face helps nobody. If you use it, read the code before you trust it.

It's MIT licensed (see [LICENSE](LICENSE)), so you're welcome to take it, fork it, or mine it for parts.

## What it is

QuixelStylizer 2.0.0 turns Quixel/Fab PBR texture sets into painterly,
Unreal-ready maps, and can reduce and re-fit the matching meshes so the
stylized textures still sit correctly on them.

It has two faces over one headless core:

- A PySide6, Lightroom-style desktop shell: a local material library,
  non-destructive per-material recipes, a live Exodus-look viewport, and
  bulk apply / fit / export.
- A CLI (`qs`) for one-off stylizing, batch runs, and calibration renders.

Textures get a Kuwahara-family painted look (GPU compute when available,
CPU fallback), an Oklab grade, and Exodus channel packing. Meshes can be
previewed on the actual asset, decimated in-process, and — when the
decimation has stretched the textures too far — re-unwrapped at the
decimated triangle count with the hi-poly maps and geometry normals
baked back onto them in Blender.

The old v1.5 Tk shell (`stylizer_gui.py`, `quixel_stylize.py`) is gone.
The current code lives in `qs/` (`qs/core` for the headless kernels,
`qs/app` for the Qt shell), with `qs_blender_server.py` as the warm
Blender worker.

## Setup

This is a Windows program. It has only ever been run on Windows 10 with
Python 3.12, so that is the supported shape even where nothing is
explicitly Windows-only.

You need:

- Python 3.12 (`winget install -e --id Python.Python.3.12`)
- Optional but recommended: Blender 4.2–5.2. Standard Windows
  installations are auto-detected, or point `BLENDER_EXE` (or
  `BLENDER`) at `blender.exe`. Blender is only needed for mesh
  conversion when no loadable mesh is present and for the rewrap bake;
  texture stylizing works without it.

From a clone:

```
git clone <repository-url>
cd <repository-folder>
powershell -File bootstrap.ps1
```

`bootstrap.ps1` creates `.venv` and installs the pinned
`requirements.txt`. Then run everything through that interpreter:

| How | What happens |
|---|---|
| `QuixelStylizer.bat` | GUI; dragging one material folder onto it stylizes that folder instead |
| `.\.venv\Scripts\python.exe run.py --gui` | GUI (running `run.py` with no command also opens the GUI) |
| `.\.venv\Scripts\python.exe -m qs` | same window |
| `.\.venv\Scripts\python.exe run.py stylize <folder> [--out DIR] [--size N] [--preset FILE]` | one material, no window |
| `.\.venv\Scripts\python.exe run.py batch <folders...> [--out DIR] [--recursive]` | many materials |
| `.\.venv\Scripts\python.exe run.py render --out sphere.png [--width N] [--height N] [--hdri FILE]` | Exodus-look calibration render |
| `.\.venv\Scripts\python.exe run.py blender [--port 18765]` | start the warm headless Blender server on localhost |
| `.\.venv\Scripts\python.exe run.py serve` | alias for the GUI |

`QuixelStylizer.exe` (not in the Git repo) is an optional windowed host
so the taskbar can pin the program instead of `python.exe`; it runs the
`run.py` next to it. Build it with `tools\compile-launcher.ps1` (needs
Visual Studio C++ tools). It is a convenience, not a requirement —
the `.bat` and the venv interpreter run the same code.

If the GUI starts with packages missing, it opens a dialog pointing back
at `bootstrap.ps1`.

## Inputs

A material set is a folder of PBR maps recognised by filename suffix
(`_BaseColor`, `_Normal`, `_Roughness`, `_AO`, `_Height`/`_Displacement`,
`_Metallic`, opacity, cavity, and the usual aliases). Supported image
formats are PNG, TIFF, TGA, JPG, BMP and EXR. Only an albedo/base-colour
map is mandatory; missing maps are substituted or defaulted and the
substitution is recorded in the `<Name>_stylize.json` sidecar written
with every export.

Meshes are optional. `.glb`/`.gltf` are read directly; `.fbx` and `.obj`
are converted to GLB in the warm Blender process when no loadable mesh
is present. A set without a mesh is previewed on a sphere, cube or
plane instead.

## Library

The catalogue is SQLite at `library/library.sqlite` (WAL mode). Source
maps and meshes stay in place in the material's own folder; the library
only records where they are. Per-material recipes live in
`library/recipes/<material_id>.json`, undo snapshots in
`library/undo/<material_id>/`, and generated rewrap output in
`library/baked/<material_id>/`. The whole `library/` tree is local
runtime data and is not in Git — a fresh clone starts with an empty
library.

The library panel supports search, an asset-class filter, an
over-budget filter, multi-select, inline display-name editing, and bulk
apply-recipe / fit-to-budget / export. Import a folder or a zip from
the panel, or drop zips into `library/inbox` and they are imported on
the next launch. A material whose source folder has gone missing is
shown greyed, not crashed on.

**Remove from library** deletes the catalogue entry, its recipe, its
undo history and its generated `baked/` folder. It deliberately never
touches the source textures or meshes on disk.

Recipes are a non-destructive stack (paint, grade, maps, packing,
optional UV-island mask) with output sizes from 256 to 2048 and paint
filters `kuwahara`, `kuwahara_classic`, `anisotropic`, `median` and
`bilateral`. Edits save as that set's recipe and are undoable (Ctrl+Z
steps back through that material's recipe snapshots; the chain is
capped). Sliders commit on release, so dragging does not flood the
undo history.

## Meshes and budgets

Every material has an asset class, and `budgets.json` (created on first
run, plain JSON, edit it freely) holds the per-class ceilings:

| Class | Triangles | Texture |
|---|---:|---:|
| hero | 20 000 | 2048 |
| prop | 3 000 | 1024 |
| foliage | 800 | 512 |
| modular | 1 500 | 1024 |
| terrain | 5 000 | 1024 |
| surface | 2 | 1024 |

Library rows colour amber within 85% of a cap and red over it, and the
budget meter shows where the mesh sits against every class ceiling.
**Fit to budget** decimates the mesh to the class triangle cap and sets
the recipe's output size to the largest standard size under the class
texture cap. GUI exports are likewise capped by the class texture
ceiling, so a prop never leaves at 2048 just because the recipe says so.

The decimate slider (5%–100%) applies on release. Decimation is an
in-process PyMeshLab quadric edge collapse on a position-welded copy of
the mesh: Quixel GLBs store one closed surface split into many UV
patches, and collapsing those patches unwelded tears the surface into
sheets, so the weld is what keeps the decimated mesh in one piece. The
result is split back apart by UV for rendering, with displacement
sampled from a height shared across coincident vertices so seams do
not crack open in the viewport.

If decimation stretches the textures too far, **Rewrap & bake from
hi-poly** is the fix, and it is one button: it takes the mesh at its
current (decimated) triangle count, smart-unwraps it in Blender, bakes
the hi-poly material maps onto the new UVs, and bakes the hi-poly
geometry into a true tangent-space normal map. The low mesh and its
baked maps are stored together under `library/baked/<material_id>/`
and take over as that material's mesh and maps — the same library
entry, no duplicates, and the original source files are never
overwritten. **Reset to default** undoes the whole thing: default
recipe, decimate back to 100%, original mesh, original maps, baked
state cleared. (It also recognises old `_reuv.glb` files left next to
a source mesh and points Reset back at the true original.)

Two implementation notes worth knowing before you go looking:

- PyMeshLab deliberately runs on the Qt GUI thread, not in the worker
  pool — its native code heap-corrupted (`0xc0000374`) on worker
  threads. Mesh loads and decimates can therefore briefly block the
  window. That is a workaround for a real crash, not an oversight.
- The warm Blender process speaks JSON-lines on `127.0.0.1:18765` and
  handshakes a protocol version (`rewrap-bake-6` at time of writing);
  a stale server from an older build is asked to quit and replaced
  rather than trusted. Blender never runs on a slider tick.

## Outputs

GUI exports go into `<export root>/<display name>/` (the picker starts
at the saved export root). CLI exports default to `<set folder>/Stylized`
unless `--out` says otherwise. Each export writes:

| File | Channels |
|---|---|
| `<Name>_D.png` | stylised diffuse, sRGB |
| `<Name>_MROD.png` | R metallic, G roughness, B AO, A displacement (min 1/255) |
| `<Name>_ORM.png` | R AO, G roughness, B metallic — when packing is set to ORM |
| `<Name>_ORMD.png` | ORM plus A displacement — when packing is ORMD |
| `<Name>_N.png` | tangent-space normal, DirectX green by default (GL is a recipe option) |
| `<Name>_Opacity.png` | only if the set has an opacity map |
| `<Name>_stylize.json` | settings, inputs, defaults used, and output notes for the run |

PNG is the default format; a recipe/preset can select 24/32-bit TGA
instead. The default MROD layout matches the Exodus master material.

One boundary worth knowing: texture exports contain maps and the JSON
sidecar only — they do not copy a mesh. A rewrap bake stores its low
mesh at `library/baked/<material_id>/low.glb`; if you need that mesh in
Unreal, import that file alongside the exported maps.

Unreal import helper: `tools/unreal/import_stylized.py`, run inside the
Unreal Editor. It imports an export folder, sets sRGB and compression
correctly per texture, and can spawn a material instance from a parent
material you name.

## The Exodus look

The viewport approximates the Exodus render target rather than a
generic PBR preview: ShaderShift Burley diffuse with a dual-lobe GGX
specular, split-sum style image-based lighting from a studio
environment, AgX Punchy tonemapping, and no auto-exposure.
`qs/core/exodus_look.py` reads the relevant values from
`Exodus/Config/DefaultShaderShift.ini` and `DefaultEngine.ini` when the
project is found via `EXODUS_ROOT`; without it, built-in defaults
matching those files are used.
The headless `qs render` uses the same look on the CPU and finishes
with SMAA 1x. A ground-truth capture script and diff against the real
Unreal Editor live in `tools/groundtruth/` — the target there is a
documented error band, not pixel identity, since the viewport has no
reflection captures or shadows.

The viewport itself is one OpenGL 4.3 context owned by the Qt widget on
the GUI thread (tessellation-based height displacement on primitives;
asset meshes are displaced per-vertex and not tessellated, with a
no-tessellation fallback if 4.3 is unavailable). Live preview is capped
at 1024 and one paint pass regardless of the recipe, because it exists
to judge the look, not to render the export; full passes and full size
apply at export. Paint and finish run as ModernGL compute on that
context and fall back to the CPU kernels if the GPU path fails; the
`anisotropic` filter and the stroke control genuinely need the GPU
path, and on the CPU fallback they degrade to a plain Kuwahara pass
rather than pretending otherwise.

## Tests

```
.\.venv\Scripts\python.exe -m pytest
```

The suite covers the map/packing/normal kernels against CPU oracles,
the recipe and library round-trips, the welded decimation (including
geometric boundary-edge checks on the real ram skull and tree stump),
the Blender protocol handshake, and the GUI end-to-end. The
Blender-dependent test in `tests/test_rewrap_blender.py` decimates a
real ram skull mesh from a private local asset library, runs an
actual rewrap bake, and asserts the exported mesh still lines up with
its input — it skips itself when
Blender or that library asset is not present, which on a fresh clone
it will not be. Deeper system notes live in `documentation/README.md`.

One honest caveat: the full suite currently has one known
order-dependent failure (a GPU-vs-CPU Kuwahara comparison that passes
in isolation). It is a test-harness artefact under investigation, not
a shipping-code failure — but you deserve to hear it from the README
rather than discover it.

## Licence

MIT — see [LICENSE](LICENSE). Take it, fork it, mine it for parts. No warranty, obviously; see the vibe-coded disclaimer at the top.
