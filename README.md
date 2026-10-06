# QuixelStylizer

Turns a Quixel Megascans / Fab surface set into a hand-painted (Dishonored-style) material and
writes packed maps for Unreal Engine 5. Offline, Python 3.12 + numpy + OpenCV (+ Pillow for the GUI preview).

## Run it

| How | What happens |
|---|---|
| Double-click `QuixelStylizer.bat` | GUI with live preview: add folder(s), move sliders, the preview updates while you drag. **CONVERT** writes the files. Save/Load preset |
| Drag one or more folders onto `QuixelStylizer.bat` | CLI with default settings, window stays open with the log |
| `python quixel_stylize.py <folder> [<folder> ...] [options]` | CLI |

A folder can be a single material set, or a parent folder: then every direct subfolder that contains an
albedo is processed (`--recursive` searches all depths). Output goes to `<set folder>\Stylized\`
or to `--out <folder>`. Source files are only read, never written.

Python used by the .bat: `%LOCALAPPDATA%\Programs\Python\Python312\python.exe` (falls back to `py -3.12`).
Setup if missing: `winget install -e --id Python.Python.3.12`, then `python -m pip install numpy opencv-python pillow moderngl`.
`moderngl` is the GPU paint path (OpenGL 4.3 compute). Without it, Kuwahara stays on the CPU. Anisotropic Kuwahara needs the GPU path.

## v1.3

The window is dark. Colour grading runs in Oklab: saturation scales chroma, posterize steps lightness and chroma, and a palette of up to eight swatches can pull colours toward those swatches (`palette_strength` 0 leaves the grade unsnapped). Per-map mix sliders keep a map closer to the source. Detail restore puts source texture back on the albedo. Offset rolls the preview by half a tile and the status line shows the measured seam difference. Seam fade is off until its slider moves. The lit view is Smith GGX plus a small built-in environment, with an exposure slider. Paint mask (amount / smooth / erase) is a per-pixel stylize amount, saved beside a preset as `*_mask.png`, and CONVERT bakes it into the maps.

Examples
```
python quixel_stylize.py "C:\Users\BigMoney\Downloads\uncut_grass_oeecl0_1k"
python quixel_stylize.py "D:\Megascans" --recursive --size 2048 --radius 8 --out "D:\Stylized"
python quixel_stylize.py <folder> --preset presets\my_look.json --edge-strength 0.4
python quixel_stylize.py <folder> --list            # only show detected maps
python quixel_stylize.py <folder> --save-preset presets\my_look.json --radius 9 --saturation 1.3
```

## Live preview (GUI, v1.2.3)

- The preview loads as soon as a folder is added. Pick another set with **Preview set**, or click a folder in the list.
- Any slider or option re-renders the preview while you drag. Rendering runs in a background thread and only the newest settings are rendered, so the UI never freezes. Dragging a paint-filter slider refreshes about every 0.3 s at 512. The top-right corner shows "rendering..." and the last update time with a per-stage breakdown.
- **Preview size** (256/512/1024, default 512) only affects the preview. The paint radius is scaled to the preview size, so it looks like the final output scaled down. The preview size is capped at the Output size.
- The preview uses the same code as CONVERT. Preview at 1024 matched the converted `_D` RGB exactly (0 differing pixels) on the grass test set.
- Stages are cached:
  - Load + resize is cached per set.
  - The paint filter is cached per (filter, radius, passes, sharpness, size).
  - Colour/AO/edge/tint/posterize/roughness/normal sliders only rerun the cheap finishing stage.
- **View:**
  - Diffuse, Normal, Roughness, AO, Displacement, Metallic.
  - **Lit**: the stylised diffuse shaded with the stylised normal and roughness under one directional light (Lambert plus GGX-style spec). This is for judging the look only, not UE's renderer.
- **Split / After / Before.** Left-drag on the image moves the split line. Right-drag sets the light direction, and so does left-drag in Lit view when not in Split mode. **Tile 2x2** is for checking seams. **Zoom** Fit/1x/2x (centre crop).
- Measured on Helios (Ryzen 9 5950X, grass set, preview 512): filter change about 280 ms render (about 430 ms from slider move to new image, including the 120 ms throttle); colour change about 90 ms render (about 260 ms total); light/view/tile change about 75-90 ms. Preview 1024: about 0.9 s paint plus 0.35 s finish.

## Outputs (8-bit PNG)

| File | Channels |
|---|---|
| `<Name>_D.png` | RGB = stylised diffuse (sRGB), no alpha |
| `<Name>_MROD.png` | R = Metallic, G = Roughness, B = AO, A = Displacement/height (all linear) |
| `<Name>_N.png` | tangent-space normal, DirectX (green down) by default |
| `<Name>_Opacity.png` | only if the set has an opacity map |

`--format tga` (GUI: Format) writes the texture outputs as uncompressed TGA instead of PNG. The preview stays PNG.
MROD alpha is clamped to at least 1/255 so no pixel is ever fully transparent (see the UE note below). Emissive maps aren't used (they're listed as ignored).
| `<Name>_preview.png` | 2x2: albedo before/after, normal before/after |
| `<Name>_stylize.json` | settings, inputs used, defaulted maps, notes. Loadable as a preset (`--preset`). |

`<Name>` comes from the albedo filename with the map suffix and resolution token removed
(`Uncut_Grass_oeecL0_1K_BaseColor.jpg` -> `Uncut_Grass_oeecL0`).

## Input detection (filename suffix, case-insensitive; resolution tokens like `_1K`/`_2048` are ignored)

Albedo/BaseColor/Base_Color/Diffuse/Color, Roughness (or Gloss -> inverted), Metalness/Metallic,
AO/AmbientOcclusion/Occlusion, Cavity, Normal / NormalDX / NormalGL (also `nor_dx`, `nor_gl`),
Displacement/Height (Bump used as height if no displacement), Opacity,
packed ORM/ARM (R=AO, G=Roughness, B=Metallic; used only for channels with no separate map).
Specular/Translucency are ignored (reported). If several files exist for one map, PNG > TIF > TGA > JPG > EXR.

Missing maps are defaulted and reported: metallic 0, AO 1 (white),
roughness 0.6, height 0.5 (flat, MROD alpha 128), normal derived from height (or flat if there is no height).

**Normal convention.** NormalDX is preferred over NormalGL. An unsuffixed `_Normal` is auto-detected by
correlating its green channel with the height gradient (result is in the JSON notes). The three Megascans
sets tested on this PC were all detected as OpenGL. With no height to check, unsuffixed normals are
assumed OpenGL. Override with `--normal-in dx|gl`, or `--flip-green` to invert the source green.
Output is DirectX unless `--normal-out gl`.

## What it does

1. Resize (area when downscaling, Lanczos when upscaling) so the longest side = `--size`. Keeps aspect ratio.
2. Albedo paint filter: generalized Kuwahara (8 smooth sectors, Gaussian weighted), `--passes` times.
   Wraps at the borders (`--tile`, on by default) so tileable sets stay seamless.
   The per-pixel sector weights from the albedo are reused on normal, height, roughness, metallic, AO
   and cavity, so all maps share the same painted shapes. Alternatives: `--filter kuwahara_classic|median|bilateral`.
3. Colour: value compression toward mean luminance, saturation, warm-highlight / cool-shadow tint,
   optional soft posterize on luminance.
4. Baked lighting: AO multiplied in linear space (`--ao-strength`); curvature edge highlight/crevice
   darkening (`--edge-strength`). Curvature from Cavity if present, else divergence of the filtered normal
   (`--curvature-source`).
5. Normal: paint-filtered, extra Gaussian low-pass (`--normal-soften`), XY scale (`--normal-strength`),
   renormalised. `--normal-mode height` rebuilds it from the filtered height instead.
6. Roughness pulled toward its mean (`--roughness-flatten`), metallic pushed toward 0/1
   (`--metal-binarize`), height filtered like the albedo (+ `--height-soften`). Height is not renormalised.

Radii/blur widths are in pixels at 1024 and scale with `--size`, so a preset looks the same at any size.

## Defaults (`presets/default.json`)

size 1024, filter kuwahara, radius 6 (px @1024), passes 2, sharpness 8, saturation 1.15,
value_compression 0.15, posterize 0 (off), tint_strength 0.35, ao_strength 0.5, edge_strength 0.25,
edge_blur 2, curvature auto, roughness_flatten 0.8, roughness_bias 0, metal_binarize 0.5,
normal_soften 0.5, normal_strength 1.0, normal_mode filter, normal_in auto, normal_out dx,
height_soften 0.5, format png, tile on, preview on.

Tuning: bigger `radius` / more `passes` = bigger brush shapes. Lower `edge_strength` if the crevice
darkening looks spotty (busy surfaces like grass); raise it for stone/plaster/wood.

## Importing in UE 5.7

| Texture | sRGB | Compression Settings |
|---|---|---|
| `_D` | on | Default (BC1, since there's no alpha) or BC7 |
| `_MROD` | **off** | **BC7 (DX11, optional A)** recommended, to keep displacement precision in alpha. "Masks (no sRGB)" also works: per Epic's compression docs it uses BC3/DXT5 when the alpha is kept, with RGB at BC1 quality. |
| `_N` | (off) | **Normalmap (BC5)**. BC5 stores only R/G, and Z is rebuilt in the shader. Output is DirectX. If you exported with `--normal-out gl`, tick "Flip Green Channel" on the texture. |

The Masks/BC1/BC3 behaviour comes from Epic's texture compression docs (written for UE4.27) and community
guides. I haven't checked it in 5.7 specifically. The texture editor shows the format it actually used, so check there.

Material hookup: `_D.rgb` -> Base Color. `_MROD.r` -> Metallic, `.g` -> Roughness, `.b` -> Ambient Occlusion,
`.a` = displacement/height (for displacement, parallax or Nanite tessellation if you want it). For `_MROD`, turn sRGB off
and set the sampler type to Linear Color. AO is already
baked into the diffuse at `ao_strength`. Plugging `_MROD.b` into Ambient Occlusion as well applies it a second time, to indirect light.
Use `--ao-strength 0` if you only want material AO.

**PNG alpha and UE's "infill" (checked in UE 5.7.4 source, `TextureImportUtils.cpp` / `EditorFactories.cpp`).**
On PNG import UE can rewrite the RGB of pixels that are exactly white with zero alpha (255,255,255,0), filling them from their neighbours.
- The setting is "When to infill RGB in transparent white PNG" (`PNGInfill`; legacy key `[TextureImporter] FillPNGZeroAlpha`, default true).
- The default is `OnlyOnBinaryTransparency`, which skips any PNG that has other partially or fully transparent pixels.
- TGA files are never infilled.
- Exodus doesn't override this setting.
- Because MROD alpha is clamped to at least 1/255, no pixel is ever fully transparent, so the infill can't touch MROD under any setting.

## Output size, versions and old files (v1.2.3)
- **Output size:** "Output size" (GUI) / `--size` (CLI) sets the dimensions of every file written: `_D`, `_MROD`, `_N`, `_Opacity` and `_preview`. Sources are upscaled with Lanczos when the size is larger than the source (e.g. a 1k set at 2048).
  - The preview PNG is a 2x2 sheet (albedo before/after, normal before/after) with the same dimensions as the maps.
  - The size is the longest side, so non-square sources keep their aspect ratio.
  - "Preview size" only affects the on-screen live preview, never the files.
- **Checks after writing:** each file's header is read back after it's written (PNG IHDR colour type 2 = RGB for `_D`/`_N`, 6 = RGBA for `_MROD`; TGA 24/32-bit, plus the dimensions). The convert fails loudly if anything doesn't match. The results go to the log panel, `stylizer.log` and `files_checked` in `_stylize.json`.
- **Mouse wheel:** it no longer changes dropdown values. Before v1.2.3, scrolling the settings panel with the pointer over a dropdown such as Output size silently changed it.
- **Version logging:** the window title, the log panel and every `stylizer.log` start/convert line show the version and the code path.
- **Old windows:** an already-open older Quixel Stylizer window keeps running the code it started with. The GUI and CLI warn when one is open; close it.
- **Old-layout files:** `<Name>_MRO*`/`<Name>_MROE*` files left by older versions are deleted when you convert. This only happens in the tool's own output folder, next to `<Name>_stylize.json`, and never in the source folder. Each deletion is logged.

## Limitations

- Generalized Kuwahara, not anisotropic (no stroke direction following the structure tensor).
- Outputs are 8-bit. EXR/16-bit inputs are read as-is (clipped to 0-1, no colour transform), so JPG/PNG sources are preferred.
- The GUI has no drag-and-drop target. Drag folders onto the .bat instead (CLI, defaults).
- Changing a paint-filter setting at preview 1024 takes about 1.3 s per update. Use 512 while tuning.
