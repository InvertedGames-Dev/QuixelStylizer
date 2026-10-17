"""Import a QuixelStylizer export folder into Exodus and spawn a material instance.

Run inside the Unreal Editor (PythonScriptPlugin):

    import sys
    sys.path.append(r"C:\\Users\\BigMoney\\Projects\\QuixelStylizer")
    from tools.unreal.import_stylized import import_folder
    import_folder(r"C:\\Users\\BigMoney\\Projects\\Exodus\\Content\\MyRock")
"""
from __future__ import annotations

import os
from pathlib import Path

PACKING_SRGB = {
    "D": True,
    "N": False,
    "MROD": False,
    "ORM": False,
    "ORMD": False,
    "Opacity": False,
}


def _suffix(name: str) -> str | None:
    stem = Path(name).stem
    for key in ("MROD", "ORMD", "ORM", "Opacity", "D", "N"):
        if stem.upper().endswith("_" + key.upper()) or stem.endswith("_" + key):
            return key
    return None


def _content_path(abs_folder: str) -> str:
    """C:/.../Exodus/Content/Foo -> /Game/Foo"""
    p = Path(abs_folder).resolve()
    parts = p.parts
    if "Content" not in parts:
        raise ValueError(f"{abs_folder} is not under a Content folder")
    i = parts.index("Content")
    rest = parts[i + 1 :]
    return "/Game" + (("/" + "/".join(rest)) if rest else "")


def import_folder(folder: str, packing: str = "mrod", parent_material: str | None = None) -> dict:
    import unreal

    folder = os.path.abspath(folder)
    dest = _content_path(folder)
    packing = (packing or "mrod").upper()
    tasks = []
    files = {}
    for name in sorted(os.listdir(folder)):
        if not name.lower().endswith((".png", ".tga")):
            continue
        key = _suffix(name)
        if key is None:
            continue
        path = os.path.join(folder, name)
        task = unreal.AssetImportTask()
        task.set_editor_property("filename", path)
        task.set_editor_property("destination_path", dest)
        task.set_editor_property("automated", True)
        task.set_editor_property("save", True)
        task.set_editor_property("replace_existing", True)
        tasks.append(task)
        files[key] = dest + "/" + Path(name).stem
    if not tasks:
        raise FileNotFoundError(f"no stylized maps in {folder}")
    unreal.AssetToolsHelpers.get_asset_tools().import_asset_tasks(tasks)

    imported = {}
    for key, asset_path in files.items():
        tex = unreal.load_asset(asset_path)
        if tex is None:
            continue
        srgb = PACKING_SRGB.get(key, False)
        tex.set_editor_property("srgb", srgb)
        if key == "N":
            tex.set_editor_property(
                "compression_settings", unreal.TextureCompressionSettings.TC_NORMALMAP)
        elif key in ("MROD", "ORM", "ORMD", "Opacity"):
            tex.set_editor_property(
                "compression_settings", unreal.TextureCompressionSettings.TC_MASKS)
        else:
            tex.set_editor_property(
                "compression_settings", unreal.TextureCompressionSettings.TC_DEFAULT)
        unreal.EditorAssetLibrary.save_loaded_asset(tex)
        imported[key] = asset_path

    mi_path = None
    if parent_material:
        name = Path(folder).name + "_Inst"
        factory = unreal.MaterialInstanceConstantFactoryNew()
        mi = unreal.AssetToolsHelpers.get_asset_tools().create_asset(
            name, dest, unreal.MaterialInstanceConstant, factory)
        parent = unreal.load_asset(parent_material)
        if parent is not None:
            mi.set_editor_property("parent", parent)
        def _set_tex(param, key):
            if key in imported:
                unreal.MaterialEditingLibrary.set_material_instance_texture_parameter_value(
                    mi, param, unreal.load_asset(imported[key]))
        _set_tex("Diffuse", "D")
        _set_tex("Normal", "N")
        pack_key = packing if packing in imported else (
            "MROD" if "MROD" in imported else ("ORM" if "ORM" in imported else "ORMD"))
        _set_tex("Packed", pack_key)
        unreal.EditorAssetLibrary.save_loaded_asset(mi)
        mi_path = dest + "/" + name
    return {"textures": imported, "material": mi_path, "path": dest}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Print the import plan (editor runs the import).")
    parser.add_argument("folder")
    parser.add_argument("--packing", default="mrod")
    parser.add_argument("--parent")
    args = parser.parse_args()
    print(_content_path(args.folder))
    for name in sorted(os.listdir(args.folder)):
        print(" ", name, "->", _suffix(name))
