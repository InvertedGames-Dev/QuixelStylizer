"""SQLite material library. One recipe.json + undo file per material."""
from __future__ import annotations

import json
import os
import shutil
import sqlite3
import time
from pathlib import Path

from qs.core.extract import extract_zip, sha256_file
from qs.core.maps import find_sets
from qs.core.recipe import Recipe
from qs.logging import log_line
from qs.version import TOOL_DIR

LIBRARY = Path(TOOL_DIR) / "library"
DB_PATH = LIBRARY / "library.sqlite"
SOURCES = LIBRARY / "sources"
INBOX = LIBRARY / "inbox"
MASKS = LIBRARY / "masks"
RECIPES = LIBRARY / "recipes"
THUMBS = LIBRARY / "thumbs"
UNDO = LIBRARY / "undo"
_ILLEGAL = set('\\/:*?"<>|')

SCHEMA = """
CREATE TABLE IF NOT EXISTS materials (
    id TEXT PRIMARY KEY,
    display_name TEXT NOT NULL,
    quixel_name TEXT,
    source_folder TEXT NOT NULL,
    kind TEXT NOT NULL DEFAULT 'surface',
    asset_class TEXT NOT NULL DEFAULT 'prop',
    missing INTEGER NOT NULL DEFAULT 0,
    tris INTEGER,
    verts INTEGER,
    islands INTEGER,
    tex_w INTEGER,
    tex_h INTEGER,
    mesh_path TEXT,
    texel_density REAL,
    original_mesh_path TEXT,
    baked_mesh_path TEXT,
    baked_maps_folder TEXT,
    updated REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS imported (
    sha256 TEXT PRIMARY KEY,
    zip_path TEXT,
    extracted TEXT
);
CREATE TABLE IF NOT EXISTS settings (
    key TEXT PRIMARY KEY,
    value TEXT
);
"""


_SCHEMA_READY: set[Path] = set()


def ensure_dirs():
    for p in (LIBRARY, SOURCES, INBOX, MASKS, RECIPES, THUMBS, UNDO):
        p.mkdir(parents=True, exist_ok=True)


def connect() -> sqlite3.Connection:
    ensure_dirs()
    db_file = Path(DB_PATH)
    con = sqlite3.connect(db_file, timeout=30.0)
    con.row_factory = sqlite3.Row
    if db_file not in _SCHEMA_READY:
        con.execute("PRAGMA journal_mode=WAL")
        con.executescript(SCHEMA)
        cols = {r[1] for r in con.execute("PRAGMA table_info(materials)")}
        if "mesh_path" not in cols:
            con.execute("ALTER TABLE materials ADD COLUMN mesh_path TEXT")
        if "texel_density" not in cols:
            con.execute("ALTER TABLE materials ADD COLUMN texel_density REAL")
        if "original_mesh_path" not in cols:
            con.execute("ALTER TABLE materials ADD COLUMN original_mesh_path TEXT")
        if "baked_mesh_path" not in cols:
            con.execute("ALTER TABLE materials ADD COLUMN baked_mesh_path TEXT")
        if "baked_maps_folder" not in cols:
            con.execute("ALTER TABLE materials ADD COLUMN baked_maps_folder TEXT")
        con.commit()
        _SCHEMA_READY.add(db_file)
    return con


def name_error(name: str) -> str:
    name = "" if name is None else str(name).strip()
    if not name or name in (".", ".."):
        return "The name cannot be empty."
    if any(c in _ILLEGAL or ord(c) < 32 for c in name):
        return 'That name cannot be a folder. Avoid \\ / : * ? " < > |'
    if name.endswith(".") or name.endswith(" "):
        return "A folder name cannot end with a space or a dot."
    return ""


def _mid(folder: str) -> str:
    import hashlib
    key = os.path.normcase(os.path.abspath(folder))
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]


def recipe_path(mid: str) -> Path:
    return RECIPES / f"{mid}.json"


def load_recipe(mid: str) -> dict:
    path = recipe_path(mid)
    if path.is_file():
        try:
            return Recipe.from_dict(json.loads(path.read_text(encoding="utf-8"))).to_dict()
        except (OSError, ValueError):
            pass
    return Recipe().to_dict()


def save_recipe(mid: str, st: dict, record_undo: bool = True) -> None:
    ensure_dirs()
    text = json.dumps(Recipe.from_dict(st).to_dict(), indent=2)
    recipe_path(mid).write_text(text, encoding="utf-8")
    if not record_undo:
        return
    undo_dir = UNDO / mid
    undo_dir.mkdir(parents=True, exist_ok=True)
    snaps = sorted(undo_dir.glob("*.json"))
    if snaps:
        try:
            if snaps[-1].read_text(encoding="utf-8") == text:
                return
        except OSError:
            pass
    stamp = f"{time.strftime('%Y%m%d-%H%M%S')}-{time.time_ns() % 1_000_000_000:09d}"
    (undo_dir / f"{stamp}.json").write_text(text, encoding="utf-8")
    snaps = sorted(undo_dir.glob("*.json"))
    for old in snaps[:-40]:
        old.unlink(missing_ok=True)


def get_setting(key: str, default: str = "") -> str:
    with connect() as con:
        row = con.execute("SELECT value FROM settings WHERE key=?", (key,)).fetchone()
    return row["value"] if row else default


def set_setting(key: str, value: str) -> None:
    with connect() as con:
        con.execute("INSERT OR REPLACE INTO settings(key, value) VALUES (?,?)", (key, value))
        con.commit()


def list_materials(over_budget=None, asset_class=None) -> list[dict]:
    sql = "SELECT * FROM materials"
    args = []
    if asset_class:
        sql += " WHERE asset_class=?"
        args.append(asset_class)
    sql += " ORDER BY display_name COLLATE NOCASE"
    with connect() as con:
        rows = [dict(r) for r in con.execute(sql, args)]
    for row in rows:
        row["source_ok"] = os.path.isdir(row["source_folder"])
        row["missing"] = int(not row["source_ok"])
        row["recipe"] = load_recipe(row["id"])
    return rows


def upsert_material(folder: str, display_name: str | None = None, kind="surface") -> str:
    s = None
    sets = find_sets([folder], recursive=False) or find_sets([folder], recursive=True)
    if sets:
        s = sets[0]
    mid = _mid(folder)
    name = display_name or (s["name"] if s else Path(folder).name)
    now = time.time()
    missing = 0 if os.path.isdir(folder) else 1
    with connect() as con:
        con.execute(
            """INSERT INTO materials(id, display_name, quixel_name, source_folder, kind, missing, updated)
               VALUES (?,?,?,?,?,?,?)
               ON CONFLICT(id) DO UPDATE SET
                 display_name=excluded.display_name,
                 source_folder=excluded.source_folder,
                 missing=excluded.missing,
                 updated=excluded.updated""",
            (mid, name, s["name"] if s else name, os.path.abspath(folder), kind, missing, now),
        )
        con.commit()
    if not recipe_path(mid).is_file():
        save_recipe(mid, Recipe().to_dict())
    return mid


def rename_material(mid: str, name: str) -> None:
    err = name_error(name)
    if err:
        raise ValueError(err)
    with connect() as con:
        con.execute("UPDATE materials SET display_name=?, updated=? WHERE id=?", (name, time.time(), mid))
        con.commit()


def set_class(mid: str, asset_class: str) -> None:
    with connect() as con:
        con.execute("UPDATE materials SET asset_class=?, updated=? WHERE id=?", (asset_class, time.time(), mid))
        con.commit()


def update_mesh_stats(mid: str, tris=None, verts=None, islands=None, tex_w=None, tex_h=None,
                      mesh_path=None, texel_density=None, original_mesh_path=None):
    with connect() as con:
        row = con.execute("SELECT * FROM materials WHERE id=?", (mid,)).fetchone()
        if row is None:
            return
        con.execute(
            """UPDATE materials SET tris=?, verts=?, islands=?, tex_w=?, tex_h=?, updated=? WHERE id=?""",
            (
                row["tris"] if tris is None else tris,
                row["verts"] if verts is None else verts,
                row["islands"] if islands is None else islands,
                row["tex_w"] if tex_w is None else tex_w,
                row["tex_h"] if tex_h is None else tex_h,
                time.time(),
                mid,
            ),
        )
        extra = con.execute("PRAGMA table_info(materials)").fetchall()
        cols = {r[1] for r in extra}
        if "mesh_path" not in cols:
            con.execute("ALTER TABLE materials ADD COLUMN mesh_path TEXT")
        if "texel_density" not in cols:
            con.execute("ALTER TABLE materials ADD COLUMN texel_density REAL")
        if "original_mesh_path" not in cols:
            con.execute("ALTER TABLE materials ADD COLUMN original_mesh_path TEXT")
        if mesh_path is not None:
            con.execute("UPDATE materials SET mesh_path=? WHERE id=?", (mesh_path, mid))
        if texel_density is not None:
            con.execute("UPDATE materials SET texel_density=? WHERE id=?", (texel_density, mid))
        if original_mesh_path is not None:
            con.execute("UPDATE materials SET original_mesh_path=? WHERE id=?",
                        (original_mesh_path, mid))
        con.commit()


def baked_dir(mid: str) -> Path:
    """Generated rewrap-and-bake output for one material, inside the library.

    Kept out of the user's source folder so the original maps and mesh are
    never overwritten. The folder holds the unwrapped low mesh and its
    baked maps together, so `detect_maps` can load them as one set.
    """
    return Path(LIBRARY) / "baked" / str(mid)


def set_baked_state(mid: str, mesh_path: str, maps_folder: str) -> None:
    """Record a rewrap-and-bake result against the existing material row."""
    with connect() as con:
        cols = {r[1] for r in con.execute("PRAGMA table_info(materials)")}
        if "baked_mesh_path" not in cols:
            con.execute("ALTER TABLE materials ADD COLUMN baked_mesh_path TEXT")
        if "baked_maps_folder" not in cols:
            con.execute("ALTER TABLE materials ADD COLUMN baked_maps_folder TEXT")
        con.execute(
            "UPDATE materials SET baked_mesh_path=?, baked_maps_folder=?, updated=? WHERE id=?",
            (str(mesh_path), str(maps_folder), time.time(), mid),
        )
        con.commit()


def clear_baked_state(mid: str) -> None:
    """Drop the baked state so the material returns to its original maps and mesh."""
    with connect() as con:
        cols = {r[1] for r in con.execute("PRAGMA table_info(materials)")}
        if "baked_mesh_path" not in cols or "baked_maps_folder" not in cols:
            return
        con.execute(
            "UPDATE materials SET baked_mesh_path=NULL, baked_maps_folder=NULL, updated=? WHERE id=?",
            (time.time(), mid),
        )
        con.commit()


def delete_material(mid: str) -> bool:
    """Remove a material from the library: catalog row, recipe, undo snapshots.

    This is "remove from library" only. The user's source texture and mesh
    files on disk are left exactly where they are. The generated baked
    folder for this material (inside the library, not a source folder) is
    removed with the entry, since nothing else can reference it.
    """
    with connect() as con:
        cur = con.execute("DELETE FROM materials WHERE id=?", (mid,))
        con.commit()
        removed = cur.rowcount > 0
    recipe_path(mid).unlink(missing_ok=True)
    undo_dir = UNDO / mid
    if undo_dir.is_dir():
        shutil.rmtree(undo_dir, ignore_errors=True)
    baked = baked_dir(mid)
    if baked.is_dir():
        shutil.rmtree(baked, ignore_errors=True)
    return removed


def import_folder(folder: str) -> list[str]:
    added = []
    for s in find_sets([folder], recursive=True):
        added.append(upsert_material(s["folder"], s["name"]))
    return added


def import_zip(zip_path: str) -> list[str]:
    ensure_dirs()
    zip_path = os.path.abspath(zip_path)
    digest = sha256_file(zip_path)
    with connect() as con:
        if con.execute("SELECT 1 FROM imported WHERE sha256=?", (digest,)).fetchone():
            return []
    stem = Path(zip_path).stem
    stem = "".join(c if c.isalnum() or c in "-_" else "_" for c in stem)[:40] or "material"
    dest = SOURCES / f"{stem}_{digest[:8]}"
    extract_zip(zip_path, str(dest))
    with connect() as con:
        con.execute(
            "INSERT INTO imported(sha256, zip_path, extracted) VALUES (?,?,?)",
            (digest, zip_path, str(dest)),
        )
        con.commit()
    return import_folder(str(dest))


def scan_inbox() -> list[str]:
    ensure_dirs()
    added = []
    for name in sorted(os.listdir(INBOX)):
        if not name.lower().endswith(".zip"):
            continue
        path = INBOX / name
        try:
            added.extend(import_zip(str(path)))
        except Exception as exc:
            log_line(f"library: could not import {path}: {exc}")
    return added


def migrate_catalog_json() -> int:
    """One-shot import of the v1.5 catalog.json if the sqlite db is empty."""
    old = LIBRARY / "catalog.json"
    if not old.is_file():
        return 0
    with connect() as con:
        n = con.execute("SELECT COUNT(*) AS c FROM materials").fetchone()["c"]
        if n:
            return 0
    data = json.loads(old.read_text(encoding="utf-8"))
    count = 0
    for mid, mat in (data.get("materials") or {}).items():
        folder = mat.get("source_folder")
        if not folder:
            continue
        upsert_material(folder, mat.get("display_name"))
        if mat.get("settings"):
            save_recipe(_mid(folder), mat["settings"])
        count += 1
    if data.get("export_root"):
        set_setting("export_root", data["export_root"])
    log_line(f"migrated {count} materials from catalog.json")
    return count
