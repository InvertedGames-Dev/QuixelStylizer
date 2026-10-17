from PySide6.QtWidgets import QApplication

from qs import budgets, library
from qs.app.library_panel import BudgetMeter
from qs.core.recipe import Recipe


def test_library_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(library, "LIBRARY", tmp_path / "library")
    monkeypatch.setattr(library, "DB_PATH", tmp_path / "library" / "library.sqlite")
    monkeypatch.setattr(library, "SOURCES", tmp_path / "library" / "sources")
    monkeypatch.setattr(library, "INBOX", tmp_path / "library" / "inbox")
    monkeypatch.setattr(library, "MASKS", tmp_path / "library" / "masks")
    monkeypatch.setattr(library, "RECIPES", tmp_path / "library" / "recipes")
    monkeypatch.setattr(library, "THUMBS", tmp_path / "library" / "thumbs")
    monkeypatch.setattr(library, "UNDO", tmp_path / "library" / "undo")
    folder = tmp_path / "src"
    folder.mkdir()
    mid = library.upsert_material(str(folder), "Rock")
    library.save_recipe(mid, Recipe(saturation=1.4).to_dict())
    rec = library.load_recipe(mid)
    assert abs(rec["saturation"] - 1.4) < 1e-9
    library.update_mesh_stats(mid, tris=1200, verts=800, tex_w=1024, tex_h=1024)
    library.update_mesh_stats(mid, tex_w=2048)  # must not wipe tris
    rows = library.list_materials()
    assert rows[0]["tris"] == 1200
    assert rows[0]["tex_w"] == 2048
    assert library.name_error("bad:name")


def test_budget_status():
    assert budgets.status(100, 512, "prop") == "ok"
    assert budgets.status(4000, 512, "prop") == "over"
    cap = budgets.ceiling("hero")
    assert cap["tris"] > budgets.ceiling("prop")["tris"]


def test_class_table_has_six_classes():
    table = budgets.class_table()
    for key in ("hero", "prop", "foliage", "modular", "terrain", "surface"):
        assert "tris" in table[key] and "tex" in table[key]
    assert budgets.CLASS_ORDER[0] == "surface"
    from PySide6.QtWidgets import QApplication
    from qs.app.library_panel import BudgetMeter
    app = QApplication.instance() or QApplication([])
    meter = BudgetMeter()
    meter.set_values(1152, 256, "foliage")
    assert "hero" in meter.legend_text()
    grouped = {row["cap"]: row["names"] for row in meter.collapsed_ticks("tex")}
    assert set(grouped[1024]) >= {"prop", "modular", "terrain", "surface"}


def test_budget_meter_ticks_between_classes():
    app = QApplication.instance() or QApplication([])
    meter = BudgetMeter()
    meter.set_values(1152, 1024, "foliage")
    for key in ("hero", "prop", "foliage", "modular", "terrain", "surface"):
        assert key in meter.class_names()
    ticks = meter.tick_positions("tris")
    fill = meter.fill_pos("tris")
    assert ticks["foliage"] < fill < ticks["modular"]
    assert abs(ticks["foliage"] - 0.0) > 0.05
    gaps = [ticks["modular"] - ticks["foliage"], ticks["prop"] - ticks["modular"],
            ticks["terrain"] - ticks["prop"], ticks["hero"] - ticks["terrain"]]
    assert max(gaps) - min(gaps) < 0.02
    tex_ticks = meter.tick_positions("tex")
    assert tex_ticks["foliage"] < tex_ticks["hero"]
    app.processEvents()
