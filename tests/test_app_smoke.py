from PySide6.QtGui import QSurfaceFormat
from PySide6.QtWidgets import QApplication

from qs.app.viewport import default_gl_format
from qs.app.window import MainWindow, _missing_packages
from qs.version import TOOL_VERSION


def test_window_constructs():
    QSurfaceFormat.setDefaultFormat(default_gl_format())
    app = QApplication.instance() or QApplication([])
    win = MainWindow()
    assert TOOL_VERSION in win.windowTitle()
    assert win.library is not None
    assert win.view is not None
    assert win.develop is not None
    rec = win.develop.recipe()
    assert rec["packing"] in ("mrod", "orm", "ormd")
    assert win.develop._spins["radius"].toolTip()
    assert win.develop.reset.toolTip()
    assert win.develop.displace.isChecked()
    assert win.develop.displace.toolTip()
    assert "palette_strength" in rec and "palette_hardness" in rec
    assert rec["palette"] == []
    win.develop.set_recipe({**rec, "palette": [[0.2, 0.4, 0.8]], "palette_strength": 0.6, "palette_hardness": 12})
    got = win.develop.recipe()
    assert got["palette"] == [[0.2, 0.4, 0.8]]
    assert abs(got["palette_strength"] - 0.6) < 1e-6
    from qs.app.library_panel import BudgetMeter
    meter = BudgetMeter()
    meter.set_values(1152, 256, "foliage")
    fill = meter.fill_pos("tris")
    ticks = meter.tick_positions("tris")
    assert ticks["foliage"] < fill < ticks["modular"]
    legend = meter.legend_text()
    assert "hero" in legend and "foliage" in legend and "surface" in legend
    tex_groups = {row["cap"]: row["names"] for row in meter.collapsed_ticks("tex")}
    assert "hero" in tex_groups[2048]
    assert len(tex_groups[1024]) >= 2
    assert "hero" in meter.fits_classes()
    assert "surface" not in meter.fits_classes()
    win.view.shutdown_gl()
    win.close()
    app.processEvents()


def test_missing_packages_empty_when_installed():
    assert _missing_packages() == []
