"""Headless stylize library: maps in, maps out. No UI."""
from qs.core.recipe import Recipe, DEFAULTS
from qs.core.stylize import stylize, load_stage, paint_stage, finish_stage, process_set

__all__ = [
    "Recipe",
    "DEFAULTS",
    "stylize",
    "load_stage",
    "paint_stage",
    "finish_stage",
    "process_set",
]
