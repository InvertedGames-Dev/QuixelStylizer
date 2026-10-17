"""qs CLI: stylize, batch, render, serve."""
from __future__ import annotations

import argparse
import os
import sys

from qs.logging import log_line, log_start, phase
from qs.version import TOOL_VERSION


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(prog="qs", description="QuixelStylizer")
    parser.add_argument("--gui", action="store_true", help="open the desktop window")
    sub = parser.add_subparsers(dest="cmd")

    p_st = sub.add_parser("stylize", help="stylize one folder")
    p_st.add_argument("folder")
    p_st.add_argument("--out")
    p_st.add_argument("--size", type=int, default=1024)
    p_st.add_argument("--preset")

    p_batch = sub.add_parser("batch", help="stylize many folders")
    p_batch.add_argument("folders", nargs="+")
    p_batch.add_argument("--out")
    p_batch.add_argument("--recursive", action="store_true")

    p_render = sub.add_parser("render", help="render a ground-truth PNG")
    p_render.add_argument("--out", required=True)
    p_render.add_argument("--hdri")
    p_render.add_argument("--width", type=int, default=512)
    p_render.add_argument("--height", type=int, default=512)

    p_serve = sub.add_parser("serve", help="start the GUI")

    p_bl = sub.add_parser("blender", help="start the warm Blender server")
    p_bl.add_argument("--port", type=int, default=18765)

    args = parser.parse_args(argv)
    if args.gui or args.cmd in (None, "serve"):
        log_start("gui")
        from qs.app.window import run_app
        return run_app()
    if args.cmd == "stylize":
        return _stylize(args)
    if args.cmd == "batch":
        return _batch(args)
    if args.cmd == "render":
        from qs.core.exodus_look import render_png
        phase("render")
        render_png(args.out, width=args.width, height=args.height, hdri=args.hdri)
        return 0
    if args.cmd == "blender":
        from qs.blender import client
        c = client()
        c.port = args.port
        if not c.start():
            print("could not start Blender", file=sys.stderr)
            return 2
        print(f"blender server on 127.0.0.1:{c.port}")
        return 0
    parser.print_help()
    return 1


def _stylize(args):
    from qs.core.maps import find_sets
    from qs.core.recipe import DEFAULTS, Recipe
    from qs.core.stylize import load_preset, process_set
    phase("stylize", args.folder)
    st = dict(DEFAULTS, size=args.size)
    if args.preset:
        st.update(load_preset(args.preset))
        st["size"] = args.size or st["size"]
    sets = find_sets([args.folder], recursive=False)
    if not sets:
        print(f"no material set in {args.folder}", file=sys.stderr)
        return 2
    for s in sets:
        process_set(s, st, write=True, out_dir=args.out)
    return 0


def _batch(args):
    from qs.core.maps import find_sets
    from qs.core.recipe import DEFAULTS
    from qs.core.stylize import process_set
    phase("batch")
    sets = find_sets(args.folders, recursive=args.recursive)
    for s in sets:
        dest = os.path.join(args.out, s["name"]) if args.out else None
        process_set(s, dict(DEFAULTS), write=True, out_dir=dest)
        print(s["name"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
