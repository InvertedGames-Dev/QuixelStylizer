"""Entry next to QuixelStylizer.exe. Reports missing packages in a dialog."""
from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
os.chdir(ROOT)


if __name__ == "__main__":
    from qs.cli import main
    raise SystemExit(main())
