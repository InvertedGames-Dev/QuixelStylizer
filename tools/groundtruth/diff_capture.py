"""Compare a Unreal HighResShot of the calibration sphere to the stylizer render."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from qs.core.exodus_look import render_sphere  # noqa: E402
from qs.core.io import imwrite, to_u8  # noqa: E402


def load(path):
    im = Image.open(path).convert("RGB")
    return np.asarray(im, np.float32) / 255.0


def sphere_mask(h, w):
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    u = (xx + 0.5) / w * 2 - 1
    v = 1 - (yy + 0.5) / h * 2
    u *= w / max(h, 1)
    return (u * u + v * v) <= 1.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("ue_png")
    p.add_argument("--out", default=str(Path(__file__).with_name("stylizer_sphere.png")))
    args = p.parse_args()
    ue = load(args.ue_png)
    h, w = ue.shape[:2]
    ours = render_sphere(w, h, smaa=True)
    imwrite(args.out, to_u8(ours))
    mask = sphere_mask(h, w)
    err = np.abs(ue - ours)
    inside = err[mask]
    print(f"size {w}x{h}")
    print(f"mean abs (sphere) {float(inside.mean()):.5f}")
    print(f"max abs (sphere)  {float(inside.max()):.5f}")
    print(f"mean abs (full)   {float(err.mean()):.5f}")
    print(f"wrote {args.out}")
    print("Band: this is a look check, not pixel identity (Unreal shadows / captures differ).")


if __name__ == "__main__":
    main()
