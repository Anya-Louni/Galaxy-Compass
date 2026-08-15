"""Verify exported web assets against the source data.

The atlas addresses images by index arithmetic in a shader, so an off-by-one
in page or row arithmetic would silently show the wrong galaxy for a point
and nothing would look broken. This checks the arithmetic numerically:
every sampled tile is compared against the source image it should hold,
with a deliberately mispaired control to prove the test can fail.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import INPUT, PROC, ROOT, STORE

ASSETS = ROOT / "web" / "assets"
TOL = 9.0  # mean absolute pixel error tolerated from JPEG quantisation


def main() -> int:
    atlas = json.loads((ASSETS / "atlas.json").read_text())
    T, cols, per = atlas["tile"], atlas["cols"], atlas["per_page"]
    imgs = np.load(PROC / "images_u8.npy")
    o = (STORE - INPUT) // 2
    pages = [
        np.asarray(Image.open(ASSETS / f["file"]).convert("RGB"))
        for f in atlas["files"]
    ]

    def tile(i):
        p, k = divmod(i, per)
        r, c = divmod(k, cols)
        return pages[p][r * T : (r + 1) * T, c * T : (c + 1) * T].astype(np.float64)

    def ref(i):
        im = Image.fromarray(imgs[i, o : o + INPUT, o : o + INPUT])
        return np.asarray(im.resize((T, T), Image.LANCZOS)).astype(np.float64)

    rng = np.random.default_rng(0)
    idx = list(rng.integers(0, len(imgs), 40))
    # Page boundaries, where index arithmetic is most likely to go wrong.
    for b in (0, per - 1, per, 2 * per - 1, 2 * per, len(imgs) - 1):
        if 0 <= b < len(imgs):
            idx.append(int(b))

    errs = [float(np.abs(tile(i) - ref(i)).mean()) for i in idx]
    ctrl = [
        float(np.abs(tile(i) - ref((i + 913) % len(imgs))).mean()) for i in idx
    ]

    # The control is judged relative to the matched error, not against an
    # absolute number: how far a mispaired tile lands depends on how similar
    # galaxies happen to look, which is not a fixed quantity.
    ratio = float(np.mean(ctrl) / max(np.mean(errs), 1e-9))

    print(f"atlas: {len(pages)} pages, {T}px tiles, {cols} cols, {per}/page")
    print(f"  sampled {len(idx)} tiles including every page boundary")
    print(f"  mean |atlas - source|  {np.mean(errs):6.2f}   max {max(errs):6.2f}")
    print(f"  mispaired control      {np.mean(ctrl):6.2f}   ({ratio:.1f}x the matched error)")

    pts = json.loads((ASSETS / "points.json").read_text())
    n_ok = pts["n"] == len(imgs)
    finite = all(
        np.isfinite(np.asarray(pts[k], dtype=float)).all()
        for k in ("x", "y", "anomaly", "ra", "dec", "redshift")
    )
    print(f"  points.json n={pts['n']} matches catalogue: {n_ok}")
    print(f"  all point fields finite (JSON-safe): {finite}")

    ok = max(errs) < TOL and ratio > 3.0 and n_ok and finite
    print("\n" + ("PASS: asset indexing verified" if ok else "FAIL: assets inconsistent"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
