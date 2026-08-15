"""Package model outputs into assets the static site can load.

Produces:

  atlas_*.jpg   Texture atlas pages holding every galaxy thumbnail, in
                catalogue order, so a WebGL point can address its own image
                by index arithmetic alone with no per-object lookup.

  points.json   Per-galaxy UMAP position, Galaxy Zoo label, anomaly scores
                and sky coordinates.

  metrics.json  Every quantitative claim the page makes, copied from the
                files the experiments wrote, so the page never hard-codes a
                number that is not backed by a result file.

Thumbnails are the same 65x65 centre crop the network sees, resampled to the
tile size. No contrast stretch beyond what the survey pipeline already
applied is added, so what the atlas shows is what the model was given.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import CLASS_NAMES, INPUT, PROC, ROOT, STORE

RESULTS = ROOT / "results"
MAX_DIM = 4096


def build_atlas(images: np.ndarray, tile: int, out: Path, quality: int):
    n = len(images)
    cols = MAX_DIM // tile
    rows_per_page = MAX_DIM // tile
    per_page = cols * rows_per_page
    pages = math.ceil(n / per_page)
    o = (STORE - INPUT) // 2

    info = []
    for p in range(pages):
        s = p * per_page
        e = min(s + per_page, n)
        cnt = e - s
        rows = math.ceil(cnt / cols)
        sheet = Image.new("RGB", (cols * tile, rows * tile), (0, 0, 0))
        for i in range(s, e):
            k = i - s
            r, c = divmod(k, cols)
            im = Image.fromarray(images[i, o : o + INPUT, o : o + INPUT])
            if tile != INPUT:
                im = im.resize((tile, tile), Image.LANCZOS)
            sheet.paste(im, (c * tile, r * tile))
        f = out / f"atlas_{p}.jpg"
        sheet.save(f, quality=quality, optimize=True, progressive=True)
        mb = f.stat().st_size / 1e6
        info.append({"page": p, "file": f.name, "count": cnt, "rows": rows,
                     "width": sheet.width, "height": sheet.height, "mb": round(mb, 2)})
        print(f"  {f.name}: {sheet.width}x{sheet.height}, {cnt} tiles, {mb:.2f} MB")
    return {"tile": tile, "cols": cols, "per_page": per_page,
            "pages": pages, "n": n, "files": info}


def read_json(p: Path):
    return json.loads(p.read_text()) if p.exists() else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tile", type=int, default=48)
    ap.add_argument("--quality", type=int, default=86)
    ap.add_argument("--outdir", default=str(ROOT / "web" / "assets"))
    a = ap.parse_args()

    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)

    images = np.load(PROC / "images_u8.npy")
    labels = np.load(PROC / "labels.npy")
    meta = dict(np.load(PROC / "meta.npz"))
    manifest = read_json(PROC / "manifest.json")

    print(f"building atlas at {a.tile}px ...")
    atlas = build_atlas(images, a.tile, out, a.quality)
    total_mb = sum(f["mb"] for f in atlas["files"])
    print(f"  atlas total {total_mb:.2f} MB")

    ssl_dir = RESULTS / "ssl"
    xy = np.load(ssl_dir / "umap2d.npy")
    # Normalise to a centred unit box so the viewer needs no data-dependent
    # camera setup.
    c = (xy.max(0) + xy.min(0)) / 2
    s = (xy.max(0) - xy.min(0)).max() / 2
    xyn = (xy - c) / s

    anom_dir = RESULTS / "anomaly"
    anomalies = read_json(anom_dir / "anomalies.json")
    combined = np.load(anom_dir / "combined_score.npy")
    iso = np.load(anom_dir / "isolation.npy")
    rec = np.load(RESULTS / "vae" / "recon_error.npy")

    def rank01(x):
        o = np.argsort(x)
        r = np.empty(len(x))
        r[o] = np.arange(len(x))
        return r / (len(x) - 1)

    split = np.zeros(len(labels), dtype=np.uint8)
    split[meta["val"]] = 1
    split[meta["test"]] = 2

    # 92 of the 17,736 galaxies carry no measured redshift. JSON has no NaN
    # literal, so those are exported as the sentinel -1 and rendered as
    # "unknown" rather than silently becoming a spurious value of zero.
    z = np.asarray(meta["redshift"], dtype=np.float64)
    n_missing_z = int((~np.isfinite(z)).sum())
    z = np.where(np.isfinite(z), z, -1.0)

    points = {
        "n": int(len(labels)),
        "class_names": CLASS_NAMES,
        "x": np.round(xyn[:, 0], 4).tolist(),
        "y": np.round(xyn[:, 1], 4).tolist(),
        "label": labels.astype(int).tolist(),
        "split": split.tolist(),
        "anomaly": np.round(combined, 4).tolist(),
        "recon_rank": np.round(rank01(rec), 4).tolist(),
        "iso_rank": np.round(rank01(iso), 4).tolist(),
        "ra": np.round(meta["ra"], 5).tolist(),
        "dec": np.round(meta["dec"], 5).tolist(),
        "redshift": np.round(z, 5).tolist(),
        "n_missing_redshift": n_missing_z,
    }
    # allow_nan=False turns any surviving non-finite value into a loud error
    # rather than emitting a NaN literal that no JSON parser will accept.
    (out / "points.json").write_text(
        json.dumps(points, separators=(",", ":"), allow_nan=False)
    )
    print(f"  points.json: {(out / 'points.json').stat().st_size / 1e6:.2f} MB")

    metrics = {
        "dataset": manifest,
        "equivariance_audit": read_json(RESULTS / "equivariance_audit.json"),
        "sweep": read_json(RESULTS / "sweep_summary.json"),
        "rotation_robustness": read_json(RESULTS / "rotation_robustness.json"),
        "tta": (lambda t: {"summary": t["summary"], "verdict": t["verdict"]}
                if t else None)(read_json(RESULTS / "tta_baseline.json")),
        "ssl": read_json(ssl_dir / "embedding_metrics.json"),
        "ssl_training": read_json(ssl_dir / "simclr_train.json"),
        "vae_training": read_json(RESULTS / "vae" / "vae_train.json"),
        "anomaly_composition": anomalies["composition"] if anomalies else None,
        "anomaly_rank_correlation": anomalies["signal_rank_correlation"] if anomalies else None,
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=1))
    print(f"  metrics.json written")

    if anomalies:
        (out / "anomalies.json").write_text(
            json.dumps({"composition": anomalies["composition"],
                        "entries": anomalies["entries"]},
                       separators=(",", ":"), allow_nan=False)
        )
        print(f"  anomalies.json: {len(anomalies['entries'])} entries")

    for sub in ("interp", "rotation_demo"):
        src = RESULTS / sub
        if not src.exists():
            print(f"  skipped {sub} (not generated yet)")
            continue
        dst = out / sub
        dst.mkdir(exist_ok=True)
        n = 0
        for f in src.glob("*"):
            if f.is_file():
                shutil.copy2(f, dst / f.name)
                n += 1
        print(f"  copied {n} {sub} asset(s)")

    (out / "atlas.json").write_text(json.dumps(atlas, indent=1))
    print(f"\nassets written to {out}")


if __name__ == "__main__":
    main()
