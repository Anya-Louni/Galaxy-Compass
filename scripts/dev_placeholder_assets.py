"""Development only: build web assets before the models have finished.

Produces a real texture atlas from the real images, but substitutes a cheap
PCA-of-pixels layout for the UMAP coordinates and zeros for the model
metrics. This exists so the WebGL renderer can be debugged against real
data volumes while training is still running. Running the real
scripts/export_web.py afterwards overwrites everything here.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.export_web import build_atlas
from src.data import CLASS_NAMES, INPUT, PROC, ROOT, STORE

out = ROOT / "web" / "assets"
out.mkdir(parents=True, exist_ok=True)

images = np.load(PROC / "images_u8.npy")
labels = np.load(PROC / "labels.npy")
meta = dict(np.load(PROC / "meta.npz"))

print("building atlas (real images) ...")
atlas = build_atlas(images, 48, out, 86)
(out / "atlas.json").write_text(json.dumps(atlas, indent=1))

print("cheap PCA layout as a stand-in for UMAP ...")
o = (STORE - INPUT) // 2
small = images[:, o : o + INPUT : 4, o : o + INPUT : 4, :].mean(axis=3)
X = small.reshape(len(images), -1).astype(np.float32) / 255.0
X -= X.mean(0)
_, _, Vt = np.linalg.svd(X[:: 3], full_matrices=False)
xy = X @ Vt[:2].T
c = (xy.max(0) + xy.min(0)) / 2
s = (xy.max(0) - xy.min(0)).max() / 2
xy = (xy - c) / s

rng = np.random.default_rng(0)
fake = rng.random(len(labels))
split = np.zeros(len(labels), dtype=np.uint8)
split[meta["val"]] = 1
split[meta["test"]] = 2

points = {
    "n": int(len(labels)),
    "class_names": CLASS_NAMES,
    "x": np.round(xy[:, 0], 4).tolist(),
    "y": np.round(xy[:, 1], 4).tolist(),
    "label": labels.astype(int).tolist(),
    "split": split.tolist(),
    "anomaly": np.round(fake, 4).tolist(),
    "recon_rank": np.round(fake, 4).tolist(),
    "iso_rank": np.round(fake, 4).tolist(),
    "ra": np.round(meta["ra"], 5).tolist(),
    "dec": np.round(meta["dec"], 5).tolist(),
    "redshift": np.round(
        np.where(np.isfinite(meta["redshift"]), meta["redshift"], -1.0), 5
    ).tolist(),
}
(out / "points.json").write_text(
    json.dumps(points, separators=(",", ":"), allow_nan=False)
)

metrics = {
    "dataset": json.loads((PROC / "manifest.json").read_text()),
    "equivariance_audit": json.loads(
        (ROOT / "results" / "equivariance_audit.json").read_text()
    ),
    "sweep": None, "ssl": None, "ssl_training": None, "vae_training": None,
    "anomaly_composition": None, "anomaly_rank_correlation": None,
}
(out / "metrics.json").write_text(json.dumps(metrics, indent=1))
print(f"placeholder assets written to {out}  (PLACEHOLDER LAYOUT, not UMAP)")
