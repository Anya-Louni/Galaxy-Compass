"""Latent interpolation between morphology classes.

Two things are produced for every path:

  frames   Decoded images along the path, written as a sprite sheet.

  trace    The trained C8-steerable classifier's class-probability vector
           evaluated on each decoded frame.

The second is what makes this an experiment rather than an animation. If the
latent space has learned morphology as a continuous quantity, the classifier
should hand off smoothly between classes along a path, and the handover
should happen at a sensible place. If the space were merely memorising, the
probabilities would jump discontinuously or pass through unrelated classes.
The trace is rendered alongside the animation so the viewer sees the
evidence and the picture together.

Anchors are class medoids: the real galaxy closest to its class centroid in
latent space. Every anchor is therefore an actual catalogued object, while
every intermediate frame is a model output and is labelled as such.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from .data import CLASS_NAMES, PROC, ROOT
from .models import EquivariantCNN, load_weights
from .vae import VAE

VAE_DIR = ROOT / "results" / "vae"
OUT = ROOT / "results" / "interp"

# A closed tour through morphology. It ends where it began so the animation
# loops seamlessly, and it visits the transitions that are physically
# interesting: round to flattened, flattened to spiral, spiral to edge-on.
LOOP = [2, 3, 5, 7, 6, 9, 8, 4, 2]

PAIRS = [
    (2, 5),   # round smooth -> barred spiral
    (8, 2),   # edge-on without bulge -> round smooth
    (7, 1),   # unbarred loose spiral -> merging
    (4, 6),   # cigar shaped -> unbarred tight spiral
]


def slerp(a: np.ndarray, b: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Spherical interpolation, which respects the roughly Gaussian shell
    that VAE latents occupy. Straight lines cut through the low-density
    interior and decode to washed-out images."""
    an, bn = a / np.linalg.norm(a), b / np.linalg.norm(b)
    omega = np.arccos(np.clip(an @ bn, -1, 1))
    if omega < 1e-6:
        return np.outer(1 - t, a) + np.outer(t, b)
    so = np.sin(omega)
    w0 = np.sin((1 - t) * omega) / so
    w1 = np.sin(t * omega) / so
    # Interpolate direction on the sphere, magnitude linearly.
    dirs = np.outer(w0, an) + np.outer(w1, bn)
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    mags = (1 - t) * np.linalg.norm(a) + t * np.linalg.norm(b)
    return dirs * mags[:, None]


def to_display(x: torch.Tensor, mean, std) -> np.ndarray:
    """Undo normalisation and return uint8 HWC images."""
    y = x * std.view(1, -1, 1, 1) + mean.view(1, -1, 1, 1)
    y = y.clamp(0, 1).permute(0, 2, 3, 1).numpy()
    return (y * 255).round().astype(np.uint8)


def sprite_sheet(frames: np.ndarray, cols: int = 12):
    n, h, w, c = frames.shape
    rows = int(np.ceil(n / cols))
    sheet = np.zeros((rows * h, cols * w, c), dtype=np.uint8)
    for i in range(n):
        r, cc = divmod(i, cols)
        sheet[r * h : (r + 1) * h, cc * w : (cc + 1) * w] = frames[i]
    return sheet, rows, cols


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--per-segment", type=int, default=16)
    p.add_argument("--clf", default="")
    a = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    labels = np.load(PROC / "labels.npy")
    meta = dict(np.load(PROC / "meta.npz"))
    mean = torch.tensor(meta["mean"], dtype=torch.float32)
    std = torch.tensor(meta["std"], dtype=torch.float32)

    blob = torch.load(VAE_DIR / "vae.pt", map_location="cpu", weights_only=False)
    vae = VAE(blob["latent"])
    vae.load_state_dict(blob["model"])
    vae.eval()

    lat = np.load(VAE_DIR / "latents.npy")

    clf = None
    if a.clf:
        cb = torch.load(a.clf, map_location="cpu", weights_only=False)
        clf = EquivariantCNN(group="C8")
        load_weights(clf, cb)
        print(f"loaded classifier {a.clf}")

    # Class medoids: the real galaxy nearest its class centroid.
    anchors, anchor_idx = {}, {}
    for c in range(len(CLASS_NAMES)):
        m = labels == c
        sub = lat[m]
        centroid = sub.mean(0)
        j = int(np.argmin(np.linalg.norm(sub - centroid, axis=1)))
        gi = int(np.where(m)[0][j])
        anchors[c] = lat[gi]
        anchor_idx[c] = gi
    print("class medoids selected")

    def render(seq, name, closed):
        pts, segs = [], []
        t = np.linspace(0, 1, a.per_segment, endpoint=False)
        for i in range(len(seq) - 1):
            z = slerp(anchors[seq[i]], anchors[seq[i + 1]], t)
            pts.append(z)
            segs += [(seq[i], seq[i + 1])] * a.per_segment
        if not closed:
            # Append the reverse so a non-closed path still loops.
            fwd = np.concatenate(pts)
            pts = [fwd, fwd[::-1]]
            segs = segs + segs[::-1]
        z = np.concatenate(pts).astype(np.float32)

        with torch.no_grad():
            dec = vae.dec(torch.from_numpy(z))
            frames = to_display(dec, mean, std)
            probs = None
            if clf is not None:
                probs = torch.softmax(clf(dec), dim=1).numpy()

        sheet, rows, cols = sprite_sheet(frames)
        from PIL import Image

        Image.fromarray(sheet).save(OUT / f"{name}.png", optimize=True)

        rec = {
            "name": name,
            "n_frames": int(len(frames)),
            "frame_size": int(frames.shape[1]),
            "rows": rows,
            "cols": cols,
            "closed_loop": closed,
            "anchor_classes": [CLASS_NAMES[c] for c in seq],
            "anchor_indices": [anchor_idx[c] for c in seq],
            "segments": [[int(u), int(v)] for u, v in segs],
            "class_probabilities": probs.round(4).tolist() if probs is not None else None,
        }
        (OUT / f"{name}.json").write_text(json.dumps(rec))
        print(f"  {name}: {len(frames)} frames, sheet {sheet.shape[1]}x{sheet.shape[0]}")
        return rec

    index = {"class_names": CLASS_NAMES, "paths": []}
    index["paths"].append(render(LOOP, "loop_morphology", True))
    for u, v in PAIRS:
        index["paths"].append(
            render([u, v], f"pair_{u}_{v}", False)
        )

    (OUT / "index.json").write_text(json.dumps(index, indent=2))
    print(f"wrote {OUT / 'index.json'}")


if __name__ == "__main__":
    main()
