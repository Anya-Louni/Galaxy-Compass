"""Precompute the rotation-response of both trained models.

For a handful of real test galaxies, both trained classifiers are evaluated
at 36 orientations spanning a full turn. The resulting probability vectors
let the web page animate a rotating galaxy beside two live bar charts: the
dense CNN's bars move as the galaxy turns, the steerable model's do not.

This is the project's central claim reduced to something a reader can watch
happen, and every number in it is a real model output on a real galaxy
rather than an illustration.

Note on fairness: the dense model was trained with full-circle rotation
augmentation, so any instability it shows here is what augmentation failed
to buy, not the result of withholding it.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.augment import eval_view
from src.data import CLASS_NAMES, PROC, ROOT
from src.models import BaselineCNN, EquivariantCNN, load_weights

SWEEP = ROOT / "results" / "sweep"
OUT = ROOT / "results" / "rotation_demo"
N_ANGLES = 36


def latest_ckpt(model: str, frac: float = 1.0):
    tag = "C8" if model == "equivariant" else "dense"
    c = sorted(SWEEP.glob(f"{model}_{tag}_f{frac:g}_s*.pt"))
    if not c:
        raise FileNotFoundError(f"no checkpoint for {model} at frac {frac}")
    return c[0]


@torch.no_grad()
def response(model, imgs_u8, mean, std, idx):
    """Probability vectors for each galaxy at each of N_ANGLES orientations."""
    model.eval()
    out = np.zeros((len(idx), N_ANGLES, len(CLASS_NAMES)), dtype=np.float32)
    for a in range(N_ANGLES):
        ang = 2 * math.pi * a / N_ANGLES
        x = eval_view(imgs_u8[idx], mean, std, angle=ang)
        out[:, a] = torch.softmax(model(x), dim=1).numpy()
    return out


@torch.no_grad()
def rotation_instability(model, imgs_u8, mean, std, idx, exact: bool, bs: int = 128):
    """Mean total-variation movement of the predicted distribution under rotation.

    For each galaxy the predicted distribution is evaluated at several
    orientations and compared against its own angular mean, giving
    0.5 * sum_c |p_c(theta) - mean_theta p_c|. Averaging over angles and
    galaxies yields one number in [0, 1]: how far the prediction moves when
    only the orientation changes.

    Total variation over the whole distribution is used rather than the
    true-class probability alone. A model that saturates at p = 1 has zero
    movement in the true class no matter how it behaves, so a true-class
    metric rewards overconfidence rather than invariance.

    exact=True uses only the four grid rotations, which are lossless pixel
    permutations on an odd-sized frame. That isolates the architecture from
    resampling: any movement is the model's, not the interpolator's.
    exact=False uses twelve evenly spaced angles, which require bilinear
    resampling and therefore also carry interpolation error, equally for
    both models.
    """
    model.eval()
    vals = []
    for s in range(0, len(idx), bs):
        b = idx[s : s + bs]
        if exact:
            x = eval_view(imgs_u8[b], mean, std)
            ps = torch.stack(
                [torch.softmax(model(torch.rot90(x, k, dims=(2, 3))), 1) for k in range(4)]
            )
        else:
            # force_resample keeps angle 0 on the same bilinear path as every
            # other angle, so this measures orientation and not sharpness.
            ps = torch.stack(
                [
                    torch.softmax(
                        model(
                            eval_view(
                                imgs_u8[b], mean, std,
                                angle=2 * math.pi * a / 12, force_resample=True,
                            )
                        ), 1
                    )
                    for a in range(12)
                ]
            )
        tv = 0.5 * (ps - ps.mean(0, keepdim=True)).abs().sum(-1)
        vals.append(tv.mean(0).numpy())
    return float(np.concatenate(vals).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--frac", type=float, default=1.0)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    images = np.load(PROC / "images_u8.npy")
    labels = np.load(PROC / "labels.npy")
    meta = dict(np.load(PROC / "meta.npz"))
    imgs = torch.from_numpy(images).permute(0, 3, 1, 2).contiguous()
    mean = torch.tensor(meta["mean"], dtype=torch.float32)
    std = torch.tensor(meta["std"], dtype=torch.float32)

    eq = EquivariantCNN(group="C8")
    load_weights(eq, torch.load(latest_ckpt("equivariant", a.frac), map_location="cpu"))
    bl = BaselineCNN()
    load_weights(bl, torch.load(latest_ckpt("baseline", a.frac), map_location="cpu"))
    print("loaded both classifiers at 100% labels")

    # Pick visually distinct, correctly classified test galaxies so the demo
    # is about stability rather than about a model being wrong.
    te = meta["test"]
    want = [5, 7, 2, 8, 1, 4]  # barred spiral, loose spiral, round, edge-on, merging, cigar
    chosen = []
    with torch.no_grad():
        for c in want[: a.n]:
            cand = te[labels[te] == c]
            x = eval_view(imgs[cand], mean, std)
            pe = torch.softmax(eq(x), 1)
            pb = torch.softmax(bl(x), 1)
            ok = (pe.argmax(1).numpy() == c) & (pb.argmax(1).numpy() == c)
            score = (pe[:, c] + pb[:, c]).numpy() * ok
            chosen.append(int(cand[int(np.argmax(score))]))
    chosen = np.array(chosen)
    print(f"selected galaxies: {chosen.tolist()}")

    re_ = response(eq, imgs, mean, std, chosen)
    rb = response(bl, imgs, mean, std, chosen)

    # Sprite sheet of the rotated views actually fed to the models.
    frames = []
    for a_i in range(N_ANGLES):
        ang = 2 * math.pi * a_i / N_ANGLES
        x = eval_view(imgs[chosen], mean, std, angle=ang)
        y = x * std.view(1, -1, 1, 1) + mean.view(1, -1, 1, 1)
        frames.append((y.clamp(0, 1).permute(0, 2, 3, 1).numpy() * 255).astype(np.uint8))
    frames = np.stack(frames, axis=1)  # (galaxy, angle, H, W, 3)

    g, na, h, w, _ = frames.shape
    sheet = np.zeros((g * h, na * w, 3), dtype=np.uint8)
    for i in range(g):
        for j in range(na):
            sheet[i * h : (i + 1) * h, j * w : (j + 1) * w] = frames[i, j]
    Image.fromarray(sheet).save(OUT / "rotation_frames.jpg", quality=90, optimize=True)

    # The headline stability numbers are measured over a large random sample of
    # the test split, not over the handful of galaxies the animation shows. The
    # displayed galaxies are chosen for visual variety; drawing a statistic from
    # them would be drawing it from a hand-picked sample.
    rng = np.random.default_rng(0)
    sample = te[rng.choice(len(te), min(400, len(te)), replace=False)]
    print(f"measuring rotation stability over {len(sample)} random test galaxies ...")
    stability = {}
    for nm, mdl in (("equivariant", eq), ("baseline", bl)):
        stability[nm] = {
            "tv_exact_grid_rotations": rotation_instability(
                mdl, imgs, mean, std, sample, exact=True
            ),
            "tv_continuous_rotations": rotation_instability(
                mdl, imgs, mean, std, sample, exact=False
            ),
        }
        print(
            f"    {nm:<12} exact {stability[nm]['tv_exact_grid_rotations']:.6f}   "
            f"continuous {stability[nm]['tv_continuous_rotations']:.6f}"
        )

    rec = {
        "n_angles": N_ANGLES,
        "frame_size": int(h),
        "class_names": CLASS_NAMES,
        "galaxies": [
            {
                "index": int(gi),
                "label": int(labels[gi]),
                "class_name": CLASS_NAMES[int(labels[gi])],
                "ra": float(meta["ra"][gi]),
                "dec": float(meta["dec"][gi]),
            }
            for gi in chosen
        ],
        "equivariant": re_.round(4).tolist(),
        "baseline": rb.round(4).tolist(),
        "n_stability_sample": int(len(sample)),
        "stability": stability,
    }
    (OUT / "rotation_demo.json").write_text(json.dumps(rec, separators=(",", ":")))
    e = stability["equivariant"]["tv_exact_grid_rotations"]
    b = stability["baseline"]["tv_exact_grid_rotations"]
    print(
        f"\n  on exact grid rotations the dense CNN's prediction moves "
        f"{b / max(e, 1e-12):,.0f}x more than the steerable model's"
    )
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
