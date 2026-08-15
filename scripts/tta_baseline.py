"""Test-time augmentation: the cheap way to buy rotation invariance.

The obvious objection to this whole project is that you do not need a
steerable architecture to get an invariant prediction. Take an ordinary CNN,
run it on every rotation of the input, and average the probabilities. The
average over a group orbit is exactly invariant, by construction, for the
same reason group pooling is. It costs nothing to implement and |G| times
more compute at inference.

If that closes the gap, the honest claim shrinks from "architectural
equivariance is worth several times the labels" to "...or you can pay 8x at
inference instead". So it has to be measured.

What is measured here
---------------------
For every checkpoint, predictions are averaged over the exact D4 orbit:
four quarter turns and their reflections, eight transforms in total. On an
odd-sized frame every one is a lossless permutation of pixels, so no
interpolation error enters and the comparison is clean.

  plain    a single forward pass                         1x inference
  C4-TTA   averaged over four rotations                  4x inference
  D4-TTA   averaged over four rotations and reflections  8x inference

Both architectures get the identical treatment. For the C8-steerable model
the rotation half of the orbit is redundant, since it is already invariant
to those, but the reflections are not in C8 and are a genuine augmentation
for it too. Giving TTA only to the baseline would be as unfair as
withholding it.

The question the numbers answer: does an averaged dense CNN reach the
steerable model's accuracy, and if so at what label budget and what
inference cost?
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.augment import eval_view
from src.data import PROC, ROOT
from src.models import BaselineCNN, EquivariantCNN, load_weights

SWEEP = ROOT / "results" / "sweep"
OUT = ROOT / "results"


def d4_transforms(x: torch.Tensor):
    """The eight elements of the dihedral group acting on a square grid.

    Each is an exact permutation of pixels: no resampling, no interpolation,
    no information lost. Yields (rotation index, flipped) with the tensor.
    """
    for k in range(4):
        r = torch.rot90(x, k, dims=(2, 3))
        yield k, False, r
        yield k, True, torch.flip(r, dims=(3,))


@torch.no_grad()
def orbit_probabilities(model, imgs, idx, mean, std, bs: int = 256):
    """Softmax outputs for every element of the D4 orbit. Shape (8, N, C)."""
    model.eval()
    chunks = []
    for s in range(0, len(idx), bs):
        b = idx[s : s + bs]
        x = eval_view(imgs[b], mean, std)  # exact centre crop
        per = [torch.softmax(model(t), dim=1) for _, _, t in d4_transforms(x)]
        chunks.append(torch.stack(per))
    return torch.cat(chunks, dim=1).numpy()


def score(probs: np.ndarray, y: np.ndarray) -> dict:
    pred = probs.argmax(1)
    return {
        "accuracy": float((pred == y).mean()),
        "macro_f1": float(f1_score(y, pred, average="macro")),
    }


def main():
    images = np.load(PROC / "images_u8.npy")
    labels = np.load(PROC / "labels.npy")
    meta = dict(np.load(PROC / "meta.npz"))
    imgs = torch.from_numpy(images).permute(0, 3, 1, 2).contiguous()
    mean = torch.tensor(meta["mean"], dtype=torch.float32)
    std = torch.tensor(meta["std"], dtype=torch.float32)
    te = meta["test"]
    y = labels[te]

    ckpts = sorted(SWEEP.glob("*.pt"))
    print(f"evaluating {len(ckpts)} checkpoints over the 8-element D4 orbit "
          f"on {len(te):,} test galaxies\n")

    rows = []
    for c in ckpts:
        name = c.stem
        is_eq = name.startswith("equivariant")
        model = EquivariantCNN(group="C8") if is_eq else BaselineCNN()
        load_weights(model, torch.load(c, map_location="cpu"))

        p = orbit_probabilities(model, imgs, te, mean, std)  # (8, N, C)
        # Index 0 is (rotation 0, unflipped): the ordinary single pass.
        plain = score(p[0], y)
        c4 = score(p[[0, 2, 4, 6]].mean(0), y)   # rotations only
        d4 = score(p.mean(0), y)                 # rotations and reflections

        frac = float(name.split("_f")[1].split("_s")[0])
        seed = int(name.split("_s")[-1])
        rows.append({
            "checkpoint": name,
            "architecture": "equivariant" if is_eq else "baseline",
            "label_fraction": frac, "seed": seed,
            "plain": plain, "c4_tta": c4, "d4_tta": d4,
        })
        print(f"  {name:<34} plain {plain['accuracy']:.4f}  "
              f"C4 {c4['accuracy']:.4f}  D4 {d4['accuracy']:.4f}")

    # ------------------------------------------------------------ aggregate
    agg = defaultdict(lambda: defaultdict(list))
    for r in rows:
        for mode in ("plain", "c4_tta", "d4_tta"):
            agg[(r["architecture"], r["label_fraction"])][mode].append(
                r[mode]["accuracy"]
            )

    fracs = sorted({r["label_fraction"] for r in rows})
    summary = {}
    print(f"\n{'':<14}{'labels':>8}{'plain':>10}{'C4-TTA':>10}{'D4-TTA':>10}"
          f"{'TTA gain':>10}")
    for arch in ("baseline", "equivariant"):
        for f in fracs:
            d = agg[(arch, f)]
            if not d:
                continue
            m = {k: float(np.mean(v)) for k, v in d.items()}
            s = {k: float(np.std(v, ddof=1)) if len(v) > 1 else 0.0
                 for k, v in d.items()}
            summary[f"{arch}_{f:g}"] = {"mean": m, "std": s, "n_seeds": len(d["plain"])}
            print(f"{arch:<14}{f * 100:>7.0f}%{m['plain']:>10.4f}"
                  f"{m['c4_tta']:>10.4f}{m['d4_tta']:>10.4f}"
                  f"{m['d4_tta'] - m['plain']:>+10.4f}")

    # ------------------------------------------------- the decisive question
    def mean_at(arch, f, mode):
        k = f"{arch}_{f:g}"
        return summary[k]["mean"][mode] if k in summary else None

    bl_d4_full = mean_at("baseline", 1.0, "d4_tta")
    eq_plain = [(f, mean_at("equivariant", f, "plain")) for f in fracs]

    def crossing(points, target):
        pts = [(f, v) for f, v in points if v is not None]
        if not pts or target is None:
            return None
        if pts[0][1] >= target:
            return pts[0][0]
        for i in range(1, len(pts)):
            if pts[i][1] >= target:
                (f0, v0), (f1, v1) = pts[i - 1], pts[i]
                if v1 == v0:
                    return f1
                t = (target - v0) / (v1 - v0)
                return float(np.exp(np.log(f0) + t * (np.log(f1) - np.log(f0))))
        return None

    cf_plain = crossing(eq_plain, mean_at("baseline", 1.0, "plain"))
    cf_tta = crossing(eq_plain, bl_d4_full)

    verdict = {
        "baseline_100pct_plain": mean_at("baseline", 1.0, "plain"),
        "baseline_100pct_d4_tta": bl_d4_full,
        "equivariant_100pct_plain": mean_at("equivariant", 1.0, "plain"),
        "equivariant_100pct_d4_tta": mean_at("equivariant", 1.0, "d4_tta"),
        "crossing_vs_plain_baseline": cf_plain,
        "crossing_vs_d4_tta_baseline": cf_tta,
        "label_efficiency_vs_plain": (1 / cf_plain) if cf_plain else None,
        "label_efficiency_vs_d4_tta": (1 / cf_tta) if cf_tta else None,
        "tta_inference_cost_multiplier": 8,
    }

    print("\n" + "=" * 68)
    print("Does averaging over rotations rescue the dense CNN?")
    print("=" * 68)
    print(f"  dense CNN, all labels, single pass       "
          f"{verdict['baseline_100pct_plain']:.4f}")
    print(f"  dense CNN, all labels, D4-TTA (8x cost)  "
          f"{verdict['baseline_100pct_d4_tta']:.4f}   "
          f"(+{(bl_d4_full - verdict['baseline_100pct_plain']) * 100:.2f} pp)")
    print(f"  steerable, all labels, single pass       "
          f"{verdict['equivariant_100pct_plain']:.4f}")
    print(f"  steerable, all labels, D4-TTA            "
          f"{verdict['equivariant_100pct_d4_tta']:.4f}")
    print()
    if cf_plain:
        print(f"  steerable matches the plain dense CNN at "
              f"{cf_plain * 100:.1f}% of labels ({1 / cf_plain:.1f}x)")
    if cf_tta:
        print(f"  steerable matches the D4-TTA dense CNN at "
              f"{cf_tta * 100:.1f}% of labels ({1 / cf_tta:.1f}x), "
              f"and the TTA model pays 8x inference")
    else:
        print("  the steerable model does NOT reach the D4-TTA dense CNN "
              "within the sampled range")

    (OUT / "tta_baseline.json").write_text(
        json.dumps({"per_checkpoint": rows, "summary": summary,
                    "verdict": verdict}, indent=2)
    )
    print(f"\nwrote {OUT / 'tta_baseline.json'}")


if __name__ == "__main__":
    main()
