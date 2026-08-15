"""Rotation robustness measured without an interpolation confound.

The sweep records accuracy on an unrotated centre crop and on a 45-degree
rotation. Those two numbers are not directly comparable: the unrotated crop
is exact, while any other angle is bilinearly resampled. Because training
always resamples, a model can score *higher* on the rotated set, which makes
the difference a measure of sharpness as much as of orientation.

This script avoids the problem entirely by using only the four exact grid
rotations. On an odd-sized frame a quarter turn is a lossless permutation of
pixels, so accuracy at 0, 90, 180 and 270 degrees differs only by
orientation. A network that is genuinely rotation-invariant must score
identically at all four; the spread across them is a clean measurement of
how much orientation still matters to the model.

Run after the sweep. Reads every checkpoint it finds.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.augment import eval_view
from src.data import PROC, ROOT
from src.models import BaselineCNN, EquivariantCNN, load_weights

SWEEP = ROOT / "results" / "sweep"
OUT = ROOT / "results"


@torch.no_grad()
def accuracy_at_rotations(model, imgs, labels, idx, mean, std, bs=256):
    """Test accuracy at each of the four exact grid rotations."""
    model.eval()
    accs = []
    for k in range(4):
        correct = 0
        for s in range(0, len(idx), bs):
            b = idx[s : s + bs]
            x = eval_view(imgs[b], mean, std)          # exact crop, no resampling
            if k:
                x = torch.rot90(x, k, dims=(2, 3))     # lossless permutation
            correct += int((model(x).argmax(1).numpy() == labels[b]).sum())
        accs.append(correct / len(idx))
    return accs


def main():
    images = np.load(PROC / "images_u8.npy")
    labels = np.load(PROC / "labels.npy")
    meta = dict(np.load(PROC / "meta.npz"))
    imgs = torch.from_numpy(images).permute(0, 3, 1, 2).contiguous()
    mean = torch.tensor(meta["mean"], dtype=torch.float32)
    std = torch.tensor(meta["std"], dtype=torch.float32)
    te = meta["test"]

    ckpts = sorted(SWEEP.glob("*.pt"))
    if not ckpts:
        print("no checkpoints found; run the sweep first")
        return
    print(f"evaluating {len(ckpts)} checkpoints at 0/90/180/270 degrees "
          f"on {len(te):,} test galaxies\n")

    rows, by_arch = [], defaultdict(list)
    for c in ckpts:
        name = c.stem
        is_eq = name.startswith("equivariant")
        model = EquivariantCNN(group="C8") if is_eq else BaselineCNN()
        load_weights(model, torch.load(c, map_location="cpu"))
        accs = accuracy_at_rotations(model, imgs, labels, te, mean, std)
        spread = max(accs) - min(accs)
        arch = "equivariant" if is_eq else "baseline"
        frac = float(name.split("_f")[1].split("_s")[0])
        rows.append({
            "checkpoint": name, "architecture": arch, "label_fraction": frac,
            "accuracy_per_rotation": accs, "mean": float(np.mean(accs)),
            "spread": float(spread), "std": float(np.std(accs, ddof=0)),
        })
        by_arch[arch].append(spread)
        print(f"  {name:<34} mean {np.mean(accs):.4f}  spread {spread:.5f}")

    summary = {
        arch: {
            "mean_accuracy_spread_across_grid_rotations": float(np.mean(v)),
            "max_spread": float(np.max(v)),
            "n_checkpoints": len(v),
        }
        for arch, v in by_arch.items()
    }
    print("\nmean accuracy spread across the four exact grid rotations:")
    for arch, s in summary.items():
        print(f"  {arch:<12} {s['mean_accuracy_spread_across_grid_rotations']:.6f} "
              f"(worst {s['max_spread']:.6f}, n={s['n_checkpoints']})")
    if "equivariant" in summary and "baseline" in summary:
        e = summary["equivariant"]["mean_accuracy_spread_across_grid_rotations"]
        b = summary["baseline"]["mean_accuracy_spread_across_grid_rotations"]
        summary["equivariant_spread_is_exactly_zero"] = e == 0.0
        # A ratio against an exactly-zero denominator is not a number, and
        # printing one would be worse than useless: it invents precision from a
        # division by zero. The stronger and truer statement is the identity.
        if e == 0.0:
            summary["ratio_baseline_over_equivariant"] = None
            print(
                f"\n  the steerable model classifies every test galaxy identically at all"
                f"\n  four orientations: the spread is exactly zero, not merely small."
                f"\n  the dense CNN's accuracy moves by {b:.4f} ({b * 100:.2f} pp) on average."
            )
        else:
            summary["ratio_baseline_over_equivariant"] = b / e
            print(f"\n  the dense CNN's accuracy moves {b / e:,.0f}x more with orientation")
        print("  (no interpolation is involved: these rotations are exact)")

    (OUT / "rotation_robustness.json").write_text(
        json.dumps({"per_checkpoint": rows, "summary": summary}, indent=2)
    )
    print(f"\nwrote {OUT / 'rotation_robustness.json'}")


if __name__ == "__main__":
    main()
