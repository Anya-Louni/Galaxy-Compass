"""Validate the preprocessing geometry against the data itself.

Two questions, answered with measurements rather than by eye:

  1. Does the 65x65 network input actually contain the galaxy, or is the
     crop clipping extended structure? Answered with the stacked radial
     surface-brightness profile: if the profile has fallen to the sky floor
     well inside the crop radius, the crop is safe.

  2. Does rotation augmentation ever pull undefined pixels into frame?
     Answered by rotating the stored frame through a full turn and checking
     the corner coverage of the cropped region.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import CLASS_NAMES, INPUT, PROC, ROOT, STORE, load

FIG = ROOT / "results" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

images, labels, meta = load()
print(f"stored images: {images.shape}  dtype={images.dtype}")

# ------------------------------------------------------- radial light profile
lum = images.astype(np.float32).mean(axis=3) / 255.0  # N, S, S
yy, xx = np.mgrid[0:STORE, 0:STORE]
c = (STORE - 1) / 2
r = np.sqrt((xx - c) ** 2 + (yy - c) ** 2)
rbin = r.astype(np.int32)
nb = rbin.max() + 1

prof = np.zeros(nb)
for b in range(nb):
    m = rbin == b
    if m.sum():
        prof[b] = lum[:, m].mean()

crop_r = (INPUT - 1) / 2
# Integrate only inside the circle inscribed in the stored frame. Beyond it
# the annuli are only partially sampled by the square image, and their large
# area would let pure sky noise dominate a cumulative sum.
r_max = int(STORE / 2)
sky = prof[r_max - 6 : r_max].mean()
peak = prof[0]
excess = np.clip(prof[: r_max + 1] - sky, 0, None)
w = np.array([(rbin == b).sum() for b in range(r_max + 1)], dtype=np.float64)
cum = np.cumsum(excess * w)
cum /= cum[-1]

enclosed = float(cum[int(crop_r)])
# Radius at which the surface-brightness excess has decayed to 2% of central.
frac_peak = excess / excess[0]
r_faint = float(np.argmax(frac_peak < 0.02))

print(f"  sky floor (normalised)             {sky:.4f}")
print(f"  central excess above sky           {excess[0]:.4f}")
print(f"  crop half-width                    {crop_r:.1f} px")
print(f"  radius where excess < 2% of peak   {r_faint:.1f} px")
print(f"  excess at crop edge                {frac_peak[int(crop_r)] * 100:.1f}% of peak")
print(f"  enclosed light within crop         {enclosed * 100:.1f}%")
print(
    "  VERDICT: crop retains the galaxy; residual flux at the edge is at sky level"
    if enclosed > 0.95
    else "  VERDICT: crop may be clipping extended light"
)
r99 = r_faint

# ----------------------------------------------------- rotation coverage check
# The stored frame must circumscribe the cropped square at every angle.
safe = STORE / math.sqrt(2.0)
print(f"\n  stored size {STORE}, inscribed circle diameter {safe:.2f}, crop {INPUT}")
print(
    "  VERDICT: rotation never introduces undefined pixels"
    if safe >= INPUT
    else "  VERDICT: rotation WILL introduce undefined corners"
)

# ------------------------------------------------------------------ montage
rng = np.random.default_rng(0)
ncol = 8
fig, axes = plt.subplots(10, ncol, figsize=(ncol * 1.15, 10 * 1.25))
o = (STORE - INPUT) // 2
for k in range(10):
    idx = np.where(labels == k)[0]
    pick = rng.choice(idx, ncol, replace=False)
    for j, i in enumerate(pick):
        ax = axes[k, j]
        ax.imshow(images[i, o : o + INPUT, o : o + INPUT])
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
    axes[k, 0].set_ylabel(
        f"{k}. {CLASS_NAMES[k]}", rotation=0, ha="right", va="center", fontsize=8
    )
fig.suptitle(
    f"Galaxy10 DECaLS, {INPUT}x{INPUT} network input (crop {STORE}->{INPUT})",
    fontsize=11,
)
fig.tight_layout(rect=(0, 0, 1, 0.98))
fig.savefig(FIG / "class_montage.png", dpi=130)
print(f"\nwrote {FIG / 'class_montage.png'}")

# ------------------------------------------------------------ profile figure
fig, ax = plt.subplots(figsize=(6, 3.6))
ax.plot(np.arange(nb), prof, lw=1.8, color="#4c9be8")
ax.axhline(sky, ls=":", lw=1, color="0.5", label="sky floor")
ax.axvline(crop_r, ls="--", lw=1.2, color="#e8734c", label=f"crop radius ({crop_r:.0f} px)")
ax.axvline(r99, ls="-.", lw=1.2, color="#7ce84c", label=f"excess < 2% of peak ({r99:.0f} px)")
ax.set_xlabel("radius from centre (px)")
ax.set_ylabel("mean normalised surface brightness")
ax.set_title(f"Stacked radial profile, all {len(images):,} galaxies")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(FIG / "radial_profile.png", dpi=140)
print(f"wrote {FIG / 'radial_profile.png'}")
