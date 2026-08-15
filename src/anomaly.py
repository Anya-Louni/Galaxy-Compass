"""Rank the catalogue by strangeness.

Two independent notions of "unusual" are combined, because each alone has a
well known failure mode:

  reconstruction error   How poorly the VAE redraws the galaxy. Sensitive to
                         objects with structure the model never learned to
                         express. Fails by also firing on anything merely
                         bright, large or noisy.

  local isolation        Mean cosine distance to the k nearest neighbours in
                         the self-supervised embedding. Sensitive to objects
                         with no close analogues in the survey. Fails by
                         ranking the sparse tail of common classes highly.

The two failure modes are largely uncorrelated, so an object scoring highly
on both is a stronger candidate than one scoring highly on either. The
combined rank is the mean of the two percentile ranks; the individual ranks
are kept so a reader can see which signal drove each detection. Local
Outlier Factor is computed alongside as an established reference method.

This mirrors what anomaly searches in survey astronomy actually do. The
output is a candidate list for human inspection, not a claim of discovery,
and every entry carries its sky coordinates so it can be checked against the
survey imaging directly.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors

from .data import CLASS_NAMES, PROC, ROOT
from .vae import VAE, encode_dataset

SSL = ROOT / "results" / "ssl"
VAE_DIR = ROOT / "results" / "vae"
OUT = ROOT / "results" / "anomaly"

# DESI Legacy Imaging Surveys sky browser. Every candidate links to the real
# image at its coordinates so the gallery can be verified independently.
VIEWER = "https://www.legacysurvey.org/viewer?ra={ra:.6f}&dec={dec:.6f}&layer=ls-dr9&zoom=14"


def percentile_rank(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x)
    r = np.empty(len(x), dtype=np.float64)
    r[order] = np.arange(len(x))
    return r / (len(x) - 1)


def isolation_score(z: np.ndarray, k: int = 20) -> np.ndarray:
    """Mean cosine distance to the k nearest neighbours."""
    zn = z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-9)
    nn = NearestNeighbors(n_neighbors=k + 1, metric="cosine").fit(zn)
    d, _ = nn.kneighbors(zn)
    return d[:, 1:].mean(axis=1)  # drop self-match


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--top", type=int, default=100)
    p.add_argument("--k", type=int, default=20)
    a = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    images = np.load(PROC / "images_u8.npy")
    labels = np.load(PROC / "labels.npy")
    meta = dict(np.load(PROC / "meta.npz"))
    imgs = torch.from_numpy(images).permute(0, 3, 1, 2).contiguous()
    mean = torch.tensor(meta["mean"], dtype=torch.float32)
    std = torch.tensor(meta["std"], dtype=torch.float32)

    blob = torch.load(VAE_DIR / "vae.pt", map_location="cpu", weights_only=False)
    model = VAE(blob["latent"])
    model.load_state_dict(blob["model"])
    print(f"loaded VAE from step {blob['step']}")

    print("computing latents and reconstruction error ...")
    mu, rec_err = encode_dataset(model, imgs, mean, std)
    np.save(VAE_DIR / "latents.npy", mu)
    np.save(VAE_DIR / "recon_error.npy", rec_err)

    z = np.load(SSL / "embeddings.npy")
    print("computing local isolation in the self-supervised embedding ...")
    iso = isolation_score(z, k=a.k)

    zn = z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-9)
    lof = -LocalOutlierFactor(n_neighbors=a.k, metric="cosine").fit(zn).negative_outlier_factor_

    r_rec, r_iso = percentile_rank(rec_err), percentile_rank(iso)
    combined = 0.5 * (r_rec + r_iso)

    corr = float(np.corrcoef(r_rec, r_iso)[0, 1])
    print(f"  rank correlation between the two signals: {corr:.3f}")

    order = np.argsort(-combined)[: a.top]
    ra, dec, zred = meta["ra"], meta["dec"], meta["redshift"]

    # 92 galaxies in the catalogue have no measured redshift. JSON has no NaN
    # literal, so a bare float() here produces a file no browser will parse.
    def z_or_none(v):
        v = float(v)
        return v if np.isfinite(v) else None

    entries = []
    for rank, i in enumerate(order, 1):
        entries.append(
            {
                "rank": rank,
                "index": int(i),
                "label": int(labels[i]),
                "class_name": CLASS_NAMES[int(labels[i])],
                "ra": float(ra[i]),
                "dec": float(dec[i]),
                "redshift": z_or_none(zred[i]),
                "recon_error": float(rec_err[i]),
                "recon_percentile": float(r_rec[i]),
                "isolation": float(iso[i]),
                "isolation_percentile": float(r_iso[i]),
                "lof": float(lof[i]),
                "combined_score": float(combined[i]),
                "viewer_url": VIEWER.format(ra=float(ra[i]), dec=float(dec[i])),
            }
        )

    # Class composition of the candidate list versus the parent catalogue,
    # which shows whether the ranking is just rediscovering a rare class.
    top_counts = np.bincount(labels[order], minlength=10)
    all_counts = np.bincount(labels, minlength=10)
    composition = [
        {
            "class_name": CLASS_NAMES[c],
            "in_top": int(top_counts[c]),
            "expected_if_random": float(a.top * all_counts[c] / len(labels)),
            "enrichment": float(
                (top_counts[c] / a.top) / (all_counts[c] / len(labels))
            ),
        }
        for c in range(10)
    ]

    res = {
        "n_total": int(len(labels)),
        "top_n": a.top,
        "k_neighbours": a.k,
        "signal_rank_correlation": corr,
        "composition": composition,
        "entries": entries,
    }
    # allow_nan=False makes any surviving non-finite value a loud failure here
    # rather than a silently unparseable asset later.
    (OUT / "anomalies.json").write_text(json.dumps(res, indent=2, allow_nan=False))
    np.save(OUT / "combined_score.npy", combined.astype(np.float32))
    np.save(OUT / "isolation.npy", iso.astype(np.float32))

    print(f"\n  top-{a.top} class enrichment (vs parent catalogue):")
    for c in sorted(composition, key=lambda d: -d["enrichment"])[:5]:
        print(
            f"    {c['class_name']:<26} {c['in_top']:3d} found, "
            f"{c['expected_if_random']:5.1f} expected, {c['enrichment']:5.2f}x"
        )
    print(f"\nwrote {OUT / 'anomalies.json'}")


if __name__ == "__main__":
    main()
