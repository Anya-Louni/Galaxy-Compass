"""Galaxy10 DECaLS data pipeline.

Source dataset
--------------
Galaxy10 DECaLS (Leung & Bovy, astroNN), Zenodo record 10845026.
17,736 RGB images, 256x256, composed from DESI Legacy Imaging Surveys
g/r/z bands, labelled into 10 morphology classes from Galaxy Zoo votes.

The HDF5 file exposes: images (17736, 256, 256, 3) uint8, ans (labels),
ra, dec, redshift, pxscale.

Geometry
--------
Images are stored at 95x95 but the network sees 65x65. The margin is not
waste, it is what makes rotation augmentation honest: rotating a 95x95
frame by any angle leaves the central 65x65 fully covered by real sky,
since 95/sqrt(2) = 67.2 > 65. Without the margin, arbitrary rotations drag
undefined corners into the field, and a network can learn to read the
corner artefact instead of the morphology.

65 is odd on purpose. A quarter turn of an odd grid fixes the centre pixel
and maps the stride-2 subsampling lattice to itself, which is what lets the
steerable network stay equivariant end to end rather than only layer by
layer. See scripts/probe_pooling.py and the note in src/models.py.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

# Class names as published by astroNN for the DECaLS variant, with the
# per-class counts stated in the dataset documentation. The counts are
# asserted at preprocessing time so a corrupted or substituted file fails
# loudly instead of silently changing the science.
CLASS_NAMES = [
    "Disturbed",
    "Merging",
    "Round Smooth",
    "In-between Round Smooth",
    "Cigar Shaped Smooth",
    "Barred Spiral",
    "Unbarred Tight Spiral",
    "Unbarred Loose Spiral",
    "Edge-on without Bulge",
    "Edge-on with Bulge",
]

EXPECTED_COUNTS = [1081, 1853, 2645, 2027, 334, 2043, 1829, 2628, 1423, 1873]
EXPECTED_TOTAL = 17736
EXPECTED_SHA256 = "19aefc477c41bb7f77ff07599a6b82a038dc042f889a111b0d4d98bb755c1571"

# The DECaLS cutouts are 256 px at 0.262"/px (a 67" field). Trimming the
# outer border drops mostly empty sky before downsampling; the value is
# checked against measured light profiles in scripts/inspect_data.py.
CROP = 224
STORE = 95
INPUT = 65

SEED = 42
TEST_FRAC = 0.20
VAL_FRAC = 0.10

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw" / "Galaxy10_DECals.h5"
PROC = ROOT / "data" / "processed"


def sha256(path: Path, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _resize_block(block: np.ndarray, size: int) -> np.ndarray:
    """Downsample a uint8 NHWC block to size x size.

    Uses exact area averaging when the ratio is integral (the correct
    operation for binning photon counts, with no ringing around bright
    cores) and Lanczos otherwise.
    """
    n, h, w, c = block.shape
    if h == w and h % size == 0:
        f = h // size
        return (
            block.reshape(n, size, f, size, f, c)
            .mean(axis=(2, 4))
            .round()
            .clip(0, 255)
            .astype(np.uint8)
        )
    from PIL import Image

    out = np.empty((n, size, size, c), dtype=np.uint8)
    for i in range(n):
        out[i] = np.asarray(
            Image.fromarray(block[i]).resize((size, size), Image.LANCZOS)
        )
    return out


def preprocess(verify_hash: bool = True) -> None:
    """Read the raw HDF5 once and write processed arrays to data/processed."""
    import h5py

    PROC.mkdir(parents=True, exist_ok=True)

    if verify_hash:
        print("verifying sha256 of source file ...")
        digest = sha256(RAW)
        if digest.lower() != EXPECTED_SHA256:
            raise RuntimeError(
                f"SHA256 mismatch for {RAW}\n  got      {digest}\n"
                f"  expected {EXPECTED_SHA256}"
            )
        print(f"  ok: {digest}")

    with h5py.File(RAW, "r") as f:
        labels = np.asarray(f["ans"][:], dtype=np.int64)
        ra = np.asarray(f["ra"][:], dtype=np.float64)
        dec = np.asarray(f["dec"][:], dtype=np.float64)
        redshift = np.asarray(f["redshift"][:], dtype=np.float64)
        n = labels.shape[0]

        if n != EXPECTED_TOTAL:
            raise RuntimeError(f"expected {EXPECTED_TOTAL} rows, found {n}")
        counts = np.bincount(labels, minlength=10).tolist()
        if counts != EXPECTED_COUNTS:
            raise RuntimeError(
                f"class counts differ from published values\n"
                f"  got      {counts}\n  expected {EXPECTED_COUNTS}"
            )
        print(f"row count and per-class counts match published values (n={n})")

        lo = (256 - CROP) // 2
        hi = lo + CROP
        images = np.empty((n, STORE, STORE, 3), dtype=np.uint8)

        step = 512
        dset = f["images"]
        for s in range(0, n, step):
            e = min(s + step, n)
            block = np.asarray(dset[s:e, lo:hi, lo:hi, :], dtype=np.uint8)
            images[s:e] = _resize_block(block, STORE)
            print(f"  resized {e}/{n}", end="\r", flush=True)
        print()

    rng = np.random.default_rng(SEED)
    idx_train, idx_val, idx_test = [], [], []
    for c in range(10):
        idx = np.where(labels == c)[0]
        rng.shuffle(idx)
        n_c = len(idx)
        n_test = int(round(TEST_FRAC * n_c))
        n_val = int(round(VAL_FRAC * n_c))
        idx_test.append(idx[:n_test])
        idx_val.append(idx[n_test : n_test + n_val])
        idx_train.append(idx[n_test + n_val :])

    splits = {
        "train": np.sort(np.concatenate(idx_train)),
        "val": np.sort(np.concatenate(idx_val)),
        "test": np.sort(np.concatenate(idx_test)),
    }

    # Channel statistics come from the training split only, and from the
    # central region the network actually sees, so neither test information
    # nor sky border leaks into normalisation.
    o = (STORE - INPUT) // 2
    tr = images[splits["train"]][:, o : o + INPUT, o : o + INPUT, :]
    tr = tr.astype(np.float32) / 255.0
    mean = tr.mean(axis=(0, 1, 2))
    std = tr.std(axis=(0, 1, 2))
    del tr

    np.save(PROC / "images_u8.npy", images)
    np.save(PROC / "labels.npy", labels)
    np.savez(
        PROC / "meta.npz",
        ra=ra,
        dec=dec,
        redshift=redshift,
        train=splits["train"],
        val=splits["val"],
        test=splits["test"],
        mean=mean,
        std=std,
    )

    manifest = {
        "source": "Galaxy10 DECaLS (astroNN), Zenodo record 10845026",
        "sha256": EXPECTED_SHA256,
        "n_total": int(n),
        "crop_from_256": CROP,
        "stored_size": STORE,
        "network_input_size": INPUT,
        "seed": SEED,
        "class_names": CLASS_NAMES,
        "class_counts": counts,
        "split_sizes": {k: int(len(v)) for k, v in splits.items()},
        "channel_mean": [float(x) for x in mean],
        "channel_std": [float(x) for x in std],
    }
    (PROC / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest["split_sizes"], indent=2))
    print(f"wrote {PROC}")


def load():
    """Return (images_uint8_NHWC at STORE resolution, labels, meta dict)."""
    images = np.load(PROC / "images_u8.npy")
    labels = np.load(PROC / "labels.npy")
    meta = dict(np.load(PROC / "meta.npz"))
    return images, labels, meta


if __name__ == "__main__":
    preprocess()
