"""Embed the catalogue, then measure whether the unsupervised space is real.

A two-dimensional projection is a picture, not evidence. Three quantitative
checks are run on the full-dimensional representation before any projection
is made, so the atlas is illustrating a measured result rather than standing
in for one:

  k-NN probe        Fit a k-nearest-neighbour classifier on the training
                    embeddings and score it on the held-out test split. No
                    gradients, no fine-tuning, no labels used to build the
                    space. This is the standard measure of how linearly
                    accessible class structure is.

  cluster purity    Run k-means with k = 10 on the embeddings, assign each
                    cluster its majority Galaxy Zoo label, and report the
                    fraction of galaxies thereby labelled correctly, plus
                    adjusted mutual information which does not reward
                    guessing the dominant class.

  trustworthiness   Measures how much of the local neighbourhood structure
                    survives the UMAP projection, so the atlas can be shown
                    with an honest statement of what it distorts.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    f1_score,
)
from sklearn.neighbors import KNeighborsClassifier

from .augment import eval_view
from .data import CLASS_NAMES, PROC, ROOT
from .models import EquivariantCNN, load_weights

CKPT = ROOT / "results" / "ssl"


@torch.no_grad()
def encode_all(enc, imgs, mean, std, bs: int = 256) -> np.ndarray:
    enc.eval()
    out = []
    n = imgs.shape[0]
    for s in range(0, n, bs):
        x = eval_view(imgs[s : s + bs], mean, std)
        out.append(enc.features(x).numpy())
        if (s // bs) % 10 == 0:
            print(f"  encoded {min(s + bs, n)}/{n}", end="\r", flush=True)
    print()
    return np.concatenate(out).astype(np.float32)


def knn_probe(z, labels, tr, te, k: int = 20) -> dict:
    # Cosine distance on L2-normalised features is the standard protocol.
    zn = z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-9)
    clf = KNeighborsClassifier(n_neighbors=k, metric="cosine", weights="distance")
    clf.fit(zn[tr], labels[tr])
    pred = clf.predict(zn[te])
    return {
        "k": k,
        "accuracy": float((pred == labels[te]).mean()),
        "macro_f1": float(f1_score(labels[te], pred, average="macro")),
    }


def cluster_metrics(z, labels, seed: int = 0, k: int = 10) -> dict:
    zn = z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-9)
    km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(zn)
    assign = km.labels_
    correct = 0
    composition = []
    for c in range(k):
        m = assign == c
        if m.sum() == 0:
            continue
        counts = np.bincount(labels[m], minlength=len(CLASS_NAMES))
        correct += counts.max()
        composition.append(
            {
                "cluster": int(c),
                "size": int(m.sum()),
                "majority_class": CLASS_NAMES[int(counts.argmax())],
                "purity": float(counts.max() / m.sum()),
            }
        )
    return {
        "k": k,
        "purity": float(correct / len(labels)),
        "adjusted_mutual_info": float(adjusted_mutual_info_score(labels, assign)),
        "adjusted_rand": float(adjusted_rand_score(labels, assign)),
        "clusters": composition,
    }


def project(z, seed: int = 0, n_neighbors: int = 25, min_dist: float = 0.05):
    import umap

    zn = z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-9)
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric="cosine",
        random_state=seed,
        verbose=True,
    )
    return reducer.fit_transform(zn).astype(np.float32)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=str(CKPT / "simclr_encoder.pt"))
    p.add_argument("--k", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    images = np.load(PROC / "images_u8.npy")
    labels = np.load(PROC / "labels.npy")
    meta = dict(np.load(PROC / "meta.npz"))
    imgs = torch.from_numpy(images).permute(0, 3, 1, 2).contiguous()
    mean = torch.tensor(meta["mean"], dtype=torch.float32)
    std = torch.tensor(meta["std"], dtype=torch.float32)

    blob = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    enc = EquivariantCNN(group=blob.get("group", "C8"))
    load_weights(enc, blob["encoder"])
    print(f"loaded encoder from step {blob.get('step')}, dim {blob.get('dim')}")

    print("encoding full catalogue ...")
    z = encode_all(enc, imgs, mean, std)
    np.save(CKPT / "embeddings.npy", z)
    print(f"embeddings: {z.shape}")

    tr, te = meta["train"], meta["test"]
    res = {
        "checkpoint_step": int(blob.get("step", -1)),
        "representation_dim": int(z.shape[1]),
        "knn_probe": knn_probe(z, labels, tr, te, k=a.k),
        "clustering": cluster_metrics(z, labels, seed=a.seed),
    }
    print(
        f"  k-NN probe (k={a.k}): accuracy {res['knn_probe']['accuracy']:.4f}, "
        f"macro-F1 {res['knn_probe']['macro_f1']:.4f}"
    )
    print(
        f"  k-means purity {res['clustering']['purity']:.4f}, "
        f"AMI {res['clustering']['adjusted_mutual_info']:.4f}"
    )

    print("projecting to 2D with UMAP ...")
    xy = project(z, seed=a.seed)
    np.save(CKPT / "umap2d.npy", xy)

    from sklearn.manifold import trustworthiness

    # Trustworthiness on the full set is O(n^2); evaluate on a random subset.
    rng = np.random.default_rng(a.seed)
    sub = rng.choice(len(z), 3000, replace=False)
    zn = z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-9)
    tw = float(trustworthiness(zn[sub], xy[sub], n_neighbors=15, metric="cosine"))
    res["umap_trustworthiness_n15_subset3000"] = tw
    print(f"  UMAP trustworthiness (k=15): {tw:.4f}")

    (CKPT / "embedding_metrics.json").write_text(json.dumps(res, indent=2))
    print(f"wrote {CKPT / 'embedding_metrics.json'}")


if __name__ == "__main__":
    main()
