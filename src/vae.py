"""Convolutional VAE: latent geometry and reconstruction-based anomaly score.

Why this model is not equivariant
---------------------------------
The classifier is invariant to rotation by construction, which means it
deliberately destroys orientation information. That is exactly the right
prior for morphology, and exactly the wrong one for reconstruction: to
redraw a specific galaxy you must know which way it was pointing. Invariance
and reconstruction are incompatible objectives, so this is a separate,
orientation-aware model. Stating that plainly is more useful than forcing
one architecture to do both jobs badly.

The model serves two downstream products:

  latent interpolation  Decoding a path between two encoded galaxies probes
                        whether the latent space is smooth and whether
                        morphology varies continuously along it.

  anomaly ranking       Per-galaxy reconstruction error, which is high for
                        objects unlike anything the model saw often. This
                        is the standard construction used for anomaly
                        discovery in survey astronomy.

A small KL weight is used rather than the unit weight of a strict VAE. At
unit weight the reconstructions of 65x65 galaxy cutouts collapse to blurred
ellipses and the anomaly score stops discriminating; the reduced weight
keeps the latent smooth enough to interpolate through while retaining
structure in the decoded images.
"""

from __future__ import annotations

import argparse
import json
import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .augment import eval_view, train_view
from .data import INPUT, PROC, ROOT

CKPT = ROOT / "results" / "vae"
WIDTHS = (32, 64, 128, 256)


class Encoder(nn.Module):
    def __init__(self, latent: int = 64, in_ch: int = 3):
        super().__init__()
        layers, prev = [], in_ch
        for w in WIDTHS:
            layers += [
                nn.Conv2d(prev, w, 3, stride=2, padding=1),
                nn.BatchNorm2d(w),
                nn.SiLU(inplace=True),
            ]
            prev = w
        self.body = nn.Sequential(*layers)
        self.flat = WIDTHS[-1] * 5 * 5
        self.mu = nn.Linear(self.flat, latent)
        self.logvar = nn.Linear(self.flat, latent)

    def forward(self, x):
        h = self.body(x).flatten(1)
        return self.mu(h), self.logvar(h).clamp(-8, 8)


class Decoder(nn.Module):
    def __init__(self, latent: int = 64, out_ch: int = 3):
        super().__init__()
        self.fc = nn.Linear(latent, WIDTHS[-1] * 5 * 5)
        rev = list(WIDTHS[::-1])
        layers = []
        for i in range(len(rev) - 1):
            # ConvTranspose with k=3, s=2, p=1 maps n -> 2n-1: 5,9,17,33,65.
            layers += [
                nn.ConvTranspose2d(rev[i], rev[i + 1], 3, stride=2, padding=1),
                nn.BatchNorm2d(rev[i + 1]),
                nn.SiLU(inplace=True),
            ]
        self.body = nn.Sequential(*layers)
        self.out = nn.ConvTranspose2d(rev[-1], out_ch, 3, stride=2, padding=1)

    def forward(self, z):
        h = self.fc(z).view(-1, WIDTHS[-1], 5, 5)
        return self.out(self.body(h))


class VAE(nn.Module):
    def __init__(self, latent: int = 64):
        super().__init__()
        self.latent = latent
        self.enc = Encoder(latent)
        self.dec = Decoder(latent)

    def forward(self, x):
        mu, logvar = self.enc(x)
        std = torch.exp(0.5 * logvar)
        z = mu + std * torch.randn_like(std)
        return self.dec(z), mu, logvar


def train(steps: int, bs: int, lr: float, beta: float, latent: int, seed: int,
          log_every: int = 100) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    CKPT.mkdir(parents=True, exist_ok=True)

    images = np.load(PROC / "images_u8.npy")
    meta = dict(np.load(PROC / "meta.npz"))
    imgs = torch.from_numpy(images).permute(0, 3, 1, 2).contiguous()
    mean = torch.tensor(meta["mean"], dtype=torch.float32)
    std_ = torch.tensor(meta["std"], dtype=torch.float32)
    tr_idx = meta["train"]
    print(f"training VAE on {len(tr_idx):,} images, latent dim {latent}")

    model = VAE(latent)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    gen = torch.Generator().manual_seed(seed)
    warmup = max(50, steps // 25)

    def lr_at(t):
        if t < warmup:
            return lr * t / warmup
        p = (t - warmup) / max(1, steps - warmup)
        return lr * 0.5 * (1 + math.cos(math.pi * p))

    hist, t0 = [], time.perf_counter()
    for step in range(1, steps + 1):
        model.train()
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        b = tr_idx[torch.randint(len(tr_idx), (bs,), generator=gen).numpy()]
        # Rotation and flip act as augmentation here too: the reconstruction
        # target is the rotated view, so the model must model orientation
        # rather than memorise a canonical one.
        x = train_view(imgs[b], mean, std_, gen=gen)
        xhat, mu, logvar = model(x)
        rec = F.mse_loss(xhat, x, reduction="none").flatten(1).sum(1).mean()
        kl = (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp())).sum(1).mean()
        loss = rec + beta * kl
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()

        if step % log_every == 0 or step == steps:
            el = (time.perf_counter() - t0) / 60
            hist.append(
                {"step": step, "recon": float(rec.detach()), "kl": float(kl.detach())}
            )
            print(
                f"  step {step:5d}/{steps}  recon {float(rec.detach()):8.2f}  "
                f"kl {float(kl.detach()):7.2f}  "
                f"[{el:.1f} min, {el / step * (steps - step):.0f} min left]",
                flush=True,
            )
            torch.save(
                {"model": model.state_dict(), "latent": latent, "step": step},
                CKPT / "vae.pt",
            )

    rec_out = {
        "steps": steps,
        "batch_size": bs,
        "lr": lr,
        "beta": beta,
        "latent": latent,
        "seed": seed,
        "epochs_equivalent": steps * bs / len(tr_idx),
        "history": hist,
        "minutes": (time.perf_counter() - t0) / 60,
    }
    (CKPT / "vae_train.json").write_text(json.dumps(rec_out, indent=2))
    print(f"done in {rec_out['minutes']:.1f} min")
    return rec_out


@torch.no_grad()
def encode_dataset(model, imgs, mean, std_, bs: int = 256):
    """Latent means and per-image reconstruction error, on unrotated crops."""
    model.eval()
    mus, errs = [], []
    n = imgs.shape[0]
    for s in range(0, n, bs):
        x = eval_view(imgs[s : s + bs], mean, std_)
        mu, logvar = model.enc(x)
        xhat = model.dec(mu)
        e = F.mse_loss(xhat, x, reduction="none").flatten(1).mean(1)
        mus.append(mu.numpy())
        errs.append(e.numpy())
        if (s // bs) % 10 == 0:
            print(f"  {min(s + bs, n)}/{n}", end="\r", flush=True)
    print()
    return np.concatenate(mus).astype(np.float32), np.concatenate(errs).astype(np.float32)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--bs", type=int, default=96)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--beta", type=float, default=0.05)
    p.add_argument("--latent", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    train(a.steps, a.bs, a.lr, a.beta, a.latent, a.seed)


if __name__ == "__main__":
    main()
