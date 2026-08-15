"""Self-supervised representation learning on galaxy morphology.

SimCLR with a C8-steerable encoder. The pairing is deliberate rather than
incidental:

  * Rotation invariance is supplied by the architecture, exactly, so the
    contrastive objective never has to spend capacity rediscovering that
    orientation is meaningless. Two views differing only by a rotation are
    already mapped to the same point.

  * C8 contains rotations but no reflections, so parity remains a genuine
    augmentation with a real gradient. Together with pointing, seeing-scale
    and noise realisation, the pretext task retains substantial signal.

  * Colour is never jittered. The channels are DECaLS g, r and z, and their
    ratios carry stellar population and dust information. The encoder is
    left free to use them.

Training uses the training split only. The validation and test images are
embedded afterwards but never contribute a gradient, so the k-NN probe
reported later is not measuring memorisation.
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

from .augment import simclr_views
from .data import PROC, ROOT
from .models import EquivariantCNN

CKPT = ROOT / "results" / "ssl"


class Projector(nn.Module):
    """Two-layer projection head, discarded after training."""

    def __init__(self, d_in: int, d_hidden: int = 256, d_out: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.BatchNorm1d(d_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(d_hidden, d_out, bias=False),
        )

    def forward(self, x):
        return self.net(x)


def nt_xent(z1: torch.Tensor, z2: torch.Tensor, tau: float = 0.2) -> torch.Tensor:
    """Normalised temperature-scaled cross entropy over a 2B batch."""
    b = z1.shape[0]
    z = F.normalize(torch.cat([z1, z2], dim=0), dim=1)
    sim = (z @ z.t()) / tau
    sim.fill_diagonal_(float("-inf"))
    # Positive of index i is i+b, and of i+b is i.
    target = torch.cat([torch.arange(b, 2 * b), torch.arange(0, b)]).to(z.device)
    return F.cross_entropy(sim, target)


def train(steps: int, bs: int, lr: float, tau: float, group: str, seed: int,
          log_every: int = 100) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    CKPT.mkdir(parents=True, exist_ok=True)

    images = np.load(PROC / "images_u8.npy")
    meta = dict(np.load(PROC / "meta.npz"))
    imgs = torch.from_numpy(images).permute(0, 3, 1, 2).contiguous()
    mean = torch.tensor(meta["mean"], dtype=torch.float32)
    std = torch.tensor(meta["std"], dtype=torch.float32)
    tr_idx = meta["train"]
    print(f"self-supervised training on {len(tr_idx):,} unlabelled images")

    enc = EquivariantCNN(group=group)
    d = enc.n_features
    proj = Projector(d)
    print(f"encoder representation dimension: {d}")

    params = list(enc.parameters()) + list(proj.parameters())
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    gen = torch.Generator().manual_seed(seed)
    warmup = max(50, steps // 25)

    def lr_at(t):
        if t < warmup:
            return lr * t / warmup
        p = (t - warmup) / max(1, steps - warmup)
        return lr * 0.5 * (1 + math.cos(math.pi * p))

    hist = []
    t0 = time.perf_counter()
    for step in range(1, steps + 1):
        enc.train()
        proj.train()
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        b = tr_idx[torch.randint(len(tr_idx), (bs,), generator=gen).numpy()]
        v1, v2 = simclr_views(imgs[b], mean, std, gen=gen)
        z1, z2 = proj(enc.features(v1)), proj(enc.features(v2))
        loss = nt_xent(z1, z2, tau)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 5.0)
        opt.step()

        if step % log_every == 0 or step == steps:
            el = (time.perf_counter() - t0) / 60
            eta = el / step * (steps - step)
            hist.append({"step": step, "loss": float(loss.detach())})
            print(
                f"  step {step:5d}/{steps}  ntxent {float(loss.detach()):.4f}  "
                f"[{el:.1f} min elapsed, {eta:.0f} min left]",
                flush=True,
            )
            torch.save(
                {"encoder": enc.state_dict(), "group": group, "dim": d, "step": step},
                CKPT / "simclr_encoder.pt",
            )

    rec = {
        "steps": steps,
        "batch_size": bs,
        "lr": lr,
        "temperature": tau,
        "group": group,
        "seed": seed,
        "representation_dim": d,
        "n_train_images": int(len(tr_idx)),
        "epochs_equivalent": steps * bs / len(tr_idx),
        "final_loss": hist[-1]["loss"] if hist else None,
        "history": hist,
        "minutes": (time.perf_counter() - t0) / 60,
    }
    (CKPT / "simclr_train.json").write_text(json.dumps(rec, indent=2))
    print(f"done in {rec['minutes']:.1f} min, final NT-Xent {rec['final_loss']:.4f}")
    return rec


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=6000)
    p.add_argument("--bs", type=int, default=96)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--tau", type=float, default=0.2)
    p.add_argument("--group", default="C8")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    train(a.steps, a.bs, a.lr, a.tau, a.group, a.seed)


if __name__ == "__main__":
    main()
