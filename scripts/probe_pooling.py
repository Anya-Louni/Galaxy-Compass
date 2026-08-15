"""Find a downsampling configuration that preserves rotation invariance.

A steerable convolution is exactly equivariant, but stride-2 subsampling is
not: on an even-sized grid there is no pixel at the centre of rotation, so
the subsampling lattice is not mapped to itself by a 90 degree turn. The
error compounds with depth. This probe sweeps input size and pooling
operator and reports the resulting feature-space invariance error.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from e2cnn import gspaces
from e2cnn import nn as enn

WIDTHS = (32, 64, 96, 128)
KERNELS = (5, 3, 3, 3)


def build(size: int, pool: str, n_pool: int = 4, group_n: int = 8):
    gs = gspaces.Rot2dOnR2(N=group_n)
    in_t = enn.FieldType(gs, 3 * [gs.trivial_repr])
    blocks, prev = [], in_t
    for i, (w, k) in enumerate(zip(WIDTHS, KERNELS)):
        out_t = enn.FieldType(gs, (w // group_n) * [gs.regular_repr])
        blocks += [
            enn.R2Conv(prev, out_t, kernel_size=k, padding=k // 2, bias=False),
            enn.InnerBatchNorm(out_t),
            enn.ReLU(out_t, inplace=True),
        ]
        if i < n_pool:
            if pool == "antialiased":
                blocks.append(
                    enn.PointwiseAvgPoolAntialiased(out_t, sigma=0.66, stride=2)
                )
            elif pool == "avg":
                blocks.append(enn.PointwiseAvgPool(out_t, kernel_size=2, stride=2))
            elif pool == "avg3":
                # 3x3 window, stride 2, padding 1: on an odd grid this window
                # is centred on the retained pixels, so the operator commutes
                # with a 90 degree turn about the central pixel.
                blocks.append(
                    enn.PointwiseAvgPool(out_t, kernel_size=3, stride=2, padding=1)
                )
        prev = out_t
    body = enn.SequentialModule(*blocks)
    gpool = enn.GroupPooling(prev)
    return gs, in_t, body, gpool


@torch.no_grad()
def invariance(size: int, pool: str, seed: int = 0) -> tuple[float, tuple]:
    torch.manual_seed(seed)
    gs, in_t, body, gpool = build(size, pool)
    body.eval()
    x = torch.randn(16, 3, size, size)

    def feats(t):
        g = enn.GeometricTensor(t, in_t)
        return gpool(body(g)).tensor.mean(dim=(2, 3))

    # Record the spatial size reaching the head, which reveals whether the
    # grid stayed odd all the way down.
    g = enn.GeometricTensor(x, in_t)
    shape = tuple(body(g).tensor.shape[2:])

    f = feats(x)
    errs = []
    for k in (1, 2, 3):
        fr = feats(torch.rot90(x, k, dims=(2, 3)))
        errs.append(((fr - f).norm(dim=1) / f.norm(dim=1)).mean().item())
    return max(errs), shape


print(f"{'input':>6} {'pooling':>13} {'final map':>11} {'max rel-L2 error':>18}")
print("-" * 52)
best = None
for size in (63, 64, 65):
    for pool in ("antialiased", "avg", "avg3"):
        try:
            err, shape = invariance(size, pool)
        except Exception as e:  # some size/operator pairs are degenerate
            print(f"{size:>6} {pool:>13} {'-':>11} {type(e).__name__}: {e}")
            continue
        print(f"{size:>6} {pool:>13} {str(shape):>11} {err:>18.3e}")
        if best is None or err < best[0]:
            best = (err, size, pool)

print("-" * 52)
print(f"best: input {best[1]}, pooling '{best[2]}', error {best[0]:.3e}")
