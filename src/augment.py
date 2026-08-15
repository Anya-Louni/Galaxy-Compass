"""Astronomy-appropriate augmentation, applied identically to both models.

What is augmented and what is not
---------------------------------
Rotation, reflection and small translations are applied freely: a galaxy
has no canonical orientation or position in its cutout, so these are exact
symmetries of the labelling function.

Colour is deliberately left alone. The three channels are DECaLS g, r and z
photometric bands, and their ratios encode stellar population age, dust and
redshift. Jittering brightness, contrast or hue, standard practice on
natural images, would destroy physically meaningful information and teach
the network to ignore a real signal. Only additive noise is used, which
models detector and sky background rather than corrupting colour.

The supervised study gives both the steerable and the dense model exactly
this pipeline. That is what makes the comparison a test of where symmetry
is encoded, architecture or data pipeline, rather than a strawman in which
only one model is told that rotations exist.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from .data import INPUT, STORE


def rigid_sample(
    x: torch.Tensor,
    angles: torch.Tensor,
    flip: torch.Tensor,
    tx: torch.Tensor,
    ty: torch.Tensor,
    out_size: int = INPUT,
) -> torch.Tensor:
    """Per-sample rotation, reflection, translation and centre crop in one op.

    x is (B, C, S, S) float. Angles are radians. flip is a 0/1 tensor
    selecting a horizontal reflection. tx, ty are translations in units of
    the output half-width. Sampling is bilinear and every output pixel is
    drawn from inside the stored frame, so no undefined corner is ever
    introduced.
    """
    b = x.shape[0]
    s = out_size / x.shape[-1]
    cos, sin = torch.cos(angles), torch.sin(angles)
    sx = torch.where(flip > 0, -torch.ones_like(cos), torch.ones_like(cos))

    theta = torch.zeros(b, 2, 3, dtype=x.dtype, device=x.device)
    theta[:, 0, 0] = s * cos * sx
    theta[:, 0, 1] = -s * sin
    theta[:, 0, 2] = tx * s
    theta[:, 1, 0] = s * sin * sx
    theta[:, 1, 1] = s * cos
    theta[:, 1, 2] = ty * s

    grid = F.affine_grid(theta, (b, x.shape[1], out_size, out_size), align_corners=False)
    return F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)


def center_crop(x: torch.Tensor, out_size: int = INPUT) -> torch.Tensor:
    o = (x.shape[-1] - out_size) // 2
    return x[..., o : o + out_size, o : o + out_size]


def normalize(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return (x - mean.view(1, -1, 1, 1)) / std.view(1, -1, 1, 1)


def train_view(
    x_u8: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
    gen: torch.Generator | None = None,
    max_shift: float = 0.05,
    noise_std: float = 0.0,
) -> torch.Tensor:
    """Full-circle rotation, random reflection, small shift, optional noise."""
    x = x_u8.float().div_(255.0)
    b = x.shape[0]
    dev = x.device
    angles = torch.rand(b, generator=gen, device=dev) * (2 * math.pi)
    flip = (torch.rand(b, generator=gen, device=dev) < 0.5).float()
    tx = (torch.rand(b, generator=gen, device=dev) * 2 - 1) * max_shift
    ty = (torch.rand(b, generator=gen, device=dev) * 2 - 1) * max_shift
    x = rigid_sample(x, angles, flip, tx, ty)
    if noise_std > 0:
        x = x + torch.randn(x.shape, generator=gen, device=dev) * noise_std
    return normalize(x, mean, std)


def eval_view(
    x_u8: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
    angle: float = 0.0,
    force_resample: bool = False,
) -> torch.Tensor:
    """Deterministic centre crop, optionally at a fixed rotation.

    By default angle=0 takes an exact centre crop with no resampling at all,
    so reported accuracy is not degraded by interpolation blur.

    That default is a trap when comparing across angles. Every non-zero angle
    goes through bilinear resampling, so a sweep over angles that includes
    zero compares one sharp image against many slightly blurred ones, and the
    resulting "rotation sensitivity" is partly just resampling sensitivity.
    Since training always resamples, models are in fact better matched to the
    blurred inputs, and the artefact can even make accuracy rise with
    rotation. Pass force_resample=True to put every angle, including zero,
    through the identical sampling path so that only orientation differs.
    """
    x = x_u8.float().div_(255.0)
    if angle == 0.0 and not force_resample:
        return normalize(center_crop(x), mean, std)
    b = x.shape[0]
    dev = x.device
    z = torch.zeros(b, device=dev)
    a = torch.full((b,), float(angle), device=dev)
    return normalize(rigid_sample(x, a, z, z, z), mean, std)


def simclr_views(
    x_u8: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
    gen: torch.Generator | None = None,
    noise_std: float = 0.06,
    max_shift: float = 0.12,
    scale_range: tuple[float, float] = (0.85, 1.15),
) -> tuple[torch.Tensor, torch.Tensor]:
    """Two independent views of each image for contrastive training.

    The transformations the encoder is asked to be invariant to are exactly
    the ones a telescope could have produced differently: orientation,
    parity, pointing, seeing-scale and noise realisation. Colour is
    untouched, so the encoder is free to keep using it.
    """
    x = x_u8.float().div_(255.0)
    b, dev = x.shape[0], x.device

    def one():
        angles = torch.rand(b, generator=gen, device=dev) * (2 * math.pi)
        flip = (torch.rand(b, generator=gen, device=dev) < 0.5).float()
        tx = (torch.rand(b, generator=gen, device=dev) * 2 - 1) * max_shift
        ty = (torch.rand(b, generator=gen, device=dev) * 2 - 1) * max_shift
        lo, hi = scale_range
        sc = torch.rand(b, generator=gen, device=dev) * (hi - lo) + lo
        # Fold the zoom into the rotation matrix by scaling the angle-derived
        # entries, keeping everything to a single resampling pass.
        cos, sin = torch.cos(angles) * sc, torch.sin(angles) * sc
        s = INPUT / x.shape[-1]
        sx = torch.where(flip > 0, -torch.ones_like(cos), torch.ones_like(cos))
        theta = torch.zeros(b, 2, 3, dtype=x.dtype, device=dev)
        theta[:, 0, 0] = s * cos * sx
        theta[:, 0, 1] = -s * sin
        theta[:, 0, 2] = tx * s
        theta[:, 1, 0] = s * sin * sx
        theta[:, 1, 1] = s * cos
        theta[:, 1, 2] = ty * s
        grid = F.affine_grid(
            theta, (b, x.shape[1], INPUT, INPUT), align_corners=False
        )
        v = F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
        v = v + torch.randn(v.shape, generator=gen, device=dev) * noise_std
        return normalize(v, mean, std)

    return one(), one()
