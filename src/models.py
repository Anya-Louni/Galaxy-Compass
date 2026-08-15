"""Rotation-equivariant and matched baseline classifiers.

Design contract
---------------
The two models are built from the same width schedule, where "width" means
the number of *output channels* of each block. For the steerable model a
block of width W is realised as W/|G| copies of the regular representation
of G, which occupies exactly W channels. The dense baseline uses W plain
channels.

Consequences, both deliberate:

  * Convolutional FLOPs are identical. e2cnn expands its steerable basis
    into a dense [W_out, W_in, k, k] filter bank and calls the same
    F.conv2d the baseline calls, so the two models do the same number of
    multiply-accumulates per image.

  * The baseline has strictly more parameters. Weight sharing across the
    group is what buys equivariance, so the steerable model spends roughly
    |G| times fewer weights on each convolution.

The baseline is therefore not handicapped. It is given the larger parameter
budget and the same compute, and in training it is given the same rotation
and flip augmentation. Any remaining gap is attributable to where the
symmetry lives: in the architecture, or in the data pipeline.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from e2cnn import gspaces
from e2cnn import nn as enn

# Width schedule and kernel sizes, chosen to fit a CPU training budget while
# keeping four downsampling stages.
WIDTHS = (32, 64, 128, 256)
KERNELS = (5, 3, 3, 3)

# Images are 65x65, not 64x64, and the odd size is load-bearing.
#
# A steerable convolution is exactly equivariant, but stride-2 subsampling
# is not automatically so: on an even grid there is no pixel at the centre
# of rotation, so the retained lattice is not mapped to itself by a quarter
# turn, and the error compounds with depth. On an odd grid the centre pixel
# is fixed and the lattice is preserved. Measured end-to-end invariance
# error falls from 4.7e-02 (64x64) to 8.8e-08 (65x65); see
# scripts/probe_pooling.py for the sweep this was chosen from.
INPUT_SIZE = 65

# Antialiasing parameters of e2cnn's PointwiseAvgPoolAntialiased, replicated
# exactly for the dense baseline so both models share a downsampling
# operator and produce identical spatial sizes at every stage.
BLUR_SIGMA = 0.66
BLUR_KERNEL = 5


def make_gspace(group: str):
    """'C8' -> 8 rotations; 'D4' -> 4 rotations plus reflections."""
    if group.startswith("C"):
        n = int(group[1:])
        return gspaces.Rot2dOnR2(N=n), n
    if group.startswith("D"):
        n = int(group[1:])
        return gspaces.FlipRot2dOnR2(N=n), 2 * n
    raise ValueError(f"unknown group {group!r}")


class EquivariantCNN(nn.Module):
    """Steerable CNN with an output invariant to the action of `group`.

    Invariance is produced by two ingredients and nothing else: every
    convolution is G-steerable, and the final feature map is reduced by a
    max over the group axis followed by a global spatial average. No
    rotation augmentation is required for the invariance to hold; it holds
    for an untrained network at initialisation.
    """

    def __init__(
        self,
        n_classes: int = 10,
        group: str = "C8",
        widths=WIDTHS,
        kernels=KERNELS,
        in_ch: int = 3,
    ):
        super().__init__()
        self.gspace, self.order = make_gspace(group)
        self.group = group
        self.widths = tuple(widths)

        for w in widths:
            if w % self.order != 0:
                raise ValueError(
                    f"width {w} is not divisible by |G|={self.order}; "
                    "channel counts could not be matched to the baseline"
                )

        in_type = enn.FieldType(self.gspace, in_ch * [self.gspace.trivial_repr])
        self.in_type = in_type

        blocks = []
        prev = in_type
        for i, (w, k) in enumerate(zip(widths, kernels)):
            mult = w // self.order
            out_type = enn.FieldType(self.gspace, mult * [self.gspace.regular_repr])
            blocks += [
                enn.R2Conv(prev, out_type, kernel_size=k, padding=k // 2, bias=False),
                enn.InnerBatchNorm(out_type),
                enn.ReLU(out_type, inplace=True),
                # Antialiased average pooling. Strided max pooling is not
                # compatible with equivariance; blur-then-subsample is the
                # standard equivariant downsampling operator.
                enn.PointwiseAvgPoolAntialiased(out_type, sigma=0.66, stride=2),
            ]
            prev = out_type
        self.body = enn.SequentialModule(*blocks)
        self.out_type = prev
        self.final_mult = widths[-1] // self.order

        # Invariant readout. A group element permutes the |G| channels within
        # each regular field, so any symmetric function of that axis is
        # exactly invariant. Taking both the max and the mean gives two
        # invariants per field instead of the one that GroupPooling alone
        # would give, which matters here: with |G|=8 a plain group max would
        # shrink a 256-channel feature map to just 32 numbers before the
        # classifier ever sees it.
        n_feat = 2 * self.final_mult

        self.head = nn.Sequential(
            nn.BatchNorm1d(n_feat),
            nn.Dropout(0.3),
            nn.Linear(n_feat, n_classes),
        )
        self.n_features = n_feat

    def features(self, x: torch.Tensor) -> torch.Tensor:
        x = enn.GeometricTensor(x, self.in_type)
        t = self.body(x).tensor
        b, _, h, w = t.shape
        # Field-major layout: (field, group element) -> split the two axes.
        t = t.view(b, self.final_mult, self.order, h, w)
        inv = torch.cat([t.amax(dim=2), t.mean(dim=2)], dim=1)
        # Global spatial average completes translation invariance.
        return inv.mean(dim=(2, 3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))


class GaussianBlurPool(nn.Module):
    """Blur-then-subsample, numerically identical to e2cnn's antialiased pool.

    Implemented as a depthwise convolution with a fixed, normalised Gaussian
    kernel. This is also a standard technique for ordinary CNNs (Zhang 2019,
    "Making Convolutional Networks Shift-Invariant Again"), so using it in
    the baseline is not an equivariance-specific favour: it simply removes
    the downsampling operator as a confounder between the two models.
    """

    def __init__(self, channels: int, sigma: float = BLUR_SIGMA, size: int = BLUR_KERNEL):
        super().__init__()
        r = torch.arange(size, dtype=torch.float64) - (size - 1) / 2
        g = torch.exp(-(r**2) / (2 * sigma**2))
        k = torch.outer(g, g)
        k = (k / k.sum()).to(torch.float32)
        self.register_buffer("weight", k.expand(channels, 1, size, size).contiguous())
        self.channels = channels
        self.padding = size // 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(x, self.weight, stride=2, padding=self.padding, groups=self.channels)


class BaselineCNN(nn.Module):
    """Dense CNN with the same channel widths, kernels and downsampling.

    Identical convolutional FLOPs to EquivariantCNN, strictly more
    parameters, and the same global-average-pool head.
    """

    def __init__(
        self,
        n_classes: int = 10,
        widths=WIDTHS,
        kernels=KERNELS,
        in_ch: int = 3,
    ):
        super().__init__()
        self.widths = tuple(widths)
        layers = []
        prev = in_ch
        for w, k in zip(widths, kernels):
            layers += [
                nn.Conv2d(prev, w, kernel_size=k, padding=k // 2, bias=False),
                nn.BatchNorm2d(w),
                nn.ReLU(inplace=True),
                GaussianBlurPool(w),
            ]
            prev = w
        self.body = nn.Sequential(*layers)
        n_feat = widths[-1]
        self.head = nn.Sequential(
            nn.BatchNorm1d(n_feat),
            nn.Dropout(0.3),
            nn.Linear(n_feat, n_classes),
        )
        self.n_features = n_feat

    def features(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x).mean(dim=(2, 3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def conv_macs(
    widths=WIDTHS, kernels=KERNELS, size: int = INPUT_SIZE, in_ch: int = 3
) -> int:
    """Multiply-accumulates in the convolution stack for one image.

    Identical for both models by construction; computed analytically so the
    claim of matched compute is checkable rather than asserted. Stride-2
    pooling with padding (k-1)/2 maps an odd size n to (n+1)/2.
    """
    total = 0
    prev, res = in_ch, size
    for w, k in zip(widths, kernels):
        total += res * res * w * prev * k * k
        prev, res = w, (res + 1) // 2
    return total


@torch.no_grad()
def feature_invariance(model: nn.Module, x: torch.Tensor) -> dict:
    """Relative L2 drift of the penultimate feature vector under grid rotations.

    Measured on the representation rather than on class probabilities: an
    untrained network emits a near-uniform softmax for every input, which
    makes probability-space comparisons vacuous at initialisation.
    """
    model.eval()
    f = model.features(x)
    errs = []
    for k in (1, 2, 3):
        fr = model.features(torch.rot90(x, k, dims=(2, 3)))
        errs.append(((fr - f).norm(dim=1) / f.norm(dim=1).clamp_min(1e-12)).mean().item())
    return {"max_rel_l2": max(errs), "per_rotation": errs}


@torch.no_grad()
def rotation_consistency(model: nn.Module, x: torch.Tensor) -> dict:
    """Measure how much a model's logits move when the input is rotated.

    Uses the three exact grid rotations (90, 180, 270 degrees), which are
    lossless on a square pixel grid and therefore isolate the model's
    behaviour from any interpolation error. Returns the mean absolute
    deviation of the softmax distribution from the unrotated one, and the
    fraction of images whose predicted class changes.
    """
    model.eval()
    base = F.softmax(model(x), dim=1)
    base_pred = base.argmax(1)
    devs, flips = [], []
    for kk in (1, 2, 3):
        p = F.softmax(model(torch.rot90(x, kk, dims=(2, 3))), dim=1)
        devs.append((p - base).abs().mean().item())
        flips.append((p.argmax(1) != base_pred).float().mean().item())
    return {
        "mean_abs_prob_deviation": float(sum(devs) / len(devs)),
        "prediction_flip_rate": float(sum(flips) / len(flips)),
        "per_rotation_deviation": devs,
    }


# e2cnn caches the expanded filter bank as a buffer, but only while the module
# is in eval mode: train() deletes it and eval() rebuilds it from the steerable
# weights. A checkpoint therefore contains these derived buffers or not
# depending on which mode the model was in when it was saved, and a strict load
# fails in one direction or the other. They are pure functions of the real
# parameters, so the correct move is to ignore them on load and force a rebuild.
_DERIVED_BUFFERS = ("filter", "expanded_bias")


def load_weights(model: nn.Module, state: dict) -> nn.Module:
    """Load a checkpoint regardless of the mode it was saved in.

    Raises if anything other than the derived buffers fails to match, so a
    genuinely incompatible checkpoint is still a loud error.
    """
    model.train()  # drop any cached derived buffers
    res = model.load_state_dict(state, strict=False)
    leaf = lambda k: k.rsplit(".", 1)[-1]
    missing = [k for k in res.missing_keys if leaf(k) not in _DERIVED_BUFFERS]
    unexpected = [k for k in res.unexpected_keys if leaf(k) not in _DERIVED_BUFFERS]
    if missing or unexpected:
        raise RuntimeError(
            f"checkpoint does not match model\n  missing: {missing}\n"
            f"  unexpected: {unexpected}"
        )
    model.eval()  # rebuild the filter bank from the weights just loaded
    return model


def build(name: str, n_classes: int = 10, group: str = "C8", **kw) -> nn.Module:
    if name == "equivariant":
        return EquivariantCNN(n_classes=n_classes, group=group, **kw)
    if name == "baseline":
        return BaselineCNN(n_classes=n_classes, **kw)
    raise ValueError(f"unknown model {name!r}")
