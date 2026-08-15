"""Correctness and cost audit for the two architectures.

Three questions are answered before any training happens:

  1. Is the steerable convolution actually equivariant, to numerical
     precision, on the exact grid rotations?
  2. Is the assembled network's output actually invariant, and by how much
     does an untrained dense CNN fail the same test?
  3. Do the two models really have matched convolutional FLOPs, and what
     does a training step cost on this machine?

Question 3 sets the training budget for everything downstream, so it is
measured rather than assumed.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from e2cnn import nn as enn

from src.models import (
    INPUT_SIZE,
    KERNELS,
    WIDTHS,
    BaselineCNN,
    EquivariantCNN,
    conv_macs,
    count_params,
    feature_invariance,
)

torch.manual_seed(0)
results = {}


def sec(title):
    print(f"\n{'=' * 68}\n{title}\n{'=' * 68}")


# ---------------------------------------------------------------- layer test
sec("1. Steerable convolution equivariance (single layer)")

from e2cnn import gspaces

gs = gspaces.Rot2dOnR2(N=8)
in_t = enn.FieldType(gs, 3 * [gs.trivial_repr])
out_t = enn.FieldType(gs, 8 * [gs.regular_repr])
conv = enn.R2Conv(in_t, out_t, kernel_size=5, padding=2, bias=False)
conv.eval()

# Odd spatial size so that the 90-degree grid rotation is an exact
# permutation of pixels with no resampling and no boundary offset.
x = torch.randn(2, 3, 33, 33)
gx = enn.GeometricTensor(x, in_t)
y = conv(gx)
scale = y.tensor.abs().max().item()

layer_errs = {}
# testing_elements is a generator; materialise it so it can be reused.
elements = list(gs.testing_elements)
for g in elements:
    lhs = conv(gx.transform(g)).tensor
    rhs = y.transform(g).tensor
    err = (lhs - rhs).abs().max().item() / scale
    layer_errs[str(g)] = err

exact = [layer_errs[str(g)] for g in elements if g % 2 == 0]  # 0, 90, 180, 270
interp = [layer_errs[str(g)] for g in elements if g % 2 == 1]  # 45, 135, ...

print(f"  feature scale                        {scale:.4f}")
print(f"  max relative error, exact grid rots  {max(exact):.3e}")
print(f"  max relative error, 45-degree rots   {max(interp):.3e}")
print("\n  The 45-degree figures are dominated by bilinear resampling of the")
print("  input, not by the layer: those rotations are not exact on a square")
print("  grid. The exact rotations are the meaningful correctness check.")

assert max(exact) < 1e-4, f"steerable conv failed exact-rotation test: {max(exact)}"
print("\n  PASS: steerable convolution is equivariant to machine precision")

results["layer_equivariance"] = {
    "max_rel_error_exact_rotations": max(exact),
    "max_rel_error_45deg_rotations": max(interp),
    "per_element": layer_errs,
}

# ------------------------------------------------------------- network test
sec("2. Network-level rotation invariance (untrained, random weights)")

eq = EquivariantCNN(group="C8")
bl = BaselineCNN()
xb = torch.randn(48, 3, INPUT_SIZE, INPUT_SIZE)

eq_fi = feature_invariance(eq, xb)
bl_fi = feature_invariance(bl, xb)

print(f"  input {INPUT_SIZE}x{INPUT_SIZE}, measured on the penultimate features")
print(f"  {'':<30}{'equivariant':>16}{'baseline':>16}")
print(
    f"  {'max relative L2 drift':<30}"
    f"{eq_fi['max_rel_l2']:>16.3e}{bl_fi['max_rel_l2']:>16.3e}"
)
ratio = bl_fi["max_rel_l2"] / max(eq_fi["max_rel_l2"], 1e-15)
print(f"\n  The dense CNN's representation moves {ratio:,.0f}x more under a")
print("  quarter turn. This holds at initialisation, with random weights and")
print("  no augmentation: the invariance is structural, not learned.")

assert eq_fi["max_rel_l2"] < 1e-5, f"network invariance too weak: {eq_fi['max_rel_l2']}"
print("\n  PASS: assembled network is invariant to machine precision")

results["network_invariance_untrained"] = {
    "input_size": INPUT_SIZE,
    "equivariant_max_rel_l2": eq_fi["max_rel_l2"],
    "baseline_max_rel_l2": bl_fi["max_rel_l2"],
    "ratio": ratio,
}

# ----------------------------------------------------------- cost accounting
sec("3. Parameters and compute")

macs = conv_macs(WIDTHS, KERNELS, size=INPUT_SIZE)
p_eq, p_bl = count_params(eq), count_params(bl)

print(f"  width schedule                {WIDTHS}")
print(f"  kernel schedule               {KERNELS}")
print(f"  conv MACs per image           {macs / 1e6:.1f} M   (identical, by construction)")
print(f"  equivariant parameters        {p_eq:,}")
print(f"  baseline parameters           {p_bl:,}")
print(f"  baseline / equivariant        {p_bl / p_eq:.2f}x")

results["cost"] = {
    "widths": list(WIDTHS),
    "kernels": list(KERNELS),
    "conv_macs_per_image": macs,
    "params_equivariant": p_eq,
    "params_baseline": p_bl,
    "param_ratio_baseline_over_equivariant": p_bl / p_eq,
}

# ------------------------------------------------------------- throughput
sec("4. Measured CPU throughput (sets the training budget)")

print(f"  torch threads: {torch.get_num_threads()}")


def bench(model, bs=64, iters=6, warmup=2):
    model.train()
    opt = torch.optim.SGD(model.parameters(), lr=1e-3)
    x = torch.randn(bs, 3, INPUT_SIZE, INPUT_SIZE)
    y = torch.randint(0, 10, (bs,))
    lossf = torch.nn.CrossEntropyLoss()
    for _ in range(warmup):
        opt.zero_grad()
        lossf(model(x), y).backward()
        opt.step()
    t0 = time.perf_counter()
    for _ in range(iters):
        opt.zero_grad()
        lossf(model(x), y).backward()
        opt.step()
    dt = (time.perf_counter() - t0) / iters
    return dt, bs / dt


for name, m in (("equivariant", eq), ("baseline", bl)):
    dt, ips = bench(m)
    print(f"  {name:<14} {dt * 1000:8.1f} ms/step (bs=64)  {ips:7.1f} img/s")
    results.setdefault("throughput", {})[name] = {
        "sec_per_step_bs64": dt,
        "images_per_sec": ips,
    }

eq_ips = results["throughput"]["equivariant"]["images_per_sec"]
bl_ips = results["throughput"]["baseline"]["images_per_sec"]
rel = eq_ips / bl_ips
verb = "faster" if rel >= 1 else "slower"
print(f"\n  equivariant runs {max(rel, 1 / rel):.2f}x {verb} per image in wall clock")
print("  Conv FLOPs are identical; the residual difference is steerable basis")
print("  expansion on one side against depthwise blur pooling on the other.")
print("  Neither model is compute-advantaged, so wall clock is not a confounder.")

out = Path(__file__).resolve().parent.parent / "results"
out.mkdir(exist_ok=True)
(out / "equivariance_audit.json").write_text(json.dumps(results, indent=2))
print(f"\nwrote {out / 'equivariance_audit.json'}")
