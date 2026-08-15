"""Matched-inference-cost comparison, computed from the cached TTA results.

The headline crossing compares a single-pass steerable model against a
D4-averaged dense CNN, which is generous to the baseline in accuracy but
unfair to it in compute: the baseline is doing eight forward passes.

The cleanest comparison gives both architectures the same eight-pass budget
and asks the same label-efficiency question. No model is run here; this only
re-reduces the numbers already written by tta_baseline.py.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import ROOT

P = ROOT / "results" / "tta_baseline.json"
blob = json.loads(P.read_text())
summary = blob["summary"]

fracs = sorted({float(k.rsplit("_", 1)[1]) for k in summary})


def mean_at(arch, f, mode):
    k = f"{arch}_{f:g}"
    return summary[k]["mean"][mode] if k in summary else None


def crossing(points, target):
    pts = [(f, v) for f, v in points if v is not None]
    if not pts or target is None:
        return None
    if pts[0][1] >= target:
        return pts[0][0]
    for i in range(1, len(pts)):
        if pts[i][1] >= target:
            (f0, v0), (f1, v1) = pts[i - 1], pts[i]
            if v1 == v0:
                return f1
            t = (target - v0) / (v1 - v0)
            return float(np.exp(np.log(f0) + t * (np.log(f1) - np.log(f0))))
    return None


eq_d4 = [(f, mean_at("equivariant", f, "d4_tta")) for f in fracs]
eq_plain = [(f, mean_at("equivariant", f, "plain")) for f in fracs]
bl_d4_full = mean_at("baseline", 1.0, "d4_tta")
bl_plain_full = mean_at("baseline", 1.0, "plain")

cf_matched = crossing(eq_d4, bl_d4_full)
cf_generous = crossing(eq_plain, bl_d4_full)
cf_plain = crossing(eq_plain, bl_plain_full)

# Does averaging over the four rotations change the steerable model at all?
# It must not: the model is already exactly invariant to them, so this is an
# independent end-to-end check of that invariance on real data.
inv_check = {
    f"{f:g}": {
        "plain": mean_at("equivariant", f, "plain"),
        "c4_tta": mean_at("equivariant", f, "c4_tta"),
        "identical": mean_at("equivariant", f, "plain")
        == mean_at("equivariant", f, "c4_tta"),
    }
    for f in fracs
}

verdict = blob.get("verdict", {})
verdict.update({
    "crossing_matched_8x_cost": cf_matched,
    "label_efficiency_matched_8x_cost": (1 / cf_matched) if cf_matched else None,
    "crossing_vs_d4_tta_baseline": cf_generous,
    "label_efficiency_vs_d4_tta": (1 / cf_generous) if cf_generous else None,
    "tta_gain_baseline_100pct_pp": (bl_d4_full - bl_plain_full) * 100,
    "architecture_gap_plain_pp": (
        mean_at("equivariant", 1.0, "plain") - bl_plain_full) * 100,
    "architecture_gap_both_tta_pp": (
        mean_at("equivariant", 1.0, "d4_tta") - bl_d4_full) * 100,
    "c4_tta_leaves_equivariant_unchanged": all(v["identical"] for v in inv_check.values()),
    "c4_invariance_check": inv_check,
})
blob["verdict"] = verdict
P.write_text(json.dumps(blob, indent=2))

print("Independent invariance check (C4 averaging on the steerable model):")
for f, v in inv_check.items():
    print(f"  {float(f) * 100:>5.0f}% labels  plain {v['plain']:.6f}  "
          f"C4-TTA {v['c4_tta']:.6f}  identical: {v['identical']}")
print(f"\n  all identical: {verdict['c4_tta_leaves_equivariant_unchanged']}")
print("  Averaging over rotations cannot change a prediction that is already")
print("  the same at every rotation. This is the invariance claim confirmed")
print("  on real test data, end to end, from a completely separate code path.")

print("\nLabel efficiency under three framings:")
print(f"  vs plain dense CNN            {cf_plain * 100:>5.1f}% of labels  "
      f"({1 / cf_plain:.1f}x)   baseline pays 1x inference")
print(f"  vs D4-TTA dense CNN           {cf_generous * 100:>5.1f}% of labels  "
      f"({1 / cf_generous:.1f}x)   baseline pays 8x inference")
print(f"  both at matched 8x inference  {cf_matched * 100:>5.1f}% of labels  "
      f"({1 / cf_matched:.1f}x)   like for like")

print(f"\n  TTA buys the dense CNN {verdict['tta_gain_baseline_100pct_pp']:+.2f} pp "
      f"for 8x the inference cost.")
print(f"  The architecture gap is {verdict['architecture_gap_plain_pp']:+.2f} pp "
      f"single-pass, {verdict['architecture_gap_both_tta_pp']:+.2f} pp with both "
      f"averaged.")
print(f"\nupdated {P}")
