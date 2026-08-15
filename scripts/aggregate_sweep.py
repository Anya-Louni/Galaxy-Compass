"""Aggregate the label-fraction sweep into the headline result.

The claim under test is a statement about label efficiency: how small a
labelled fraction does the steerable model need before it matches what the
dense model achieves with all of them?

Answering that honestly requires care in three places.

  1. The comparison target is the dense model's mean at 100% of labels. Its
     seed-to-seed spread is carried through, so "matches" means the
     steerable mean reaches the dense mean, and the reader can see whether
     that is inside the noise.

  2. The crossing fraction is obtained by linear interpolation on log-
     fraction between the two bracketing measured points, not by declaring
     the nearest sampled point a match. If the curve never reaches the
     target within the sampled range, that is reported instead of
     extrapolated.

  3. Both accuracy and macro-F1 are reported. With class counts spanning
     334 to 2645, accuracy alone can be moved by the majority classes.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import CLASS_NAMES, ROOT

SWEEP = ROOT / "results" / "sweep"
OUT = ROOT / "results"
FIG = OUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)

BG = "#0a0d14"
FG = "#e8ecf4"
GRID = "#232838"
EQ_C = "#5ec8f5"
BL_C = "#f5915e"


def load_records():
    recs = []
    for f in sorted(SWEEP.glob("*.json")):
        recs.append(json.loads(f.read_text()))
    return recs


def curve(recs, model, metric):
    by = defaultdict(list)
    for r in recs:
        if r["model"] == model:
            by[r["label_fraction"]].append(r[metric])
    fr = sorted(by)
    mu = np.array([np.mean(by[f]) for f in fr])
    sd = np.array([np.std(by[f], ddof=1) if len(by[f]) > 1 else 0.0 for f in fr])
    n = np.array([len(by[f]) for f in fr])
    return np.array(fr), mu, sd, n


def crossing_fraction(fr, mu, target):
    """Smallest fraction whose interpolated mean reaches target."""
    if mu[0] >= target:
        return float(fr[0]), "at or below the smallest sampled fraction"
    for i in range(1, len(fr)):
        if mu[i] >= target:
            x0, x1 = np.log(fr[i - 1]), np.log(fr[i])
            y0, y1 = mu[i - 1], mu[i]
            if y1 == y0:
                return float(fr[i]), "interpolated"
            t = (target - y0) / (y1 - y0)
            return float(np.exp(x0 + t * (x1 - x0))), "interpolated"
    return None, "never reached within the sampled range"


def main():
    recs = load_records()
    if not recs:
        print("no sweep records found")
        return
    seeds = sorted({r["seed"] for r in recs})
    print(f"{len(recs)} runs, seeds {seeds}")

    summary = {"n_runs": len(recs), "seeds": seeds, "metrics": {}}

    for metric, nice in (("test_accuracy", "accuracy"), ("test_macro_f1", "macro-F1")):
        fe, me, se, ne = curve(recs, "equivariant", metric)
        fb, mb, sb, nb = curve(recs, "baseline", metric)
        if not len(fe) or not len(fb):
            continue

        target = float(mb[-1])
        target_sd = float(sb[-1])
        cf, how = crossing_fraction(fe, me, target)

        entry = {
            "equivariant": {
                "fractions": fe.tolist(), "mean": me.tolist(),
                "std": se.tolist(), "n_seeds": ne.tolist(),
            },
            "baseline": {
                "fractions": fb.tolist(), "mean": mb.tolist(),
                "std": sb.tolist(), "n_seeds": nb.tolist(),
            },
            "baseline_full_label_mean": target,
            "baseline_full_label_std": target_sd,
            "equivariant_crossing_fraction": cf,
            "crossing_method": how,
            "label_efficiency_factor": (1.0 / cf) if cf else None,
        }
        summary["metrics"][metric] = entry

        print(f"\n{nice}")
        print(f"  {'frac':>6} {'equivariant':>18} {'baseline':>18}   delta")
        for i, f in enumerate(fe):
            j = int(np.where(fb == f)[0][0]) if f in fb else None
            b = f"{mb[j]:.4f} +/- {sb[j]:.4f}" if j is not None else "-"
            d = f"{me[i] - mb[j]:+.4f}" if j is not None else ""
            print(f"  {f:>6.2f} {me[i]:.4f} +/- {se[i]:.4f} {b:>18}   {d}")
        print(f"  baseline at 100% labels: {target:.4f} +/- {target_sd:.4f}")
        if cf:
            print(
                f"  equivariant reaches it at {cf * 100:.1f}% of labels "
                f"({1 / cf:.1f}x label efficiency, {how})"
            )
        else:
            print(f"  equivariant {how}")

    # ------------------------------------------------- per-seed crossing
    #
    # A single crossing fraction computed from the pooled means hides how
    # uncertain it is. The crossing lands on the steep part of the curve,
    # where a small vertical shift moves it a long way horizontally, so it
    # is computed independently per seed (each seed's steerable curve
    # against that same seed's own all-labels baseline) and reported as a
    # range rather than a point.
    for metric in ("test_accuracy", "test_macro_f1"):
        if metric not in summary["metrics"]:
            continue
        per_seed = []
        for s in seeds:
            e = [(r["label_fraction"], r[metric]) for r in recs
                 if r["model"] == "equivariant" and r["seed"] == s]
            b1 = [r[metric] for r in recs if r["model"] == "baseline"
                  and r["seed"] == s and r["label_fraction"] == 1.0]
            if not e or not b1:
                continue
            e.sort()
            fr = np.array([x[0] for x in e])
            mu = np.array([x[1] for x in e])
            cf, _ = crossing_fraction(fr, mu, b1[0])
            if cf:
                per_seed.append({"seed": s, "crossing_fraction": cf,
                                 "label_efficiency": 1.0 / cf})
        if per_seed:
            cfs = [d["crossing_fraction"] for d in per_seed]
            summary["metrics"][metric]["per_seed_crossing"] = {
                "runs": per_seed,
                "mean_crossing_fraction": float(np.mean(cfs)),
                "min": float(np.min(cfs)), "max": float(np.max(cfs)),
                "mean_label_efficiency": float(np.mean([1 / c for c in cfs])),
                "min_label_efficiency": float(np.min([1 / c for c in cfs])),
                "max_label_efficiency": float(np.max([1 / c for c in cfs])),
            }
            print(
                f"\n{metric}: per-seed crossing "
                + ", ".join(f"s{d['seed']}={d['crossing_fraction'] * 100:.1f}%"
                            for d in per_seed)
                + f"  -> mean {np.mean(cfs) * 100:.1f}% "
                f"(range {np.min(cfs) * 100:.1f}-{np.max(cfs) * 100:.1f}%), "
                f"efficiency {np.mean([1 / c for c in cfs]):.1f}x "
                f"({np.min([1 / c for c in cfs]):.1f}-{np.max([1 / c for c in cfs]):.1f}x)"
            )

    # ------------------------------------------- paired win count per cell
    #
    # Stronger than comparing means: hold seed and label fraction fixed and
    # ask which architecture won that specific matched pair. Under the null
    # that the two are equivalent, wins are a fair coin, so the count has an
    # exact sign-test p-value.
    paired = {}
    for metric in ("test_accuracy", "test_macro_f1"):
        wins = losses = 0
        for s in seeds:
            for f in sorted({r["label_fraction"] for r in recs}):
                e = [r[metric] for r in recs if r["model"] == "equivariant"
                     and r["seed"] == s and r["label_fraction"] == f]
                b = [r[metric] for r in recs if r["model"] == "baseline"
                     and r["seed"] == s and r["label_fraction"] == f]
                if e and b:
                    wins += e[0] > b[0]
                    losses += e[0] < b[0]
        n = wins + losses
        # Two-sided exact binomial tail at p=0.5.
        from math import comb

        pval = sum(comb(n, k) for k in range(wins, n + 1)) / (2 ** n) * 2 if n else 1.0
        paired[metric] = {"equivariant_wins": wins, "baseline_wins": losses,
                          "n_pairs": n, "sign_test_two_sided_p": min(pval, 1.0)}
        print(f"\npaired by (seed, label fraction) on {metric}: "
              f"steerable wins {wins}/{n}, sign test p = {min(pval, 1.0):.2e}")
    summary["paired_comparison"] = paired

    # Accuracy change at 45 degrees, reported with an explicit caveat.
    #
    # This number is NOT a clean measure of rotation robustness. The unrotated
    # evaluation uses an exact centre crop while any rotated evaluation is
    # bilinearly resampled, and since training always resamples, models are
    # better matched to the blurred input. The difference therefore mixes
    # orientation sensitivity with resampling sensitivity, and can come out
    # negative. The clean measurement, using only lossless 90 degree
    # rotations, is produced by scripts/rotation_robustness.py.
    rob = {}
    for m in ("equivariant", "baseline"):
        rr = [r for r in recs if r["model"] == m]
        if not rr:
            continue
        d = [r["test_accuracy"] - r["test_accuracy_rot45"] for r in rr]
        rob[m] = {
            "mean_accuracy_change_at_45deg": float(np.mean(d)),
            "std": float(np.std(d, ddof=1)) if len(d) > 1 else 0.0,
            "n_runs": len(rr),
            "caveat": "confounded by bilinear resampling; see rotation_robustness.json",
        }
        print(
            f"\n{m}: accuracy change under a 45 degree rotation "
            f"{np.mean(d) * 100:+.2f} pp over {len(rr)} runs "
            f"(confounded by resampling, see rotation_robustness.json)"
        )
    summary["rotation_at_45deg_confounded"] = rob

    (OUT / "sweep_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {OUT / 'sweep_summary.json'}")

    # ------------------------------------------------------------- figure
    acc = summary["metrics"].get("test_accuracy")
    if not acc:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), facecolor=BG)
    for ax, key, nice in zip(
        axes, ("test_accuracy", "test_macro_f1"), ("Test accuracy", "Test macro-F1")
    ):
        e = summary["metrics"][key]["equivariant"]
        b = summary["metrics"][key]["baseline"]
        ax.set_facecolor(BG)
        for spine in ax.spines.values():
            spine.set_color(GRID)
        ax.grid(True, color=GRID, lw=0.6, alpha=0.7)
        ax.set_axisbelow(True)

        for d, c, lab in ((e, EQ_C, "C8-steerable"), (b, BL_C, "dense CNN")):
            f = np.array(d["fractions"])
            m = np.array(d["mean"])
            s = np.array(d["std"])
            ax.plot(f * 100, m, "-o", color=c, lw=2, ms=5, label=lab, zorder=3)
            ax.fill_between(f * 100, m - s, m + s, color=c, alpha=0.18, lw=0, zorder=2)

        tgt = summary["metrics"][key]["baseline_full_label_mean"]
        cf = summary["metrics"][key]["equivariant_crossing_fraction"]
        ax.axhline(tgt, color=BL_C, ls="--", lw=1.1, alpha=0.8, zorder=1)
        if cf:
            ax.axvline(cf * 100, color=EQ_C, ls=":", lw=1.4, alpha=0.9, zorder=1)
            ax.annotate(
                f"{cf * 100:.0f}% of labels",
                xy=(cf * 100, tgt),
                xytext=(cf * 100 * 1.15, tgt - 0.075),
                color=EQ_C, fontsize=9,
                arrowprops=dict(arrowstyle="->", color=EQ_C, lw=1),
            )
        ax.set_xscale("log")
        ax.set_xticks([5, 10, 20, 50, 100])
        ax.set_xticklabels(["5%", "10%", "20%", "50%", "100%"])
        ax.set_xlabel("fraction of training labels used", color=FG)
        ax.set_ylabel(nice, color=FG)
        ax.tick_params(colors=FG)
        ax.set_title(nice, color=FG, fontsize=11)
        ax.legend(facecolor=BG, edgecolor=GRID, labelcolor=FG, fontsize=9, loc="lower right")

    fig.suptitle(
        "Label efficiency on Galaxy10 DECaLS: architectural rotation invariance "
        "vs. the same symmetry taught by augmentation",
        color=FG, fontsize=12,
    )
    fig.text(
        0.5, 0.005,
        "Both models: identical convolutional FLOPs, identical augmentation "
        "(full-circle rotation and flips), shaded band = 1 s.d. over seeds. "
        "The dense CNN carries 11.8x more parameters.",
        ha="center", color="#9aa4bb", fontsize=8.5,
    )
    fig.tight_layout(rect=(0, 0.035, 1, 0.94))
    fig.savefig(FIG / "sample_efficiency.png", dpi=160, facecolor=BG)
    print(f"wrote {FIG / 'sample_efficiency.png'}")


if __name__ == "__main__":
    main()
