"""Drive the full label-fraction sweep.

Ordered seed-major so that a complete (if noisy) curve for both
architectures exists after the first pass, and error bars accumulate with
later passes. Runs are skipped if their JSON record already exists, so the
sweep is resumable after an interruption.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.train_supervised import RESULTS, load_cached, run

FRACTIONS = [0.05, 0.1, 0.2, 0.5, 1.0]
MODELS = ["equivariant", "baseline"]


def tag(model, group, frac, seed):
    return f"{model}_{group if model == 'equivariant' else 'dense'}_f{frac:g}_s{seed}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--fractions", type=float, nargs="+", default=FRACTIONS)
    p.add_argument("--steps", type=int, default=2500)
    p.add_argument("--group", default="C8")
    p.add_argument("--bs", type=int, default=64)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--eval-every", type=int, default=150)
    p.add_argument("--patience", type=int, default=6)
    a = p.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)
    print("loading dataset into memory ...", flush=True)
    load_cached()

    jobs = [
        (m, f, s)
        for s in a.seeds
        for f in a.fractions
        for m in MODELS
    ]
    todo = [j for j in jobs if not (RESULTS / f"{tag(j[0], a.group, j[1], j[2])}.json").exists()]
    print(f"{len(jobs)} jobs, {len(jobs) - len(todo)} already done, {len(todo)} to run\n")

    t0 = time.perf_counter()
    for i, (m, f, s) in enumerate(todo, 1):
        el = (time.perf_counter() - t0) / 60
        eta = (el / (i - 1) * (len(todo) - i + 1)) if i > 1 else float("nan")
        print(
            f"[{i}/{len(todo)}] {m} frac={f:g} seed={s}   "
            f"elapsed {el:.0f} min, eta {eta:.0f} min",
            flush=True,
        )
        run(m, f, s, a.steps, a.group, a.bs, a.lr, a.eval_every, a.patience)
        print(flush=True)

    print(f"sweep complete in {(time.perf_counter() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
