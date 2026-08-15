"""Run every stage that depends on the finished supervised sweep.

Ordered so that each stage's inputs exist when it starts:

  simclr              self-supervised encoder
  embed               encode catalogue, k-NN probe, cluster purity, UMAP
  vae                 generative model for latent geometry and reconstruction
  anomaly             needs both the VAE latents and the SimCLR embedding
  interpolate         needs the VAE and a trained classifier for the trace
  rotation_demo       needs both trained classifiers
  rotation_robustness needs every checkpoint
  aggregate_sweep     headline figure and summary
  export_web          package assets
  check_assets        verify the packaging numerically

Each stage is timed and failures stop the run with the offending command
printed, so a long unattended run does not silently produce partial output.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable


def stage(name: str, cmd: list[str]) -> float:
    print(f"\n{'=' * 70}\n{name}\n{'=' * 70}", flush=True)
    print("  " + " ".join(cmd[1:]), flush=True)
    t0 = time.perf_counter()
    r = subprocess.run(cmd, cwd=ROOT)
    dt = (time.perf_counter() - t0) / 60
    if r.returncode != 0:
        print(f"\nFAILED: {name} (exit {r.returncode}) after {dt:.1f} min")
        raise SystemExit(r.returncode)
    print(f"  [{name} finished in {dt:.1f} min]", flush=True)
    return dt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--simclr-steps", type=int, default=5000)
    p.add_argument("--vae-steps", type=int, default=3000)
    p.add_argument("--top", type=int, default=100)
    p.add_argument("--clf", default="results/sweep/equivariant_C8_f1_s0.pt")
    a = p.parse_args()

    if not (ROOT / a.clf).exists():
        raise SystemExit(f"classifier checkpoint not found: {a.clf}")

    t0 = time.perf_counter()
    stage("1/10 self-supervised encoder (SimCLR)",
          [PY, "-u", "-m", "src.simclr", "--steps", str(a.simclr_steps)])
    stage("2/10 embed catalogue, k-NN probe, purity, UMAP",
          [PY, "-u", "-m", "src.embed"])
    stage("3/10 convolutional VAE",
          [PY, "-u", "-m", "src.vae", "--steps", str(a.vae_steps)])
    stage("4/10 anomaly ranking",
          [PY, "-u", "-m", "src.anomaly", "--top", str(a.top)])
    stage("5/10 latent interpolation",
          [PY, "-u", "-m", "src.interpolate", "--clf", a.clf])
    stage("6/10 rotation demonstration",
          [PY, "-u", "scripts/rotation_demo.py", "--frac", "1.0"])
    stage("7/10 rotation robustness over all checkpoints",
          [PY, "-u", "scripts/rotation_robustness.py"])
    stage("8/10 aggregate sweep",
          [PY, "-u", "scripts/aggregate_sweep.py"])
    stage("9/10 export web assets",
          [PY, "-u", "scripts/export_web.py"])
    stage("10/10 verify assets",
          [PY, "-u", "scripts/check_assets.py"])

    print(f"\nALL STAGES COMPLETE in {(time.perf_counter() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
