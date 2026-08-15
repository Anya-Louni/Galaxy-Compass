"""Label-fraction sample-efficiency study.

One run trains one architecture on one stratified fraction of the training
split with one seed, and writes a JSON record of its test performance.

Protocol notes that matter for interpreting the result
------------------------------------------------------
* Both architectures receive the identical augmentation pipeline, including
  full-circle rotation and reflection. The dense CNN is therefore told about
  the symmetry in the strongest way a data pipeline can tell it. The
  question under test is whether being told is as good as being built that
  way.

* Both architectures are trained under an identical step budget, not an
  identical epoch budget. At a fixed number of epochs a run on 100% of the
  labels would see twenty times more gradient updates than a run on 5%,
  which would confound label efficiency with optimisation budget.

* The validation set used for model selection is held fixed at 10% of the
  data for every label fraction. A strictly realistic low-label protocol
  would shrink the validation set too. Keeping it fixed slightly flatters
  the low-fraction runs, but it does so identically for both architectures,
  so the comparison between them is unaffected. This is stated as a
  limitation rather than hidden.

* Class counts range from 334 to 2645, so macro-F1 is reported alongside
  accuracy and is the more honest headline for the rare classes.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, f1_score

from .augment import eval_view, train_view
from .data import CLASS_NAMES, PROC, ROOT
from .models import build, count_params

RESULTS = ROOT / "results" / "sweep"


def stratified_subset(labels: np.ndarray, idx: np.ndarray, frac: float, seed: int):
    """Draw a class-stratified subset, keeping at least a few of every class."""
    if frac >= 1.0:
        return idx
    rng = np.random.default_rng(seed)
    keep = []
    for c in range(len(CLASS_NAMES)):
        ci = idx[labels[idx] == c]
        rng.shuffle(ci)
        n = max(4, int(round(frac * len(ci))))
        keep.append(ci[:n])
    out = np.concatenate(keep)
    rng.shuffle(out)
    return out


@torch.no_grad()
def evaluate(model, imgs, labels, idx, mean, std, bs=256, angle=0.0):
    model.eval()
    preds = np.empty(len(idx), dtype=np.int64)
    for s in range(0, len(idx), bs):
        b = idx[s : s + bs]
        x = eval_view(imgs[b], mean, std, angle=angle)
        preds[s : s + len(b)] = model(x).argmax(1).numpy()
    y = labels[idx]
    return preds, float((preds == y).mean()), float(f1_score(y, preds, average="macro"))


_CACHE = {}


def load_cached():
    """Load the 0.5 GB image tensor once per process, not once per run."""
    if not _CACHE:
        images = np.load(PROC / "images_u8.npy")
        _CACHE["imgs"] = torch.from_numpy(images).permute(0, 3, 1, 2).contiguous()
        _CACHE["labels"] = np.load(PROC / "labels.npy")
        meta = dict(np.load(PROC / "meta.npz"))
        _CACHE["meta"] = meta
        _CACHE["mean"] = torch.tensor(meta["mean"], dtype=torch.float32)
        _CACHE["std"] = torch.tensor(meta["std"], dtype=torch.float32)
    return _CACHE


def run(model_name: str, frac: float, seed: int, steps: int, group: str,
        bs: int, lr: float, eval_every: int, patience: int) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)

    c = load_cached()
    imgs, labels, meta = c["imgs"], c["labels"], c["meta"]
    mean, std = c["mean"], c["std"]

    tr_idx = stratified_subset(labels, meta["train"], frac, seed)
    va_idx, te_idx = meta["val"], meta["test"]

    model = build(model_name, n_classes=len(CLASS_NAMES), group=group)
    n_par = count_params(model)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    gen = torch.Generator().manual_seed(seed)

    warmup = max(20, steps // 20)

    def lr_at(t):
        if t < warmup:
            return lr * t / warmup
        p = (t - warmup) / max(1, steps - warmup)
        return lr * 0.5 * (1 + math.cos(math.pi * p))

    best = {"f1": -1.0, "state": None, "step": 0}
    stale = 0
    history = []
    t0 = time.perf_counter()

    for step in range(1, steps + 1):
        model.train()
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        b = tr_idx[torch.randint(len(tr_idx), (bs,), generator=gen).numpy()]
        x = train_view(imgs[b], mean, std, gen=gen)
        y = torch.from_numpy(labels[b])
        opt.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(x), y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()

        if step % eval_every == 0 or step == steps:
            _, acc, f1 = evaluate(model, imgs, labels, va_idx, mean, std)
            history.append({"step": step, "loss": float(loss), "val_acc": acc, "val_f1": f1})
            el = time.perf_counter() - t0
            print(
                f"  step {step:5d}/{steps}  loss {float(loss):.3f}  "
                f"val acc {acc:.4f}  val macroF1 {f1:.4f}  [{el / 60:.1f} min]",
                flush=True,
            )
            if f1 > best["f1"]:
                best = {
                    "f1": f1,
                    "state": {k: v.clone() for k, v in model.state_dict().items()},
                    "step": step,
                }
                stale = 0
            else:
                stale += 1
                if stale >= patience:
                    print(f"  early stop at step {step} (best step {best['step']})")
                    break

    model.load_state_dict(best["state"])
    preds, acc, f1 = evaluate(model, imgs, labels, te_idx, mean, std)
    y = labels[te_idx]

    # Robustness to an orientation the model never saw in this exact form.
    _, acc45, f145 = evaluate(model, imgs, labels, te_idx, mean, std, angle=math.pi / 4)

    rec = {
        "model": model_name,
        "group": group if model_name == "equivariant" else None,
        "label_fraction": frac,
        "seed": seed,
        "n_train_labels": int(len(tr_idx)),
        "n_params": int(n_par),
        "steps_run": int(history[-1]["step"]) if history else 0,
        "best_step": int(best["step"]),
        "test_accuracy": acc,
        "test_macro_f1": f1,
        "test_accuracy_rot45": acc45,
        "test_macro_f1_rot45": f145,
        "per_class_f1": f1_score(y, preds, average=None).tolist(),
        "confusion_matrix": confusion_matrix(y, preds).tolist(),
        "history": history,
        "minutes": (time.perf_counter() - t0) / 60,
    }

    RESULTS.mkdir(parents=True, exist_ok=True)
    tag = f"{model_name}_{group if model_name == 'equivariant' else 'dense'}_f{frac:g}_s{seed}"
    (RESULTS / f"{tag}.json").write_text(json.dumps(rec, indent=2))
    torch.save(best["state"], RESULTS / f"{tag}.pt")
    print(
        f"  -> test acc {acc:.4f}  macroF1 {f1:.4f}  "
        f"(rot45: {acc45:.4f} / {f145:.4f})  {rec['minutes']:.1f} min"
    )
    return rec


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=["equivariant", "baseline"])
    p.add_argument("--frac", type=float, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=2500)
    p.add_argument("--group", default="C8")
    p.add_argument("--bs", type=int, default=64)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--eval-every", type=int, default=150)
    p.add_argument("--patience", type=int, default=6)
    a = p.parse_args()
    print(f"[{a.model} {a.group if a.model == 'equivariant' else 'dense'}] "
          f"frac={a.frac} seed={a.seed}")
    run(a.model, a.frac, a.seed, a.steps, a.group, a.bs, a.lr, a.eval_every, a.patience)


if __name__ == "__main__":
    main()
