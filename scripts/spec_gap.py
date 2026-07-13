"""Gap analysis: how predictable is the big model's latent trajectory?

For each anchor step t and horizon k, compare predictors of x_{t+k}:
  identity      x_t                       (skip steps, no walk)
  taylor1       x_t + k(x_t - x_{t-1})    (walk blind along last delta)
  affine        per-(t,k) least-squares   a*x_t + b (fit train, eval held out)
Metric: relL2 = ||pred - x_{t+k}|| / ||x_{t+k} - x_t||  (1.0 = as bad as not moving)

Run: .venv/bin/python scripts/spec_gap.py --steps 16
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

BASE = Path("/sdb-disk/latentteleport-spec")


def load(steps: int, size: int):
    files = sorted((BASE / f"trajs-{steps}step-{size}").glob("*.pt"))
    trajs = [torch.load(f, map_location="cpu", weights_only=False)["latents"].float() for f in files]
    return trajs


def rel_l2(pred, target, anchor):
    move = (target - anchor).flatten(1).norm(dim=1).clamp_min(1e-8)
    err = (pred - target).flatten(1).norm(dim=1)
    return (err / move).mean().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=16)
    ap.add_argument("--size", type=int, default=512)
    args = ap.parse_args()

    trajs = load(args.steps, args.size)
    n_train = int(len(trajs) * 0.8)
    train, test = trajs[:n_train], trajs[n_train:]
    print(f"{len(trajs)} trajs ({len(train)} train / {len(test)} test), shape {trajs[0].shape}")

    S = trajs[0].shape[0]
    results = {}
    for k in (1, 2, 3, 4):
        for t in range(1, S - k):
            a_tr = torch.stack([tr[t] for tr in train])
            y_tr = torch.stack([tr[t + k] for tr in train])
            a_te = torch.stack([tr[t] for tr in test])
            y_te = torch.stack([tr[t + k] for tr in test])
            prev_te = torch.stack([tr[t - 1] for tr in test])

            row = {
                "identity": rel_l2(a_te, y_te, a_te),
                "taylor1": rel_l2(a_te + k * (a_te - prev_te), y_te, a_te),
            }
            # per-(t,k) scalar affine: y ≈ a*x + b, least squares over train set
            xf, yf = a_tr.flatten(), y_tr.flatten()
            xc, yc = xf - xf.mean(), yf - yf.mean()
            a = (xc @ yc) / (xc @ xc).clamp_min(1e-8)
            b = yf.mean() - a * xf.mean()
            row["affine"] = rel_l2(a * a_te + b, y_te, a_te)
            results[f"t{t}+k{k}"] = {m: round(v, 4) for m, v in row.items()}

    for k in (1, 2, 3, 4):
        rows = {key: v for key, v in results.items() if key.endswith(f"k{k}")}
        avg = {m: round(sum(r[m] for r in rows.values()) / len(rows), 4) for m in ("identity", "taylor1", "affine")}
        print(f"k={k}: {avg}")

    out = BASE / f"gap-{args.steps}step-{args.size}.json"
    out.write_text(json.dumps(results, indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
