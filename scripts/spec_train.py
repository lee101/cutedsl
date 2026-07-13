"""Train the LatentWalker (drafts single steps, rolled out) and GapInterpolator
(teleports draft endpoints onto the big trajectory) on captured trajectories.

Walker: predict x_{t+1} from x_t (teacher forcing + short scheduled rollout).
Interp: predict x_{t+k} from (anchor x_t, walker draft x̂_{t+k}).

Run: .venv/bin/python scripts/spec_train.py --steps 16 --epochs 8
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))
from latentteleport.speculative import GapInterpolator, LatentWalker, SpecConfig  # noqa: E402

BASE = Path("/sdb-disk/latentteleport-spec")


def load(steps: int, size: int):
    files = sorted((BASE / f"trajs-{steps}step-{size}").glob("*.pt"))
    return [torch.load(f, map_location="cpu", weights_only=False)["latents"].float() for f in files]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=16)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--max-k", type=int, default=4)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    trajs = load(args.steps, args.size)
    n_train = int(len(trajs) * 0.8)
    train, test = trajs[:n_train], trajs[n_train:]
    S = trajs[0].shape[0]
    dev = args.device
    print(f"{len(train)} train / {len(test)} test trajs, S={S}")

    cfg = SpecConfig()
    walker = LatentWalker(cfg).to(dev)
    interp = GapInterpolator(cfg).to(dev)
    opt = torch.optim.AdamW(list(walker.parameters()) + list(interp.parameters()), lr=args.lr)
    nsteps = torch.tensor([float(args.steps)], device=dev)

    def batch_from(trajset, bs):
        anchors, targets, drafts_gt, ts, ks = [], [], [], [], []
        for _ in range(bs):
            tr = random.choice(trajset)
            k = random.randint(1, args.max_k)
            t = random.randint(1, S - 1 - k)
            anchors.append(tr[t])
            targets.append(tr[t + k])
            drafts_gt.append(tr[t : t + k + 1])
            ts.append(t)
            ks.append(k)
        return (
            torch.stack(anchors).to(dev),
            torch.stack(targets).to(dev),
            [d.to(dev) for d in drafts_gt],
            torch.tensor(ts, device=dev, dtype=torch.float32),
            torch.tensor(ks, device=dev, dtype=torch.float32),
        )

    iters = max(1, 60 * len(train) // args.batch)
    hist = []
    for ep in range(args.epochs):
        walker.train(); interp.train()
        t0, tot_w, tot_i = time.time(), 0.0, 0.0
        for it in range(iters):
            anchors, targets, drafts_gt, ts, ks = batch_from(train, args.batch)
            tfrac = ts / max(S - 1, 1)
            kfrac = ks / max(S - 1, 1)

            # walker: teacher-forced single steps at every offset in the window
            loss_w = 0.0
            for j in range(args.max_k):
                xs, ys, mask = [], [], []
                for b, d in enumerate(drafts_gt):
                    if j + 1 < d.shape[0]:
                        xs.append(d[j]); ys.append(d[j + 1]); mask.append(b)
                if not xs:
                    continue
                x = torch.stack(xs); y = torch.stack(ys)
                tf = (ts[mask] + j) / max(S - 1, 1)
                pred = walker(x, tf, nsteps.expand(x.shape[0]))
                loss_w = loss_w + F.mse_loss(pred, y)

            # interp: correct the walker's own rollout endpoint (detached)
            with torch.no_grad():
                walker.eval()
                draft_ends = []
                for b in range(anchors.shape[0]):
                    outs = walker.rollout(anchors[b : b + 1], int(ts[b]), int(ks[b]), args.steps)
                    draft_ends.append(outs[-1][0])
                walker.train()
            draft_end = torch.stack(draft_ends)
            pred = interp(anchors, draft_end, tfrac, kfrac)
            loss_i = F.mse_loss(pred, targets)

            loss = loss_w + loss_i
            opt.zero_grad(); loss.backward(); opt.step()
            tot_w += float(loss_w); tot_i += float(loss_i)

        walker.eval(); interp.eval()
        with torch.no_grad():
            anchors, targets, _, ts, ks = batch_from(test, 64)
            draft_ends = torch.stack([
                walker.rollout(anchors[b : b + 1], int(ts[b]), int(ks[b]), args.steps)[-1][0]
                for b in range(anchors.shape[0])
            ])
            pred = interp(anchors, draft_ends, ts / max(S - 1, 1), ks / max(S - 1, 1))
            move = (targets - anchors).flatten(1).norm(dim=1).clamp_min(1e-8)
            rel_interp = ((pred - targets).flatten(1).norm(dim=1) / move).mean().item()
            rel_walker = ((draft_ends - targets).flatten(1).norm(dim=1) / move).mean().item()
        print(
            f"ep{ep}: walker {tot_w/iters:.4f} interp {tot_i/iters:.4f} "
            f"| test relL2 walker {rel_walker:.3f} interp {rel_interp:.3f} ({time.time()-t0:.0f}s)",
            flush=True,
        )
        hist.append({"epoch": ep, "rel_walker": rel_walker, "rel_interp": rel_interp})

    ckpt_dir = BASE / f"ckpt-{args.steps}step-{args.size}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"walker": walker.state_dict(), "interp": interp.state_dict(), "cfg": cfg.__dict__, "hist": hist}, ckpt_dir / "spec.pt")
    (ckpt_dir / "hist.json").write_text(json.dumps(hist, indent=1))
    print(f"saved {ckpt_dir}/spec.pt")


if __name__ == "__main__":
    main()
