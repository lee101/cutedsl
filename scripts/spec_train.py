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
        # one shared (t, k) per batch so walker rollouts and teacher steps run
        # batched. Window starts at t-1 so the anchor's incoming delta exists;
        # excludes t=0 (pure-noise anchor) and the duplicated no-op final latent.
        k = random.randint(1, args.max_k)
        t = random.randint(1, S - 2 - k)
        trs = [random.choice(trajset) for _ in range(bs)]
        window = torch.stack([tr[t - 1 : t + k + 1] for tr in trs]).to(dev)  # [B, k+2, C, H, W]
        return window, t, k

    iters = max(1, 60 * len(train) // args.batch)
    hist = []
    for ep in range(args.epochs):
        walker.train(); interp.train()
        t0, tot_w, tot_i = time.time(), 0.0, 0.0
        for it in range(iters):
            window, t, k = batch_from(train, args.batch)
            anchors, targets = window[:, 1], window[:, -1]
            anchor_delta = window[:, 1] - window[:, 0]
            B = anchors.shape[0]
            tfrac = torch.full((B,), t / max(S - 1, 1), device=dev)
            kfrac = torch.full((B,), k / max(S - 1, 1), device=dev)

            # walker: teacher-forced single steps at every offset (batched),
            # delta = the real incoming movement at each offset
            loss_w = 0.0
            for j in range(k):
                tf = torch.full((B,), (t + j) / max(S - 1, 1), device=dev)
                d = window[:, j + 1] - window[:, j]
                pred = walker(window[:, j + 1], d, tf, nsteps.expand(B))
                loss_w = loss_w + F.mse_loss(pred, window[:, j + 2])

            # interp: correct the walker's own batched rollout endpoint (detached)
            with torch.no_grad():
                walker.eval()
                draft_end = walker.rollout(anchors, anchor_delta, t, k, args.steps)[-1]
                walker.train()
            pred = interp(anchors, anchor_delta, draft_end, tfrac, kfrac)
            loss_i = F.mse_loss(pred, targets)

            loss = loss_w + loss_i
            opt.zero_grad(); loss.backward(); opt.step()
            tot_w += float(loss_w); tot_i += float(loss_i)

        walker.eval(); interp.eval()
        with torch.no_grad():
            rels_w, rels_i, rels_t = [], [], []
            for _ in range(16):
                window, t, k = batch_from(test, 32)
                anchors, targets = window[:, 1], window[:, -1]
                anchor_delta = window[:, 1] - window[:, 0]
                B = anchors.shape[0]
                draft_end = walker.rollout(anchors, anchor_delta, t, k, args.steps)[-1]
                pred = interp(anchors, anchor_delta, draft_end,
                              torch.full((B,), t / max(S - 1, 1), device=dev),
                              torch.full((B,), k / max(S - 1, 1), device=dev))
                move = (targets - anchors).flatten(1).norm(dim=1).clamp_min(1e-8)
                rels_i.append(((pred - targets).flatten(1).norm(dim=1) / move).mean().item())
                rels_w.append(((draft_end - targets).flatten(1).norm(dim=1) / move).mean().item())
                tay = anchors + k * anchor_delta
                rels_t.append(((tay - targets).flatten(1).norm(dim=1) / move).mean().item())
            rel_interp = sum(rels_i) / len(rels_i)
            rel_walker = sum(rels_w) / len(rels_w)
            rel_taylor = sum(rels_t) / len(rels_t)
        print(
            f"ep{ep}: walker {tot_w/iters:.4f} interp {tot_i/iters:.4f} "
            f"| test relL2 walker {rel_walker:.3f} interp {rel_interp:.3f} taylor {rel_taylor:.3f} ({time.time()-t0:.0f}s)",
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
