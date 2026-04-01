"""Training loop for the neural latent combiner."""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from latentteleport.cache import LatentCache
from latentteleport.combiner import NeuralCombinerNet
from latentteleport.config import CombinerConfig, TrainConfig
from latentteleport.tokenizer import VisualUnit

log = logging.getLogger(__name__)


class PairDataset(Dataset):
    """Dataset of (unit_a_latent, unit_b_latent, compound_latent) triples."""

    def __init__(self, metadata_path: str, cache: LatentCache, target_step: int):
        self.cache = cache
        self.target_step = target_step
        self.pairs = []
        with open(metadata_path) as f:
            for line in f:
                rec = json.loads(line)
                self.pairs.append(rec)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        rec = self.pairs[idx]
        unit_a = VisualUnit.from_text(rec["unit_a"])
        unit_b = VisualUnit.from_text(rec["unit_b"])
        unit_compound = VisualUnit.from_text(rec["compound"])

        lat_a = self.cache.load_latent(unit_a, self.target_step)
        lat_b = self.cache.load_latent(unit_b, self.target_step)
        lat_target = self.cache.load_latent(unit_compound, self.target_step)

        emb_a = self.cache.load_text_embedding(unit_a)
        emb_b = self.cache.load_text_embedding(unit_b)

        if any(x is None for x in [lat_a, lat_b, lat_target, emb_a, emb_b]):
            # Return zeros as fallback; collate will handle
            shape = (16, 1, 64, 64)
            return (
                torch.zeros(shape), torch.zeros(shape), torch.zeros(shape),
                torch.zeros(2560), torch.zeros(2560), False,
            )

        return lat_a, lat_b, lat_target, emb_a, emb_b, True


def collate_pairs(batch):
    lat_a, lat_b, lat_t, emb_a, emb_b, valid = zip(*batch)
    valid = torch.tensor(valid)
    return (
        torch.stack(lat_a), torch.stack(lat_b), torch.stack(lat_t),
        torch.stack(emb_a), torch.stack(emb_b), valid,
    )


def train_combiner(
    train_config: TrainConfig,
    combiner_config: CombinerConfig,
    cache: LatentCache,
    metadata_path: str,
    device: str = "cuda",
):
    target_step = int(20 * combiner_config.teleport_timestep)
    dataset = PairDataset(metadata_path, cache, target_step)
    loader = DataLoader(
        dataset, batch_size=train_config.batch_size,
        shuffle=True, collate_fn=collate_pairs, num_workers=0,
    )

    latent_dim = combiner_config.latent_channels * combiner_config.latent_h * combiner_config.latent_w
    net = NeuralCombinerNet(
        latent_dim=latent_dim,
        clip_dim=combiner_config.clip_dim,
        hidden_dim=combiner_config.neural_hidden_dim,
        num_layers=combiner_config.neural_num_layers,
    ).to(device)

    optimizer = torch.optim.AdamW(net.parameters(), lr=train_config.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=train_config.epochs * len(loader),
    )

    ckpt_dir = Path(train_config.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    param_count = sum(p.numel() for p in net.parameters())
    log.info(f"Neural combiner: {param_count:,} params, {len(dataset)} pairs")

    best_loss = float("inf")
    history = []

    for epoch in range(train_config.epochs):
        net.train()
        epoch_loss = 0.0
        num_valid = 0
        t0 = time.time()

        for batch_idx, (lat_a, lat_b, lat_t, emb_a, emb_b, valid) in enumerate(loader):
            if not valid.any():
                continue
            mask = valid.to(device)
            lat_a = lat_a[mask].to(device, torch.float32)
            lat_b = lat_b[mask].to(device, torch.float32)
            lat_t = lat_t[mask].to(device, torch.float32)
            emb_a = emb_a[mask].to(device, torch.float32)
            emb_b = emb_b[mask].to(device, torch.float32)

            pred = net(lat_a, lat_b, emb_a, emb_b)
            loss_mse = F.mse_loss(pred, lat_t.reshape(pred.shape))
            loss = train_config.weight_mse * loss_mse

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item() * lat_a.shape[0]
            num_valid += lat_a.shape[0]

            if batch_idx % train_config.log_interval == 0:
                log.info(f"  [{batch_idx}/{len(loader)}] loss={loss.item():.6f}")

        avg_loss = epoch_loss / max(num_valid, 1)
        elapsed = time.time() - t0
        log.info(f"Epoch {epoch+1}/{train_config.epochs}: loss={avg_loss:.6f} ({elapsed:.1f}s)")

        history.append({"epoch": epoch + 1, "loss": avg_loss, "lr": scheduler.get_last_lr()[0]})

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(net.state_dict(), ckpt_dir / "best_combiner.pt")

        if (epoch + 1) % train_config.save_interval == 0:
            torch.save(net.state_dict(), ckpt_dir / f"combiner_epoch{epoch+1}.pt")

    torch.save(net.state_dict(), ckpt_dir / "final_combiner.pt")
    (ckpt_dir / "train_history.json").write_text(json.dumps(history, indent=2))
    log.info(f"Training done. Best loss: {best_loss:.6f}")
    return net


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    parser = argparse.ArgumentParser(description="Train neural latent combiner")
    parser.add_argument("--metadata", required=True, help="Path to pair_metadata.jsonl")
    parser.add_argument("--cache-dir", default="/vfast/latentteleport/cache")
    parser.add_argument("--checkpoint-dir", default="/vfast/latentteleport/checkpoints")
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--teleport-timestep", type=float, default=0.3)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    cache = LatentCache(args.cache_dir, resolution=(args.height, args.width))
    train_cfg = TrainConfig(
        batch_size=args.batch_size, lr=args.lr, epochs=args.epochs,
        checkpoint_dir=args.checkpoint_dir,
    )
    combiner_cfg = CombinerConfig(
        method="neural", teleport_timestep=args.teleport_timestep,
        latent_h=args.height // 8, latent_w=args.width // 8,
    )
    train_combiner(train_cfg, combiner_cfg, cache, args.metadata, args.device)


if __name__ == "__main__":
    main()
