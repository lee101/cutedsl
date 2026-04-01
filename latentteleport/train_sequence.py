"""Train the order-aware sequence combiner.

Takes visual unit embedding sequences -> target combined embedding.
Uses the sequence combiner transformer or the latent-level combiner.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from latentteleport.cache import LatentCache
from latentteleport.sequence_combiner import (
    SequenceCombinerTransformer,
    SequenceCombinerLatent,
    PositionalWeightedMean,
)
from latentteleport.tokenizer import VisualUnit

log = logging.getLogger(__name__)


class SequenceDataset(Dataset):
    """Dataset: ordered visual unit embeddings -> compound target embedding/latent."""

    def __init__(self, metadata_path: str, cache: LatentCache, target_step: int, max_units: int = 8):
        self.cache = cache
        self.target_step = target_step
        self.max_units = max_units
        self.records = []
        with open(metadata_path) as f:
            for line in f:
                rec = json.loads(line)
                self.records.append(rec)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        # Parse units from the compound prompt
        units_text = [rec.get("unit_a", ""), rec.get("unit_b", "")]
        units_text = [t for t in units_text if t]
        compound_text = rec.get("compound", ", ".join(units_text))

        # Load embeddings and latents
        unit_embs = []
        unit_lats = []
        for ut in units_text:
            u = VisualUnit.from_text(ut)
            emb = self.cache.load_text_embedding(u)
            lat = self.cache.load_latent(u, self.target_step)
            if emb is not None and lat is not None:
                unit_embs.append(emb)
                unit_lats.append(lat)

        target_unit = VisualUnit.from_text(compound_text)
        target_emb = self.cache.load_text_embedding(target_unit)
        target_lat = self.cache.load_latent(target_unit, self.target_step)

        if not unit_embs or target_emb is None or target_lat is None:
            n = len(units_text) or 2
            return (
                torch.zeros(n, 2560),
                torch.zeros(n, 16, 1, 64, 64),
                torch.zeros(2560),
                torch.zeros(16, 1, 64, 64),
                torch.zeros(n, dtype=torch.bool),
                False,
            )

        # Pad to max_units
        n = len(unit_embs)
        mask = torch.ones(n, dtype=torch.bool)
        emb_stack = torch.stack(unit_embs)
        lat_stack = torch.stack(unit_lats)

        return emb_stack, lat_stack, target_emb, target_lat, mask, True


def train_sequence_combiner(
    cache: LatentCache,
    metadata_path: str,
    mode: str = "embedding",  # "embedding" or "latent"
    clip_dim: int = 2560,
    num_heads: int = 8,
    num_layers: int = 2,
    hidden_dim: int = 512,
    lr: float = 1e-4,
    epochs: int = 50,
    batch_size: int = 8,
    checkpoint_dir: str = "/vfast/latentteleport/checkpoints",
    device: str = "cuda",
    teleport_timestep: float = 0.3,
):
    target_step = int(20 * teleport_timestep)

    dataset = SequenceDataset(metadata_path, cache, target_step)
    # Can't easily batch variable-length sequences, use batch_size=1 for now
    loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)

    if mode == "embedding":
        model = SequenceCombinerTransformer(
            embed_dim=clip_dim, num_heads=num_heads, num_layers=num_layers,
        ).to(device)
    elif mode == "latent":
        model = SequenceCombinerLatent(
            latent_channels=16, latent_spatial=64*64, clip_dim=clip_dim,
            hidden_dim=hidden_dim, num_heads=num_heads,
        ).to(device)
    else:
        model = PositionalWeightedMean(embed_dim=clip_dim).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    ckpt_dir = Path(checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    param_count = sum(p.numel() for p in model.parameters())
    log.info(f"Sequence combiner ({mode}): {param_count:,} params")

    best_loss = float("inf")
    history = []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        num_valid = 0
        t0 = time.time()

        for batch_idx, (unit_embs, unit_lats, target_emb, target_lat, mask, valid) in enumerate(loader):
            if not valid.any():
                continue

            unit_embs = unit_embs.to(device, torch.float32)
            target_emb = target_emb.to(device, torch.float32)
            mask = mask.to(device)

            if mode == "embedding":
                pred = model(unit_embs, mask)  # (1, D)
                loss = F.mse_loss(pred, target_emb)
            elif mode == "latent":
                unit_lats = unit_lats.to(device, torch.float32)
                target_lat = target_lat.to(device, torch.float32)
                pred = model(unit_lats, unit_embs, mask)  # (1, C, H, W)
                loss = F.mse_loss(pred, target_lat.squeeze(0))
            else:
                pred = model(unit_embs, mask)
                loss = F.mse_loss(pred, target_emb)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            num_valid += 1

        avg_loss = epoch_loss / max(num_valid, 1)
        elapsed = time.time() - t0
        log.info(f"Epoch {epoch+1}/{epochs}: loss={avg_loss:.6f} ({elapsed:.1f}s)")
        history.append({"epoch": epoch + 1, "loss": avg_loss})

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), ckpt_dir / f"best_sequence_{mode}.pt")

    torch.save(model.state_dict(), ckpt_dir / f"final_sequence_{mode}.pt")
    (ckpt_dir / f"sequence_{mode}_history.json").write_text(json.dumps(history, indent=2))
    log.info(f"Done. Best loss: {best_loss:.6f}")
    return model


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    parser = argparse.ArgumentParser(description="Train sequence combiner")
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--cache-dir", default="/vfast/latentteleport/cache")
    parser.add_argument("--checkpoint-dir", default="/vfast/latentteleport/checkpoints")
    parser.add_argument("--mode", default="embedding", choices=["embedding", "latent", "weighted_mean"])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--teleport-timestep", type=float, default=0.3)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    cache = LatentCache(args.cache_dir, resolution=(512, 512))
    train_sequence_combiner(
        cache, args.metadata, mode=args.mode,
        lr=args.lr, epochs=args.epochs,
        checkpoint_dir=args.checkpoint_dir,
        device=args.device, teleport_timestep=args.teleport_timestep,
    )


if __name__ == "__main__":
    main()
