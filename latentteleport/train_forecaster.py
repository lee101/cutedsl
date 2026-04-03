"""Train a neural latent step forecaster from cached 20-step trajectories."""

from __future__ import annotations

import argparse
import json
import logging
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from latentteleport.cache import LatentCache
from latentteleport.step_forecaster import LatentStepForecaster, StepForecasterConfig
from latentteleport.tokenizer import VisualUnit

log = logging.getLogger(__name__)


class TrajectoryDataset(Dataset):
    def __init__(self, cache: LatentCache, min_step: int = 0, max_step: int = 18):
        self.cache = cache
        self.examples: list[tuple[str, int]] = []
        for unit_id, _text, num_cached_steps in cache.list_units():
            if num_cached_steps < 2:
                continue
            for step in range(min_step, max_step + 1):
                self.examples.append((unit_id, step))

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int):
        unit_id, step = self.examples[idx]
        unit = self.cache.load_unit_by_id(unit_id)
        if unit is None:
            raise IndexError(unit_id)
        latent = self.cache.load_latent(unit, step)
        next_latent = self.cache.load_latent(unit, step + 1)
        text_emb = self.cache.load_text_embedding(unit)
        if latent is None or next_latent is None:
            raise IndexError((unit_id, step))
        if text_emb is None:
            text_emb = torch.zeros(2560)
        return latent.float(), next_latent.float(), text_emb.float(), torch.tensor(float(step))


def train_forecaster(
    cache: LatentCache,
    checkpoint_dir: str,
    device: str = "cuda",
    batch_size: int = 8,
    epochs: int = 10,
    lr: float = 1e-4,
) -> LatentStepForecaster:
    dataset = TrajectoryDataset(cache)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    model = LatentStepForecaster().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    ckpt_dir = Path(checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    history = []
    best = float("inf")

    for epoch in range(epochs):
        model.train()
        total = 0.0
        count = 0
        t0 = time.time()
        for latent, next_latent, text_emb, timestep in loader:
            latent = latent.to(device)
            next_latent = next_latent.to(device)
            text_emb = text_emb.to(device)
            timestep = timestep.to(device)
            pred = model(latent, timestep, text_emb)
            loss = F.mse_loss(pred, next_latent)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item() * latent.shape[0]
            count += latent.shape[0]
        avg = total / max(count, 1)
        history.append({"epoch": epoch + 1, "loss": avg})
        log.info(f"epoch {epoch+1}/{epochs} loss={avg:.6f} time={time.time()-t0:.1f}s")
        if avg < best:
            best = avg
            torch.save(model.state_dict(), ckpt_dir / "best_forecaster.pt")

    torch.save(model.state_dict(), ckpt_dir / "final_forecaster.pt")
    (ckpt_dir / "forecaster_history.json").write_text(json.dumps(history, indent=2))
    return model


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    parser = argparse.ArgumentParser(description="Train latent step forecaster")
    parser.add_argument("--cache-dir", default="/vfast/latentteleport/cache")
    parser.add_argument("--checkpoint-dir", default="/vfast/latentteleport/checkpoints")
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()

    cache = LatentCache(args.cache_dir, resolution=(args.height, args.width))
    train_forecaster(
        cache,
        checkpoint_dir=args.checkpoint_dir,
        device=args.device,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
    )


if __name__ == "__main__":
    main()

