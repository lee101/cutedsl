"""Configuration dataclasses for latent teleportation experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


@dataclass
class TeleportConfig:
    model_id: str = "Tongyi-MAI/Z-Image-Turbo"
    device: str = "cuda"
    dtype: str = "bfloat16"
    height: int = 512
    width: int = 512
    num_steps: int = 20
    guidance_scale: float = 0.0
    cache_dir: str = "/vfast/latentteleport/cache"
    seed: int = 42


@dataclass
class TokenizerConfig:
    strategy: Literal["nlp", "curated", "clip"] = "nlp"
    vocab_size: int = 1000
    spacy_model: str = "en_core_web_sm"
    clip_model_id: str = ""
    gobed_binary: str = ""
    curated_vocab_path: str = ""


@dataclass
class CombinerConfig:
    method: Literal["slerp", "neural", "tree"] = "slerp"
    refinement_steps: int = 5
    slerp_t: float = 0.5
    neural_hidden_dim: int = 1024
    neural_num_layers: int = 4
    teleport_timestep: float = 0.3
    latent_channels: int = 16
    latent_h: int = 64
    latent_w: int = 64
    clip_dim: int = 2560


@dataclass
class TrainConfig:
    batch_size: int = 8
    lr: float = 1e-4
    epochs: int = 50
    weight_mse: float = 1.0
    weight_perceptual: float = 0.1
    weight_lpips: float = 0.05
    checkpoint_dir: str = "/vfast/latentteleport/checkpoints"
    dataset_dir: str = "/vfast/latentteleport/datasets"
    log_interval: int = 50
    save_interval: int = 5


@dataclass
class EvalConfig:
    metrics: list[str] = field(default_factory=lambda: ["fid", "lpips", "ssim", "psnr"])
    num_samples: int = 1000
    reference_dir: str = "/vfast/latentteleport/eval/references"
    output_dir: str = "/vfast/latentteleport/eval"
    batch_size: int = 4


@dataclass
class AblationConfig:
    tokenizer_strategies: list[str] = field(default_factory=lambda: ["nlp", "curated", "clip"])
    combiner_methods: list[str] = field(default_factory=lambda: ["slerp", "neural", "tree"])
    refinement_steps_list: list[int] = field(default_factory=lambda: [0, 1, 3, 5, 7, 10])
    teleport_timesteps: list[float] = field(default_factory=lambda: [0.1, 0.3, 0.5])
    vocab_sizes: list[int] = field(default_factory=lambda: [100, 500, 1000, 5000])
