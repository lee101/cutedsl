"""CuteAnima: accelerated inference for the Anima 2.9B Cosmos diffusion transformer."""

from .loader import load_pipeline, load_transformer, remap_checkpoint
from .patch import apply_fused_blocks, remove_fused_blocks
from .runner import AnimaRunner

__all__ = [
    "load_pipeline",
    "load_transformer",
    "remap_checkpoint",
    "apply_fused_blocks",
    "remove_fused_blocks",
    "AnimaRunner",
]
