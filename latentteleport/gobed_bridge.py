"""Python bridge to gobed for fast 512-dim text embeddings and similarity."""

from __future__ import annotations

import json
import subprocess
import shutil
from pathlib import Path

import numpy as np


def find_gobed_binary() -> str | None:
    """Locate the gobed binary."""
    for candidate in [
        shutil.which("bed"),
        str(Path.home() / "go" / "bin" / "bed"),
        str(Path.home() / "code" / "gobed" / "bed" / "bed"),
    ]:
        if candidate and Path(candidate).exists():
            return candidate
    return None


def embed_text(text: str, binary: str | None = None) -> np.ndarray | None:
    """Get 512-dim embedding for text using gobed."""
    binary = binary or find_gobed_binary()
    if not binary:
        return None
    try:
        result = subprocess.run(
            [binary, "embed", "--json", text],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            data = json.loads(result.stdout)
            return np.array(data["embedding"], dtype=np.float32)
    except (subprocess.TimeoutExpired, json.JSONDecodeError, KeyError):
        pass
    return None


def batch_embed(texts: list[str], binary: str | None = None) -> list[np.ndarray | None]:
    """Embed multiple texts."""
    return [embed_text(t, binary) for t in texts]


def similarity(text_a: str, text_b: str, binary: str | None = None) -> float | None:
    """Compute cosine similarity between two texts via gobed."""
    emb_a = embed_text(text_a, binary)
    emb_b = embed_text(text_b, binary)
    if emb_a is None or emb_b is None:
        return None
    norm_a = np.linalg.norm(emb_a)
    norm_b = np.linalg.norm(emb_b)
    if norm_a < 1e-8 or norm_b < 1e-8:
        return 0.0
    return float(np.dot(emb_a, emb_b) / (norm_a * norm_b))
