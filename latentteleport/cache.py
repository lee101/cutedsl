"""Latent cache: safetensors storage + SQLite index for visual units."""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file, save_file

from latentteleport.tokenizer import VisualUnit


class LatentCache:
    def __init__(self, cache_dir: str, resolution: tuple[int, int] = (512, 512)):
        self.cache_dir = Path(cache_dir)
        self.resolution = resolution
        self._res_dir = self.cache_dir / f"{resolution[0]}x{resolution[1]}"
        self._units_dir = self._res_dir / "units"
        self._units_dir.mkdir(parents=True, exist_ok=True)
        self._db_path = self._res_dir / "index.sqlite"
        self._embedding_index_dirty = True
        self._embedding_index: dict[int, tuple[np.ndarray, list[tuple[str, str]]]] = {}
        self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(str(self._db_path))
        conn.execute("""
            CREATE TABLE IF NOT EXISTS units (
                unit_id TEXT PRIMARY KEY,
                unit_text TEXT NOT NULL,
                file_path TEXT NOT NULL,
                clip_embedding BLOB,
                gobed_embedding BLOB,
                num_cached_steps INTEGER DEFAULT 0,
                created_at REAL,
                metadata TEXT
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_text ON units(unit_text)")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS bigrams (
                bigram_id TEXT PRIMARY KEY,
                unit_a_id TEXT NOT NULL,
                unit_b_id TEXT NOT NULL,
                unit_a_text TEXT NOT NULL,
                unit_b_text TEXT NOT NULL,
                file_path TEXT NOT NULL,
                num_cached_steps INTEGER DEFAULT 0,
                created_at REAL,
                metadata TEXT,
                FOREIGN KEY (unit_a_id) REFERENCES units(unit_id),
                FOREIGN KEY (unit_b_id) REFERENCES units(unit_id)
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_bigram_pair ON bigrams(unit_a_id, unit_b_id)")
        conn.commit()
        conn.close()

    def _conn(self) -> sqlite3.Connection:
        return sqlite3.connect(str(self._db_path))

    def _unit_path(self, unit: VisualUnit) -> Path:
        return self._units_dir / unit.unit_id

    def unit_dir(self, unit: VisualUnit) -> Path:
        d = self._unit_path(unit)
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _get_embedding_index(self, embedding_dim: int) -> tuple[np.ndarray, list[tuple[str, str]]]:
        if self._embedding_index_dirty:
            self._embedding_index.clear()
            self._embedding_index_dirty = False

        cached = self._embedding_index.get(embedding_dim)
        if cached is not None:
            return cached

        conn = self._conn()
        rows = conn.execute(
            "SELECT unit_id, unit_text, clip_embedding FROM units WHERE clip_embedding IS NOT NULL"
        ).fetchall()
        conn.close()

        metadata: list[tuple[str, str]] = []
        embeddings: list[np.ndarray] = []
        for uid, text, blob in rows:
            emb = np.frombuffer(blob, dtype=np.float32)
            if emb.shape[0] != embedding_dim:
                continue
            metadata.append((uid, text))
            embeddings.append(emb)

        if embeddings:
            matrix = np.stack(embeddings, axis=0)
            matrix = matrix / np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-8)
        else:
            matrix = np.empty((0, embedding_dim), dtype=np.float32)

        cached = (matrix, metadata)
        self._embedding_index[embedding_dim] = cached
        return cached

    def has_unit(self, unit: VisualUnit) -> bool:
        conn = self._conn()
        row = conn.execute("SELECT 1 FROM units WHERE unit_id=?", (unit.unit_id,)).fetchone()
        conn.close()
        return row is not None

    def store_latents(
        self,
        unit: VisualUnit,
        latents: dict[int, torch.Tensor],
        text_embedding: torch.Tensor | None = None,
        gobed_embedding: np.ndarray | None = None,
        metadata: dict | None = None,
    ):
        d = self.unit_dir(unit)
        tensors = {}
        for step_idx, lat in latents.items():
            tensors[f"latent_t{step_idx}"] = lat.contiguous().cpu()
        if text_embedding is not None:
            if text_embedding.dim() == 2:
                tensors["text_embedding"] = text_embedding.mean(0).contiguous().cpu()
                tensors["text_embedding_full"] = text_embedding.contiguous().cpu()
            else:
                tensors["text_embedding"] = text_embedding.contiguous().cpu()
        save_file(tensors, str(d / "latents.safetensors"))

        clip_blob = None
        if text_embedding is not None:
            clip_blob = tensors.get("text_embedding", text_embedding.mean(0) if text_embedding.dim() == 2 else text_embedding).cpu().numpy().tobytes()

        gobed_blob = gobed_embedding.tobytes() if gobed_embedding is not None else None

        conn = self._conn()
        conn.execute(
            """INSERT OR REPLACE INTO units
               (unit_id, unit_text, file_path, clip_embedding, gobed_embedding,
                num_cached_steps, created_at, metadata)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                unit.unit_id,
                unit.text,
                str(d / "latents.safetensors"),
                clip_blob,
                gobed_blob,
                len(latents),
                time.time(),
                json.dumps(metadata or {}),
            ),
        )
        conn.commit()
        conn.close()
        self._embedding_index_dirty = True

    def load_latent(self, unit: VisualUnit, step_idx: int) -> torch.Tensor | None:
        d = self._unit_path(unit)
        path = d / "latents.safetensors"
        if not path.exists():
            return None
        data = load_file(str(path))
        key = f"latent_t{step_idx}"
        return data.get(key)

    def load_text_embedding(self, unit: VisualUnit) -> torch.Tensor | None:
        d = self._unit_path(unit)
        path = d / "latents.safetensors"
        if not path.exists():
            return None
        data = load_file(str(path))
        return data.get("text_embedding")

    def load_all_latents(self, unit: VisualUnit) -> dict[int, torch.Tensor]:
        d = self._unit_path(unit)
        path = d / "latents.safetensors"
        if not path.exists():
            return {}
        data = load_file(str(path))
        result = {}
        for k, v in data.items():
            if k.startswith("latent_t"):
                step = int(k[len("latent_t"):])
                result[step] = v
        return result

    def find_nearest(
        self, query_embedding: torch.Tensor | np.ndarray, top_k: int = 5
    ) -> list[tuple[str, str, float]]:
        """Find nearest cached units by CLIP cosine similarity."""
        if top_k <= 0:
            return []

        if isinstance(query_embedding, torch.Tensor):
            query = query_embedding.cpu().float().numpy()
        else:
            query = query_embedding.astype(np.float32, copy=False)
        query = np.asarray(query, dtype=np.float32).reshape(-1)
        query = query / (np.linalg.norm(query) + 1e-8)

        matrix, metadata = self._get_embedding_index(query.shape[0])
        if matrix.shape[0] == 0:
            return []

        similarities = matrix @ query

        top_k = min(top_k, similarities.shape[0])
        if top_k == similarities.shape[0]:
            top_idx = np.argsort(-similarities)
        else:
            top_idx = np.argpartition(-similarities, top_k - 1)[:top_k]
            top_idx = top_idx[np.argsort(-similarities[top_idx])]

        return [(*metadata[idx], float(similarities[idx])) for idx in top_idx.tolist()]

    def list_units(self) -> list[tuple[str, str, int]]:
        conn = self._conn()
        rows = conn.execute(
            "SELECT unit_id, unit_text, num_cached_steps FROM units"
        ).fetchall()
        conn.close()
        return rows

    def load_unit_by_id(self, unit_id: str) -> VisualUnit | None:
        conn = self._conn()
        row = conn.execute(
            "SELECT unit_text FROM units WHERE unit_id=?",
            (unit_id,),
        ).fetchone()
        conn.close()
        if row is None:
            return None
        return VisualUnit(text=row[0], unit_id=unit_id)

    # --- Bigram Cache ---

    @staticmethod
    def bigram_id(unit_a: VisualUnit, unit_b: VisualUnit) -> str:
        """Deterministic bigram ID from ordered pair."""
        import hashlib
        key = f"{unit_a.unit_id}:{unit_b.unit_id}"
        return hashlib.sha256(key.encode()).hexdigest()[:16]

    def has_bigram(self, unit_a: VisualUnit, unit_b: VisualUnit) -> bool:
        bid = self.bigram_id(unit_a, unit_b)
        conn = self._conn()
        row = conn.execute("SELECT 1 FROM bigrams WHERE bigram_id=?", (bid,)).fetchone()
        conn.close()
        return row is not None

    def store_bigram(
        self,
        unit_a: VisualUnit,
        unit_b: VisualUnit,
        latents: dict[int, torch.Tensor],
        metadata: dict | None = None,
    ):
        """Store pre-computed latent for an ordered pair of visual units."""
        bid = self.bigram_id(unit_a, unit_b)
        d = self._units_dir / f"bigram_{bid}"
        d.mkdir(parents=True, exist_ok=True)
        tensors = {f"latent_t{step}": lat.contiguous().cpu() for step, lat in latents.items()}
        path = d / "latents.safetensors"
        save_file(tensors, str(path))

        conn = self._conn()
        conn.execute(
            """INSERT OR REPLACE INTO bigrams
               (bigram_id, unit_a_id, unit_b_id, unit_a_text, unit_b_text,
                file_path, num_cached_steps, created_at, metadata)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (bid, unit_a.unit_id, unit_b.unit_id, unit_a.text, unit_b.text,
             str(path), len(latents), time.time(), json.dumps(metadata or {})),
        )
        conn.commit()
        conn.close()

    def load_bigram_latent(
        self, unit_a: VisualUnit, unit_b: VisualUnit, step_idx: int,
    ) -> torch.Tensor | None:
        """Load pre-cached bigram latent. Returns None if not cached."""
        bid = self.bigram_id(unit_a, unit_b)
        d = self._units_dir / f"bigram_{bid}"
        path = d / "latents.safetensors"
        if not path.exists():
            return None
        data = load_file(str(path))
        return data.get(f"latent_t{step_idx}")

    def lookup_best(
        self, units: list[VisualUnit], step_idx: int,
    ) -> tuple[torch.Tensor | None, str]:
        """Try bigram first, then individual units. Returns (latent, method)."""
        # Try bigram for first two units
        if len(units) >= 2:
            lat = self.load_bigram_latent(units[0], units[1], step_idx)
            if lat is not None:
                return lat, "bigram"
        # Fall back to individual unit
        if units:
            lat = self.load_latent(units[0], step_idx)
            if lat is not None:
                return lat, "unit"
        return None, "miss"

    def list_bigrams(self) -> list[tuple[str, str, str, int]]:
        conn = self._conn()
        rows = conn.execute(
            "SELECT bigram_id, unit_a_text, unit_b_text, num_cached_steps FROM bigrams"
        ).fetchall()
        conn.close()
        return rows

    def stats(self) -> dict:
        conn = self._conn()
        count = conn.execute("SELECT COUNT(*) FROM units").fetchone()[0]
        total_steps = conn.execute("SELECT SUM(num_cached_steps) FROM units").fetchone()[0] or 0
        bigram_count = conn.execute("SELECT COUNT(*) FROM bigrams").fetchone()[0]
        conn.close()
        return {
            "num_units": count,
            "total_cached_steps": total_steps,
            "num_bigrams": bigram_count,
            "resolution": self.resolution,
        }
