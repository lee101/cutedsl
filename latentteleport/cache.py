"""Latent cache: safetensors storage + SQLite index for visual units."""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file, save_file

from latentteleport.tokenizer import VisualUnit


class LatentCache:
    _REQUIRED_COLUMNS = {
        "units": {
            "unit_id",
            "unit_text",
            "file_path",
            "clip_embedding",
            "gobed_embedding",
            "num_cached_steps",
            "created_at",
            "last_accessed",
            "file_bytes",
            "metadata",
        },
        "bigrams": {
            "bigram_id",
            "unit_a_id",
            "unit_b_id",
            "unit_a_text",
            "unit_b_text",
            "file_path",
            "num_cached_steps",
            "created_at",
            "metadata",
        },
        "prompts": {
            "prompt_hash",
            "prompt",
            "exact_unit_id",
            "unit_ids",
            "unit_texts",
            "seed",
            "width",
            "height",
            "steps",
            "created_at",
            "metadata",
        },
    }
    _COLUMN_DEFINITIONS = {
        "units": {
            "unit_id": "TEXT",
            "unit_text": "TEXT",
            "file_path": "TEXT",
            "clip_embedding": "BLOB",
            "gobed_embedding": "BLOB",
            "num_cached_steps": "INTEGER DEFAULT 0",
            "created_at": "REAL",
            "last_accessed": "REAL",
            "file_bytes": "INTEGER DEFAULT 0",
            "metadata": "TEXT",
        },
        "bigrams": {
            "bigram_id": "TEXT",
            "unit_a_id": "TEXT",
            "unit_b_id": "TEXT",
            "unit_a_text": "TEXT",
            "unit_b_text": "TEXT",
            "file_path": "TEXT",
            "num_cached_steps": "INTEGER DEFAULT 0",
            "created_at": "REAL",
            "metadata": "TEXT",
        },
        "prompts": {
            "prompt_hash": "TEXT",
            "prompt": "TEXT",
            "exact_unit_id": "TEXT",
            "unit_ids": "TEXT",
            "unit_texts": "TEXT",
            "seed": "INTEGER",
            "width": "INTEGER",
            "height": "INTEGER",
            "steps": "INTEGER",
            "created_at": "REAL",
            "metadata": "TEXT",
        },
    }

    def __init__(
        self,
        cache_dir: str,
        resolution: tuple[int, int] = (512, 512),
        *,
        max_entries: int | None = None,
        max_bytes: int | None = None,
        prune_interval_s: float = 60.0,
    ):
        self.cache_dir = Path(cache_dir)
        self.resolution = resolution
        self.max_entries = max_entries if max_entries is None else max(1, int(max_entries))
        self.max_bytes = max_bytes if max_bytes is None else max(1, int(max_bytes))
        self.prune_interval_s = max(0.0, float(prune_interval_s))
        self._res_dir = self.cache_dir / f"{resolution[0]}x{resolution[1]}"
        self._units_dir = self._res_dir / "units"
        self._units_dir.mkdir(parents=True, exist_ok=True)
        self._db_path = self._res_dir / "index.sqlite"
        self._schema_lock = threading.RLock()
        self._embedding_index_dirty = True
        self._embedding_index: dict[int, tuple[np.ndarray, list[tuple[str, str]]]] = {}
        self._last_prune_at = 0.0
        self._touch_times: dict[str, float] = {}
        self._init_db()

    def _init_db(self):
        with self._schema_lock:
            try:
                conn = self._open_conn()
                try:
                    self._create_schema(conn)
                    conn.commit()
                finally:
                    conn.close()
            except sqlite3.DatabaseError:
                self._replace_corrupt_db()
                conn = self._open_conn()
                try:
                    self._create_schema(conn)
                    conn.commit()
                finally:
                    conn.close()

    def _open_conn(self) -> sqlite3.Connection:
        return sqlite3.connect(str(self._db_path), timeout=30.0)

    def _conn(self) -> sqlite3.Connection:
        conn = self._open_conn()
        try:
            schema_missing = self._schema_missing(conn)
        except sqlite3.DatabaseError:
            conn.close()
            self._replace_corrupt_db()
            self._init_db()
            return self._open_conn()

        if schema_missing:
            conn.close()
            self._init_db()
            conn = self._open_conn()
        return conn

    def _replace_corrupt_db(self):
        with self._schema_lock:
            if not self._db_path.exists():
                return
            ts = int(time.time() * 1000)
            replacement = self._db_path.with_suffix(f".corrupt-{ts}.sqlite")
            try:
                os.replace(self._db_path, replacement)
            except FileNotFoundError:
                return

    def _create_schema(self, conn: sqlite3.Connection):
        conn.execute("""
            CREATE TABLE IF NOT EXISTS units (
                unit_id TEXT PRIMARY KEY,
                unit_text TEXT NOT NULL,
                file_path TEXT NOT NULL,
                clip_embedding BLOB,
                gobed_embedding BLOB,
                num_cached_steps INTEGER DEFAULT 0,
                created_at REAL,
                last_accessed REAL,
                file_bytes INTEGER DEFAULT 0,
                metadata TEXT
            )
        """)
        self._migrate_schema(conn)
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
        self._migrate_schema(conn)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_bigram_pair ON bigrams(unit_a_id, unit_b_id)")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS prompts (
                prompt_hash TEXT PRIMARY KEY,
                prompt TEXT NOT NULL,
                exact_unit_id TEXT,
                unit_ids TEXT,
                unit_texts TEXT,
                seed INTEGER,
                width INTEGER,
                height INTEGER,
                steps INTEGER,
                created_at REAL,
                metadata TEXT
            )
        """)
        self._migrate_schema(conn)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_prompt_exact ON prompts(exact_unit_id)")

    def _migrate_schema(self, conn: sqlite3.Connection):
        for table_name, columns in self._COLUMN_DEFINITIONS.items():
            if not self._table_exists(conn, table_name):
                continue
            existing_columns = {
                row[1]
                for row in conn.execute(f"PRAGMA table_info({table_name})").fetchall()
            }
            for column_name, column_sql in columns.items():
                if column_name not in existing_columns:
                    conn.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_sql}")

    def _table_exists(self, conn: sqlite3.Connection, table_name: str) -> bool:
        row = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
            (table_name,),
        ).fetchone()
        return row is not None

    def _schema_missing(self, conn: sqlite3.Connection) -> bool:
        required = set(self._REQUIRED_COLUMNS)
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name IN (?, ?, ?)",
            tuple(sorted(required)),
        ).fetchall()
        present = {row[0] for row in rows}
        if present != required:
            return True
        for table_name, required_columns in self._REQUIRED_COLUMNS.items():
            columns = {
                row[1]
                for row in conn.execute(f"PRAGMA table_info({table_name})").fetchall()
            }
            if not required_columns <= columns:
                return True
        return False

    def _execute_write(self, sql: str, params: tuple):
        for attempt in range(2):
            conn = self._conn()
            try:
                conn.execute(sql, params)
                conn.commit()
                return
            except sqlite3.OperationalError as exc:
                conn.rollback()
                if attempt == 0 and "no such table" in str(exc).lower():
                    self._init_db()
                    continue
                raise
            finally:
                conn.close()

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

    def _touch_unit(self, unit_id: str) -> None:
        """Persist LRU recency without turning repeated tensor reads into writes."""
        now = time.time()
        if now - self._touch_times.get(unit_id, 0.0) < 60.0:
            return
        self._touch_times[unit_id] = now
        self._execute_write(
            "UPDATE units SET last_accessed=? WHERE unit_id=?",
            (now, unit_id),
        )

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
            tensors[f"latent_t{step_idx}"] = lat.detach().contiguous().cpu()
        if text_embedding is not None:
            # Cached tensors are inference artifacts, never autograd inputs.
            # Detach at the serialization boundary so callers cannot retain a
            # graph or fail when the similarity embedding crosses into NumPy.
            text_embedding = text_embedding.detach().float()
            if text_embedding.dim() == 2:
                tensors["text_embedding"] = text_embedding.mean(0).contiguous().cpu()
                tensors["text_embedding_full"] = text_embedding.contiguous().cpu()
            else:
                tensors["text_embedding"] = text_embedding.contiguous().cpu()
        latent_path = d / "latents.safetensors"
        save_file(tensors, str(latent_path))
        file_bytes = latent_path.stat().st_size

        clip_blob = None
        if text_embedding is not None:
            # The key is always populated above. Avoid dict.get(..., default):
            # Python evaluates its default eagerly, which previously created a
            # grad-tracked fallback even when the cached tensor was present.
            clip_blob = tensors["text_embedding"].numpy().tobytes()

        gobed_blob = gobed_embedding.tobytes() if gobed_embedding is not None else None

        self._execute_write(
            """INSERT OR REPLACE INTO units
               (unit_id, unit_text, file_path, clip_embedding, gobed_embedding,
                num_cached_steps, created_at, last_accessed, file_bytes, metadata)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                unit.unit_id,
                unit.text,
                str(latent_path),
                clip_blob,
                gobed_blob,
                len(latents),
                time.time(),
                time.time(),
                file_bytes,
                json.dumps(metadata or {}),
            ),
        )
        self._embedding_index_dirty = True
        self._maybe_prune()

    def _maybe_prune(self) -> None:
        if self.max_entries is None and self.max_bytes is None:
            return
        now = time.monotonic()
        if now - self._last_prune_at < self.prune_interval_s:
            return
        self._last_prune_at = now
        self.prune()

    def prune(
        self,
        *,
        max_entries: int | None = None,
        max_bytes: int | None = None,
        vacuum: bool = False,
    ) -> dict[str, int]:
        """Evict oldest regenerable units until count and byte limits are met.

        File targets come from the cache index but are still constrained to the
        resolution directory before unlinking. ``vacuum`` is intentionally
        opt-in because it can be expensive; it is useful for one-time cleanup
        of a previously unbounded cache.
        """
        entry_limit = self.max_entries if max_entries is None else max(1, int(max_entries))
        byte_limit = self.max_bytes if max_bytes is None else max(1, int(max_bytes))
        if entry_limit is None and byte_limit is None:
            return {
                "removed_entries": 0,
                "removed_bigrams": 0,
                "removed_bytes": 0,
                "remaining_entries": 0,
                "remaining_bytes": 0,
            }

        conn = self._conn()
        rows = conn.execute(
            """SELECT unit_id, file_path, COALESCE(last_accessed, created_at, 0),
                      COALESCE(file_bytes, 0)
               FROM units
               ORDER BY COALESCE(last_accessed, created_at, 0) ASC"""
        ).fetchall()

        indexed: list[tuple[str, str, float, int]] = []
        size_updates: list[tuple[int, str]] = []
        for unit_id, file_path, accessed, file_bytes in rows:
            size = int(file_bytes or 0)
            if size <= 0:
                try:
                    size = Path(file_path).stat().st_size
                except OSError:
                    size = 0
                size_updates.append((size, unit_id))
            indexed.append((unit_id, file_path, float(accessed or 0), size))
        if size_updates:
            conn.executemany("UPDATE units SET file_bytes=? WHERE unit_id=?", size_updates)

        bigram_rows = conn.execute(
            """SELECT bigram_id, unit_a_id, unit_b_id, file_path,
                      COALESCE(created_at, 0)
               FROM bigrams
               ORDER BY COALESCE(created_at, 0) ASC"""
        ).fetchall()
        indexed_bigrams: list[tuple[str, str, str, str, float, int]] = []
        for bigram_id, unit_a_id, unit_b_id, file_path, created_at in bigram_rows:
            try:
                size = Path(file_path).stat().st_size
            except OSError:
                size = 0
            indexed_bigrams.append(
                (bigram_id, unit_a_id, unit_b_id, file_path, float(created_at or 0), size)
            )

        bigrams_by_unit: dict[str, list[tuple[str, str, str, str, float, int]]] = {}
        for bigram in indexed_bigrams:
            bigrams_by_unit.setdefault(bigram[1], []).append(bigram)
            bigrams_by_unit.setdefault(bigram[2], []).append(bigram)

        remaining_entries = len(indexed)
        remaining_bytes = sum(row[3] for row in indexed) + sum(row[5] for row in indexed_bigrams)
        victims: list[tuple[str, str, float, int]] = []
        victim_unit_ids: set[str] = set()
        victim_bigrams: list[tuple[str, str, str, str, float, int]] = []
        victim_bigram_ids: set[str] = set()

        def plan_bigram(row: tuple[str, str, str, str, float, int]) -> None:
            nonlocal remaining_bytes
            if row[0] in victim_bigram_ids:
                return
            victim_bigram_ids.add(row[0])
            victim_bigrams.append(row)
            remaining_bytes -= row[5]

        def plan_unit(row: tuple[str, str, float, int]) -> None:
            nonlocal remaining_entries, remaining_bytes
            if row[0] in victim_unit_ids:
                return
            victim_unit_ids.add(row[0])
            victims.append(row)
            remaining_entries -= 1
            remaining_bytes -= row[3]
            for bigram in bigrams_by_unit.get(row[0], []):
                plan_bigram(bigram)

        # The entry cap applies to reusable single-prompt units. Removing a
        # unit also removes any pair cache that depends on it.
        for row in indexed:
            if entry_limit is None or remaining_entries <= entry_limit:
                break
            plan_unit(row)

        # Pair entries are lower-value and can otherwise grow without bound,
        # so reclaim the oldest pairs before dropping more single units.
        if byte_limit is not None:
            for row in indexed_bigrams:
                if remaining_bytes <= byte_limit:
                    break
                plan_bigram(row)

            for row in indexed:
                if remaining_bytes <= byte_limit:
                    break
                plan_unit(row)

        cache_root = self._res_dir.resolve()
        removed_bytes = 0
        for bigram_id, _unit_a_id, _unit_b_id, file_path, _created_at, file_bytes in victim_bigrams:
            path = Path(file_path).resolve()
            if path.is_relative_to(cache_root):
                try:
                    path.unlink()
                    removed_bytes += file_bytes
                except FileNotFoundError:
                    pass
                try:
                    path.parent.rmdir()
                except OSError:
                    pass
            conn.execute("DELETE FROM bigrams WHERE bigram_id=?", (bigram_id,))

        for unit_id, file_path, _accessed, file_bytes in victims:
            path = Path(file_path).resolve()
            if path.is_relative_to(cache_root):
                try:
                    path.unlink()
                    removed_bytes += file_bytes
                except FileNotFoundError:
                    pass
                try:
                    path.parent.rmdir()
                except OSError:
                    pass
            conn.execute("DELETE FROM bigrams WHERE unit_a_id=? OR unit_b_id=?", (unit_id, unit_id))
            conn.execute("DELETE FROM prompts WHERE exact_unit_id=?", (unit_id,))
            conn.execute("DELETE FROM units WHERE unit_id=?", (unit_id,))
            self._touch_times.pop(unit_id, None)
        conn.commit()
        if vacuum and (victims or victim_bigrams):
            conn.execute("VACUUM")
        conn.close()
        if victims:
            self._embedding_index_dirty = True
        return {
            "removed_entries": len(victims),
            "removed_bigrams": len(victim_bigrams),
            "removed_bytes": removed_bytes,
            "remaining_entries": remaining_entries,
            "remaining_bytes": remaining_bytes,
        }

    def record_prompt(
        self,
        prompt: str,
        exact_unit: VisualUnit | None,
        units: list[VisualUnit],
        seed: int | None = None,
        width: int | None = None,
        height: int | None = None,
        steps: int | None = None,
        metadata: dict | None = None,
    ):
        import hashlib

        ph = hashlib.sha256(prompt.strip().lower().encode()).hexdigest()[:16]
        self._execute_write(
            """INSERT OR REPLACE INTO prompts
               (prompt_hash, prompt, exact_unit_id, unit_ids, unit_texts,
                seed, width, height, steps, created_at, metadata)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                ph,
                prompt,
                exact_unit.unit_id if exact_unit else None,
                json.dumps([u.unit_id for u in units]),
                json.dumps([u.text for u in units]),
                seed,
                width,
                height,
                steps,
                time.time(),
                json.dumps(metadata or {}),
            ),
        )

    def record_prompt(
        self,
        prompt: str,
        exact_unit: VisualUnit | None,
        units: list[VisualUnit],
        seed: int | None = None,
        width: int | None = None,
        height: int | None = None,
        steps: int | None = None,
        metadata: dict | None = None,
    ):
        import hashlib

        ph = hashlib.sha256(prompt.strip().lower().encode()).hexdigest()[:16]
        conn = self._conn()
        conn.execute(
            """INSERT OR REPLACE INTO prompts
               (prompt_hash, prompt, exact_unit_id, unit_ids, unit_texts,
                seed, width, height, steps, created_at, metadata)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                ph,
                prompt,
                exact_unit.unit_id if exact_unit else None,
                json.dumps([u.unit_id for u in units]),
                json.dumps([u.text for u in units]),
                seed,
                width,
                height,
                steps,
                time.time(),
                json.dumps(metadata or {}),
            ),
        )
        conn.commit()
        conn.close()

    def load_latent(self, unit: VisualUnit, step_idx: int) -> torch.Tensor | None:
        d = self._unit_path(unit)
        path = d / "latents.safetensors"
        if not path.exists():
            return None
        data = load_file(str(path))
        key = f"latent_t{step_idx}"
        value = data.get(key)
        if value is not None:
            self._touch_unit(unit.unit_id)
        return value

    def load_text_embedding(self, unit: VisualUnit) -> torch.Tensor | None:
        d = self._unit_path(unit)
        path = d / "latents.safetensors"
        if not path.exists():
            return None
        data = load_file(str(path))
        value = data.get("text_embedding")
        if value is not None:
            self._touch_unit(unit.unit_id)
        return value

    def load_text_embedding_full(self, unit: VisualUnit) -> torch.Tensor | None:
        d = self._unit_path(unit)
        path = d / "latents.safetensors"
        if not path.exists():
            return None
        data = load_file(str(path))
        value = data.get("text_embedding_full")
        if value is not None:
            self._touch_unit(unit.unit_id)
        return value

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
        if result:
            self._touch_unit(unit.unit_id)
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

        self._execute_write(
            """INSERT OR REPLACE INTO bigrams
               (bigram_id, unit_a_id, unit_b_id, unit_a_text, unit_b_text,
                file_path, num_cached_steps, created_at, metadata)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (bid, unit_a.unit_id, unit_b.unit_id, unit_a.text, unit_b.text,
             str(path), len(latents), time.time(), json.dumps(metadata or {})),
        )

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
        prompt_count = conn.execute("SELECT COUNT(*) FROM prompts").fetchone()[0]
        conn.close()
        return {
            "num_units": count,
            "total_cached_steps": total_steps,
            "num_bigrams": bigram_count,
            "num_prompts": prompt_count,
            "resolution": self.resolution,
        }
