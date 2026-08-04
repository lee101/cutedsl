"""Tests for latent cache."""

import concurrent.futures
import sqlite3
import tempfile
from pathlib import Path

import torch

from latentteleport.cache import LatentCache
from latentteleport.tokenizer import VisualUnit


class TestLatentCache:
    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.cache = LatentCache(self.tmpdir, resolution=(512, 512))

    def test_store_and_load(self):
        unit = VisualUnit.from_text("red car")
        latents = {
            0: torch.randn(16, 1, 64, 64),
            5: torch.randn(16, 1, 64, 64),
            10: torch.randn(16, 1, 64, 64),
        }
        text_emb = torch.randn(77, 2560)
        self.cache.store_latents(unit, latents, text_embedding=text_emb)

        assert self.cache.has_unit(unit)
        loaded = self.cache.load_latent(unit, 5)
        assert loaded is not None
        assert loaded.shape == (16, 1, 64, 64)
        assert torch.allclose(loaded, latents[5], atol=1e-4)

    def test_load_missing(self):
        unit = VisualUnit.from_text("nonexistent")
        assert self.cache.load_latent(unit, 0) is None

    def test_load_all_latents(self):
        unit = VisualUnit.from_text("beach")
        latents = {i: torch.randn(16, 1, 64, 64) for i in range(20)}
        self.cache.store_latents(unit, latents)
        loaded = self.cache.load_all_latents(unit)
        assert len(loaded) == 20

    def test_text_embedding(self):
        unit = VisualUnit.from_text("sunset")
        text_emb = torch.randn(77, 2560)
        self.cache.store_latents(unit, {0: torch.randn(16, 1, 64, 64)}, text_embedding=text_emb)
        loaded = self.cache.load_text_embedding(unit)
        assert loaded is not None
        assert loaded.shape == (2560,)

    def test_store_latents_detaches_grad_tracked_embeddings_across_prompt_lengths(self):
        prompts = [
            "portrait",
            "cinematic portrait with soft window light and detailed natural textures",
            " ".join(["cinematic portrait with detailed natural textures"] * 40),
        ]

        for prompt in prompts:
            unit = VisualUnit.from_text(prompt)
            embedding = torch.randn(8, 16, requires_grad=True)
            expected = embedding.detach().float().mean(0)

            self.cache.store_latents(
                unit,
                {0: torch.randn(1, 1, 2, 2, requires_grad=True)},
                text_embedding=embedding,
            )

            stored = self.cache.load_text_embedding(unit)
            assert stored is not None
            assert not stored.requires_grad
            assert torch.equal(stored, expected)

    def test_find_nearest(self):
        for name in ["cat", "dog", "car"]:
            unit = VisualUnit.from_text(name)
            emb = torch.randn(2560)
            self.cache.store_latents(
                unit, {0: torch.randn(16, 1, 64, 64)}, text_embedding=emb,
            )
        query = torch.randn(2560)
        results = self.cache.find_nearest(query, top_k=2)
        assert len(results) == 2

    def test_find_nearest_handles_matrix_queries_and_large_top_k(self):
        embeddings = {
            "cat": torch.tensor([1.0, 0.0, 0.0, 0.0]),
            "dog": torch.tensor([0.0, 1.0, 0.0, 0.0]),
            "car": torch.tensor([0.0, 0.0, 1.0, 0.0]),
        }
        for name, emb in embeddings.items():
            unit = VisualUnit.from_text(name)
            self.cache.store_latents(
                unit,
                {0: torch.randn(1, 1, 2, 2)},
                text_embedding=emb,
            )

        query = torch.tensor([[3.0, 0.0, 0.0, 0.0]])
        results = self.cache.find_nearest(query, top_k=10)

        assert [text for _, text, _ in results] == ["cat", "dog", "car"]
        assert len(results) == 3

    def test_bounded_cache_evicts_least_recently_used_unit(self):
        cache = LatentCache(
            tempfile.mkdtemp(),
            resolution=(512, 512),
            max_entries=2,
            prune_interval_s=0,
        )
        first = VisualUnit.from_text("first")
        second = VisualUnit.from_text("second")
        third = VisualUnit.from_text("third")
        latent = {0: torch.randn(1, 1, 2, 2)}

        cache.store_latents(first, latent)
        cache.store_latents(second, latent)
        assert cache.load_latent(first, 0) is not None
        cache.store_latents(third, latent)

        assert cache.has_unit(first)
        assert not cache.has_unit(second)
        assert cache.has_unit(third)
        assert cache.stats()["num_units"] == 2

    def test_prune_constrains_index_paths_to_resolution_directory(self):
        cache = LatentCache(tempfile.mkdtemp(), resolution=(512, 512))
        unit = VisualUnit.from_text("unsafe-index")
        cache.store_latents(unit, {0: torch.randn(1, 1, 2, 2)})
        outside = Path(tempfile.mkdtemp()) / "keep.safetensors"
        outside.write_bytes(b"keep")
        conn = cache._conn()
        conn.execute("UPDATE units SET file_path=? WHERE unit_id=?", (str(outside), unit.unit_id))
        conn.commit()
        conn.close()

        result = cache.prune(max_entries=1, max_bytes=1)

        assert result["removed_entries"] == 1
        assert outside.read_bytes() == b"keep"

    def test_prune_removes_bigram_file_when_a_dependency_is_evicted(self):
        cache = LatentCache(tempfile.mkdtemp(), resolution=(512, 512))
        first = VisualUnit.from_text("first")
        second = VisualUnit.from_text("second")
        latent = {0: torch.randn(1, 1, 2, 2)}
        cache.store_latents(first, latent)
        cache.store_latents(second, latent)
        cache.store_bigram(first, second, latent)
        bigram_path = cache._units_dir / f"bigram_{cache.bigram_id(first, second)}" / "latents.safetensors"
        assert bigram_path.exists()

        result = cache.prune(max_entries=1)

        assert result["removed_entries"] == 1
        assert result["removed_bigrams"] == 1
        assert not bigram_path.exists()
        assert cache.stats()["num_bigrams"] == 0

    def test_byte_limit_reclaims_old_bigram_before_single_units(self):
        cache = LatentCache(tempfile.mkdtemp(), resolution=(512, 512))
        first = VisualUnit.from_text("first")
        second = VisualUnit.from_text("second")
        latent = {0: torch.randn(1, 1, 8, 8)}
        cache.store_latents(first, latent)
        cache.store_latents(second, latent)
        cache.store_bigram(first, second, latent)
        unit_bytes = sum(path.stat().st_size for path in cache._units_dir.glob("*/latents.safetensors") if "bigram_" not in path.parent.name)

        result = cache.prune(max_entries=10, max_bytes=unit_bytes)

        assert result["removed_entries"] == 0
        assert result["removed_bigrams"] == 1
        assert cache.stats()["num_units"] == 2
        assert cache.stats()["num_bigrams"] == 0

    def test_stats(self):
        unit = VisualUnit.from_text("tree")
        self.cache.store_latents(unit, {0: torch.randn(16, 1, 64, 64), 1: torch.randn(16, 1, 64, 64)})
        s = self.cache.stats()
        assert s["num_units"] == 1
        assert s["total_cached_steps"] == 2

    def test_list_units(self):
        for name in ["a", "b", "c"]:
            unit = VisualUnit.from_text(name)
            self.cache.store_latents(unit, {0: torch.randn(16, 1, 64, 64)})
        units = self.cache.list_units()
        assert len(units) == 3

    def test_load_unit_by_id(self):
        unit = VisualUnit.from_text("red car")
        self.cache.store_latents(unit, {0: torch.randn(16, 1, 64, 64)})
        loaded = self.cache.load_unit_by_id(unit.unit_id)
        assert loaded is not None
        assert loaded.text == unit.text

    def test_store_latents_recreates_truncated_index_across_prompt_lengths(self):
        db_path = Path(self.tmpdir) / "512x512" / "index.sqlite"
        db_path.write_bytes(b"")

        prompts = [
            "gargoyle",
            "ancient stone gargoyle statue sculpture",
            " ".join(["cinematic fantasy stone gargoyle with moss and rim light"] * 12),
            " ".join(["high quality coherent gothic cathedral gargoyle sculpture texture"] * 28),
        ]

        for prompt in prompts:
            unit = VisualUnit.from_text(prompt)
            self.cache.store_latents(unit, {0: torch.randn(1, 1, 2, 2)})
            assert self.cache.has_unit(unit)

        assert self.cache.stats()["num_units"] == len(prompts)

    def test_store_latents_recovers_corrupt_index(self):
        db_path = Path(self.tmpdir) / "512x512" / "index.sqlite"
        db_path.write_bytes(b"not a sqlite database")

        unit = VisualUnit.from_text("ancient stone gargoyle statue sculpture")
        self.cache.store_latents(unit, {0: torch.randn(1, 1, 2, 2)})

        assert self.cache.has_unit(unit)
        assert self.cache.stats()["num_units"] == 1
        assert list(db_path.parent.glob("index.corrupt-*.sqlite"))

    def test_existing_index_missing_new_tables_is_migrated(self):
        old_dir = Path(self.tmpdir) / "640x640"
        old_dir.mkdir(parents=True)
        conn = sqlite3.connect(old_dir / "index.sqlite")
        conn.execute("""
            CREATE TABLE units (
                unit_id TEXT PRIMARY KEY,
                unit_text TEXT NOT NULL,
                file_path TEXT NOT NULL,
                clip_embedding BLOB,
                num_cached_steps INTEGER DEFAULT 0,
                created_at REAL,
                metadata TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE bigrams (
                bigram_id TEXT PRIMARY KEY,
                unit_a_id TEXT NOT NULL,
                unit_b_id TEXT NOT NULL,
                unit_a_text TEXT NOT NULL,
                unit_b_text TEXT NOT NULL,
                file_path TEXT NOT NULL,
                num_cached_steps INTEGER DEFAULT 0,
                created_at REAL,
                metadata TEXT
            )
        """)
        conn.commit()
        conn.close()

        cache = LatentCache(self.tmpdir, resolution=(640, 640))
        unit = VisualUnit.from_text("medium prompt with migrated schema")
        cache.store_latents(unit, {0: torch.randn(1, 1, 2, 2)})
        cache.record_prompt(unit.text, unit, [unit], width=640, height=640, steps=1)

        stats = cache.stats()
        assert stats["num_units"] == 1
        assert stats["num_prompts"] == 1

    def test_concurrent_store_latents_initializes_schema_once(self):
        tmpdir = tempfile.mkdtemp()
        prompts = [
            "short gargoyle",
            "ancient stone gargoyle statue sculpture",
            " ".join(["cinematic fantasy gargoyle with moss and rim light"] * 8),
            " ".join(["high quality coherent gothic cathedral gargoyle sculpture texture"] * 20),
        ]

        def write_prompt(prompt: str):
            cache = LatentCache(tmpdir, resolution=(768, 768))
            unit = VisualUnit.from_text(prompt)
            cache.store_latents(unit, {0: torch.randn(1, 1, 2, 2)})
            return cache.has_unit(unit)

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            assert all(pool.map(write_prompt, prompts))

        cache = LatentCache(tmpdir, resolution=(768, 768))
        assert cache.stats()["num_units"] == len(prompts)
