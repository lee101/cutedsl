"""Tests for latent cache."""

import tempfile

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
