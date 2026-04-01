"""Tests for bigram cache functionality."""

import tempfile

import torch

from latentteleport.cache import LatentCache
from latentteleport.tokenizer import VisualUnit


class TestBigramCache:
    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.cache = LatentCache(self.tmpdir, resolution=(512, 512))

    def test_store_and_load_bigram(self):
        a = VisualUnit.from_text("red car")
        b = VisualUnit.from_text("beach")
        latents = {6: torch.randn(16, 1, 64, 64)}
        self.cache.store_bigram(a, b, latents)

        assert self.cache.has_bigram(a, b)
        loaded = self.cache.load_bigram_latent(a, b, 6)
        assert loaded is not None
        assert loaded.shape == (16, 1, 64, 64)

    def test_bigram_order_matters(self):
        a = VisualUnit.from_text("cat")
        b = VisualUnit.from_text("mat")
        self.cache.store_bigram(a, b, {0: torch.randn(16, 1, 64, 64)})

        assert self.cache.has_bigram(a, b)
        assert not self.cache.has_bigram(b, a)

    def test_lookup_best_prefers_bigram(self):
        a = VisualUnit.from_text("dragon")
        b = VisualUnit.from_text("mountain")
        # Store individual unit
        self.cache.store_latents(a, {6: torch.randn(16, 1, 64, 64)})
        # Store bigram
        self.cache.store_bigram(a, b, {6: torch.randn(16, 1, 64, 64)})

        lat, method = self.cache.lookup_best([a, b], 6)
        assert method == "bigram"
        assert lat is not None

    def test_lookup_best_falls_back_to_unit(self):
        a = VisualUnit.from_text("wizard")
        b = VisualUnit.from_text("cave")
        self.cache.store_latents(a, {6: torch.randn(16, 1, 64, 64)})

        lat, method = self.cache.lookup_best([a, b], 6)
        assert method == "unit"

    def test_lookup_best_miss(self):
        a = VisualUnit.from_text("unicorn")
        lat, method = self.cache.lookup_best([a], 6)
        assert method == "miss"

    def test_list_bigrams(self):
        a = VisualUnit.from_text("sun")
        b = VisualUnit.from_text("moon")
        self.cache.store_bigram(a, b, {0: torch.randn(16, 1, 64, 64)})
        bigrams = self.cache.list_bigrams()
        assert len(bigrams) == 1

    def test_stats_includes_bigrams(self):
        a = VisualUnit.from_text("fire")
        b = VisualUnit.from_text("ice")
        self.cache.store_bigram(a, b, {0: torch.randn(16, 1, 64, 64)})
        s = self.cache.stats()
        assert s["num_bigrams"] == 1
