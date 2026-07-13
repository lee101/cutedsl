"""Tests for latent teleportation dataset helpers."""

import torch

from latentteleport.dataset import extract_text_embedding


class _ListEncodePipe:
    def encode_prompt(self, prompt: str, do_classifier_free_guidance: bool):
        return ([torch.randn(1, 4, 8)], None)


def test_extract_text_embedding_handles_zimage_prompt_embed_lists():
    embedding = extract_text_embedding(_ListEncodePipe(), "red car")

    assert embedding.shape == (4, 8)
    assert embedding.dtype == torch.float32
