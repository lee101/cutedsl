"""Prompt adherence and reference similarity judges for teleportation evals."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from cutezimage.image_metrics import compare_images, pil_to_tensor


@dataclass
class JudgeResult:
    prompt_score: float | None = None
    reference_score: float | None = None
    combined_score: float | None = None


class CLIPImageJudge:
    """CLIP-based local proxy for VLM judging when no external judge is configured."""

    def __init__(self, model_id: str = "openai/clip-vit-base-patch32"):
        from transformers import CLIPModel, CLIPProcessor

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = CLIPModel.from_pretrained(model_id).to(self._device).eval()
        self._processor = CLIPProcessor.from_pretrained(model_id)

    @torch.no_grad()
    def score_prompt(self, prompt: str, image: Image.Image) -> float:
        batch = self._processor(text=[prompt], images=[image], return_tensors="pt", padding=True)
        batch = {k: v.to(self._device) for k, v in batch.items()}
        out = self._model(**batch)
        image_emb = out.image_embeds / out.image_embeds.norm(dim=-1, keepdim=True)
        text_emb = out.text_embeds / out.text_embeds.norm(dim=-1, keepdim=True)
        return float((image_emb * text_emb).sum().item())

    @torch.no_grad()
    def score_pair(self, prompt: str, image: Image.Image, reference: Image.Image | None) -> JudgeResult:
        prompt_score = self.score_prompt(prompt, image)
        reference_score = None
        if reference is not None:
            metrics = compare_images(pil_to_tensor(image), pil_to_tensor(reference))
            reference_score = 1.0 - float(metrics["mean_abs_error"])
        combined = prompt_score if reference_score is None else 0.5 * (prompt_score + reference_score)
        return JudgeResult(prompt_score=prompt_score, reference_score=reference_score, combined_score=combined)


def create_judge(enabled: bool) -> CLIPImageJudge | None:
    if not enabled:
        return None
    try:
        return CLIPImageJudge()
    except Exception:
        return None

