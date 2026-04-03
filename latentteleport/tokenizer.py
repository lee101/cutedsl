"""Visual unit tokenizer: decompose prompts into compositional visual units."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from latentteleport.config import TokenizerConfig


@dataclass(frozen=True)
class VisualUnit:
    text: str
    unit_id: str  # deterministic hash

    @staticmethod
    def from_text(text: str) -> VisualUnit:
        normalized = text.strip().lower()
        uid = hashlib.sha256(normalized.encode()).hexdigest()[:16]
        return VisualUnit(text=normalized, unit_id=uid)


class TokenizerStrategy(Protocol):
    def tokenize(self, prompt: str) -> list[VisualUnit]: ...


# --- NLP Strategy (spaCy noun chunks) ---

class NLPTokenizer:
    def __init__(self, model_name: str = "en_core_web_sm"):
        import spacy
        self._nlp = spacy.load(model_name)

    def tokenize(self, prompt: str) -> list[VisualUnit]:
        doc = self._nlp(prompt)
        chunks = [chunk.text for chunk in doc.noun_chunks]
        if not chunks:
            chunks = [prompt.strip()]
        return [VisualUnit.from_text(c) for c in chunks if c.strip()]


# --- Curated Dictionary Strategy ---

class CuratedTokenizer:
    def __init__(self, vocab_path: str | None = None, aliases_path: str | None = None):
        self._vocab: list[str] = []
        if vocab_path and Path(vocab_path).exists():
            self._vocab = json.loads(Path(vocab_path).read_text())
        if not self._vocab:
            self._vocab = _DEFAULT_VISUAL_VOCAB
        self._aliases = dict(_DEFAULT_ALIAS_MAP)
        if aliases_path and Path(aliases_path).exists():
            self._aliases.update(json.loads(Path(aliases_path).read_text()))
        self._vocab = [self._canonicalize(term) for term in self._vocab]
        self._sorted = sorted(self._vocab, key=len, reverse=True)

    def _canonicalize(self, text: str) -> str:
        normalized = re.sub(r"\s+", " ", text.strip().lower())
        normalized = re.sub(r"[^a-z0-9\s-]", "", normalized)
        words = []
        for word in normalized.split():
            alias = self._aliases.get(word, word)
            if alias.endswith("es") and len(alias) > 4 and alias[:-2] in self._vocab:
                alias = alias[:-2]
            elif alias.endswith("s") and len(alias) > 3 and not alias.endswith("ss"):
                singular = alias[:-1]
                if singular in self._vocab or singular in _DEFAULT_VISUAL_VOCAB_SET:
                    alias = singular
            words.append(alias)
        normalized = " ".join(words)
        return self._aliases.get(normalized, normalized)

    def tokenize(self, prompt: str) -> list[VisualUnit]:
        lower = self._canonicalize(prompt)
        units = []
        remaining = lower
        for term in self._sorted:
            if term in remaining:
                units.append(VisualUnit.from_text(term))
                remaining = remaining.replace(term, " ", 1)
        remaining = remaining.strip()
        if remaining and len(remaining) > 2:
            for part in re.split(r"\s{2,}|,\s*", remaining):
                part = self._canonicalize(part)
                if len(part) > 2:
                    units.append(VisualUnit.from_text(part))
        if not units:
            units = [VisualUnit.from_text(self._canonicalize(prompt.strip()))]
        deduped = []
        seen: set[str] = set()
        for unit in units:
            if unit.text in seen:
                continue
            deduped.append(unit)
            seen.add(unit.text)
        return deduped


# --- CLIP-Clustered Strategy ---

class CLIPClusteredTokenizer:
    """Cluster text embeddings to discover visual units. Requires pre-computed centroids."""

    def __init__(self, centroids_path: str, labels_path: str):
        self._centroids = np.load(centroids_path)  # (K, embed_dim)
        with open(labels_path) as f:
            self._labels = json.load(f)  # list of representative texts per cluster
        self._nlp_fallback = NLPTokenizer()

    def tokenize(self, prompt: str) -> list[VisualUnit]:
        chunks = self._nlp_fallback.tokenize(prompt)
        return chunks  # TODO: map chunks to nearest centroid labels

    def build_centroids(
        self, prompts: list[str], n_clusters: int, clip_model, output_dir: str
    ):
        """One-time: embed prompts, k-means cluster, save centroids."""
        import torch
        from sklearn.cluster import MiniBatchKMeans

        embeddings = []
        for p in prompts:
            with torch.no_grad():
                emb = clip_model.encode_text(p)
            embeddings.append(emb.cpu().numpy())
        X = np.vstack(embeddings)
        km = MiniBatchKMeans(n_clusters=n_clusters, batch_size=256)
        km.fit(X)
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        np.save(out / "centroids.npy", km.cluster_centers_)
        labels_map = {}
        for i, label in enumerate(km.labels_):
            labels_map.setdefault(int(label), []).append(prompts[i])
        representative = {k: vs[0] for k, vs in labels_map.items()}
        (out / "labels.json").write_text(json.dumps(representative))


def create_tokenizer(config: TokenizerConfig) -> TokenizerStrategy:
    if config.strategy == "nlp":
        return NLPTokenizer(config.spacy_model)
    elif config.strategy == "curated":
        return CuratedTokenizer(config.curated_vocab_path, config.curated_aliases_path)
    elif config.strategy == "clip":
        return CLIPClusteredTokenizer(
            centroids_path=config.curated_vocab_path.replace(".json", "_centroids.npy"),
            labels_path=config.curated_vocab_path,
        )
    raise ValueError(f"Unknown strategy: {config.strategy}")


_DEFAULT_VISUAL_VOCAB = [
    "person", "man", "woman", "child", "girl", "boy",
    "cat", "dog", "bird", "horse", "fish", "butterfly",
    "tree", "flower", "forest", "mountain", "river", "ocean", "lake", "waterfall",
    "grass", "box", "cube",
    "sky", "cloud", "sun", "moon", "star", "rainbow", "sunset", "sunrise",
    "house", "building", "castle", "tower", "bridge", "road", "path",
    "car", "train", "boat", "ship", "airplane", "bicycle",
    "apple", "bread", "cake", "wine", "coffee",
    "sword", "shield", "crown", "book", "lamp", "candle",
    "fire", "water", "ice", "snow", "rain", "lightning",
    "garden", "field", "desert", "beach", "island", "cave",
    "city", "village", "market", "temple", "church", "library",
    "portrait", "landscape", "still life", "abstract",
    "red", "blue", "green", "golden", "silver", "dark", "bright",
    "old", "ancient", "modern", "futuristic", "magical", "ethereal",
    "hand", "eye", "face", "wing", "tail", "horn",
    "robot", "dragon", "fairy", "angel", "demon", "ghost",
    "astronaut", "knight", "wizard", "pirate", "samurai",
    "painting", "photograph", "illustration", "sketch", "watercolor",
    "cinematic", "dramatic", "peaceful", "mysterious", "whimsical",
]

_DEFAULT_VISUAL_VOCAB_SET = set(_DEFAULT_VISUAL_VOCAB)

_DEFAULT_ALIAS_MAP = {
    "grasses": "grass",
    "grasslands": "grass",
    "boxes": "box",
    "cubes": "cube",
    "cars": "car",
    "cats": "cat",
    "dogs": "dog",
    "wolves": "wolf",
    "mountains": "mountain",
    "trees": "tree",
    "flowers": "flower",
    "books": "book",
    "robots": "robot",
    "dragons": "dragon",
    "fairies": "fairy",
    "astronauts": "astronaut",
    "knights": "knight",
    "wizards": "wizard",
    "ships": "ship",
    "boats": "boat",
    "houses": "house",
    "beaches": "beach",
    "forests": "forest",
    "windowsills": "windowsill",
    "sunsets": "sunset",
    "sunrises": "sunrise",
}
