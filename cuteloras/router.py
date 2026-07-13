"""Prompt-to-LoRA routing — embedding similarity with keyword fallback.

Embeds each record's trigger word, name, and keywords with a small static
sentence-transformer (loaded in a background thread); falls back to keyword scoring
until embeddings are ready or if sentence-transformers is unavailable.
"""

from __future__ import annotations

import logging
import re
import threading
from dataclasses import dataclass

import numpy as np

from cuteloras.registry import LoRARecord, LoRARegistry

logger = logging.getLogger("cuteloras")

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/static-retrieval-mrl-en-v1"

_STOPWORDS = {"a", "an", "and", "the", "with", "for", "to", "of", "in", "on", "by", "at", "is"}
_ADULT_QUERY_TERMS = {
    "nsfw",
    "adult",
    "explicit",
    "porn",
    "hentai",
    "nude",
    "naked",
    "breast",
    "breasts",
    "ahegao",
    "erotic",
    "sensual",
    "xxx",
}


def _normalize_token(token: str) -> str:
    token = re.sub(r"^[^\w]+|[^\w]+$", "", token.lower())
    for suffix in ("istic", "ism", "ing", "ness", "tion", "es", "ed", "s"):
        if len(token) > len(suffix) + 3 and token.endswith(suffix):
            return token[: -len(suffix)]
    return token


def _query_terms(text: str) -> list[str]:
    terms = []
    for raw in re.split(r"[\s,;:/|()\[\]{}]+", text.lower()):
        term = _normalize_token(raw)
        if len(term) > 2 and term not in _STOPWORDS:
            terms.append(term)
    return terms


def _normalized_phrase(text: str) -> str:
    return " ".join(_query_terms(text))


def query_allows_adult(query: str) -> bool:
    return any(term in set(_query_terms(query)) for term in _ADULT_QUERY_TERMS)


@dataclass
class RouteResult:
    record: LoRARecord
    score: float
    match_type: str


class LoRARouter:
    def __init__(
        self,
        registry: LoRARegistry,
        embedding_model: str | None = DEFAULT_EMBEDDING_MODEL,
        min_score: float = 0.45,
    ):
        self.registry = registry
        self.embedding_model = embedding_model
        self.min_score = min_score
        self._embeddings: dict[str, np.ndarray] = {}
        self._neg_embeddings: dict[str, np.ndarray] = {}
        self._query_cache: dict[str, np.ndarray] = {}
        self._model = None
        self._ready = False
        self._lock = threading.Lock()
        if embedding_model:
            threading.Thread(target=self._load_and_precompute, daemon=True).start()

    def _load_and_precompute(self):
        try:
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer(self.embedding_model, device="cpu")
        except Exception as e:
            logger.warning("embedding model load failed, keyword routing only: %s", e)
            return
        with self._lock:
            self._model = model
        self.refresh_embeddings()

    def refresh_embeddings(self):
        with self._lock:
            model = self._model
        if model is None:
            return
        records = self.registry.all()
        for record in records:
            texts = [t for t in [record.trigger_word, record.name, *record.keywords] if t]
            if texts:
                embs = model.encode(texts, convert_to_numpy=True)
                self._embeddings[record.id] = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-8)
            if record.negative_keywords:
                neg = model.encode(record.negative_keywords, convert_to_numpy=True)
                self._neg_embeddings[record.id] = neg / (np.linalg.norm(neg, axis=1, keepdims=True) + 1e-8)
        with self._lock:
            self._ready = True
        logger.info("router embeddings ready for %d loras", len(records))

    def _embed_query(self, query: str) -> np.ndarray | None:
        with self._lock:
            model = self._model
        if model is None:
            return None
        cached = self._query_cache.get(query)
        if cached is not None:
            return cached
        emb = model.encode(query, convert_to_numpy=True)
        emb = emb / (np.linalg.norm(emb) + 1e-8)
        if len(self._query_cache) > 1000:
            self._query_cache.clear()
        self._query_cache[query] = emb
        return emb

    def search(self, query: str, top_k: int = 5, allow_adult: bool | None = None) -> list[RouteResult]:
        if allow_adult is None:
            allow_adult = query_allows_adult(query)
        with self._lock:
            ready = self._ready
        if ready:
            results = self._search_embeddings(query, allow_adult)
        else:
            results = self._search_keywords(query, allow_adult)
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]

    def route(self, prompt: str, allow_adult: bool | None = None) -> LoRARecord | None:
        results = self.search(prompt, top_k=1, allow_adult=allow_adult)
        if results and results[0].score >= self.min_score:
            return results[0].record
        return None

    def _search_embeddings(self, query: str, allow_adult: bool) -> list[RouteResult]:
        query_emb = self._embed_query(query)
        if query_emb is None:
            return self._search_keywords(query, allow_adult)
        query_norm = _normalized_phrase(query)
        query_lower = query.lower()
        results = []
        for record in self.registry.all():
            if record.is_adult and not allow_adult:
                continue
            embs = self._embeddings.get(record.id)
            if embs is None:
                continue
            sims = embs @ query_emb
            if record.trigger_word:
                sims = sims.copy()
                sims[0] *= 1.5
            score = float(sims.max())
            neg = self._neg_embeddings.get(record.id)
            if neg is not None:
                score -= float((neg @ query_emb).max())
            score = max(score, 0.0)
            match_type = "embedding"
            if record.trigger_word and _normalized_phrase(record.trigger_word) in query_norm:
                score += 0.35
            if _normalized_phrase(record.name) in query_norm:
                score += 0.25
            score *= 1.2
            if record.trigger_word and record.trigger_word.lower() in query_lower:
                match_type = "trigger"
                score += 2.0
            elif record.name.lower() in query_lower:
                match_type = "name"
                score += 1.0
            if score > 0:
                results.append(RouteResult(record=record, score=score, match_type=match_type))
        return results

    def _search_keywords(self, query: str, allow_adult: bool) -> list[RouteResult]:
        query_norm = _normalized_phrase(query)
        query_word_set = set(_query_terms(query))
        results = []
        for record in self.registry.all():
            if record.is_adult and not allow_adult:
                continue
            score = 0.0
            match_type = "keyword"
            trigger_norm = _normalized_phrase(record.trigger_word)
            if trigger_norm and trigger_norm in query_norm:
                score += 3.0
                match_type = "trigger"
            name_norm = _normalized_phrase(record.name)
            if name_norm and name_norm in query_norm:
                score += 2.0
                if match_type == "keyword":
                    match_type = "name"
            for name_word in set(_query_terms(record.name)):
                if name_word in query_word_set:
                    score += 1.5
                    if match_type == "keyword":
                        match_type = "name"
            matched = set()
            for kw in record.keywords:
                kw_terms = _query_terms(kw)
                if kw_terms and " ".join(kw_terms) in query_norm:
                    score += 1.25
                for word in query_word_set:
                    if word in kw_terms and word not in matched:
                        score += 1.0
                        matched.add(word)
            for neg_kw in record.negative_keywords:
                neg_norm = _normalized_phrase(neg_kw)
                if neg_norm and neg_norm in query_norm:
                    score -= 0.5
                elif any(w in query_word_set for w in _query_terms(neg_kw)):
                    score -= 0.25
            if score > 0:
                results.append(RouteResult(record=record, score=score * 1.2, match_type=match_type))
        return results
