"""Tests for visual unit tokenizer."""

import pytest

from latentteleport.tokenizer import VisualUnit, CuratedTokenizer


def _has_spacy():
    try:
        import spacy
        spacy.load("en_core_web_sm")
        return True
    except Exception:
        return False


class TestVisualUnit:
    def test_from_text_deterministic(self):
        a = VisualUnit.from_text("red car")
        b = VisualUnit.from_text("red car")
        assert a == b
        assert a.unit_id == b.unit_id

    def test_from_text_normalized(self):
        a = VisualUnit.from_text("Red Car")
        b = VisualUnit.from_text("  red car  ")
        assert a == b

    def test_different_texts_different_ids(self):
        a = VisualUnit.from_text("red car")
        b = VisualUnit.from_text("blue car")
        assert a.unit_id != b.unit_id


class TestCuratedTokenizer:
    def test_basic(self):
        tok = CuratedTokenizer()
        units = tok.tokenize("a red car on a sunny beach")
        texts = [u.text for u in units]
        assert any("car" in t for t in texts)
        assert any("beach" in t for t in texts)

    def test_single_word(self):
        tok = CuratedTokenizer()
        units = tok.tokenize("cat")
        assert len(units) >= 1
        assert units[0].text == "cat"

    def test_empty_falls_back(self):
        tok = CuratedTokenizer()
        units = tok.tokenize("xyzzy")
        assert len(units) >= 1


@pytest.mark.skipif(not _has_spacy(), reason="spacy not installed")
class TestNLPTokenizer:
    def test_noun_chunks(self):
        from latentteleport.tokenizer import NLPTokenizer
        tok = NLPTokenizer()
        units = tok.tokenize("a red car parked on a sunny beach")
        assert len(units) >= 2
        texts = [u.text for u in units]
        assert any("car" in t for t in texts)
