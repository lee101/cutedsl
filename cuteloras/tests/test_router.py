from cuteloras.registry import LoRARecord, LoRARegistry
from cuteloras.router import LoRARouter, query_allows_adult


def _registry():
    return LoRARegistry(
        [
            LoRARecord(
                id="anime_style",
                name="Anime Style",
                trigger_word="animestyle",
                keywords=["anime", "manga", "cel shading"],
            ),
            LoRARecord(
                id="realistic_person",
                name="Realistic Person",
                trigger_word="",
                keywords=["photorealistic", "portrait", "person"],
                negative_keywords=["anime", "cartoon"],
            ),
            LoRARecord(id="nsfw_lora", name="NSFW Realistic", is_adult=True, keywords=["nsfw", "nude"]),
        ]
    )


def _router():
    return LoRARouter(_registry(), embedding_model=None)


def test_keyword_routing():
    router = _router()
    results = router.search("anime girl with manga cel shading")
    assert results[0].record.id == "anime_style"


def test_trigger_word_match():
    router = _router()
    results = router.search("animestyle warrior")
    assert results[0].record.id == "anime_style"
    assert results[0].match_type == "trigger"


def test_negative_keywords_penalize():
    router = _router()
    results = router.search("anime portrait")
    scores = {r.record.id: r.score for r in results}
    assert scores["anime_style"] > scores.get("realistic_person", 0)


def test_adult_gated_by_default():
    router = _router()
    assert all(r.record.id != "nsfw_lora" for r in router.search("a portrait"))


def test_adult_allowed_by_query():
    router = _router()
    results = router.search("nsfw nude portrait")
    assert any(r.record.id == "nsfw_lora" for r in results)


def test_adult_allowed_by_flag():
    router = _router()
    results = router.search("nude figure study", allow_adult=True)
    assert any(r.record.id == "nsfw_lora" for r in results)


def test_route_threshold():
    router = _router()
    assert router.route("qwzx unrelated gibberish") is None


def test_query_allows_adult():
    assert query_allows_adult("nsfw anime girl")
    assert not query_allows_adult("a cute cat")


def test_template_applied():
    record = LoRARecord(id="x", template="animestyle, {prompt}")
    assert record.apply_template("a cat") == "animestyle, a cat"
