"""Tests for teleportation pipeline scheduling."""

import tempfile

from latentteleport.cache import LatentCache
from latentteleport.combiner import create_combiner
from latentteleport.config import CombinerConfig, TeleportConfig, TokenizerConfig
from latentteleport.refine import TeleportPipeline
from latentteleport.tokenizer import create_tokenizer


def _pipeline_for(combiner_config: CombinerConfig) -> TeleportPipeline:
    cache = LatentCache(tempfile.mkdtemp(), resolution=(512, 512))
    tokenizer = create_tokenizer(TokenizerConfig(strategy="curated"))
    combiner = create_combiner(combiner_config)
    return TeleportPipeline(
        pipe=object(),
        cache=cache,
        tokenizer=tokenizer,
        combiner=combiner,
        config=TeleportConfig(num_steps=20),
        combiner_config=combiner_config,
    )


def test_fixed_refinement_steps_choose_matching_cache_start_step():
    pipeline = _pipeline_for(CombinerConfig(refinement_steps=8))

    assert pipeline._fixed_refinement_steps == 8
    assert pipeline._start_step == 12


def test_zero_refinement_steps_keeps_adaptive_timestep_mode():
    pipeline = _pipeline_for(CombinerConfig(refinement_steps=0, teleport_timestep=0.3))

    assert pipeline._fixed_refinement_steps is None
    assert pipeline._start_step == 6
