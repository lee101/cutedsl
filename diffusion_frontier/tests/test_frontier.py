from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import torch
from PIL import Image

from diffusion_frontier.benchmark import (
    _annotate_quality,
    _encode_conditioning,
    _pareto_flags,
    _release_text_encoders,
    _review_manifest,
    _should_stage_krea_conditioning,
)
from diffusion_frontier.catalog import MODEL_CATALOG, get_model
from diffusion_frontier.consolidate import consolidate_reports
from diffusion_frontier.flow_resume import capture_flow_latent, remaining_linear_sigmas, resume_flow_latent
from diffusion_frontier.metrics import compare_images
from diffusion_frontier.preflight import _weights_cached, inspect_model
from diffusion_frontier.sdxl import (
    DEFAULT_PROFILES,
    SDXL_PROFILES,
    _adapter_specs,
    _annotate_native_retention,
    _review_artifacts,
)


def test_catalog_covers_requested_model_families():
    assert {"zimage", "flux", "sdxl", "krea2"} <= {spec.family for spec in MODEL_CATALOG.values()}
    assert get_model("krea2-turbo").default_steps == 8
    assert get_model("flux-schnell").default_steps == 4
    assert get_model("realvisxl-v4").supports_lora


def test_identical_image_metrics_are_json_safe():
    image = Image.new("RGB", (32, 32), (10, 20, 30))
    metrics = compare_images(image, image)
    assert metrics["images_identical"] is True
    assert metrics["ssim"] == 1.0
    assert metrics["psnr_db"] == 99.0


def test_preflight_never_downloads_and_explains_uncached_model(tmp_path: Path):
    snapshot = {
        "disk_free_gib": 100.0,
        "ram_available_gib": 128.0,
        "cuda_available": True,
        "gpu_free_gib": 24.0,
    }
    result = inspect_model(get_model("flux-schnell"), snapshot, cache_dir=tmp_path, allow_download=False)
    assert result["cached"] is False
    assert result["ready"] is False
    assert any("not cached" in blocker for blocker in result["blockers"])


def test_preflight_rejects_metadata_only_cache(tmp_path: Path):
    spec = replace(get_model("flux-schnell"), approximate_download_gib=1e-9)
    snapshot = tmp_path / f"models--{spec.repo_id.replace('/', '--')}" / "snapshots" / "revision"
    snapshot.mkdir(parents=True)
    (snapshot / "model_index.json").write_text("{}")

    assert not _weights_cached(spec, snapshot.parents[1])

    (snapshot / "transformer").mkdir()
    (snapshot / "transformer" / "weights.safetensors").write_bytes(b"weights")
    assert _weights_cached(spec, snapshot.parents[1])


def test_pareto_is_scoped_per_model():
    rows = [
        {"model": "a", "wall_s": 1.0, "quality_retention": {"ssim": 0.8}},
        {"model": "a", "wall_s": 2.0, "quality_retention": {"ssim": 0.7}},
        {"model": "b", "wall_s": 3.0, "quality_retention": {"ssim": 0.6}},
    ]
    _pareto_flags(rows)
    assert [row["pareto"] for row in rows] == [True, False, True]


def test_review_manifest_is_deterministic_and_hides_model_labels():
    rows = [
        {"model": "a", "prompt_index": 0, "image_path": "a.webp"},
        {"model": "b", "prompt_index": 0, "image_path": "b.webp"},
    ]
    first = _review_manifest(rows, 42)
    second = _review_manifest(rows, 42)
    assert first == second
    assert all("model" not in row for row in first)


def test_quality_references_are_matched_by_repeat(tmp_path: Path):
    rows = []
    for run, color in ((0, 20), (1, 80)):
        for steps in (4, 20):
            path = tmp_path / f"r{run}_s{steps}.png"
            Image.new("RGB", (16, 16), (color if steps == 20 else 0, 0, 0)).save(path)
            rows.append(
                {
                    "model": "test",
                    "prompt_index": 0,
                    "seed": 42,
                    "run": run,
                    "steps": steps,
                    "image_path": str(path),
                }
            )

    _annotate_quality(rows, include_lpips=False)

    references = [row for row in rows if row["steps"] == 20]
    assert all(row["quality_retention"]["images_identical"] for row in references)


def test_sdxl_profiles_use_native40_and_official_distilled_schedulers():
    assert "native40" in DEFAULT_PROFILES
    assert SDXL_PROFILES["lcm4"].scheduler == "lcm"
    assert SDXL_PROFILES["hyper4"].scheduler == "ddim-trailing"
    assert SDXL_PROFILES["hyper8"].steps == 8
    adapters = _adapter_specs([SDXL_PROFILES[key] for key in DEFAULT_PROFILES])
    assert set(adapters) == {"lcm-sdxl", "hyper-sdxl-4"}


def test_sdxl_retention_always_uses_native40_not_largest_distilled_steps(tmp_path: Path):
    rows = []
    for profile, color in (("native40", 80), ("lcm8", 20), ("native20", 60)):
        path = tmp_path / f"{profile}.png"
        Image.new("RGB", (16, 16), (color, 0, 0)).save(path)
        rows.append(
            {
                "model": "base",
                "profile": profile,
                "prompt_index": 0,
                "seed": 42,
                "run": 0,
                "image_path": str(path),
            }
        )

    _annotate_native_retention(rows)

    reference = next(row for row in rows if row["profile"] == "native40")
    assert reference["quality_retention"]["images_identical"]
    assert all(row["quality_reference"] == "native40" for row in rows)


def test_sdxl_blind_review_has_separate_answer_key(tmp_path: Path):
    image_path = tmp_path / "source.webp"
    Image.new("RGB", (16, 16), "blue").save(image_path)
    rows = [
        {
            "model": "base",
            "profile": "hyper4",
            "prompt_index": 0,
            "seed": 42,
            "run": 0,
            "image_path": str(image_path),
        }
    ]

    manifest, answer_key = _review_artifacts(rows, tmp_path, 42)

    assert "model" not in manifest[0]
    assert "profile" not in manifest[0]
    assert answer_key[manifest[0]["candidate_id"]]["profile"] == "hyper4"


def test_large_model_prompt_conditioning_uses_family_specific_outputs():
    class FakeFluxPipe:
        def encode_prompt(self, **kwargs):
            return "flux-embeds", "flux-pooled", "flux-text-ids"

    class FakeKreaPipe:
        def encode_prompt(self, **kwargs):
            return "krea-embeds", "krea-mask"

    flux = _encode_conditioning(FakeFluxPipe(), get_model("flux-schnell"), "fox")
    krea = _encode_conditioning(FakeKreaPipe(), get_model("krea2-turbo"), "fox")

    assert flux == {"prompt_embeds": "flux-embeds", "pooled_prompt_embeds": "flux-pooled"}
    assert krea == {"prompt_embeds": "krea-embeds", "prompt_embeds_mask": "krea-mask"}


def test_flow_resume_replays_original_sigma_tail_and_captured_latent():
    class Result:
        images = ["image"]

    class FakePipe:
        calls = []

        def __call__(self, **kwargs):
            self.calls.append(kwargs)
            callback = kwargs.get("callback_on_step_end")
            if callback:
                for step in range(kwargs["num_inference_steps"]):
                    callback(self, step, None, {"latents": torch.full((1, 2), float(step))})
            return Result()

    pipe = FakePipe()
    captured = capture_flow_latent(pipe, {"prompt_embeds": "cached"}, 4, 1)
    resumed = resume_flow_latent(pipe, captured.latent, {"prompt_embeds": "cached"}, 4, 1)

    assert remaining_linear_sigmas(4, 1) == [0.5, 0.25]
    assert torch.equal(pipe.calls[1]["latents"], torch.ones(1, 2))
    assert pipe.calls[1]["sigmas"] == [0.5, 0.25]
    assert resumed.physical_steps == 2


def test_flow_resume_sigma_tail_matches_dynamic_shifted_scheduler():
    from diffusers import FlowMatchEulerDiscreteScheduler

    scheduler = FlowMatchEulerDiscreteScheduler(use_dynamic_shifting=True, shift_terminal=None)
    scheduler.set_timesteps(sigmas=[1.0, 0.75, 0.5, 0.25], mu=1.15)
    expected_tail = scheduler.timesteps[2:].clone()
    resumed = FlowMatchEulerDiscreteScheduler.from_config(scheduler.config)
    resumed.set_timesteps(sigmas=remaining_linear_sigmas(4, 1), mu=1.15)

    assert torch.equal(resumed.timesteps, expected_tail)


def test_release_text_encoders_preserves_other_pipeline_components():
    class FakePipe:
        text_encoder = object()
        text_encoder_2 = object()
        tokenizer = object()
        tokenizer_2 = None
        transformer = "keep"

    pipe = FakePipe()
    released = _release_text_encoders(pipe)

    assert released == ["text_encoder", "text_encoder_2", "tokenizer"]
    assert pipe.text_encoder is None
    assert pipe.transformer == "keep"


def test_krea_conditioning_stages_only_for_memory_safe_quantized_cuda_path():
    from types import SimpleNamespace

    spec = get_model("krea2-turbo")
    args = SimpleNamespace(
        offload="cuda",
        quantization="torchao-fp8wo",
        cache_prompt_embeds=True,
        release_text_encoders=True,
        lora=None,
    )
    assert _should_stage_krea_conditioning(spec, args)
    args.lora = "adapter.safetensors"
    assert _should_stage_krea_conditioning(spec, args)


def test_consolidation_keeps_ssim_labeled_as_within_model(tmp_path: Path):
    path = tmp_path / "results.json"
    path.write_text(
        json.dumps(
            {
                "target_latency_s": 1.0,
                "rows": [
                    {
                        "model": "base",
                        "profile": "fast",
                        "physical_steps": 4,
                        "wall_s": 0.9,
                        "quality_retention": {"ssim": 0.6},
                    },
                    {
                        "model": "base",
                        "profile": "fast",
                        "physical_steps": 4,
                        "wall_s": 1.1,
                        "quality_retention": {"ssim": 0.8},
                    },
                ],
            }
        )
    )

    report = consolidate_reports([path])

    assert report["profiles"][0]["latency_passes"] == 1
    assert report["profiles"][0]["ssim_median_within_model_reference"] == 0.7
    assert "do not rank" in report["metric_warning"]


def test_consolidation_never_mixes_resolutions(tmp_path: Path):
    path = tmp_path / "results.json"
    path.write_text(
        json.dumps(
            {
                "rows": [
                    {"model": "base", "profile": "fast", "width": 512, "height": 512, "wall_s": 0.5},
                    {"model": "base", "profile": "fast", "width": 1024, "height": 1024, "wall_s": 2.0},
                ]
            }
        )
    )

    profiles = consolidate_reports([path])["profiles"]

    assert len(profiles) == 2
    assert {tuple(item["resolutions"]) for item in profiles} == {("512x512",), ("1024x1024",)}
    assert {item["wall_s_median"] for item in profiles} == {0.5, 2.0}
