from pathlib import Path

from zimageaccelerated.sdcpp_benchmark import build_sdcpp_command, summarize_latencies


def test_build_sdcpp_command_includes_required_zimage_flags():
    command = build_sdcpp_command(
        sdcpp_bin=Path("/tmp/sd-cli"),
        diffusion_model=Path("/models/zimage.gguf"),
        vae=Path("/models/ae.safetensors"),
        llm=Path("/models/qwen.gguf"),
        prompt="test prompt",
        output_path=Path("/tmp/out.png"),
        width=768,
        height=512,
        steps=9,
        cfg_scale=1.0,
        seed=123,
        sampling_method="euler",
        scheduler="discrete",
        rng="cuda",
        offload_to_cpu=True,
        diffusion_fa=True,
        clip_on_cpu=True,
        vae_on_cpu=False,
        vae_tiling=True,
        cache_mode="cache-dit",
        cache_option="threshold=0.25",
        cache_preset="fast",
        verbose=True,
        extra_args=["--preview", "none"],
    )

    assert command[:8] == [
        "/tmp/sd-cli",
        "--diffusion-model",
        "/models/zimage.gguf",
        "--vae",
        "/models/ae.safetensors",
        "--llm",
        "/models/qwen.gguf",
        "-p",
    ]
    assert "--cfg-scale" in command
    assert "--steps" in command
    assert "-s" in command
    assert "--scheduler" in command
    assert "--rng" in command
    assert "--offload-to-cpu" in command
    assert "--diffusion-fa" in command
    assert "--clip-on-cpu" in command
    assert "--vae-tiling" in command
    assert "--cache-mode" in command
    assert "--cache-option" in command
    assert "--cache-preset" in command
    assert "-v" in command
    assert command[-2:] == ["--preview", "none"]


def test_build_sdcpp_command_omits_optional_flags_when_disabled():
    command = build_sdcpp_command(
        sdcpp_bin=Path("/tmp/sd-cli"),
        diffusion_model=Path("/models/zimage.gguf"),
        vae=Path("/models/ae.safetensors"),
        llm=Path("/models/qwen.gguf"),
        prompt="test prompt",
        output_path=Path("/tmp/out.png"),
        width=768,
        height=512,
        steps=9,
        cfg_scale=1.0,
        seed=123,
        sampling_method="euler",
        scheduler=None,
        rng=None,
        offload_to_cpu=False,
        diffusion_fa=False,
        clip_on_cpu=False,
        vae_on_cpu=False,
        vae_tiling=False,
        cache_mode=None,
        cache_option=None,
        cache_preset=None,
        verbose=False,
        extra_args=[],
    )

    for flag in [
        "--scheduler",
        "--rng",
        "--offload-to-cpu",
        "--diffusion-fa",
        "--clip-on-cpu",
        "--vae-on-cpu",
        "--vae-tiling",
        "--cache-mode",
        "--cache-option",
        "--cache-preset",
        "-v",
    ]:
        assert flag not in command


def test_summarize_latencies_reports_basic_stats():
    summary = summarize_latencies([10.0, 20.0, 30.0])

    assert summary["avg_latency_ms"] == 20.0
    assert summary["min_latency_ms"] == 10.0
    assert summary["max_latency_ms"] == 30.0
    assert summary["std_latency_ms"] > 0.0
