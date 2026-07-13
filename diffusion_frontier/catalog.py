"""Public model catalog for the diffusion latency/quality frontier.

The catalog contains only public model identifiers and runtime metadata.  It
deliberately has no product-specific routing, prompt data, or proprietary RA1
logic so the benchmarking/acceleration layer can stay open source.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ModelSpec:
    key: str
    family: str
    repo_id: str
    pipeline_class: str
    default_steps: int
    reference_steps: int
    guidance_scale: float
    approximate_download_gib: float
    approximate_resident_gib: float
    license: str
    loader: str = "pretrained"
    source_file: str | None = None
    variant: str | None = None
    max_sequence_length: int | None = None
    supports_lora: bool = True
    recommended_offload: str = "auto"
    notes: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


MODEL_CATALOG: dict[str, ModelSpec] = {
    "zimage-turbo": ModelSpec(
        key="zimage-turbo",
        family="zimage",
        repo_id="Tongyi-MAI/Z-Image-Turbo",
        pipeline_class="ZImagePipeline",
        default_steps=8,
        reference_steps=20,
        guidance_scale=0.0,
        approximate_download_gib=12.5,
        approximate_resident_gib=13.5,
        license="Apache-2.0",
        recommended_offload="cuda",
        notes="Use the CuteZImage adapter when evaluating custom kernels.",
    ),
    "flux-schnell": ModelSpec(
        key="flux-schnell",
        family="flux",
        repo_id="black-forest-labs/FLUX.1-schnell",
        pipeline_class="FluxPipeline",
        default_steps=4,
        reference_steps=4,
        guidance_scale=0.0,
        approximate_download_gib=31.5,
        approximate_resident_gib=32.5,
        license="Apache-2.0",
        max_sequence_length=256,
        recommended_offload="model",
        notes="Gated weights; accept the Hugging Face access conditions first.",
    ),
    "proteus-v0.2": ModelSpec(
        key="proteus-v0.2",
        family="sdxl",
        repo_id="dataautogpt3/ProteusV0.2",
        pipeline_class="StableDiffusionXLPipeline",
        default_steps=20,
        reference_steps=40,
        guidance_scale=7.0,
        approximate_download_gib=6.5,
        approximate_resident_gib=7.5,
        license="GPL-3.0",
        loader="single_file",
        source_file="ProteusV0.2.safetensors",
        recommended_offload="cuda",
    ),
    "proteus-v0.4": ModelSpec(
        key="proteus-v0.4",
        family="sdxl",
        repo_id="dataautogpt3/ProteusV0.4",
        pipeline_class="StableDiffusionXLPipeline",
        default_steps=20,
        reference_steps=40,
        guidance_scale=7.0,
        approximate_download_gib=6.5,
        approximate_resident_gib=7.5,
        license="GPL-3.0",
        loader="single_file",
        source_file="ProteusV0.4.safetensors",
        recommended_offload="cuda",
    ),
    "realvisxl-v4": ModelSpec(
        key="realvisxl-v4",
        family="sdxl",
        repo_id="SG161222/RealVisXL_V4.0",
        pipeline_class="StableDiffusionXLPipeline",
        default_steps=25,
        reference_steps=40,
        guidance_scale=7.0,
        approximate_download_gib=6.5,
        approximate_resident_gib=7.5,
        license="OpenRAIL++",
        variant="fp16",
        recommended_offload="cuda",
    ),
    "krea2-turbo": ModelSpec(
        key="krea2-turbo",
        family="krea2",
        repo_id="krea/Krea-2-Turbo",
        pipeline_class="Krea2Pipeline",
        default_steps=8,
        reference_steps=8,
        guidance_scale=0.0,
        approximate_download_gib=33.3,
        approximate_resident_gib=34.5,
        license="Krea 2 Community License",
        max_sequence_length=512,
        recommended_offload="model",
        notes="Requires a Diffusers source build containing Krea2Pipeline.",
    ),
}


def get_model(key: str) -> ModelSpec:
    try:
        return MODEL_CATALOG[key]
    except KeyError as exc:
        choices = ", ".join(sorted(MODEL_CATALOG))
        raise KeyError(f"unknown model {key!r}; choose one of: {choices}") from exc
