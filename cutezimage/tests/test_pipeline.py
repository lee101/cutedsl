"""Tests for cutezimage pipeline helpers."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

import cutezimage.pipeline as zpipe


class _FakeTransformer(nn.Module):
    def __init__(self, name: str = "orig"):
        super().__init__()
        self.name = name
        self.weight = nn.Parameter(torch.ones(1, dtype=torch.float32))


class _FakeCuteTransformer(_FakeTransformer):
    def __init__(self, name: str = "cute"):
        super().__init__(name=name)
        self.source = None
        self.compile_mode = None

    @classmethod
    def from_diffusers(cls, model):
        inst = cls()
        inst.source = model
        return inst

    @classmethod
    def from_diffusers_compiled(cls, model, compile_mode="reduce-overhead"):
        inst = cls()
        inst.source = model
        inst.compile_mode = compile_mode
        return inst


class _FakePipelineBase:
    def __init__(self, *, transformer=None, controlnet=None, scheduler=None, vae=None, text_encoder=None, tokenizer=None):
        self.transformer = transformer or _FakeTransformer()
        self.scheduler = scheduler or object()
        self.vae = vae or object()
        self.text_encoder = text_encoder or object()
        self.tokenizer = tokenizer or object()
        if controlnet is not None:
            self.controlnet = controlnet
        self.attention_slicing_enabled = False
        self.vae_slicing_enabled = False
        self.cpu_offload_enabled = False
        self.to_device = None
        self.last_call = None

    @property
    def components(self):
        return {
            "scheduler": self.scheduler,
            "vae": self.vae,
            "text_encoder": self.text_encoder,
            "tokenizer": self.tokenizer,
            "transformer": self.transformer,
        }

    def enable_attention_slicing(self):
        self.attention_slicing_enabled = True

    def enable_vae_slicing(self):
        self.vae_slicing_enabled = True

    def enable_model_cpu_offload(self):
        self.cpu_offload_enabled = True

    def to(self, device):
        self.to_device = str(device)
        return self

    def __call__(self, **kwargs):
        self.last_call = kwargs
        return SimpleNamespace(images=[f"image:{kwargs.get('prompt', '')}"])


class _FakeZImagePipeline(_FakePipelineBase):
    load_calls: list[tuple[str, torch.dtype, bool]] = []

    @classmethod
    def from_pretrained(cls, model_path, torch_dtype=None, low_cpu_mem_usage=False):
        cls.load_calls.append((model_path, torch_dtype, low_cpu_mem_usage))
        return cls()


class _FakeZImageImg2ImgPipeline(_FakePipelineBase):
    def __init__(self, **components):
        super().__init__(**components)


class _FakeControlNetModel:
    pretrained_calls: list[tuple[str, torch.dtype]] = []
    single_file_calls: list[tuple[str, torch.dtype]] = []

    def __init__(self, source_type: str, source: str):
        self.source_type = source_type
        self.source = source

    @classmethod
    def from_pretrained(cls, model_path, torch_dtype=None):
        cls.pretrained_calls.append((model_path, torch_dtype))
        return cls("pretrained", model_path)

    @classmethod
    def from_single_file(cls, model_path, torch_dtype=None):
        cls.single_file_calls.append((model_path, torch_dtype))
        return cls("single_file", model_path)


class _FakeZImageControlNetPipeline(_FakePipelineBase):
    def __init__(self, scheduler, vae, text_encoder, tokenizer, transformer, controlnet):
        super().__init__(
            transformer=transformer,
            controlnet=controlnet,
            scheduler=scheduler,
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
        )


@pytest.fixture
def fake_pipeline_env(monkeypatch):
    _FakeZImagePipeline.load_calls.clear()
    _FakeControlNetModel.pretrained_calls.clear()
    _FakeControlNetModel.single_file_calls.clear()

    zpipe.clear_pipeline_caches()
    monkeypatch.setattr(zpipe, "_IMPORT_ERROR", None)
    monkeypatch.setattr(zpipe, "ZImagePipeline", _FakeZImagePipeline)
    monkeypatch.setattr(zpipe, "ZImageImg2ImgPipeline", _FakeZImageImg2ImgPipeline)
    monkeypatch.setattr(zpipe, "ZImageControlNetPipeline", _FakeZImageControlNetPipeline)
    monkeypatch.setattr(zpipe, "ZImageControlNetModel", _FakeControlNetModel)
    monkeypatch.setattr(zpipe, "CuteZImageTransformer", _FakeCuteTransformer)
    monkeypatch.setattr(zpipe, "hf_hub_download", lambda repo_id, filename: f"/tmp/{repo_id.replace('/', '__')}__{filename}")
    yield
    zpipe.clear_pipeline_caches()


def test_accelerate_zimage_pipeline_replaces_transformer(fake_pipeline_env):
    pipe = _FakeZImagePipeline.from_pretrained("repo/zimage", torch_dtype=torch.float32, low_cpu_mem_usage=False)
    original_transformer = pipe.transformer

    zpipe.accelerate_zimage_pipeline(
        pipe,
        compile_mode="reduce-overhead",
        device="cpu",
        torch_dtype=torch.float32,
        enable_cpu_offload=False,
    )

    assert isinstance(pipe.transformer, _FakeCuteTransformer)
    assert pipe.transformer.source is original_transformer
    assert pipe.transformer.compile_mode == "reduce-overhead"
    assert pipe.attention_slicing_enabled is True
    assert pipe.vae_slicing_enabled is True
    assert pipe.to_device == "cpu"


def test_get_zimage_pipelines_caches_and_shares_transformer(fake_pipeline_env):
    text2img, img2img = zpipe.get_zimage_pipelines(
        model_path="repo/zimage",
        torch_dtype=torch.float32,
        device="cpu",
        enable_cpu_offload=False,
    )
    cached_text2img, cached_img2img = zpipe.get_zimage_pipelines(
        model_path="repo/zimage",
        torch_dtype=torch.float32,
        device="cpu",
        enable_cpu_offload=False,
    )

    assert text2img is cached_text2img
    assert img2img is cached_img2img
    assert len(_FakeZImagePipeline.load_calls) == 1
    assert isinstance(text2img.transformer, _FakeCuteTransformer)
    assert img2img.transformer is text2img.transformer


def test_get_zimage_controlnet_pipeline_loads_single_file_and_shares_transformer(fake_pipeline_env):
    controlnet_pipe = zpipe.get_zimage_controlnet_pipeline(
        model_path="repo/zimage",
        controlnet_model="alibaba-pai/controlnet",
        controlnet_filename="controlnet.safetensors",
        torch_dtype=torch.float32,
        device="cpu",
        enable_cpu_offload=False,
    )
    base_pipe, _ = zpipe.get_zimage_pipelines(
        model_path="repo/zimage",
        torch_dtype=torch.float32,
        device="cpu",
        enable_cpu_offload=False,
    )

    assert controlnet_pipe.transformer is base_pipe.transformer
    assert _FakeControlNetModel.single_file_calls == [
        ("/tmp/alibaba-pai__controlnet__controlnet.safetensors", torch.float32)
    ]
    assert controlnet_pipe.controlnet.source_type == "single_file"


def test_create_image_with_zimage_controlnet_forwards_control_image(fake_pipeline_env):
    output = zpipe.create_image_with_zimage_controlnet(
        "guided prompt",
        control_image="pose-image",
        width=640,
        height=832,
        controlnet_model="alibaba-pai/controlnet",
        controlnet_filename="controlnet.safetensors",
        torch_dtype=torch.float32,
        device="cpu",
        enable_cpu_offload=False,
        negative_prompt="bad anatomy",
    )

    pipe = zpipe.get_zimage_controlnet_pipeline(
        model_path=zpipe.ZIMAGE_MODEL_PATH,
        controlnet_model="alibaba-pai/controlnet",
        controlnet_filename="controlnet.safetensors",
        torch_dtype=torch.float32,
        device="cpu",
        enable_cpu_offload=False,
    )

    assert output == "image:guided prompt"
    assert pipe.last_call["control_image"] == "pose-image"
    assert pipe.last_call["width"] == 640
    assert pipe.last_call["height"] == 832
    assert pipe.last_call["negative_prompt"] == "bad anatomy"
