"""Load Anima 2.9B without the reference load-time overhead.

The reference path materialises a 2.9B fp32 transformer on CPU (random init, then
overwritten) and also loads the 3.9 GB base transformer it immediately discards.
This loader instantiates on `meta`, adopts the bf16 checkpoint tensors with
`assign=True`, and never reads the base transformer weights.
"""

from __future__ import annotations

import hashlib
import importlib
import os
import re
import sys
from pathlib import Path

MODEL_ID = os.getenv("ANIMA_MODEL_ID", "Gazingstars123/Anima-2.9B")
MODEL_FILE = os.getenv("ANIMA_MODEL_FILE", "Anima-2.9B-preview-v1.safetensors")
MODEL_REVISION = os.getenv("ANIMA_MODEL_REVISION", "fb00923d6a68424b731048d4c65da61eed1a6cc2")
BASE_MODEL_ID = os.getenv("ANIMA_BASE_MODEL_ID", "CalamitousFelicitousness/Anima-sdnext-diffusers")
BASE_MODEL_REVISION = os.getenv("ANIMA_BASE_MODEL_REVISION", "587e3941c37ace6234f9c0daa5c908408652870a")
SPACE_ID = "akhaliq/Anima-2.9B"
SPACE_REVISION = "88543bfa289482b451631a565f54653b22b1c1cb"
SPACE_SOURCE_HASHES = {
    "pipeline.py": "c2a963876781988ed1343788c9a3e529b076d569dd909bcc12b534271686b83a",
    "modeling_llm_adapter.py": "9fc725376c9373c0db9da3efff123996e5cbfaae7e12abb136b1ef99c31c2aff",
}

BLOCK_MAP = {
    "self_attn.q_proj": "attn1.to_q",
    "self_attn.k_proj": "attn1.to_k",
    "self_attn.v_proj": "attn1.to_v",
    "self_attn.output_proj": "attn1.to_out.0",
    "self_attn.q_norm": "attn1.norm_q",
    "self_attn.k_norm": "attn1.norm_k",
    "cross_attn.q_proj": "attn2.to_q",
    "cross_attn.k_proj": "attn2.to_k",
    "cross_attn.v_proj": "attn2.to_v",
    "cross_attn.output_proj": "attn2.to_out.0",
    "cross_attn.q_norm": "attn2.norm_q",
    "cross_attn.k_norm": "attn2.norm_k",
    "mlp.layer1": "ff.net.0.proj",
    "mlp.layer2": "ff.net.2",
    "adaln_modulation_self_attn.1": "norm1.linear_1",
    "adaln_modulation_self_attn.2": "norm1.linear_2",
    "adaln_modulation_cross_attn.1": "norm2.linear_1",
    "adaln_modulation_cross_attn.2": "norm2.linear_2",
    "adaln_modulation_mlp.1": "norm3.linear_1",
    "adaln_modulation_mlp.2": "norm3.linear_2",
}

TOP_MAP = {
    "net.x_embedder.proj.1.weight": "patch_embed.proj.weight",
    "net.t_embedding_norm.weight": "time_embed.norm.weight",
    "net.final_layer.adaln_modulation.1.weight": "norm_out.linear_1.weight",
    "net.final_layer.adaln_modulation.2.weight": "norm_out.linear_2.weight",
    "net.final_layer.linear.weight": "proj_out.weight",
}


def remap_checkpoint(state_dict: dict) -> dict:
    remapped = {}
    for key, value in state_dict.items():
        if key.startswith("net.blocks."):
            block_n, module_key = key[len("net.blocks."):].split(".", 1)
            suffix = ""
            for candidate in (".weight", ".bias"):
                if module_key.endswith(candidate):
                    module_key, suffix = module_key[: -len(candidate)], candidate
                    break
            mapped = BLOCK_MAP.get(module_key)
            if mapped:
                remapped[f"transformer_blocks.{block_n}.{mapped}{suffix}"] = value
            continue
        mapped = TOP_MAP.get(key)
        if mapped:
            remapped[mapped] = value
            continue
        match = re.match(r"net\.t_embedder\.\d+\.(linear_[12]\.weight)", key)
        if match:
            remapped[f"time_embed.t_embedder.{match.group(1)}"] = value
    return remapped


def load_pipeline_class():
    """Import the SHA-256 verified reference pipeline from the pinned Space revision."""
    from huggingface_hub import hf_hub_download

    source_dir = None
    for filename, expected in SPACE_SOURCE_HASHES.items():
        path = Path(hf_hub_download(SPACE_ID, filename, repo_type="space", revision=SPACE_REVISION))
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"pinned Space source hash mismatch for {filename}")
        source_dir = path.parent
    if source_dir is None:
        raise RuntimeError("Anima pipeline sources are unavailable")
    if str(source_dir) not in sys.path:
        sys.path.insert(0, str(source_dir))
    importlib.import_module("modeling_llm_adapter")
    return importlib.import_module("pipeline").AnimaTextToImagePipeline


def load_transformer(torch, device: str = "cuda", num_layers: int = 40):
    from diffusers import CosmosTransformer3DModel
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    config = dict(
        CosmosTransformer3DModel.load_config(
            BASE_MODEL_ID, subfolder="transformer", revision=BASE_MODEL_REVISION
        )
    )
    config["num_layers"] = num_layers
    with torch.device("meta"):
        transformer = CosmosTransformer3DModel.from_config(config)

    checkpoint = os.getenv("ANIMA_MODEL_PATH", "").strip() or hf_hub_download(
        MODEL_ID, MODEL_FILE, revision=MODEL_REVISION
    )
    state_dict = remap_checkpoint(load_file(checkpoint, device=device))
    missing, unexpected = transformer.load_state_dict(state_dict, strict=False, assign=True)
    still_meta = [name for name, value in transformer.state_dict().items() if value.is_meta]
    if missing or unexpected or still_meta:
        raise RuntimeError(
            f"Anima checkpoint remap incomplete: missing={missing[:8]} "
            f"unexpected={unexpected[:8]} uninitialized={still_meta[:8]}"
        )
    return transformer.to(dtype=torch.bfloat16)


def load_pipeline(torch, device: str = "cuda", fused: bool = True, compile_mode: str = ""):
    """Build the Anima pipeline with the CuteAnima fast paths applied."""
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    pipeline_class = load_pipeline_class()
    transformer = load_transformer(torch, device=device)
    pipe = pipeline_class.from_pretrained(
        BASE_MODEL_ID,
        revision=BASE_MODEL_REVISION,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        transformer=None,
    )
    pipe.register_modules(transformer=transformer)
    pipe.set_progress_bar_config(disable=True)
    pipe.to(device)

    if fused:
        from .patch import apply_fused_blocks

        apply_fused_blocks(pipe.transformer)
    if compile_mode:
        pipe.transformer = torch.compile(pipe.transformer, mode=compile_mode, fullgraph=False, dynamic=False)
    return pipe
