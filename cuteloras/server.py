"""Generic LoRA inference server — FastAPI app with hot-swapping and prompt routing.

Run: ``uvicorn cuteloras.server:app --port 8200`` with env:
- ``CUTELORAS_REGISTRY`` — path to a registry JSON, or a directory of .safetensors
- ``CUTELORAS_BASE_MODEL`` — backend id (default ``zimage``)
- ``CUTELORAS_MIN_ROUTE_SCORE`` — auto-routing threshold (default 0.45)
"""

from __future__ import annotations

import base64
import io
import logging
import os
import threading
import time
from typing import Any

from cuteloras.registry import LoRARegistry
from cuteloras.router import LoRARouter
from cuteloras.swapper import LoRASwapper

logger = logging.getLogger("cuteloras")


class ZImageBackend:
    name = "zimage"

    def __init__(
        self, compile_mode: str | None = None, device: str | None = None, enable_cpu_offload: bool | None = None
    ):
        from cutezimage.pipeline import get_zimage_pipelines

        kwargs: dict[str, Any] = {}
        if compile_mode is not None:
            kwargs["compile_mode"] = compile_mode
        if device is not None:
            kwargs["device"] = device
        if enable_cpu_offload is not None:
            kwargs["enable_cpu_offload"] = enable_cpu_offload
        self.pipe, _ = get_zimage_pipelines(**kwargs)
        self.transformer = self.pipe.transformer

    def generate(
        self,
        prompt: str,
        width: int = 1024,
        height: int = 1024,
        steps: int | None = None,
        guidance_scale: float | None = None,
        seed: int = 0,
    ):
        from cutezimage.pipeline import create_image_with_zimage

        kwargs: dict[str, Any] = {"seed": seed}
        if steps is not None:
            kwargs["num_inference_steps"] = steps
        if guidance_scale is not None:
            kwargs["guidance_scale"] = guidance_scale
        return create_image_with_zimage(prompt, width, height, **kwargs)


BACKENDS = {"zimage": ZImageBackend}


class LoRAServer:
    def __init__(self, backend, registry: LoRARegistry, min_route_score: float = 0.45):
        self.backend = backend
        self.registry = registry
        self.swapper = LoRASwapper(backend.transformer, registry)
        self.router = LoRARouter(registry, min_score=min_route_score)
        self._gen_lock = threading.Lock()

    def generate(
        self,
        prompt: str,
        lora_id: str | None = None,
        loras: list[tuple[str, float]] | None = None,
        scale: float | None = None,
        auto_route: bool = True,
        allow_adult: bool | None = None,
        **gen_kwargs,
    ) -> dict:
        record = None
        selection = "none"
        if loras is None:
            if lora_id:
                record = self.registry.get(lora_id)
                if record is None:
                    raise KeyError(f"unknown LoRA id: {lora_id}")
                selection = "explicit"
            elif auto_route:
                record = self.router.route(prompt, allow_adult=allow_adult)
                selection = "auto" if record else "none"
            if record is not None:
                loras = [(record.id, scale)]
                prompt = record.apply_template(prompt)
            else:
                loras = []

        with self._gen_lock:
            swap_info = self.swapper.activate(loras)
            start = time.perf_counter()
            image = self.backend.generate(prompt, **gen_kwargs)
            gen_ms = (time.perf_counter() - start) * 1000

        return {
            "image": image,
            "prompt": prompt,
            "lora": record.id if record else (loras[0][0] if loras else None),
            "selection": selection,
            "swap": swap_info,
            "gen_ms": gen_ms,
        }


def _load_registry() -> LoRARegistry:
    src = os.getenv("CUTELORAS_REGISTRY", "")
    if src and os.path.isdir(src):
        return LoRARegistry.from_directory(src)
    if src and os.path.isfile(src):
        return LoRARegistry.from_json(src)
    return LoRARegistry()


def create_app(server: LoRAServer | None = None):
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel

    app = FastAPI(title="cuteloras", version="0.1.0")
    state: dict[str, LoRAServer | None] = {"server": server}

    def get_server() -> LoRAServer:
        if state["server"] is None:
            backend_name = os.getenv("CUTELORAS_BASE_MODEL", "zimage")
            backend_cls = BACKENDS.get(backend_name)
            if backend_cls is None:
                raise HTTPException(500, f"unknown backend: {backend_name}")
            registry = _load_registry()
            state["server"] = LoRAServer(
                backend_cls(),
                registry,
                min_route_score=float(os.getenv("CUTELORAS_MIN_ROUTE_SCORE", "0.45")),
            )
        return state["server"]

    class GenerateRequest(BaseModel):
        prompt: str
        lora_id: str | None = None
        scale: float | None = None
        auto_route: bool = True
        allow_adult: bool | None = None
        width: int = 1024
        height: int = 1024
        steps: int | None = None
        guidance_scale: float | None = None
        seed: int = 0

    @app.get("/health")
    def health():
        return {"ok": True}

    @app.get("/loras")
    def list_loras():
        srv = get_server()
        return {
            "loras": [
                {
                    "id": r.id,
                    "name": r.name,
                    "base_model": r.base_model,
                    "trigger_word": r.trigger_word,
                    "keywords": r.keywords,
                    "is_adult": r.is_adult,
                }
                for r in srv.registry.all()
            ]
        }

    @app.get("/search")
    def search(q: str, top_k: int = 5):
        srv = get_server()
        return {
            "results": [
                {"id": r.record.id, "name": r.record.name, "score": r.score, "match_type": r.match_type}
                for r in srv.router.search(q, top_k=top_k)
            ]
        }

    @app.post("/generate")
    def generate(req: GenerateRequest):
        srv = get_server()
        try:
            result = srv.generate(
                req.prompt,
                lora_id=req.lora_id,
                scale=req.scale,
                auto_route=req.auto_route,
                allow_adult=req.allow_adult,
                width=req.width,
                height=req.height,
                steps=req.steps,
                guidance_scale=req.guidance_scale,
                seed=req.seed,
            )
        except KeyError as e:
            raise HTTPException(404, str(e))
        image = result.pop("image")
        buf = io.BytesIO()
        image.save(buf, format="WEBP", quality=85)
        result["image_base64"] = base64.b64encode(buf.getvalue()).decode()
        result["format"] = "webp"
        return result

    return app


app = create_app()
