from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from types import MethodType

import torch

from cutecosyvoice.runtime import CosyVoicePaths, configure_cosyvoice_imports, configure_inductor_env, timed


class CosyVoiceVCModel:
    """Thin external-CosyVoice VC wrapper for CuteDSL optimization experiments."""

    def __init__(
        self,
        paths: CosyVoicePaths | None = None,
        fp16: bool = True,
        load_trt: bool = False,
        trt_concurrent: int = 1,
    ):
        configure_inductor_env()
        self.paths = paths or CosyVoicePaths()
        self.root = configure_cosyvoice_imports(self.paths.root)
        from cosyvoice.cli.cosyvoice import AutoModel

        self.cosyvoice, self.load_seconds = timed(
            lambda: AutoModel(
                model_dir=self.paths.resolved_model_dir(),
                fp16=fp16,
                load_trt=load_trt,
                trt_concurrent=trt_concurrent,
            )
        )
        self._original_decoder_forward: Callable | None = None
        self._original_hift_inference: Callable | None = None
        self._original_estimator_forward: Callable | None = None

    @property
    def sample_rate(self) -> int:
        return self.cosyvoice.sample_rate

    @property
    def flow_estimator(self) -> torch.nn.Module:
        return self.cosyvoice.model.flow.decoder.estimator

    @flow_estimator.setter
    def flow_estimator(self, estimator: torch.nn.Module) -> None:
        self.cosyvoice.model.flow.decoder.estimator = estimator

    def compile_flow_estimator(self, mode: str = "reduce-overhead", backend: str | None = None) -> torch.nn.Module:
        estimator = self.flow_estimator
        if not isinstance(estimator, torch.nn.Module):
            raise TypeError("flow decoder estimator is not a torch module")
        if backend is None:
            self.flow_estimator = torch.compile(estimator, mode=mode, fullgraph=False, dynamic=False)
        else:
            self.flow_estimator = torch.compile(estimator, backend=backend, fullgraph=False, dynamic=False)
        return self.flow_estimator

    def set_flow_steps(self, steps: int | None) -> None:
        """Override diffusion flow Euler steps; pass None to restore the model default."""
        decoder = self.cosyvoice.model.flow.decoder
        if self._original_decoder_forward is None:
            self._original_decoder_forward = decoder.forward

        if steps is None:
            decoder.forward = self._original_decoder_forward
            return

        if steps < 1:
            raise ValueError("flow steps must be >= 1")

        original_forward = self._original_decoder_forward

        def forward_with_steps(*args, **kwargs):
            kwargs["n_timesteps"] = steps
            return original_forward(*args, **kwargs)

        decoder.forward = forward_with_steps

    def set_hift_f0_device(self, device: str | None) -> None:
        """Override HiFT f0 predictor placement; pass None or 'cpu' to restore upstream behavior."""
        hift = self.cosyvoice.model.hift
        if self._original_hift_inference is None:
            self._original_hift_inference = hift.inference

        if device is None or device == "cpu":
            hift.inference = self._original_hift_inference
            return
        if device != "cuda":
            raise ValueError("hift f0 device must be 'cpu', 'cuda', or None")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")

        @torch.inference_mode()
        def inference_cuda(self_hift, speech_feat: torch.Tensor, finalize: bool = True):
            self_hift.f0_predictor.to(speech_feat.device)
            f0 = self_hift.f0_predictor(speech_feat, finalize=finalize).to(speech_feat)
            s = self_hift.f0_upsamp(f0[:, None]).transpose(1, 2)
            s, _, _ = self_hift.m_source(s)
            s = s.transpose(1, 2)
            if finalize is True:
                generated_speech = self_hift.decode(x=speech_feat, s=s, finalize=finalize)
            else:
                generated_speech = self_hift.decode(
                    x=speech_feat[:, :, :-self_hift.f0_predictor.condnet[0].causal_padding],
                    s=s,
                    finalize=finalize,
                )
            return generated_speech, s

        hift.inference = MethodType(inference_cuda, hift)

    def set_skip_alltrue_dit_mask(self, enabled: bool = True) -> None:
        estimator = self.flow_estimator
        if self._original_estimator_forward is None:
            self._original_estimator_forward = estimator.forward

        if not enabled:
            estimator.forward = self._original_estimator_forward
            return

        original_forward = self._original_estimator_forward
        if not all(hasattr(estimator, name) for name in ("time_embed", "input_embed", "rotary_embed", "transformer_blocks", "norm_out", "proj_out")):
            raise TypeError("flow estimator does not look like a CosyVoice DiT module")

        def forward_skip_mask(self_estimator, x, mask, mu, t, spks=None, cond=None, streaming=False):
            if streaming:
                return original_forward(x, mask, mu, t, spks=spks, cond=cond, streaming=streaming)

            x = x.transpose(1, 2)
            mu = mu.transpose(1, 2)
            cond = cond.transpose(1, 2)
            spks = spks.unsqueeze(dim=1)
            batch, seq_len = x.shape[0], x.shape[1]
            if t.ndim == 0:
                t = t.repeat(batch)

            t = self_estimator.time_embed(t)
            x = self_estimator.input_embed(x, cond, mu, spks.squeeze(1))
            rope = self_estimator.rotary_embed.forward_from_seq_len(seq_len)

            if self_estimator.long_skip_connection is not None:
                residual = x

            for block in self_estimator.transformer_blocks:
                x = block(x, t, mask=None, rope=rope)

            if self_estimator.long_skip_connection is not None:
                x = self_estimator.long_skip_connection(torch.cat((x, residual), dim=-1))

            x = self_estimator.norm_out(x, t)
            return self_estimator.proj_out(x).transpose(1, 2)

        estimator.forward = MethodType(forward_skip_mask, estimator)

    def frontend_vc(self, source: str | Path | None = None, prompt: str | Path | None = None) -> dict:
        source_path = str(source) if source is not None else self.paths.resolved_source()
        prompt_path = str(prompt) if prompt is not None else self.paths.resolved_prompt()
        return self.cosyvoice.frontend.frontend_vc(source_path, prompt_path, self.sample_rate)

    def token2wav(self, model_input: dict, stream: bool = False, speed: float = 1.0) -> list[dict[str, torch.Tensor]]:
        return list(self.cosyvoice.model.tts(**model_input, stream=stream, speed=speed))

    def inference_vc(
        self,
        source: str | Path | None = None,
        prompt: str | Path | None = None,
        stream: bool = False,
        speed: float = 1.0,
    ) -> list[dict[str, torch.Tensor]]:
        return self.token2wav(self.frontend_vc(source, prompt), stream=stream, speed=speed)
