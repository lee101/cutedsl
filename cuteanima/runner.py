"""Batched classifier-free-guidance denoise loop for Anima.

The reference pipeline runs the transformer twice per step (conditional, then
unconditional). Both calls share the latent and the timestep, so they are one
batch-2 forward instead, which keeps the GEMMs full and halves launch overhead.
Prompt embeddings are cached because production traffic reuses one negative prompt.
"""

from __future__ import annotations

from collections import OrderedDict

EMBED_CACHE_SIZE = 8


class AnimaRunner:
    def __init__(self, pipe, torch, batch_cfg: bool = True, embed_cache: int = EMBED_CACHE_SIZE):
        self.pipe = pipe
        self.torch = torch
        self.batch_cfg = batch_cfg
        self.embed_cache = embed_cache
        self._embeds: "OrderedDict[str, object]" = OrderedDict()

    def encode(self, prompt: str):
        cached = self._embeds.get(prompt)
        if cached is not None:
            self._embeds.move_to_end(prompt)
            return cached
        with self.torch.inference_mode():
            embeds = self.pipe._encode_prompt(
                [prompt], self.pipe._execution_device, self.pipe.text_encoder.dtype, 512
            )
        if self.embed_cache > 0:
            self._embeds[prompt] = embeds
            while len(self._embeds) > self.embed_cache:
                self._embeds.popitem(last=False)
        return embeds

    def __call__(
        self,
        prompt: str,
        negative_prompt: str = "",
        width: int = 832,
        height: int = 1216,
        num_inference_steps: int = 28,
        guidance_scale: float = 4.0,
        generator=None,
        output_type: str = "pil",
    ):
        torch = self.torch
        pipe = self.pipe
        device = pipe._execution_device
        do_cfg = guidance_scale > 1.0

        prompt_embeds = self.encode(prompt)
        negative_prompt_embeds = self.encode(negative_prompt or "") if do_cfg else None

        pipe.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = pipe.scheduler.timesteps

        transformer_dtype = pipe.transformer.dtype
        latents = pipe.prepare_latents(
            1, pipe.transformer.config.in_channels, height, width, 1, torch.float32, device, generator, None
        )
        padding_mask = latents.new_zeros(1, 1, height, width, dtype=transformer_dtype)
        batched = do_cfg and self.batch_cfg
        encoder_hidden_states = (
            torch.cat([prompt_embeds, negative_prompt_embeds], dim=0) if batched else prompt_embeds
        )

        with torch.inference_mode():
            for index, timestep_value in enumerate(timesteps):
                sigma = pipe.scheduler.sigmas[index]
                latent_model_input = latents.to(transformer_dtype)
                if batched:
                    velocity = pipe.transformer(
                        hidden_states=latent_model_input.repeat(2, 1, 1, 1, 1),
                        timestep=sigma.expand(2).to(transformer_dtype),
                        encoder_hidden_states=encoder_hidden_states,
                        padding_mask=padding_mask,
                        return_dict=False,
                    )[0].float()
                    velocity = velocity[1:2] + guidance_scale * (velocity[0:1] - velocity[1:2])
                else:
                    timestep = sigma.expand(1).to(transformer_dtype)
                    velocity = pipe.transformer(
                        hidden_states=latent_model_input,
                        timestep=timestep,
                        encoder_hidden_states=prompt_embeds,
                        padding_mask=padding_mask,
                        return_dict=False,
                    )[0].float()
                    if do_cfg:
                        uncond = pipe.transformer(
                            hidden_states=latent_model_input,
                            timestep=timestep,
                            encoder_hidden_states=negative_prompt_embeds,
                            padding_mask=padding_mask,
                            return_dict=False,
                        )[0].float()
                        velocity = uncond + guidance_scale * (velocity - uncond)
                latents = pipe.scheduler.step(velocity, timestep_value, latents, return_dict=False)[0]

            if output_type == "latent":
                return latents[:, :, 0]
            mean = (
                torch.tensor(pipe.vae.config.latents_mean)
                .view(1, pipe.vae.config.z_dim, 1, 1, 1)
                .to(latents.device, latents.dtype)
            )
            inv_std = 1.0 / torch.tensor(pipe.vae.config.latents_std).view(
                1, pipe.vae.config.z_dim, 1, 1, 1
            ).to(latents.device, latents.dtype)
            latents = latents / inv_std + mean
            video = pipe.vae.decode(latents.to(pipe.vae.dtype), return_dict=False)[0]
            video = pipe.video_processor.postprocess_video(video, output_type=output_type)
        return [frames[0] for frames in video][0]
