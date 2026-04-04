"""Train a Z-Image ControlNet on conditioned style-transfer pairs."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from zimagecontrol.dataset import ZImageControlDataset, collate_controlnet_examples
from zimagecontrol.runtime import (
    build_zimage_controlnet,
    calculate_shift,
    encode_vae_image,
    freeze_module,
    iter_trainable_controlnet_parameters,
    parse_dtype,
)


def _precompute_latents(pipe, dataset, device, dtype):
    """Encode all images/prompts on GPU, then return CPU tensors so we can free VAE/text_encoder."""
    pipe.vae.to(device=device, dtype=dtype)
    pipe.text_encoder.to(device=device)
    freeze_module(pipe.vae)
    freeze_module(pipe.text_encoder)

    cached = []
    for idx in range(len(dataset)):
        item = dataset[idx]
        pv = item["pixel_values"].unsqueeze(0).to(device=device, dtype=dtype)
        cv = item["control_values"].unsqueeze(0).to(device=device, dtype=dtype)
        with torch.no_grad():
            latents = encode_vae_image(pipe.vae, pv, sample_mode="sample").cpu()
            control_latents = encode_vae_image(pipe.vae, cv, sample_mode="argmax").cpu()
            prompt_embeds, _ = pipe.encode_prompt(
                prompt=[item["prompt"]],
                device=device,
                do_classifier_free_guidance=False,
                max_sequence_length=512,
            )
            if isinstance(prompt_embeds, list):
                prompt_embeds = torch.cat(prompt_embeds, dim=0)
            prompt_embeds = prompt_embeds.cpu()
        cached.append({
            "latents": latents.squeeze(0),
            "control_latents": control_latents.squeeze(0),
            "prompt_embeds": prompt_embeds.squeeze(0),
        })
        if (idx + 1) % 10 == 0:
            print(f"pre-encoded {idx + 1}/{len(dataset)}")

    pipe.vae.to("cpu")
    pipe.text_encoder.to("cpu")
    del pipe.vae, pipe.text_encoder
    gc.collect()
    torch.cuda.empty_cache()
    print(f"pre-encoded {len(cached)} samples, freed VAE+text_encoder")
    return cached


class CachedLatentDataset(torch.utils.data.Dataset):
    def __init__(self, cached):
        self.cached = cached

    def __len__(self):
        return len(self.cached)

    def __getitem__(self, idx):
        return self.cached[idx]


def _collate_cached(examples):
    return {
        "latents": torch.stack([e["latents"] for e in examples]),
        "control_latents": torch.stack([e["control_latents"] for e in examples]),
        "prompt_embeds": torch.stack([e["prompt_embeds"] for e in examples]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a Z-Image ControlNet")
    parser.add_argument("--model-id", default="Tongyi-MAI/Z-Image-Turbo")
    parser.add_argument("--metadata-path", required=True)
    parser.add_argument("--output-dir", default="zimagecontrol/checkpoints/default")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-train-steps", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--save-every", type=int, default=250)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--control-layers", default="all")
    parser.add_argument("--control-refiner-layers", default="all")
    parser.add_argument("--add-control-noise-refiner", default=None)
    parser.add_argument("--sparse-control-prob", type=float, default=0.7)
    parser.add_argument("--conditioning-type", default="line", choices=["line", "canny"])
    parser.add_argument("--control-mode", default="sparse_or_full", choices=["full", "sparse", "sparse_or_full"])
    parser.add_argument("--conditioning-scale", type=float, default=1.0)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device)
    dtype = parse_dtype(args.dtype)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    from diffusers import ZImagePipeline

    pipe = ZImagePipeline.from_pretrained(args.model_id, torch_dtype=dtype)

    # pre-encode dataset on GPU then free VAE+text_encoder
    image_dataset = ZImageControlDataset(
        args.metadata_path,
        conditioning_type=args.conditioning_type,
        control_mode=args.control_mode,
        sparse_control_prob=args.sparse_control_prob,
        regenerate_sparse=True,
    )
    cached = _precompute_latents(pipe, image_dataset, device, dtype)

    # compute scheduler params from image dimensions
    image_height = image_dataset.records[0].height
    image_width = image_dataset.records[0].width
    if image_height is None or image_width is None:
        sample = image_dataset[0]["pixel_values"]
        image_height, image_width = int(sample.shape[-2]), int(sample.shape[-1])
    image_seq_len = (image_height // 2) * (image_width // 2)
    mu = calculate_shift(
        image_seq_len,
        pipe.scheduler.config.get("base_image_seq_len", 256),
        pipe.scheduler.config.get("max_image_seq_len", 4096),
        pipe.scheduler.config.get("base_shift", 0.5),
        pipe.scheduler.config.get("max_shift", 1.15),
    )
    pipe.scheduler.sigma_min = 0.0
    pipe.scheduler.set_timesteps(pipe.scheduler.config.num_train_timesteps, device=device, mu=mu)
    train_timesteps = pipe.scheduler.timesteps[:-1]
    patch_size = pipe.transformer.config.all_patch_size[0]
    f_patch_size = pipe.transformer.config.all_f_patch_size[0]

    # now move transformer to GPU and build controlnet
    pipe.transformer.to(device=device, dtype=dtype)
    freeze_module(pipe.transformer)
    if args.gradient_checkpointing and hasattr(pipe.transformer, 'enable_gradient_checkpointing'):
        try:
            pipe.transformer.enable_gradient_checkpointing()
        except (ValueError, AttributeError):
            print("gradient checkpointing not supported by transformer, skipping")

    controlnet = build_zimage_controlnet(
        pipe.transformer,
        control_layers_spec=args.control_layers,
        control_refiner_layers_spec=args.control_refiner_layers,
        add_control_noise_refiner=args.add_control_noise_refiner,
    )
    controlnet.to(device=device, dtype=dtype)
    controlnet.gradient_checkpointing = False
    controlnet.train()

    trainable_params = list(iter_trainable_controlnet_parameters(controlnet))
    if not trainable_params:
        raise ValueError("No trainable ControlNet parameters selected.")
    print(f"trainable params: {sum(p.numel() for p in trainable_params):,}")

    optimizer = torch.optim.AdamW(trainable_params, lr=args.learning_rate, weight_decay=args.weight_decay)
    loader = DataLoader(
        CachedLatentDataset(cached),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=_collate_cached,
    )

    control_in_dim = controlnet.config.control_in_dim
    scaler_enabled = device.type == "cuda" and dtype in {torch.float16, torch.bfloat16}
    autocast_dtype = dtype if scaler_enabled else torch.float32

    loss_log = output_dir / "loss.jsonl"
    global_step = 0
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(args.epochs):
        for batch in loader:
            if global_step >= args.max_train_steps:
                break

            latents = batch["latents"].to(device=device, dtype=dtype)
            control_latents = batch["control_latents"].to(device=device, dtype=dtype)
            prompt_embeds = batch["prompt_embeds"].to(device=device, dtype=dtype)

            with torch.no_grad():
                if control_latents.shape[1] != control_in_dim:
                    pad_channels = control_in_dim - control_latents.shape[1]
                    zeros = torch.zeros(
                        control_latents.shape[0], pad_channels,
                        *control_latents.shape[2:],
                        device=device, dtype=dtype,
                    )
                    control_latents = torch.cat([control_latents, zeros], dim=1)

                noise = torch.randn_like(latents)
                timestep_indices = torch.randint(0, len(train_timesteps), (latents.shape[0],), device=device)
                timesteps = train_timesteps[timestep_indices]
                noised_latents = pipe.scheduler.scale_noise(latents.float(), timesteps, noise.float()).to(dtype)
                target = (noise - latents.float()).to(torch.float32)

            timestep_input = ((1000.0 - timesteps.float()) / 1000.0).to(device=device)
            latent_list = list(noised_latents.unsqueeze(2).unbind(dim=0))
            control_list = list(control_latents.unsqueeze(2).unbind(dim=0))

            with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=scaler_enabled):
                controlnet_block_samples = controlnet(
                    x=latent_list,
                    t=timestep_input,
                    cap_feats=prompt_embeds,
                    control_context=control_list,
                    conditioning_scale=args.conditioning_scale,
                    patch_size=patch_size,
                    f_patch_size=f_patch_size,
                )
                model_out_list = pipe.transformer(
                    latent_list,
                    timestep_input,
                    prompt_embeds,
                    return_dict=False,
                    controlnet_block_samples=controlnet_block_samples,
                    patch_size=patch_size,
                    f_patch_size=f_patch_size,
                )[0]
                prediction = torch.stack([tensor.float() for tensor in model_out_list], dim=0).squeeze(2)
                loss = F.mse_loss(prediction, target) / args.gradient_accumulation_steps

            loss.backward()

            if (global_step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, args.grad_clip)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            global_step += 1
            log_row = {"step": global_step, "epoch": epoch + 1, "loss": float(loss.item() * args.gradient_accumulation_steps)}
            with loss_log.open("a") as handle:
                handle.write(json.dumps(log_row) + "\n")
            print(json.dumps(log_row))

            if global_step % args.save_every == 0:
                checkpoint_dir = output_dir / f"checkpoint-{global_step:06d}"
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                controlnet.save_pretrained(checkpoint_dir)

        if global_step >= args.max_train_steps:
            break

    controlnet.save_pretrained(output_dir / "final")
    print(json.dumps({
        "output_dir": str(output_dir),
        "final_checkpoint": str(output_dir / "final"),
        "steps": global_step,
    }, indent=2))


if __name__ == "__main__":
    main()
