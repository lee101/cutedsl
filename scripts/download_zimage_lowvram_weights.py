"""Download low-VRAM Z-Image weights with fuzzy filename matching."""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import hf_hub_download, list_repo_files


REPOS = {
    "diffusion": "leejet/Z-Image-Turbo-GGUF",
    "llm": "unsloth/Qwen3-4B-Instruct-2507-GGUF",
    "vae": "black-forest-labs/FLUX.1-schnell",
}


def pick_file(repo_id: str, pattern: str) -> str:
    files = list_repo_files(repo_id)
    matches = [path for path in files if pattern.lower() in path.lower()]
    if not matches:
        raise SystemExit(f"no file in {repo_id} matched pattern: {pattern}")
    matches.sort(key=len)
    return matches[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Download low-VRAM Z-Image weights")
    parser.add_argument("--out-dir", default="downloads/models/zimage")
    parser.add_argument("--diffusion-pattern", default="Q3_K.gguf")
    parser.add_argument("--llm-pattern", default="Q4_K_M.gguf")
    parser.add_argument("--vae-pattern", default="ae.safetensors")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    selections = {
        "diffusion": pick_file(REPOS["diffusion"], args.diffusion_pattern),
        "llm": pick_file(REPOS["llm"], args.llm_pattern),
        "vae": pick_file(REPOS["vae"], args.vae_pattern),
    }

    for key, repo_id in REPOS.items():
        downloaded = hf_hub_download(
            repo_id=repo_id,
            filename=selections[key],
            local_dir=out_dir,
            local_dir_use_symlinks=False,
        )
        print(f"{key}: {downloaded}")


if __name__ == "__main__":
    main()
