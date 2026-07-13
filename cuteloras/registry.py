"""LoRA registry — JSON-backed catalog of LoRA adapters with local paths or remote URLs."""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from urllib.parse import unquote

logger = logging.getLogger("cuteloras")


@dataclass
class LoRARecord:
    id: str
    name: str = ""
    base_model: str = "zimage"
    path: str | None = None
    url: str | None = None
    trigger_word: str = ""
    template: str = "{prompt}"
    keywords: list[str] = field(default_factory=list)
    negative_keywords: list[str] = field(default_factory=list)
    is_adult: bool = False
    scale: float = 1.0

    def __post_init__(self):
        if not self.name:
            self.name = self.id.replace("_", " ").strip()

    def apply_template(self, prompt: str) -> str:
        if "{prompt}" in self.template:
            return self.template.replace("{prompt}", prompt)
        return f"{self.template} {prompt}".strip()


class LoRARegistry:
    def __init__(self, records: list[LoRARecord] | None = None, cache_dir: str | None = None):
        self._records: dict[str, LoRARecord] = {}
        self.cache_dir = cache_dir or os.path.join(str(Path.home()), ".cache", "cuteloras")
        for r in records or []:
            self.add(r)

    def add(self, record: LoRARecord) -> None:
        self._records[record.id] = record

    def get(self, lora_id: str) -> LoRARecord | None:
        return self._records.get(lora_id)

    def all(self) -> list[LoRARecord]:
        return list(self._records.values())

    def __len__(self) -> int:
        return len(self._records)

    def __contains__(self, lora_id: str) -> bool:
        return lora_id in self._records

    @classmethod
    def from_json(cls, path: str, cache_dir: str | None = None) -> "LoRARegistry":
        with open(path) as f:
            data = json.load(f)
        records = [LoRARecord(**item) for item in (data if isinstance(data, list) else data["loras"])]
        return cls(records, cache_dir=cache_dir)

    def to_json(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump([asdict(r) for r in self._records.values()], f, indent=2)

    @classmethod
    def from_directory(cls, directory: str, base_model: str = "zimage", cache_dir: str | None = None) -> "LoRARegistry":
        registry = cls(cache_dir=cache_dir)
        for p in sorted(Path(directory).glob("*.safetensors")):
            lora_id = re.sub(r"[^\w]+", "_", p.stem).strip("_").lower()
            registry.add(LoRARecord(id=lora_id, base_model=base_model, path=str(p)))
        return registry

    def resolve_path(self, record: LoRARecord) -> str:
        if record.path and os.path.exists(record.path):
            return record.path
        if not record.url:
            raise FileNotFoundError(f"LoRA {record.id}: no local path and no url")
        os.makedirs(self.cache_dir, exist_ok=True)
        filename = unquote(record.url.split("/")[-1].split("?")[0]) or f"{record.id}.safetensors"
        local_path = os.path.join(self.cache_dir, filename)
        if os.path.exists(local_path):
            record.path = local_path
            return local_path
        import urllib.request

        logger.info("downloading LoRA %s from %s", record.id, record.url)
        req = urllib.request.Request(record.url, headers={"User-Agent": "Mozilla/5.0"})
        tmp_path = local_path + ".part"
        with urllib.request.urlopen(req) as resp, open(tmp_path, "wb") as f:
            while chunk := resp.read(1 << 20):
                f.write(chunk)
        os.replace(tmp_path, local_path)
        record.path = local_path
        return local_path
