"""Merge a PEFT LoRA adapter (trained on Qwen3_5ForCausalLM) into the original Qwen3.5 checkpoint, so
SGLang serves it exactly like the base model (same config, tokenizer, vision/MTP weights).

usage: python -m octopus.tools.tree_sft.merge_lora --adapter RUN/final --out DIR [--base PATH]
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


def lora_deltas(adapter: Path) -> dict[str, torch.Tensor]:
    """{checkpoint weight name: delta} with delta = B @ A * alpha / r, in float32."""
    cfg = json.loads((adapter / "adapter_config.json").read_text())
    scale = cfg["lora_alpha"] / cfg["r"]
    weights = load_file(str(adapter / "adapter_model.safetensors"))
    pairs: dict[str, dict[str, torch.Tensor]] = {}
    for key, tensor in weights.items():
        m = re.match(r"base_model\.model\.model\.(layers\.\d+\..+)\.lora_([AB])(?:\.default)?\.weight$", key)
        if not m:
            raise SystemExit(f"unexpected adapter key {key}")
        pairs.setdefault(f"model.language_model.{m.group(1)}.weight", {})[m.group(2)] = tensor.float()
    return {name: (ab["B"] @ ab["A"]) * scale for name, ab in pairs.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--base", default="/mnt/disk1/user/models/Qwen3.5-9B")
    args = ap.parse_args()
    base, out = Path(args.base), Path(args.out)
    deltas = lora_deltas(Path(args.adapter))
    out.mkdir(parents=True, exist_ok=True)
    index = json.loads((base / "model.safetensors.index.json").read_text())["weight_map"]
    missing = set(deltas) - set(index)
    if missing:
        raise SystemExit(f"{len(missing)} LoRA targets not in the checkpoint, e.g. {sorted(missing)[:3]}")
    shards = sorted(set(index.values()))
    merged = 0
    for shard in shards:
        tensors = load_file(str(base / shard))
        touched = [name for name in tensors if name in deltas]
        for name in touched:
            tensors[name] = (tensors[name].float() + deltas[name]).to(tensors[name].dtype)
        save_file(tensors, str(out / shard), metadata={"format": "pt"})
        merged += len(touched)
    for f in base.iterdir():
        if f.is_file() and not f.name.endswith(".safetensors"):
            shutil.copy2(f, out / f.name)
    (out / "MERGED_FROM.json").write_text(json.dumps({"base": str(base), "adapter": str(Path(args.adapter).resolve()), "weights_merged": merged}, indent=1))
    print(f"merged {merged} weights from {args.adapter} into {out}")


if __name__ == "__main__":
    main()
