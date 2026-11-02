"""Shared loading for the tree SFT scripts."""

from __future__ import annotations

import contextlib
import json

import torch


def load_base(model_path: str, dtype=torch.bfloat16):
    """Text-only Qwen3.5 (the checkpoint's vision tower is skipped) with SDPA, on cuda (bf16 by default)."""
    from transformers import AutoConfig, Qwen3_5ForCausalLM

    config = AutoConfig.from_pretrained(model_path)
    text_config = getattr(config, "text_config", config)
    model = Qwen3_5ForCausalLM.from_pretrained(
        model_path, config=text_config, dtype=dtype, attn_implementation="sdpa",
    )
    model.config.use_cache = False
    return model.cuda()


def load_trees(path: str) -> list[dict]:
    return [json.loads(line) for line in open(path) if line.strip()]


def offload(enabled: bool):
    """Saved activations (with per-layer checkpointing: each layer's inputs) go to pinned host memory.
    Same numbers, slower; lets 80k-token trees fit a 48 GB card."""
    return torch.autograd.graph.save_on_cpu(pin_memory=True) if enabled else contextlib.nullcontext()
