"""Do the rollout server (SGLang) and the trainer score the same tokens alike, with a LoRA on every target module?

Scores each sequence's tokens with SGLang (prompt logprobs) and with the trainer's model (HF + the GDN varlen patch,
bf16 base; the LoRA in fp32 here, PeftModel.from_pretrained's default autocast_adapter_dtype, while the trainer stores
and computes it in bf16: octopus/rl/grpo.yaml optimizer_dtype), with and without the adapter. Use a random adapter
(--randomize: B != 0) so a module SGLang silently skips shows up: the LoRA gap then grows to the size of the LoRA
effect instead of staying near the base gap (bf16 / kernel noise floor).

usage: python -m octopus.tools.lora_parity --randomize ZERO_ADAPTER_DIR --out RANDOM_ADAPTER_DIR [--std 0.02]
       python -m octopus.tools.lora_parity --model TEXT_MODEL_DIR --adapter ADAPTER_DIR --server 127.0.0.1:PORT
              --lora NAME [--dump LOGS/<trial>/rollout/0] [--n-seqs 4] [--max-len 4000]
(the server runs with --enable-lora --lora-paths NAME=ADAPTER_DIR --lora-strict-loading)
"""

from __future__ import annotations

import argparse
import json
import shutil
import urllib.request
from pathlib import Path

import numpy as np
import torch


def randomize(src: str, out: str, std: float) -> None:
    from safetensors.torch import load_file, save_file
    shutil.copytree(src, out, dirs_exist_ok=True)
    tensors = load_file(f"{src}/adapter_model.safetensors")
    g = torch.Generator().manual_seed(0)
    for k, v in tensors.items():
        if "lora_B" in k:
            tensors[k] = (torch.randn(v.shape, generator=g) * std).to(v.dtype)
    save_file(tensors, f"{out}/adapter_model.safetensors", metadata={"format": "pt"})
    print(f"{sum('lora_B' in k for k in tensors)} lora_B tensors ~ N(0, {std}) -> {out}")


def sglang_logprobs(server: str, ids: list[int], lora: str | None) -> np.ndarray:
    body = {"input_ids": ids, "sampling_params": {"max_new_tokens": 1, "temperature": 0},
            "return_logprob": True, "logprob_start_len": 0}
    if lora:
        body["lora_path"] = lora
    req = urllib.request.Request(f"http://{server}/generate", json.dumps(body).encode(), {"Content-Type": "application/json"})
    meta = json.load(urllib.request.urlopen(req, timeout=600))["meta_info"]
    return np.array([lp for lp, _, _ in meta["input_token_logprobs"][1:]], dtype=np.float64)


@torch.no_grad()
def hf_logprobs(model, ids: list[int]) -> np.ndarray:
    x = torch.tensor([ids]).cuda()
    logits = model(input_ids=x, use_cache=False).logits[0, :-1].float()
    return torch.log_softmax(logits, -1).gather(-1, x[0, 1:, None])[:, 0].double().cpu().numpy()


def gap(a: np.ndarray, b: np.ndarray) -> str:
    d = np.abs(a - b)
    return f"mean|d| {d.mean():.4f}  p99 {np.quantile(d, 0.99):.4f}  max {d.max():.3f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--randomize")
    ap.add_argument("--out")
    ap.add_argument("--std", type=float, default=0.02)
    ap.add_argument("--model")
    ap.add_argument("--adapter")
    ap.add_argument("--server")
    ap.add_argument("--lora")
    ap.add_argument("--dump")
    ap.add_argument("--n-seqs", type=int, default=4)
    ap.add_argument("--max-len", type=int, default=4000)
    args = ap.parse_args()
    if args.randomize:
        return randomize(args.randomize, args.out, args.std)

    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from octopus.rl.model_hooks import patch_model, tile_model
    from octopus.tools.tree_check import group_sequences, random_sequences

    tok = AutoTokenizer.from_pretrained(args.model)
    seqs = (group_sequences(args.dump, tok, args.n_seqs, args.max_len) if args.dump
            else random_sequences(args.n_seqs, args.max_len, len(tok)))
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, attn_implementation="flash_attention_2").cuda()
    model = PeftModel.from_pretrained(model, args.adapter).eval()
    patch_model(model.config)  # the trainer's GDN kernels and tiled layers
    tile_model(model)
    cat = lambda xs: np.concatenate(xs)  # noqa: E731
    hf_lora = cat([hf_logprobs(model, s) for s in seqs])
    with model.disable_adapter():
        hf_base = cat([hf_logprobs(model, s) for s in seqs])
    sg_base = cat([sglang_logprobs(args.server, s, None) for s in seqs])
    sg_lora = cat([sglang_logprobs(args.server, s, args.lora) for s in seqs])
    print(f"{len(seqs)} sequences, {len(hf_lora)} scored tokens")
    print(f"base   SGLang vs trainer: {gap(sg_base, hf_base)}   (noise floor)")
    print(f"LoRA   SGLang vs trainer: {gap(sg_lora, hf_lora)}")
    print(f"effect trainer LoRA vs base: {gap(hf_lora, hf_base)}   (what a skipped module would cost)")
    print(f"effect SGLang vs trainer: {gap(sg_lora - sg_base, hf_lora - hf_base)}   (the LoRA alone, any base gap removed)")
    floor, lora, effect = (np.abs(a - b).mean() for a, b in ((sg_base, hf_base), (sg_lora, hf_lora), (hf_lora, hf_base)))
    ok = lora <= 2 * floor + 0.1 * effect
    print("PARITY OK" if ok else "PARITY FAIL: SGLang does not apply this LoRA as the trainer does")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
