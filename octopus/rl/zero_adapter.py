"""The step-0 LoRA adapter the rollout servers start with (<lora_name>-v0): the actor's LoRA layout
with B = 0, so it serves exactly the base model until AReaL loads the first trained version.

usage: OCTOPUS_SGLANG_PY=SGLANG_VENV/bin/python python -m octopus.rl.zero_adapter --model MODEL_DIR --out ADAPTER_DIR
       [--rank 32 --alpha 32]
"""

import argparse

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM

from octopus.rl.lora_targets import resolve


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--alpha", type=int, default=32)
    args = ap.parse_args()
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16)  # the text-only checkpoint AReaL trains
    targets = resolve(model)  # every decoder linear SGLang serves LoRA on, as the actor's default
    peft = get_peft_model(model, LoraConfig(r=args.rank, lora_alpha=args.alpha, target_modules=targets, lora_dropout=0.0))
    peft.save_pretrained(args.out)
    names = [n for n, _ in peft.named_parameters() if "lora_" in n]
    print(f"targets {targets}; {len(names)} LoRA tensors, vision ones: {sum('visual' in n for n in names)}; e.g. {names[0]}")


if __name__ == "__main__":
    main()
