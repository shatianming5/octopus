"""A text-only Qwen3.5 checkpoint (Qwen3_5ForCausalLM) from the vision-language one: the language
model's weights under model.* plus lm_head, the text config, and the tokenizer and chat template.
The vision tower and MTP head are dropped. AReaL then trains it as a causal LM (FSDP transport padding
and meta-device loading work, which its vision-language path refuses). SGLang serves the original
vision-language checkpoint (rl_run.sh MODEL_VL): the same language-model weights, and it takes this
model's adapters (octopus/rl/lora_targets.py: one LoRA target set for both classes).

usage: python -m octopus.rl.text_checkpoint --src MODEL_DIR --out TEXT_MODEL_DIR
"""

import argparse
import json
import shutil
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file

PREFIX = "model.language_model."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    src, out = Path(args.src), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    config = json.loads((src / "config.json").read_text())
    text = {**config["text_config"], "architectures": ["Qwen3_5ForCausalLM"], "model_type": "qwen3_5_text"}
    for key in ("tie_word_embeddings", "torch_dtype", "dtype", "transformers_version"):
        if key in config and key not in text:
            text[key] = config[key]
    (out / "config.json").write_text(json.dumps(text, indent=2))
    for f in src.iterdir():  # tokenizer, chat template, generation config
        if f.is_file() and not f.name.endswith(".safetensors") and f.name not in {"config.json", "model.safetensors.index.json"}:
            shutil.copy2(f, out / f.name)
    weight_map, kept = {}, 0
    for shard in sorted(src.glob("*.safetensors")):
        tensors = {}
        with safe_open(shard, "pt") as g:
            for name in g.keys():
                if name.startswith(PREFIX):
                    tensors["model." + name[len(PREFIX):]] = g.get_tensor(name)
                elif name == "lm_head.weight":
                    tensors[name] = g.get_tensor(name)
        if tensors:
            save_file(tensors, out / shard.name, metadata={"format": "pt"})
            weight_map.update({n: shard.name for n in tensors})
            kept += len(tensors)
    (out / "model.safetensors.index.json").write_text(json.dumps({"metadata": {}, "weight_map": weight_map}, indent=2))
    print(f"{kept} tensors -> {out}")


if __name__ == "__main__":
    main()
