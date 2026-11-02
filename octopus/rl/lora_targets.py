"""LoRA target modules from the loaded model: the leaf names of every nn.Linear in the decoder layers that the rollout
SGLang serves a LoRA on. SGLang decides with its own tables, so that check runs in its venv (OCTOPUS_SGLANG_PY): it
maps the name to its LoRA module (get_normalized_target_modules: q/k/v_proj -> qkv_proj, ...), which its LoRA layers
must handle (_KNOWN_LORA_TARGET_MODULES) and the model must size: the class's supported_lora_modules if it declares
them, else the config-driven default dims. patches/apply_sglang_lora_targets.py adds Qwen3.5's in_proj_qkv/in_proj_z
and granite's Mamba out_proj; tests/test_lora_targets.py lists what stays out and why.

The class asked about is the trained model's (Qwen3_5ForCausalLM, octopus/rl/text_checkpoint.py); SGLang serves the
vision-language checkpoint (Qwen3_5ForConditionalGeneration), whose supported_lora_modules is the text class's list
itself (sglang/srt/models/qwen3_5.py, 0.5.19), so both give one target set.

usage: resolve(model) before get_peft_model; SGLANG_PY -m octopus.rl.lora_targets ARCH CONFIG_DIR NAME...
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from torch import nn


def decoder_linears(model) -> list[str]:
    layers = model.get_decoder().layers
    return sorted({n.rsplit(".", 1)[-1] for n, m in layers.named_modules() if isinstance(m, nn.Linear)})


def served(arch: str, config, names: list[str]) -> list[str]:
    """The names whose LoRA SGLang serves on a model of class `arch` (run in SGLang's venv)."""
    from sglang.srt.lora import utils
    from sglang.srt.models.registry import ModelRegistry

    declared = getattr(ModelRegistry.resolve_model_cls([arch])[0], "supported_lora_modules", None)

    def sized(module: str) -> bool:
        if declared is not None:
            return module in declared
        try:
            return bool(utils.get_default_hidden_dim(module, config.get_text_config(), 0))
        except NotImplementedError:
            return False

    modules = {name: utils.get_normalized_target_modules([name]).pop() for name in names}
    return [n for n, m in modules.items() if m in utils._KNOWN_LORA_TARGET_MODULES and sized(m)]


def resolve(model) -> list[str]:
    """target_modules for a LoRA on this (not yet PEFT-wrapped) model."""
    with tempfile.TemporaryDirectory() as d:
        model.config.save_pretrained(d)
        out = subprocess.run([os.environ["OCTOPUS_SGLANG_PY"], "-m", "octopus.rl.lora_targets", type(model).__name__,
                              d, *decoder_linears(model)], cwd=Path(__file__).resolve().parents[2], check=True,
                             stdout=subprocess.PIPE, text=True).stdout
    return json.loads(out.splitlines()[-1])


if __name__ == "__main__":
    from transformers import AutoConfig

    print(json.dumps(served(sys.argv[1], AutoConfig.from_pretrained(sys.argv[2]), sys.argv[3:])))
