import os

import pytest
import torch
from transformers import AutoModelForCausalLM, GraniteMoeHybridConfig, Qwen3_5TextConfig

from octopus.rl.lora_targets import decoder_linears, resolve

QWEN35 = Qwen3_5TextConfig(vocab_size=96, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                           layer_types=["linear_attention", "full_attention"], num_attention_heads=2,
                           num_key_value_heads=1, head_dim=16, linear_num_key_heads=2, linear_num_value_heads=2,
                           linear_key_head_dim=16, linear_value_head_dim=16)
GRANITE = GraniteMoeHybridConfig(vocab_size=96, hidden_size=32, intermediate_size=64, shared_intermediate_size=64,
                                 num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
                                 layer_types=["mamba", "attention"], mamba_n_heads=4, mamba_d_head=16, mamba_d_state=8)
# grpo.yaml's list before resolution: existing Qwen3.5 runs and adapters keep this exact layout
QWEN35_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "in_proj_qkv",
                  "in_proj_z", "out_proj"]


def meta_model(config):
    with torch.device("meta"):
        return AutoModelForCausalLM.from_config(config)


def test_decoder_linears():
    assert decoder_linears(meta_model(QWEN35)) == sorted(QWEN35_TARGETS + ["in_proj_a", "in_proj_b"])
    assert decoder_linears(meta_model(GRANITE)) == ["in_proj", "input_linear", "k_proj", "o_proj", "out_proj",
                                                    "output_linear", "q_proj", "v_proj"]


@pytest.mark.skipif("OCTOPUS_SGLANG_PY" not in os.environ, reason="needs the serving venv's SGLang (patched)")
def test_resolve_with_sglang():
    # Qwen3.5: not GDN in_proj_a / in_proj_b (no in_proj_ba LoRA in SGLang)
    assert resolve(meta_model(QWEN35)) == sorted(QWEN35_TARGETS)
    # granite: not Mamba in_proj (SGLang's in_proj LoRA covers 2 of its 5 partitions, Nemotron-H's layout) nor the
    # shared MLP's input_linear / output_linear (no SGLang LoRA module of that name)
    assert resolve(meta_model(GRANITE)) == ["k_proj", "o_proj", "out_proj", "q_proj", "v_proj"]
