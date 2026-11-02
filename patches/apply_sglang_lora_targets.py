"""Let SGLang 0.5.19 serve a PEFT LoRA on more of a model's linear layers (octopus/rl/lora_targets.py reads what
SGLang serves from SGLang itself, so these edits are what add the modules to the resolved targets):

Qwen3.5's GDN input projections (in_proj_qkv + in_proj_z). SGLang fuses them into in_proj_qkvz and stacks that
LoRA as four A slots (q, k, v, z) and one full-height B, but only accepts adapters with split in_proj_q/k/v/z or a
merged in_proj_qkvz; PEFT on the HF model gives in_proj_qkv and in_proj_z (startup fails with "get_hidden_dim not
implemented for in_proj_z"). Exact conversion: A = [A_qkv, A_qkv, A_qkv, A_z], B = [B_qkv; B_z].

granite-4.0-h's Mamba-2 out_proj (granitemoehybrid): a plain row-parallel linear SGLang's LoRA layers handle, but the
model class gives no LoRA buffer dims for it (startup fails with "get_hidden_dim not implemented for out_proj").

octopus/tools/lora_parity.py checks the result against the trainer.

usage: python3 patches/apply_sglang_lora_targets.py /path/to/site-packages/sglang   (idempotent)
"""

import runpy
import sys
from pathlib import Path

root = Path(sys.argv[1]) / "srt/lora"
models = Path(sys.argv[1]) / "srt/models"
EDITS = {
    root / "utils.py": [(
        '        "out_proj": "out_proj",\n',
        '        "out_proj": "out_proj",\n'
        '        "in_proj_qkv": "in_proj_qkvz",  # Qwen3.5 GDN (PEFT on the HF model)\n'
        '        "in_proj_z": "in_proj_qkvz",\n',
    )],
    root / "lora.py": [(
        '            elif "in_proj_qkvz" in weight_name and "lora_A" in weight_name:\n',
        '            elif "in_proj_qkv." in weight_name:\n'
        '                # Qwen3.5 PEFT layout in_proj_qkv + in_proj_z: q, k, v share the qkv A\n'
        '                z_name = weight_name.replace("in_proj_qkv.", "in_proj_z.")\n'
        '                if z_name not in weights:\n'
        '                    continue\n'
        '                qkv, z = weights.pop(weight_name), weights.pop(z_name)\n'
        '                parts = (qkv, qkv, qkv, z) if "lora_A" in weight_name else (qkv, z)\n'
        '                weights[weight_name.replace("in_proj_qkv.", "in_proj_qkvz.")] = torch.cat(parts, qkv.dim() - 2)\n'
        '            elif "in_proj_qkvz" in weight_name and "lora_A" in weight_name:\n',
    )],
    models / "granitemoehybrid.py": [(
        '        "input_linear": ["input_linear"],\n'
        '    }\n',
        '        "input_linear": ["input_linear"],\n'
        '    }\n'
        '    # LoRA: attention and the Mamba-2 out_proj. Not Mamba in_proj: SGLang\'s in_proj LoRA is Nemotron-H\'s layout\n'
        '    # (gate + x, 2 of the 5 partitions), not a LoRA on the whole projection; nor the shared MLP\'s\n'
        '    # input_linear/output_linear (no LoRA module names in SGLang)\n'
        '    supported_lora_modules = ["qkv_proj", "o_proj", "out_proj"]\n'
        '\n'
        '    def get_hidden_dim(self, module_name, layer_idx):\n'
        '        if module_name == "out_proj":  # Mamba-2 mixer: heads x head dim -> hidden\n'
        '            return self.config.mamba_n_heads * self.config.mamba_d_head, self.config.hidden_size\n'
        '        from sglang.srt.lora.utils import get_default_hidden_dim\n'
        '\n'
        '        return get_default_hidden_dim(module_name, self.config, layer_idx)\n',
    )],
}
found = runpy.run_path(sys.argv[1] + "/_version.py")["__version__"]
if found != "0.5.19":
    sys.exit(f"built for SGLang 0.5.19, found {found}; re-align these hooks: {', '.join(map(str, EDITS))}")
for path, edits in EDITS.items():
    text = path.read_text()
    for old, new in edits:
        if new in text:
            continue
        assert text.count(old) == 1, f"{path}: anchor not found once: {old!r}"
        text = text.replace(old, new)
    path.write_text(text)
    print(f"patched {path}")
