"""LoRA adapter of the last training step from AReaL's recover checkpoint (the saver skips a last step that is not a
multiple of saver.freq_steps): the dcp.model.*lora*.default.weight tensors of payloads/default, renamed as PEFT saves
them, checked against an earlier saved adapter (same keys and shapes), with its adapter_config.json.
scripts/eval/pool.py runs it on node-94 (copy at /home/user/octopus-sft/dcp_adapter.py).
usage: python dcp_adapter.py PAYLOADS_DIR LIKE_ADAPTER_DIR OUT_DIR"""
import shutil, sys
from pathlib import Path
import torch
import torch.distributed.checkpoint as dcp
from safetensors import safe_open
from safetensors.torch import save_file

src, like, out = map(Path, sys.argv[1:4])
meta = dcp.FileSystemReader(str(src)).read_metadata().state_dict_metadata
keys = [k for k in meta if k.startswith("dcp.model.") and ".lora_" in k and k.endswith(".default.weight")]
sd = {k: torch.empty(meta[k].size, dtype=meta[k].properties.dtype) for k in keys}
dcp.load(sd, storage_reader=dcp.FileSystemReader(str(src)), no_dist=True)
new = {k[len("dcp.model."):].replace(".default.weight", ".weight"): v.contiguous() for k, v in sd.items()}
with safe_open(str(like / "adapter_model.safetensors"), "pt") as f:
    ref = {k: f.get_tensor(k) for k in f.keys()}
assert set(new) == set(ref), (len(new), len(ref), sorted(set(new) ^ set(ref))[:4])
assert all(new[k].shape == ref[k].shape for k in ref)
changed = sum(not torch.equal(new[k].to(ref[k].dtype), ref[k]) for k in ref)
out.mkdir(parents=True, exist_ok=True)
save_file({k: v.to(ref[k].dtype) for k, v in new.items()}, str(out / "adapter_model.safetensors"))
shutil.copy(like / "adapter_config.json", out / "adapter_config.json")
print(f"{len(new)} tensors -> {out}; {changed} differ from {like.name}")
