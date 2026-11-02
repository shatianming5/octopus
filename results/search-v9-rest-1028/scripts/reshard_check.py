"""Does a recover checkpoint saved at one FSDP degree load at another? The real actor (octopus/rl/actor.py on AReaL's
FSDP engine, as results/chunk-major/actor_ab.py builds it) loads PAYLOADS (a generation's payloads/default, torch DCP)
with its optimizer, as AReaL's recover does (engine.load, dcp, with_optim), on this torchrun's N ranks. Then every LoRA
weight and every Adam state tensor (step, exp_avg, exp_avg_sq), gathered from the ranks, must equal the checkpoint's own
full tensor (read without a process group), the LoRA weights must equal ADAPTER (the step's weight_update_vN, PEFT
names; bf16 there), and the Adam moments must be non-zero. One JSON line; exit 1 on a mismatch. README "node-134 (10-30)".
usage: torchrun --nproc-per-node N reshard_check.py MODEL PAYLOADS ADAPTER [key=value config overrides]
(PYTHONPATH = the patched AReaL overlay, then this code)"""
import json, os, sys, time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.metadata import TensorStorageMetadata
from torch.distributed.checkpoint.state_dict import get_state_dict


def make_actor(model, overrides):  # results/chunk-major/actor_ab.py's
    from omegaconf import OmegaConf
    from areal.api.cli_args import to_structured_cfg
    from areal.api.io_struct import FinetuneSpec
    from octopus.rl.actor import ChunkedHeadFSDPPPOActor
    from octopus.rl.config import RQ1RLConfig
    cfg = OmegaConf.merge(OmegaConf.load(Path(__file__).parents[3] / "octopus/rl/grpo.yaml"),
                          OmegaConf.from_dotlist(overrides))
    cfg.actor.path, cfg.actor.backend = model, f"fsdp:d{int(os.environ['WORLD_SIZE'])}"
    a = ChunkedHeadFSDPPPOActor(OmegaConf.to_object(to_structured_cfg(cfg, RQ1RLConfig)).actor)
    a.create_process_group()
    a.initialize(None, FinetuneSpec(total_train_epochs=1, dataset_size=1, train_batch_size=1))
    return a


def log(m):  # phase times: the load is the slow part
    print(f"{time.strftime('%T')} rank {os.environ['RANK']}: {m}", file=sys.stderr, flush=True)


def full(t):
    return (t.full_tensor() if hasattr(t, "full_tensor") else t).detach().float().cpu()


model, payloads, adapter, *overrides = sys.argv[1:]
torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
from areal.api.io_struct import SaveLoadMeta
a = make_actor(model, overrides)
log("actor built")
a.load(SaveLoadMeta(path=payloads, weight_format="dcp", with_optim=True, tokenizer=None, processor=None))
log("checkpoint loaded")
msd, osd = get_state_dict(a.model, a.optimizer)
lora = {k: full(v) for k, v in msd.items() if ".lora_" in k}  # every rank gathers, in the same order
optim = {f"{fqn}.{s}": full(v) for fqn, st in osd["state"].items() for s, v in st.items()}
log(f"gathered {len(lora)} LoRA and {len(optim)} Adam tensors")
if dist.get_rank() == 0:
    meta = dcp.FileSystemReader(payloads).read_metadata().state_dict_metadata
    want = {**{"dcp.model." + k: v for k, v in lora.items()}, **{"dcp.optim.state." + k: v for k, v in optim.items()}}
    missing = [k for k in want if not isinstance(meta.get(k), TensorStorageMetadata)]
    ref = {k: torch.empty(meta[k].size, dtype=meta[k].properties.dtype) for k in want if k not in missing}
    dcp.load(ref, storage_reader=dcp.FileSystemReader(payloads), no_dist=True)
    log("reference read")
    differ = [k for k in ref if not torch.equal(ref[k].float(), want[k])]
    from safetensors.torch import load_file
    peft = load_file(str(Path(adapter) / "adapter_model.safetensors"))
    names = {k: k.replace(".default.weight", ".weight") for k in lora}
    vs_adapter = [k for k in lora if names[k] not in peft or not torch.equal(lora[k].to(peft[names[k]].dtype),
                                                                             peft[names[k]])]
    steps = sorted({float(v) for k, v in optim.items() if k.endswith(".step")})
    zero = [k for k, v in optim.items() if k.endswith(("exp_avg", "exp_avg_sq")) and not v.abs().sum()]
    out = dict(world_size=dist.get_world_size(), lora_tensors=len(lora), adapter_tensors=len(peft),
               optim_tensors=len(optim), missing_in_ckpt=missing[:3], n_missing=len(missing), differ=differ[:3],
               n_differ=len(differ), lora_vs_adapter_differ=vs_adapter[:3], n_lora_vs_adapter=len(vs_adapter),
               adam_steps=steps, zero_moments=len(zero),
               exp_avg_sq_norm=sum(v.norm() ** 2 for k, v in optim.items() if k.endswith("exp_avg_sq")).sqrt().item())
    print(json.dumps(out), flush=True)
    ok = lora and optim and not missing and not differ and not vs_adapter and not zero and len(peft) == len(lora)
dist.barrier()
sys.exit(0 if dist.get_rank() or ok else 1)
