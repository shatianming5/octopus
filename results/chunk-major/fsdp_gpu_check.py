"""2 ranks on GPUs (NCCL): tests/test_chunk_major.py's tiny fp64 model with LoRA, fully_shard as AReaL; ACROSS_MBS per
rank as compute_logp then ppo_update run them, per micro-batch (AReaL's) or across_micro_batches, against the unsharded
whole tree on every rank's micro-batches (README.md "FSDP once per call"). From the repository root:
  PYTHONPATH=<areal overlay>:. torchrun --nproc-per-node 2 results/chunk-major/fsdp_gpu_check.py [mixed] [nohost]
mixed: fp32 compute, fp64 reduce; nohost: chunk-major's kept tensors on the GPU."""
import contextlib
import copy
import os
import sys

import torch
import torch.distributed as dist

sys.path.insert(0, "tests")
import test_chunk_major as t  # noqa: E402
from octopus.rl import chunk_major  # noqa: E402
from octopus.rl.prefix_tree import TreeMeta, tree_head_output  # noqa: E402

rank = int(os.environ["RANK"])
torch.cuda.set_device(rank)
dist.init_process_group("nccl")
mixed, nohost = "mixed" in sys.argv, "nohost" in sys.argv
chunk_major.HOST = not nohost
from peft import LoraConfig, get_peft_model  # noqa: E402
from torch.distributed.device_mesh import init_device_mesh  # noqa: E402
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard  # noqa: E402

model = get_peft_model(t._tiny_model(setattr), LoraConfig(r=4, lora_alpha=8, lora_dropout=0.0,
                                                          target_modules=t.LORA_TARGETS))
torch.manual_seed(1)
for name, p in model.named_parameters():
    if "lora_B" in name:
        torch.nn.init.normal_(p, std=0.1)
model.cuda()
from octopus.rl import prefix_tree  # noqa: E402
import torch.nn.functional as F  # noqa: E402


def _attention_call(q, k, v, call, softmax_scale=None):  # test_chunk_major's, its mask on q's device
    _, keys, cu_q, cu_k, _, _ = call
    out = []
    for i in range(len(cu_q) - 1):
        q_ = q[cu_q[i]:cu_q[i + 1]].transpose(0, 1)
        k_, v_ = (x.index_select(0, keys[cu_k[i]:cu_k[i + 1]]).repeat_interleave(q.shape[1] // k.shape[1], 1)
                  .transpose(0, 1) for x in (k, v))
        mask = torch.ones(q_.shape[1], k_.shape[1], dtype=torch.bool, device=q.device).tril(k_.shape[1] - q_.shape[1])
        out.append(F.scaled_dot_product_attention(q_, k_, v_, attn_mask=mask, scale=softmax_scale).transpose(0, 1))
    return torch.cat(out)


prefix_tree.attention_call = _attention_call
reference = copy.deepcopy(model)
shard = dict(mesh=init_device_mesh("cuda", (2,)), mp_policy=MixedPrecisionPolicy(
    param_dtype=torch.float32, reduce_dtype=torch.float64) if mixed else MixedPrecisionPolicy())
for m in model.modules():
    if type(m).__name__ == "Qwen3_5DecoderLayer" or isinstance(m, torch.nn.Embedding):
        fully_shard(m, **shard)
fully_shard(model, **shard)
head = t._head(model)
model.forward = lambda plan: chunk_major.chunk_major_output(plan, model, head)

for mbs in t.ACROSS_MBS:
    for case, _, scale in mbs:
        tree, w = t._case(case)
        meta = TreeMeta(tree, "cuda")
        ids, pos = (torch.from_numpy(a).cuda()[None] for a in (tree.tokens, tree.position_ids))
        hidden = reference.get_decoder()(input_ids=ids, position_ids=pos, octopus_tree=meta,
                                         use_cache=False).last_hidden_state
        (tree_head_output(t._head(reference), hidden, meta) * w.cuda() * scale / 2).sum().backward()
ref = {n: p.grad for n, p in reference.named_parameters() if p.grad is not None}
mbs = [(chunk_major.ChunkPlan(t._case(c)[0], tokens, "cuda"), t._case(c)[1].cuda() * s)
       for c, tokens, s in t.ACROSS_MBS[rank]]
for mode in ("per_mb", "across", "per_mb", "across"):
    def ctx():
        return chunk_major.across_micro_batches(model, len(mbs)) if mode == "across" else contextlib.nullcontext()
    with torch.no_grad(), ctx():  # compute_logp
        for plan, _ in mbs:
            model(plan)
    chunk_major.release_host(False)
    model.zero_grad(set_to_none=True)
    with ctx():  # ppo_update
        for plan, w in mbs:
            (model(plan) * w).sum().backward()
    chunk_major.release_host(True)
    g = {n: p.grad.full_tensor() for n, p in model.named_parameters() if p.grad is not None}
    diff = max((g[k].double() - r.double()).abs().max().item() for k, r in ref.items())
    finite = all(torch.isfinite(x).all() for x in g.values())
    print(f"rank {rank} mixed={mixed} host={not nohost} {mode}: max |grad - whole tree| {diff:.2e} (largest "
          f"{max(r.abs().max().item() for r in ref.values()):.1e}), finite {finite}", flush=True)
dist.destroy_process_group()
