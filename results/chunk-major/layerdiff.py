"""Where chunk-major's forward departs from the whole tree's, layer by layer, on one GPU (no FSDP, no LoRA): each
decoder layer's output over the tree, against the whole-tree streamed layers at SEG: the whole tree at another SEG
(--ref-seg: other kernel-call bounds, the bf16 noise floor), chunk-major with the kept tensors on the GPU and in pinned
host memory. SEG and the chunk size are shrunk so that a small micro-batch spans many chunks.

usage: PYTHONPATH=... python results/chunk-major/layerdiff.py MB.pt MODEL_DIR [--seg 4096] [--chunk 8192]
"""

import argparse

import torch
from transformers import AutoModelForCausalLM

from octopus.rl import chunk_major, lm_head, prefix_tree
from octopus.rl.model_hooks import patch_model, tile_model
from octopus.rl.prefix_tree import TreeMeta, patch_attention, tree_head_output

import bench


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mb")
    ap.add_argument("model")
    ap.add_argument("--seg", type=int, default=4096)
    ap.add_argument("--chunk", type=int, default=8192)
    ap.add_argument("--ref-seg", type=int, default=32768)
    args = ap.parse_args()
    prefix_tree.SEG = args.seg
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16,
                                                 attn_implementation="flash_attention_2").cuda()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    patch_model(model.config)
    tile_model(model)
    patch_attention()
    tree, *_ = bench.micro_batch(args.mb)
    lm = model.get_decoder()
    bound = lm_head.bind_head(model, 1.0)
    head = lambda h, y: lm_head._chunked_rows(bound, h, y)  # noqa: E731
    outs = [[] for _ in lm.layers]
    hooks = [layer.register_forward_hook(lambda m, a, o, i=i: outs[i].append(o.detach()))
             for i, layer in enumerate(lm.layers)]
    def whole_tree(seg):
        prefix_tree.SEG = seg
        meta = TreeMeta(tree, "cuda")
        hidden = lm(input_ids=torch.from_numpy(tree.tokens)[None].cuda(),
                    position_ids=torch.from_numpy(tree.position_ids)[None].cuda(), octopus_tree=meta,
                    use_cache=False).last_hidden_state
        prefix_tree.SEG = args.seg
        return tree_head_output(head, hidden, meta), [o.pop()[0] for o in outs]

    def report(name, out, layer_outs):
        print(f"{name}: head output max |diff| {float((out - whole_out).abs().max()):.3g}, mean "
              f"{float((out - whole_out).abs().mean()):.3g}", flush=True)
        for i, layer in enumerate(lm.layers):
            d = (layer_outs[i].float() - whole[i].float()).abs()
            print(f"  layer {i:2d} {layer.block_type:16s} max |diff| {float(d.max()):9.3g}  mean {float(d.mean()):9.3g}"
                  f"  (|out| mean {float(whole[i].float().abs().mean()):.3g})", flush=True)

    with torch.no_grad():
        whole_out, whole = whole_tree(args.seg)
        report(f"whole tree, SEG {args.ref_seg} (the noise floor: other kernel-call bounds)", *whole_tree(args.ref_seg))
        for host in (False, True):
            chunk_major.HOST = host
            plan = chunk_major.ChunkPlan(tree, args.chunk, "cuda")
            out, _ = chunk_major.run_forward(plan, model, head)
            got = []
            for o in outs:
                g = torch.empty_like(whole[0])
                for ch, part in zip(plan.chunks, o):
                    g[torch.from_numpy(ch.tree_index).cuda()] = part[0]
                o.clear()
                got.append(g)
            report(f"chunk-major HOST={host}: {len(tree.tokens)} tree tokens, {len(plan.chunks)} chunks", out, got)
    for h in hooks:
        h.remove()


if __name__ == "__main__":
    main()
