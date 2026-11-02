"""Method A: flat / branch-conditioned SFT with TRL SFTTrainer + PEFT LoRA.

Every child of every tree becomes one ordinary sample (root + child; root labels, if any, only in the
join row). Labels come
pre-computed (-100 off the trained assistant spans) and TRL uses them as is. Only the loss is swapped
for the same chunked, labelled-positions-only CE the tree trainer uses (full-vocabulary logits for
20k-token samples do not fit), normalised the stock way: sum over the accumulated batch / its
labelled-token count. Multi-GPU: `torchrun --nproc_per_node K`; each optimizer step still covers
`--samples-per-step` samples in total.

usage: python -m octopus.tools.tree_sft.train_flat --data encoded.jsonl --out DIR [--rank 16 ...]
"""

from __future__ import annotations

import argparse
import json
import os

import torch

from datasets import Dataset
from peft import LoraConfig
from transformers import AutoTokenizer
from trl import SFTConfig, SFTTrainer

from octopus.tools.tree_sft.common import load_base, load_trees, offload
from octopus.tools.tree_sft.encode import flat_rows
from octopus.tools.tree_sft.tree_forward import flat_loss


class LabelledCETrainer(SFTTrainer):
    offload_activations = False
    skipped_oom = 0

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # The labelled CE becomes the model's forward, so under DDP every step still goes through the
        # wrapper's forward (which arms its gradient all-reduce).
        causal = self.model.base_model.model

        def labelled_forward(input_ids, labels, **_):
            with offload(self.offload_activations):
                return flat_loss(causal, input_ids[0], labels[0])

        self.model.forward = labelled_forward

    def training_step(self, model, inputs, num_items_in_batch=None):
        try:
            return super().training_step(model, inputs, num_items_in_batch)
        except torch.cuda.OutOfMemoryError:
            # a sample too long for this card: drop it (and partial gradients of this window), keep training
            self.skipped_oom += 1
            model.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            print(json.dumps({"skipped_oom": self.skipped_oom, "tokens": int(inputs["input_ids"].shape[-1])}), flush=True)
            # a zero-weighted two-token step keeps this rank in step with the others' gradient all-reduce
            ids = inputs["input_ids"][:, :2]
            loss_sum, _ = model(input_ids=ids, labels=torch.stack([torch.full_like(ids[:, 0], -100), ids[:, 1]], 1))
            self.accelerator.backward(loss_sum * 0)
            return torch.zeros((), device=self.args.device)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        loss_sum, n = model(input_ids=inputs["input_ids"], labels=inputs["labels"])
        denom = num_items_in_batch if num_items_in_batch is not None else max(n, 1)
        loss = loss_sum / denom
        if self.args.average_tokens_across_devices and num_items_in_batch is not None:
            # denom counts every rank's tokens and DDP averages gradients over ranks: undo that mean
            # (what the stock compute_loss does), so K GPUs give the single-GPU gradient
            loss = loss * self.accelerator.num_processes
        return (loss, None) if return_outputs else loss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="/mnt/disk1/user/models/Qwen3.5-9B")
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--samples-per-step", type=int, default=24)
    ap.add_argument("--warmup", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-trees", type=int, default=0)
    ap.add_argument("--offload-activations", action="store_true", help="keep checkpointed layer inputs in pinned CPU memory (long samples)")
    args = ap.parse_args()
    world = int(os.environ.get("WORLD_SIZE", "1"))
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    if args.samples_per_step % world:
        raise SystemExit(f"--samples-per-step {args.samples_per_step} is not divisible by {world} GPUs")

    rows = [r for t in load_trees(args.data)[: args.max_trees or None] for r in flat_rows(t)]
    dataset = Dataset.from_list([{"input_ids": r["input_ids"], "labels": r["labels"]} for r in rows])
    config = SFTConfig(
        output_dir=args.out,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=args.samples_per_step // world,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup,
        max_grad_norm=1.0,
        bf16=True,
        gradient_checkpointing=True,
        max_length=None,
        logging_steps=1,
        save_strategy="epoch",
        report_to="none",
        seed=args.seed,
    )
    trainer = LabelledCETrainer(
        model=load_base(args.model),
        args=config,
        processing_class=AutoTokenizer.from_pretrained(args.model),  # text only: AutoProcessor would want vision deps
        train_dataset=dataset,
        peft_config=LoraConfig(
            r=args.rank, lora_alpha=args.alpha, lora_dropout=0.05, target_modules="all-linear", task_type="CAUSAL_LM",
        ),
    )
    trainer.offload_activations = args.offload_activations
    trainer.train()
    trainer.save_model(f"{args.out}/final")


if __name__ == "__main__":
    main()
