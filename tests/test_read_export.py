"""Reads through AReaL's export to the actor's micro-batches (docs/cross-branch-read.md "Export").

The CPU test runs the proxy (patches/areal-cursor.patch) on tests/test_areal_cursor_hop.py's fake SGLang where the patched
AReaL is importable (PYTHONPATH=<areal overlay>:<this repo>) and skips elsewhere. The live test reruns that file's
forced read episode (OCTOPUS_SGLANG_URL, one mount-capable SGLang; a free GPU for the offline forwards), exports it, and
runs the actor's micro-batch tree forward and backward (no optimizer step) against the rollout's log-probs.
"""

import asyncio
import json
import os
import statistics

import pytest
import torch

pytest.importorskip("areal.infra.remote_inf_engine")
import test_areal_cursor_hop as hop  # noqa: E402  (not `from ... import test_*`: pytest would collect it here too)

from octopus.rl.prefix_tree import MOUNT_KEYS, node_id  # noqa: E402


def _batch(session):
    """The session's concat leaves as the trainer gets them: the proxy's export and serialization, the workflow's
    deserialization, workflow's to_tensor_dict and AReaL's concat into one padded batch."""
    import areal.infra.workflow_executor as workflow_executor
    from areal.experimental.openai.proxy.server import deserialize_interactions, serialize_interactions

    import octopus.rl.workflow  # noqa: F401  (its to_tensor_dict and concat)

    leaves = session.export_interactions(discount=1.0, style="concat", drop_retry_orphans=True)
    batch = workflow_executor.concat_tensor_interactions(deserialize_interactions(serialize_interactions(leaves)))
    return batch, leaves


def _row(batch, leaf_id):
    return next(i for i in range(len(batch["input_ids"])) if int(batch["octopus_node"][i, 0]) == node_id(leaf_id))


def test_proxy_export_carries_reads_to_the_actor(monkeypatch):
    """Through proxy_rollout_server.chat_completions on the fake SGLang: A's a3 has two leaves below it (a4, a4b); B's b2
    reads A's replies 1..3 and b3 continues after it; C's c1 reads the same range; E's e1 reads B's b2..b3, which B
    computed with its own read. Every reader leaf's read reaches actor.batch_rows as (at, the first leaf below the
    source, start, end) with the proxy's numbers, and the source row holds the source's tokens there, with
    OCTOPUS_READ_TOOL unset in the worker. Outside cursor mode: the batch has no read keys and every other tensor is
    the same."""
    import transformers
    from areal.experimental.openai.client import ArealOpenAI
    from areal.experimental.openai.proxy import proxy_rollout_server as srv
    from areal.experimental.openai.proxy.server import SessionData
    from test_delta_render import QWEN35

    from octopus.rl import actor

    path = next((p for p in QWEN35 if p and os.path.isdir(p)), None) or pytest.skip("no Qwen3.5 tokenizer here")
    tok = transformers.AutoTokenizer.from_pretrained(path)
    fake = hop.FakeSGLang([(tok.encode(f"Reply {k}.") + [tok.eos_token_id], "stop") for k in range(11)])
    srv._openai_client = ArealOpenAI(engine=hop.engine(monkeypatch, fake), tokenizer=tok, chat_template_type="concat")
    srv._session_cache.clear()
    srv._session_cache["s"] = session = SessionData(session_id="s")

    def ask(cursor, text, mount=None):
        body = {"max_completion_tokens": 16, "messages": [{"role": "user", "content": text}]}
        if cursor is None:
            body["messages"] = [{"role": "system", "content": "Be brief."}] + body["messages"]
        else:
            body["cursor"] = cursor
        if mount:
            body["mount"] = mount
        return asyncio.run(srv.chat_completions(body, session_id="s")).id
    a = [ask(None, "Hi.")]
    for k in (1, 2, 3):
        a.append(ask(a[-1], f"A{k}."))
    a4, a4b = ask(a[3], "A4."), ask(a[3], "A4b.")
    read_a = {"thread": "A", "from": a[1], "to": a[3]}
    b1 = ask(a[0], "B1.")
    b2 = ask(b1, "B2.", read_a)
    b3 = ask(b2, "B3.")
    c1 = ask(a[0], "C1.", read_a)
    e1 = ask(a[0], "E1.", {"thread": "B", "from": b2, "to": b3})

    monkeypatch.delenv("OCTOPUS_READ_TOOL", raising=False)  # the episode's setting, never the export's
    monkeypatch.delenv("OCTOPUS_CURSOR", raising=False)
    off, _ = _batch(session)  # first: in this process the live leaves cache their tensors
    monkeypatch.setenv("OCTOPUS_CURSOR", "1")
    on, leaves = _batch(session)
    assert set(leaves) == {a4, a4b, b3, c1, e1}
    assert set(on) == set(off) | set(MOUNT_KEYS) and not set(off) & set(MOUNT_KEYS)
    assert all(not torch.is_tensor(v) or (v.dtype == on[k].dtype and v.equal(on[k])) for k, v in off.items())

    seqs, mounts = actor.batch_rows(on)
    pairs = ((b3, b2), (c1, c1), (e1, e1))  # (reader leaf, the reader interaction in its chain)
    rec = {leaf: session.completions[reader].metadata["mount"] for leaf, reader in pairs}
    first_below = {a[3]: a4, b3: b3}  # a4 is exported before a4b
    assert mounts == {_row(on, leaf): [(m["at"], _row(on, first_below[m["source"]]), m["start"], m["end"])]
                      for leaf, m in rec.items()}
    for leaf, m in rec.items():
        [(at, j, s, e)] = mounts[_row(on, leaf)]
        assert seqs[j][s:e].tolist() == session.completions[m["source"]].model_response.input_tokens[s:e]
    assert actor.batch_rows(off)[1] is None


@hop.live
def test_live_read_trains_as_it_rolled_out(monkeypatch, tmp_path):
    """The forced read episode (tests/test_areal_cursor_hop.py, run as is), exported as the trainer gets it, split by
    the actor's allocation and run through its micro-batch tree forward (Qwen3.5-4B-text, flash-attn 2 + fla, as the
    actor's layers): each reader token's log-prob (the free replies right after a read) against SGLang's rollout
    log-prob, the same with the read dropped from the tree, and the model's own stop tokens of the forced replies
    before any read as the baseline; then a backward through the mounted tree (finite gradients, no optimizer step)."""
    import transformers
    import areal.utils.data as areal_data
    from areal.api.cli_args import MicroBatchSpec
    from areal.experimental.openai.proxy import proxy_rollout_server as srv
    from test_delta_render import QWEN35

    from octopus.rl import actor
    from octopus.rl.model_hooks import patch_model, tile_model
    from octopus.rl.prefix_tree import patch_attention, tree_inputs

    hop.test_live_forced_read_episode(monkeypatch, tmp_path)
    torch.cuda.empty_cache()
    session = srv._session_cache["s"]
    monkeypatch.setenv("OCTOPUS_CURSOR", "1")
    batch, leaves = _batch(session)
    tensors = {k: v for k, v in batch.items() if torch.is_tensor(v)}
    actor._batch_seqs, actor._batch_mounts = actor.batch_rows(tensors)
    try:  # grpo.yaml's cap
        mbs = areal_data.split_padded_tensor_dict_into_mb_list(tensors, MicroBatchSpec(max_tokens_per_mb=262144),
                                                               sync_mbs=False)
    finally:
        mounts_all, actor._batch_seqs, actor._batch_mounts = actor._batch_mounts, None, None
    readers = {x.interaction_id: x for x in session.completions.values() if x.metadata.get("mount")}
    assert len(readers) == 2 and len(mounts_all) == 2

    path = next((p for p in QWEN35 if p and os.path.isdir(p)), None) or pytest.skip("no Qwen3.5 tokenizer here")
    model = transformers.AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16,
                                                              attn_implementation="flash_attention_2").cuda()
    patch_model(model.config)
    tile_model(model)  # as the actor
    patch_attention()
    head = model.get_output_embeddings()

    def scored(tree, flat, q):
        """(log-prob, argmax) of flat tokens q, from the tree rows scoring positions q - 1."""
        meta, kw = tree
        logits = head(model.model(**kw).last_hidden_state[0][meta.pair_pos[meta.flat_pair[q - 1]]]).float()
        return torch.log_softmax(logits, -1).gather(1, flat[q][:, None])[:, 0], logits.argmax(-1)

    d = {"reader": [], "reader_no_read": [], "baseline": []}
    agree = {"reader": [], "reader_no_read": []}
    per_reader: dict[str, tuple[list, list]] = {}  # reader -> (|delta| with its read, without)
    report, seen = {"micro_batch_sequences": [len(mb["attention_mask"]) for mb in mbs.mbs]}, set()
    for mb in mbs.mbs:
        packed = areal_data.pack_tensor_dict(mb)
        kwargs = {k: v.cuda()[None] if torch.is_tensor(v) and k != "cu_seqlens" else v for k, v in packed.items()}
        kwargs["cu_seq_lens_q"] = packed["cu_seqlens"].cuda()
        cu = packed["cu_seqlens"].tolist()
        flat, rollout = kwargs["input_ids"].reshape(-1), kwargs["logprobs"].reshape(-1).float()
        mounts = actor.packed_mounts(kwargs, cu) or {}
        row_of = {int(packed["octopus_node"][c]): i for i, c in enumerate(cu[:-1])}
        reader_q, base_q, owner = [], [], []
        for lid, leaf in leaves.items():
            if (i := row_of.get(node_id(lid))) is None:
                continue
            first_read = min((m[0] for m in mounts.get(i, [])), default=None)
            x = leaf
            while x is not None:
                r, xid, x = x.model_response, x.interaction_id, x.parent
                if xid in seen:
                    continue
                seen.add(xid)
                if xid in readers:  # the free reply computed with the read: every token's log-prob is the model's
                    reader_q += [cu[i] + len(r.input_tokens) + k for k in range(len(r.output_tokens))]
                    owner += [xid] * len(r.output_tokens)
                elif first_read is None or len(r.input_tokens) < first_read:  # a forced reply before any read
                    base_q.append(cu[i] + len(r.input_tokens) + len(r.output_tokens) - 1)  # its stop token: the model's
        if not reader_q and not base_q:
            continue
        q = torch.tensor(reader_q + base_q, device="cuda")
        n = len(reader_q)
        every = {k: v for k, v in kwargs.items() if k != "loss_mask"}  # a tree with every pair: baseline q is untrained
        with torch.no_grad():
            lp, top = scored(actor.micro_batch_tree(every), flat, q)
            lp0, top0 = scored(tree_inputs(kwargs["input_ids"], kwargs["cu_seq_lens_q"]), flat, q)  # the read dropped
        d["reader"] += (lp[:n] - rollout[q[:n]]).abs().tolist()
        d["baseline"] += (lp[n:] - rollout[q[n:]]).abs().tolist()
        d["reader_no_read"] += (lp0[:n] - rollout[q[:n]]).abs().tolist()
        for xid, a, b in zip(owner, (lp[:n] - rollout[q[:n]]).abs().tolist(), (lp0[:n] - rollout[q[:n]]).abs().tolist()):
            per_reader.setdefault(xid, ([], []))[0].append(a)
            per_reader[xid][1].append(b)
        agree["reader"] += (top[:n] == flat[q[:n]]).tolist()
        agree["reader_no_read"] += (top0[:n] == flat[q[:n]]).tolist()
        if n:  # backward through the mounted tree: finite gradients, no optimizer step
            loss = -scored(actor.micro_batch_tree(every), flat, q[:n])[0].mean()
            loss.backward()
            grads = [p.grad for p in model.parameters() if p.grad is not None]
            report.setdefault("backward", []).append({
                "loss": loss.item(), "params_with_grad": len(grads),
                "finite": all(bool(torch.isfinite(g).all()) for g in grads),
                "grad_norm": float(torch.stack([g.float().norm() for g in grads]).norm())})
            model.zero_grad(set_to_none=True)
    for k, v in d.items():
        report[k] = {"tokens": len(v), "mean_abs": statistics.mean(v), "max_abs": max(v)}
    for k, v in agree.items():
        report[k]["argmax_agree"] = statistics.mean(v)
    report["per_reader"] = {readers[x].metadata["mount"]["thread"] + f" read at {readers[x].metadata['mount']['at']}":
                            {"tokens": len(a), "mean_abs": statistics.mean(a), "mean_abs_no_read": statistics.mean(b)}
                            for x, (a, b) in per_reader.items()}
    print(json.dumps(report, indent=1))
    assert report["reader"]["tokens"] == sum(len(x.model_response.output_tokens) for x in readers.values())
    assert report["reader"]["mean_abs"] <= 0.03  # #131/#132's bar
    assert all(r["mean_abs_no_read"] > 1.5 * r["mean_abs"] for r in report["per_reader"].values())  # the read is used
    assert all(b["finite"] and b["grad_norm"] > 0 for b in report["backward"])


@pytest.mark.parametrize("reader_continues", [True, False])
def test_a_read_controls_first_request_is_no_retry_of_its_readers_mount_request(reader_continues):
    """Review of #148: the read control's first request has the reader's mount request's parent and messages; only the
    mount differs. AReaL's drop_retry_orphans (grpo.yaml) grouped them as one retry, so a reader whose mount request
    was its last (it committed right after its read) lost that leaf to the control's. The mount is part of the key."""
    from areal.experimental.openai.cache import InteractionCache
    from areal.experimental.openai.types import InteractionWithTokenLogpReward as Interaction

    cache = InteractionCache()

    def add(key, parent, delta, mount=None):
        x = Interaction(messages=(parent.messages + parent.output_message_list if parent else []) + delta,
                        chat_template_type="concat", parent=parent)
        x.output_message_list, x.interaction_id = [{"role": "assistant", "content": key}], key
        if mount:
            x.metadata["mount"] = mount
        cache[key] = x
        return x
    read = add("r2", None, [{"role": "user", "content": "task"}])  # the reader's read reply
    result = [{"role": "tool", "tool_call_id": "c0", "content": "[branch-1 | step 4]\nMounted branch-0's steps ..."}]
    mounted = add("r3", read, result, {"thread": "branch-0", "source": "a3", "start": 1, "end": 9, "at": 40})
    if reader_continues:
        add("r4", mounted, [{"role": "tool", "content": "exit 0"}])
    add("c2", add("c1", read, result), [{"role": "tool", "content": "exit 0"}])  # the read control goes on
    assert cache._find_retry_orphan_ids() == set()
