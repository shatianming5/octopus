"""CPU checks of the Route A baselines (baselines/kv_read/route_a.py): parsing, template-agnostic rendering, report merge,
and on tiny random models (dense qwen3, Mamba-2 hybrid granite) cache continuation, subset mounts and one mount point."""

import json
from types import SimpleNamespace

import pytest
import torch
from tokenizers import AddedToken, Tokenizer, decoders, models, pre_tokenizers
from transformers import (AttentionInterface, GraniteMoeHybridConfig, GraniteMoeHybridForCausalLM,
                          PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM)
from transformers.integrations.sdpa_attention import sdpa_attention_forward

from baselines.kv_read import route_a
from octopus.tools.kv_read_probe import facts, render

# ChatML with Qwen3.5's rule: an assistant turn after the last user query carries an (empty) think block
CHATML = (
    "{% set ns = namespace(last=-1) %}{% for m in messages %}{% if m.role == 'user' %}{% set ns.last = loop.index0 %}"
    "{% endif %}{% endfor %}{% for m in messages %}"
    "{% if m.role == 'assistant' %}<|im_start|>assistant\n{% if loop.index0 > ns.last %}<think>\n\n</think>\n\n{% endif %}"
    "{{ m.content }}{% for c in m.tool_calls or [] %}<tool_call>\n<function={{ c.function.name }}>\n</function>\n"
    "</tool_call>{% endfor %}<|im_end|>\n"
    "{% elif m.role == 'tool' %}<|im_start|>user\n<tool_response>\n{{ m.content }}\n</tool_response><|im_end|>\n"
    "{% else %}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n{% endif %}{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n<think>\n\n</think>\n\n{% endif %}")
# granite-4.0-h's shape: role markers, a default system message, JSON tool calls
GRANITE = (
    "{% if messages[0].role != 'system' %}<|start_of_role|>system<|end_of_role|>You are Granite.<|end_of_text|>\n"
    "{% endif %}{% for m in messages %}<|start_of_role|>{{ m.role }}<|end_of_role|>{{ m.content }}"
    "{% for c in m.tool_calls or [] %}<tool_call>\n{\"name\": \"{{ c.function.name }}\", \"arguments\": "
    "{{ c.function.arguments | tojson }}}\n</tool_call>{% endfor %}<|end_of_text|>\n{% endfor %}"
    "{% if add_generation_prompt %}<|start_of_role|>assistant<|end_of_role|>{% endif %}")
TEMPLATES = {"chatml": (CHATML, ["<|im_start|>", "<|im_end|>"], "<|im_end|>"),
             "granite": (GRANITE, ["<|start_of_role|>", "<|end_of_role|>", "<|end_of_text|>"], "<|end_of_text|>")}
TOOLS = [{"type": "function", "function": {"name": "bash", "description": "run", "parameters": {
    "type": "object", "properties": {"cmd": {"type": "string"}}, "required": ["cmd"]}}}]


def tokenizer(name):
    """Byte-level tokenizer (no merges) with the template's control tokens."""
    template, specials, eos = TEMPLATES[name]
    t = Tokenizer(models.BPE(vocab={c: i for i, c in enumerate(sorted(pre_tokenizers.ByteLevel.alphabet()))}, merges=[]))
    t.pre_tokenizer, t.decoder = pre_tokenizers.ByteLevel(add_prefix_space=False), decoders.ByteLevel()
    t.add_special_tokens([AddedToken(s, special=True) for s in specials])
    return PreTrainedTokenizerFast(tokenizer_object=t, eos_token=eos, chat_template=template)


def call(i, name="bash"):
    return {"role": "assistant", "content": f"step {i}", "tool_calls": [
        {"id": f"c{i}", "type": "function", "function": {"name": name, "arguments": json.dumps({"cmd": f"ls {i}"})}}]}


def result(i, text):
    return {"role": "tool", "tool_call_id": f"c{i}", "content": text}


def test_tool_name_xml_json_and_none():
    assert route_a.tool_name("<tool_call>\n<function=str_replace>\n<parameter=path>") == "str_replace"
    assert route_a.tool_name('ok\n<tool_call>\n{"name": "bash", "arguments": {}}\n</tool_call>') == "bash"
    assert route_a.tool_name("I am done.") is None
    assert route_a.same_tool('<tool_call>{"name": "bash"}', "<function=bash>") == 1.0
    assert route_a.same_tool("done", "also done") == 0.0  # None == None is not a match


@pytest.mark.parametrize("name, user, gen", [
    ("chatml", "<|im_start|>user\nQ?<|im_end|>\n", "<|im_start|>assistant\n<think>\n\n</think>\n\n"),
    ("granite", "<|start_of_role|>user<|end_of_role|>Q?<|end_of_text|>\n", "<|start_of_role|>assistant<|end_of_role|>")])
def test_appended_user_message(name, user, gen):
    tok = tokenizer(name)
    ctx = [{"role": "user", "content": "task"}, call(1), result(1, "out"), call(2), result(2, "out 2")]
    extra = [{"role": "user", "content": "Q?"}]
    assert route_a.appended(tok, ctx, extra, TOOLS) == user
    assert route_a.appended(tok, ctx, extra, TOOLS, gen=True) == user + gen
    # what the cached context ends with, then the appended turn: B's tokens are never re-rendered
    assert render(tok, ctx, TOOLS).endswith("<|end_of_text|>\n" if name == "granite" else "<|im_end|>\n")
    if name == "chatml":  # the whole-context render is no prefix: the template re-renders turns before a new query
        assert not render(tok, ctx + extra, TOOLS).startswith(render(tok, ctx, TOOLS))


def test_appended_raises_when_not_a_suffix():
    tok = tokenizer("chatml")
    tok.chat_template = "{% for m in messages %}{{ m.content }}{% endfor %}{{ messages | length }}"
    with pytest.raises(ValueError):
        route_a.appended(tok, [{"role": "user", "content": "a"}], [{"role": "user", "content": "b"}], TOOLS)


def test_mount_layers_and_root_map():
    dense, hybrid = list(range(36)), [3, 7, 11, 15, 19, 23, 27, 31]
    assert route_a.mount_layers("all", hybrid) == hybrid
    assert route_a.mount_layers("16-24", dense) == list(range(16, 25))
    assert route_a.mount_layers("7,3", hybrid) == [3, 7]
    for bad in ("16-24", "4", "9-3"):
        with pytest.raises(ValueError):
            route_a.mount_layers(bad, hybrid)
    rm = ["../data/e11=/data/e11", "/mnt/disk4/user=/srv/z"]
    assert route_a.remap("../data/e11/t/agent", rm) == "/data/e11/t/agent"
    assert route_a.remap("/mnt/disk4/user/rl/x", rm) == "/srv/z/rl/x"
    assert route_a.remap("/other/x", rm) == "/other/x"


def fake_row(i, conds, em, gen="<function=bash>"):
    def cond(e):
        return {"facts": [{"answer": "x", "lp": -1.0, "tok_lp": [-0.5, -1.5], "n": 2, "em": e, "gen16": [],
                           "d_tot": [-9.0, -9.0, -9.0]}], "nll": 1.0, "kl": 0.1, "gen": gen}
    return {"id": i, "agent_dir": f"tree{i % 3}", "ahead_by": i - 2, "a_only": ["a_only_1"], "b_only": ["b_only_1"],
            "reply_text": "<function=bash>", "p_b": 10, "a0": 5, "M": 3, "cond": {c: cond(em[c]) for c in conds}}


def test_report_merges_files_by_id(tmp_path, capsys):
    run2 = [fake_row(i, ("none", "K2"), {"none": False, "K2": True}) for i in range(6)]
    back = [fake_row(i, ("none", "summary"), {"none": True, "summary": False}) for i in range(6)]
    for r in back:
        r["cond"]["summary"]["summary"] = {"text": "s", "gen_tokens": 40, "insert_tokens": 60, "ms": 900.0}
    back[5] = {"id": 5, "agent_dir": "tree2", "skipped": "template"}
    (tmp_path / "run2.jsonl").write_text("".join(json.dumps(r) + "\n" for r in run2))
    (tmp_path / "back.jsonl").write_text("".join(json.dumps(r) + "\n" for r in back))
    rows, files = route_a.merged([str(tmp_path / "run2.jsonl"), str(tmp_path / "back.jsonl")])
    assert list(files.values()) == [{"all"}, {"all"}]
    assert sorted(rows[0]["cond"]) == ["K2", "none", "summary"] and rows[0]["cond"]["none"]["facts"][0]["em"] is False
    assert sorted(rows[5]["cond"]) == ["K2", "none"]  # skipped in the backfill: run 2's row
    route_a.report(SimpleNamespace(results=[str(tmp_path / "run2.jsonl"), str(tmp_path / "back.jsonl")]))
    out = capsys.readouterr().out
    assert "conditions none, summary, K2" in out and "| K3 |" not in out and "text" not in out.split("paired")[1]
    assert "all: EM K2 - summary = 1.000" in out
    assert "| summary | 0.000 [0.000, 0.000] | 40.000" in out  # 5 summaries (one skipped), 40 tokens each


# ---------------------------------------------------------------- tiny models on the CPU

def bottom_right_sdpa(module, q, k, v, mask, **kwargs):
    """What flash attention does on the GPU without a mask: query i of n sees the keys up to len(k) - n + i, so
    attention layers holding different numbers of rows (a mount on a layer subset) are each causal."""
    n, m = q.shape[2], k.shape[2]
    causal = torch.ones(n, m, dtype=torch.bool).tril(m - n)[None, None]
    return sdpa_attention_forward(module, q, k, v, causal, is_causal=False, **kwargs)


AttentionInterface.register("bottom_right_sdpa", bottom_right_sdpa)


def tiny(family):
    torch.manual_seed(0)
    if family == "qwen3":
        cfg = Qwen3Config(vocab_size=320, hidden_size=64, intermediate_size=128, num_hidden_layers=4,
                          num_attention_heads=4, num_key_value_heads=2, head_dim=16, max_position_embeddings=8192)
        model = Qwen3ForCausalLM(cfg)
    else:  # granite-4.0-h's layout: Mamba-2 layers around an attention layer without positions, logits_scaling
        cfg = GraniteMoeHybridConfig(vocab_size=320, hidden_size=64, intermediate_size=64, shared_intermediate_size=64,
                                     num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
                                     layer_types=["mamba", "attention", "mamba", "attention"], mamba_n_heads=8,
                                     mamba_d_head=16, mamba_d_state=16, mamba_n_groups=1, mamba_chunk_size=8,
                                     num_local_experts=0, position_embedding_type="nope", logits_scaling=3.0)
        model = GraniteMoeHybridForCausalLM(cfg)
    model.set_attn_implementation("bottom_right_sdpa")
    torch.set_grad_enabled(False)
    return model.eval()


@pytest.fixture
def probe(monkeypatch):
    monkeypatch.setattr(route_a, "patch_model", lambda config: None)  # mamba2_varlen imports fla (CUDA hosts)
    torch.set_grad_enabled(False)

    def make(family):
        model = tiny(family)
        p = route_a.Probe(model, tokenizer("granite" if family == "granite" else "chatml"))
        p.model = model  # the reference forward
        return p
    yield make
    torch.set_grad_enabled(True)


@pytest.mark.parametrize("family", ["qwen3", "granite"])
def test_multi_token_continuation_equals_plain_forward(probe, family):
    """A cache continued by a multi-token chunk (the recurrent layers from their stored conv + SSM state) equals one
    forward over all tokens; the head applies the model's own post-head ops (granite: / logits_scaling)."""
    p = probe(family)
    ids = torch.randint(0, 300, (50,)).tolist()
    full, _ = p.run(ids, 0)
    _, c = p.run(ids[:23], 0)
    h, _ = p.run(ids[23:], 23, c)
    assert torch.allclose(h, full[23:], atol=1e-5)
    assert torch.allclose(p.logits(h), p.model(torch.tensor([ids])).logits[0, 23:], atol=1e-5)
    assert p.attn == ([0, 1, 2, 3] if family == "qwen3" else [1, 3])


@pytest.mark.parametrize("family", ["qwen3", "granite"])
def test_rotate(probe, family):
    """qwen3: keys at p re-rotated by delta are the keys at p + delta; granite (NoPE): the identity."""
    p = probe(family)
    k = torch.randn(1, 2, 5, 16)
    if family == "granite":
        assert p.rope is None and torch.equal(p.rotate(k, 777), k)
        return
    pos = torch.arange(100, 105)[None]
    at = lambda q: p.apply_rope(k, k, *p.rope(k, q))[1]  # noqa: E731
    assert torch.allclose(p.rotate(at(pos), 777), at(pos + 777), atol=1e-4)


def test_mount_of_own_rows_on_every_layer_is_the_plain_forward(probe):
    """Dense model: B's cache + its own next rows mounted (K1 positions) then a tail = one forward; a subset mount
    leaves the other layers' caches alone and changes the output."""
    p = probe("qwen3")
    ids = torch.randint(0, 300, (40,)).tolist()
    full, _ = p.run(ids, 0)
    _, c_all = p.run(ids[:30], 0)
    kv = p.kv(c_all, 20, 30)
    _, c_b = p.run(ids[:20], 0)
    h, _ = p.run(ids[30:], 30, p.mounted(c_b, kv, 0))
    assert torch.allclose(h, full[30:], atol=1e-5)
    sub = p.mounted(c_b, {i: kv[i] for i in [1, 2]}, 0)
    assert [layer.keys.shape[2] for layer in sub.layers] == [20, 30, 30, 20]
    h_sub, _ = p.run(ids[30:], 30, sub)
    assert not torch.allclose(h_sub, full[30:], atol=1e-3)


def write_tree(tmp_path):
    lines = 'File "/src/pkg/core_7.py", line 88, in render_value\nFile "/src/pkg/util_3.py", line 12, in parse_cfg_9'
    a = [{"role": "user", "content": "Task A: fix core."}, call(1), result(1, "a.txt"), call(2), result(2, lines),
         call(3)]
    b = [{"role": "user", "content": "Task B: fix docs."}, call(4), result(4, "b.txt"), call(5), result(5, "README.md"),
         call(6)]
    tree = {"tools": TOOLS, "main": [{"role": "system", "content": "You fix bugs."}, {"role": "user", "content": "Go."}],
            "forks": [{"prefix_len": 2, "branches": [{"branch_id": "branch-0", "messages": a},
                                                     {"branch_id": "branch-1", "messages": b}]}]}
    (tmp_path / "tree.json").write_text(json.dumps(tree))
    return tree


@pytest.mark.parametrize("family", ["qwen3", "granite"])
def test_mount_point_runs_every_condition(probe, tmp_path, family):
    p = probe(family)
    tree = write_tree(tmp_path)
    row = {"id": 3, "agent_dir": str(tmp_path), "fork": 0, "a": 0, "b": 1, "a_id": "branch-0", "b_id": "branch-1",
           "k": 5, "s": 3, "ja": 5, "a_only": ["core_7.py"], "b_only": ["README.md"], "facts": []}
    from octopus.tools.kv_read_probe import materialize
    m = materialize(p.tok, row, tree)
    row["facts"] = facts([tree["forks"][0]["branches"][0]["messages"][4]["content"]], m["mount"], m["b_ctx"])
    assert [f["answer"] for f in row["facts"]] == ["render_value", "parse_cfg_9"]
    x = route_a.inputs(p.tok, row, tree, list(route_a.CONDITIONS))
    sel = p.attn[1:]
    res = route_a.mount_point(p, row, x, {"render_value", "parse_cfg_9"}, list(route_a.CONDITIONS), sel, True, True)
    assert list(res["cond"]) == list(route_a.CONDITIONS) and res["mount_layers"] == sel
    assert res["sanity"]["none"]["mean_abs"] < 1e-4 and res["sanity"]["text"]["mean_abs"] < 1e-4
    assert res["sanity"]["rotate_rel_err"] < 1e-5 if family == "qwen3" else res["sanity"]["rotate_rel_err"] == 0  # NoPE
    per_layer = 2 * 2 * res["M"] * 16 * 4  # K and V x kv heads x M x head dim x fp32 bytes
    assert res["cond"]["K3"]["copy_bytes"] == len(p.attn) * per_layer
    assert res["cond"]["K2"]["copy_bytes"] == res["cond"]["kvcomm_sel"]["copy_bytes"] == len(sel) * per_layer
    s = res["cond"]["summary"]["summary"]
    assert 0 < s["gen_tokens"] <= route_a.SUMMARY_TOKENS and s["insert_tokens"] > 0
    route_a.per_mount(res)  # the report's per-mount stats take every condition
    if family == "granite":  # NoPE: mounts on the same layers are one computation
        assert res["cond"]["K2"]["same_as"] == res["cond"]["kvcomm_sel"]["same_as"] == "K1"
        assert "same_as" not in res["cond"]["K3"]
    res = route_a.mount_point(p, row, x, set(), ["none", "K3", "kvcomm_sel"], p.attn, False, False)
    assert res["cond"]["kvcomm_sel"] == {**res["cond"]["K3"], "same_as": "K3"}  # every attention layer: K3 itself
    if family == "granite":  # computed apart, K1 and K2 on every attention layer equal K3
        for k in ("K1", "K2"):
            alone = route_a.mount_point(p, row, x, set(), ["none", k], p.attn, False, False)["cond"][k]
            for key in ("nll", "kl", "gen"):
                assert alone[key] == pytest.approx(res["cond"]["K3"][key], abs=1e-6)
            assert sum((f["tok_lp"] for f in alone["facts"]), []) == pytest.approx(
                sum((f["tok_lp"] for f in res["cond"]["K3"]["facts"]), []), abs=1e-6)


def test_report_keeps_a_layer_subset_apart(tmp_path, capsys):
    """Qwen3 on layers 16-24 and on all 36: K1 / K2 / kvcomm_sel of each survive, the subset's as X@16-24."""
    every, sub = list(range(36)), list(range(16, 25))
    run_all = [fake_row(i, ("none", "K1", "K2", "K3", "kvcomm_sel"),
                        {"none": False, "K1": False, "K2": True, "K3": False, "kvcomm_sel": False}) for i in range(6)]
    run_sub = [fake_row(i, ("none", "K1", "K2", "kvcomm_sel"),
                        {"none": False, "K1": False, "K2": False, "kvcomm_sel": True}) for i in range(6)]
    for rows, layers in ((run_all, every), (run_sub, sub)):
        for r in rows:
            r.update(mount_layers=layers, attn_layers=every)
    run_all[0]["cond"]["K2"]["same_as"] = "K1"  # RoPE, peer behind: K2's positions are K1's at this mount point only
    for r in run_all:
        r["cond"]["kvcomm_sel"]["same_as"] = "K3"
    paths = [str(tmp_path / "sub.jsonl"), str(tmp_path / "all.jsonl")]
    for path, rows in zip(paths, (run_sub, run_all)):
        (tmp_path / path).write_text("".join(json.dumps(r) + "\n" for r in rows))
    by, files = route_a.merged(paths)
    assert sorted(by[0]["cond"]) == ["K1", "K1@16-24", "K2", "K2@16-24", "K3", "kvcomm_sel", "kvcomm_sel@16-24", "none"]
    assert files == {paths[0]: {"16-24"}, paths[1]: {"all"}}
    route_a.report(SimpleNamespace(results=paths))
    out = capsys.readouterr().out
    assert f"{paths[0]}: 16-24; {paths[1]}: all" in out
    assert "kvcomm_sel (= K3" in out and "(= K1" not in out
    assert "all: EM K2@16-24 - K2 = -1.000" in out and "all: EM kvcomm_sel@16-24 - K3 = 1.000" in out
    assert "all: EM kvcomm_sel@16-24 - K2@16-24 = 1.000" in out and "all: EM K2 - K3 = 1.000" in out
