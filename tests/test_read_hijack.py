"""Task-hijack measures for read_thread in Search (docs/search-read-hijack.md): Measure 1's extraction and pairing on a
fake tree (mounted vs refused runs of reads, the read/fork-only skip, relations), the judge's label aggregation with a
fake judge, and Measure 2's mount range (with the Qwen3.5 tokenizer where present) and report."""

import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_delta_render import tok  # noqa: E402,F401  (fixture)

from octopus.tools import read_hijack as rh  # noqa: E402
from octopus.tools import read_hijack_probe as rp  # noqa: E402

N = [0]


def call(name: str, **args) -> dict:
    N[0] += 1
    return {"id": f"c{N[0]}", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}


def reply(*calls, content="") -> dict:
    return {"role": "assistant", "content": content, **({"tool_calls": list(calls)} if calls else {})}


def results(r: dict, *texts: str, who: str = "branch-0", step: int | None = None) -> list[dict]:
    tag = f"[{who} | context left: 9000 | calls left: 9 | step {step}]\n" if step is not None else ""
    return [{"role": "tool", "tool_call_id": c["id"], "content": tag + t} for c, t in zip(r["tool_calls"], texts)]


def steps(who: str, first: int, n: int) -> list[dict]:
    """n search steps of a thread, numbered from first, then done (untagged result)."""
    out = []
    for k in range(first, first + n):
        r = reply(call("search", query=f"{who} query {k}"))
        out += [r] + results(r, f"[1] docid={1000 + k} | a {who} document", who=who, step=k)
    d = reply(call("done", answer=f"{who} answer", evidence="docid=1001"))
    return out + [d] + results(d, f"Final answer recorded: {who} answer")


def fake_tree() -> dict:
    sys_q = [{"role": "system", "content": "You answer a question by searching."},
             {"role": "user", "content": "Who wrote the root question?"}]
    fork = reply(call("fork", tasks=["find the author", "find the year"]))
    prefix = sys_q + [fork] + results(fork, "fork started", who="main", step=1)

    def task(name, t):
        return {"role": "user", "content": f"You are {name}.\nYour task: {t}"}
    s_msgs = prefix + [task("fork0/branch-1", "find the year")] + steps("branch-1", 2, 3)
    r1 = reply(call("read_thread", thread="fork0/branch-1"))
    r2 = reply(call("fork", tasks=["sub"]))
    a1 = reply(call("search", query="author of the book"), content="Back to my task.")
    r3 = reply(call("read_thread", thread="nope"), call("read_thread", thread="fork1/branch-0"))
    a2 = reply(call("done", answer="1999", evidence="docid=1003"))
    a3 = reply(call("search", query="more"), call("read_thread", thread="fork0/branch-0"))
    a4 = reply(content="Thinking about it.")
    r5 = reply(call("read_thread", thread="fork0/branch-1"))
    a5 = reply(call("done", answer="?", evidence="none"))
    r4 = reply(call("read_thread", thread="fork0/branch-1"))
    r_msgs = prefix + [task("fork0/branch-0", "find the author")] + [
        r1, *results(r1, "Mounted fork0/branch-1's steps 2 to 3 (120 tokens; fork0/branch-1 has readable steps 2 to "
                     "4): from your next reply on you attend to them; their full text is not copied here.", step=2),
        r2, *results(r2, "fork started", step=3),
        a1, *results(a1, "[1] docid=77 | author page", step=4),
        r3, *results(r3, "No thread nope in this episode. Nothing was read.",
                     "fork1/branch-0 has no readable step yet. Nothing was read.", step=5),
        a2, *results(a2, "Final answer recorded: 1999"),
        a3, *results(a3, "[1] docid=78 | more", "You are fork0/branch-0: read_thread reads another thread, never your "
                     "own. Nothing was read.", step=6),
        a4, r5, *results(r5, "Refused: your context is nearly used up (300 tokens left). Close now: call done with "
                         "what you have so far. Until you do, every other call is refused.", step=7),
        a5, *results(a5, "Final answer recorded: ?"),
        r4, *results(r4, "fork0/branch-1 has readable steps 2 to 4. Nothing was read.", step=8)]

    def branch(name, t, msgs, forks=()):
        return {"branch_id": name.split("/")[-1], "workdir": name, "task": t, "messages": msgs, "forks": list(forks)}
    sub = {"prefix_len": len(r_msgs) - 12, "branches": [branch("fork0/branch-0/fork0/branch-0.0", "sub", [])]}
    return {"tools": [], "main": prefix, "forks": [
        {"prefix_len": len(prefix), "branches": [branch("fork0/branch-0", "find the author", r_msgs, [sub]),
                                                 branch("fork0/branch-1", "find the year", s_msgs)]},
        {"prefix_len": len(prefix), "branches": [branch("fork1/branch-0", "check the author", prefix[:2])]}]}


def write_tree(tmp_path: Path) -> Path:
    p = tmp_path / "jobs-search-x" / "search-7-3:2-0-ab" / "7__cd" / "agent" / "tree.json"
    p.parent.mkdir(parents=True)
    p.write_text(json.dumps(fake_tree()))
    return tmp_path / "jobs-search-x"


def test_relation_and_resolve():
    r = "fork0/branch-0"
    assert [rh.relation(r, s) for s in ("fork0/branch-1", "fork1/branch-2", "fork0/branch-0/fork0/branch-0.1", "main",
                                        r, None)] == ["sibling", "cousin", "descendant", "ancestor", "self", "unknown"]
    assert rh.relation("fork0/branch-0/fork0/branch-0.1", "fork0/branch-0") == "ancestor"
    assert rh.relation("fork0/branch-0/fork0/branch-0.1", "fork0/branch-1") == "cousin"
    names = {"fork0/branch-0", "fork0/branch-1", "fork1/branch-1"}
    assert rh.resolve(" fork0/branch-1 ", names) == "fork0/branch-1"
    assert rh.resolve("branch-0", names) == "fork0/branch-0" and rh.resolve("branch-1", names) is None  # ambiguous
    assert rh.resolve("main", names) == "main" and rh.resolve(3, names) is None


def test_extract_pairs_mounted_and_refused_runs(tmp_path, capsys):
    jobs = write_tree(tmp_path)
    out = tmp_path / "units.jsonl"
    rh.main(["extract", "--out", str(out), str(jobs)])
    assert "no_action_after_reads" in capsys.readouterr().out
    units = [json.loads(x) for x in open(out)]
    assert [(u["S"], u["relation"], u["arm"], u["refusal"], u["mounted_before"], u["reply_index"]) for u in units] == [
        ("fork0/branch-1", "sibling", "mounted", None, False, 3),  # read, fork-only skipped, then the search
        ("fork1/branch-0", "cousin", "refused", "not_readable_yet", True, 5),  # "nope": no thread; the next one decides
        ("fork0/branch-0", "self", "refused", "self", True, 7),  # a substantive reply's read: the next reply's unit
        ("fork0/branch-1", "sibling", "close_now", None, True, 9)]  # the reserve's order to close, not a refusal
    assert units[0]["position"] == 3 / 10 and all(u["R"] == "fork0/branch-0" for u in units)
    assert rh.reason("main waits for your fork to close: none of its steps can be read. Other threads: x (no readable "
                     "step yet). Nothing was read.") == "ancestor_waits"
    assert rh.reason("\n[clipped: context nearly used up]\n\nYour context is nearly used up.") == "close_now"
    m = units[0]
    assert m["steps"] == [2, 3, 120] and m["action"].startswith("Back to my task.\nTool calls:\nsearch(")
    assert m["S_task"] == "find the year" and m["question"] == "Who wrote the root question?"
    msgs = fake_tree()["forks"][0]["branches"][0]["messages"]
    at = next(i for i, m in enumerate(msgs) if "author of the book" in json.dumps(m))
    assert m["cluster"] == "jobs-search-x/search-7-3"
    assert m["id"] == f"jobs-search-x/search-7-3:2-0-ab/7__cd/fork0/branch-0/{at}"
    assert msgs[m["read_reply"]]["tool_calls"][0]["function"]["name"] == "read_thread"
    assert units[1]["read"].startswith("No thread nope")
    assert units[1]["action"] == 'Tool calls:\ndone({"answer": "1999", "evidence": "docid=1003"})'
    assert units[2]["action"] == "Thinking about it."  # the trailing read (no action after it) is no unit


def test_judge_and_report(tmp_path, capsys):
    jobs = write_tree(tmp_path)
    units, labels, cases = (tmp_path / f for f in ("units.jsonl", "labels.jsonl", "cases.jsonl"))
    rh.main(["extract", "--out", str(units), str(jobs)])
    seen = []

    def fake(body):  # a judge that says S_TASK when the action searches, OWN otherwise; the first call fails once
        seen.append(body)
        if len(seen) == 1:
            raise OSError("down")
        text = body["messages"][0]["content"]
        assert "Root question:\nWho wrote the root question?" in text and "fork0/branch-1: find the year" in text
        return "S_TASK" if "search(" in text.split("<action>")[1] else " own."
    rh.judge(SimpleNamespace(items=str(units), labels=str(labels), url="x", model="judge", workers=1), post=fake)
    got = {json.loads(x)["id"].split("/")[-1]: json.loads(x)["label"] for x in open(labels)}
    assert sorted(got.values()) == ["OWN", "OWN", "OWN", "S_TASK"]
    assert [rh.parse(x) for x in ("label: OWN_USING_S", "s_task", "?")] == ["OWN_USING_S", "S_TASK", "UNPARSED"]
    rh.main(["export", "--n", "4", str(units), str(cases)])
    rows = [json.loads(x) for x in open(cases)]
    assert len(rows) == 3 and all("arm" not in r and r["label"] == "" for r in rows)
    with open(cases, "w") as f:  # the annotator disagrees with the judge on the mounted unit
        f.writelines(json.dumps(r | {"label": "OWN", "annotator": "test-llm"}) + "\n" for r in rows)
    capsys.readouterr()
    assert rh.main(["report", str(units), str(labels), "--annotated", str(cases)]) is None
    out = capsys.readouterr().out
    line = next(x for x in out.splitlines() if x.startswith("| jobs-search-x | all |"))
    assert "| 1 / 2 | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 1.000 [1.000, 1.000] |" in line  # CLOSE_NOW out
    assert "| jobs-search-x | close_now | 1 |" in out and "| jobs-search-x | refused: self | 1 |" in out
    assert "LLM-annotated labels (test-llm)" in out and "| S_TASK | 1 | 0.00 |" in out and "| OWN | 2 | 1.00 |" in out
    with open(labels, "a") as f:  # a judge error left: never a non-hijack, no result, and judge exits 1
        f.write(json.dumps({"id": units_ids(units)[0], "label": "ERROR"}) + "\n")
    assert rh.main(["report", str(units), str(labels)]) == 1
    assert "{'ERROR': 1}" in capsys.readouterr().out
    assert rh.judge(SimpleNamespace(items=str(units), labels=str(labels), url="x", model="judge", workers=1),
                    post=lambda body: (_ for _ in ()).throw(OSError("down"))) == 1


def units_ids(path) -> list[str]:
    return [json.loads(x)["id"] for x in open(path)]


def test_boot_diff_resamples_clusters():
    rows = [{"cluster": c, "m": m, "y": y} for c, m, y in
            (("a", 1, 1), ("a", 0, 0), ("b", 1, 0), ("b", 0, 0), ("c", 1, 1), ("c", 0, 1))]
    est, lo, hi = rh.boot_diff(rows, lambda u: u["y"], lambda u: u["m"])
    assert abs(est - (2 / 3 - 1 / 3)) < 1e-9 and lo <= est <= hi
    assert math.isnan(rh.boot_diff(rows[:1], lambda u: u["y"], lambda u: u["m"])[0])  # no refused row


def test_probe_mount_range(tok, tmp_path):  # noqa: F811
    jobs = write_tree(tmp_path)
    out = tmp_path / "u.jsonl"
    rh.main(["extract", "--out", str(out), str(jobs)])
    u = next(x for x in map(json.loads, open(out)) if x["arm"] == "mounted")
    src = json.loads(Path(u["tree"]).read_text())["forks"][0]["branches"][1]["messages"]
    ids = lambda msgs: tok.encode(rp.render(tok, msgs, []), add_special_tokens=False)  # noqa: E731
    at = lambda q: next(i for i, m in enumerate(src) if m["role"] == "assistant" and q in json.dumps(m))  # noqa: E731
    p = rp.point(u | {"steps": [2, 3, len(ids(src[:at("query 4")])) - len(ids(src[:at("query 2")]))]}, tok)
    text = tok.decode(p["a"][p["s"]:p["e"]])
    assert text.startswith("<|im_start|>assistant") and "query 2" in text and "docid=1003" in text
    assert "query 4" not in text  # it stops at the turn of the reply after step 3
    assert tok.decode(p["delta"]["none"]).startswith("<|im_start|>user\n<tool_response>\n[branch-0")
    assert "query 3" in tok.decode(p["delta"]["text"]) and "query 3" not in tok.decode(p["delta"]["none"])
    assert p["range_minus_read"] == 0 and p["jump"] == max(0, p["e"] - len(p["b"]))
    done = at("done")  # S has ended: step 5 is its done reply, read through that reply alone
    ended = rp.point(u | {"steps": [5, 5, len(ids(src[:done + 1])) - len(ids(src[:done]))]}, tok)
    assert "branch-1 answer" in tok.decode(ended["a"][ended["s"]:ended["e"]])
    tree = json.loads(Path(u["tree"]).read_text())  # S's step 2 mounted: rows the probe does not rebuild
    tree["forks"][0]["branches"][1]["messages"][at("query 3") - 1]["content"] = (
        "[branch-1 | context left: 9000 | calls left: 9 | step 2]\n"
        "Mounted fork0/branch-0's steps 2 to 2 (10 tokens; x)")
    Path(u["tree"]).write_text(json.dumps(tree))
    try:
        rp.point(u | {"steps": [2, 3, 1]}, tok)
        raise AssertionError("a source holding a mount is a mount point")
    except rp.Skip as x:
        assert str(x) == "source mounted before the range end"


def test_probe_parse_and_report(tmp_path, capsys):
    raw = ("Let me look.\n<tool_call>\n<function=search>\n<parameter=query>\nthe year\n</parameter>\n</function>\n"
           "</tool_call>")
    assert rp.parse_reply(raw) == 'Let me look.\nTool calls:\nsearch({"query": "the year"})'
    items, labels = tmp_path / "items.jsonl", tmp_path / "labels.jsonl"
    rows, labs = [], []
    for ck in ("v0", "v9"):
        for p in range(6):
            for cond in rp.CONDITIONS:
                for i in range(2):
                    hit = cond == "mount" and ck == "v9" or cond == "text"
                    rows.append({"id": f"{p}|{ck}|{cond}|{i}", "point": str(p), "cluster": f"q{p}", "ckpt": ck,
                                 "relation": "sibling", "cond": cond, "skip": False, "jump": p % 2})
                    labs.append({"id": rows[-1]["id"], "label": "S_TASK" if hit else "OWN"})
    items.write_text("".join(json.dumps(r) + "\n" for r in rows))
    labels.write_text("".join(json.dumps(r) + "\n" for r in labs))
    assert rp.main(["report", str(items), str(labels)]) == 0
    out = capsys.readouterr().out
    assert "| v9 | K2 jump | hijack |" in out and "| v9 | no jump | hijack |" in out
    assert "v0: mount - none 0.000 [0.000, 0.000] -> no mount hijack" in out
    assert "v9: mount - none 1.000 [1.000, 1.000] -> the mount hijacks" in out
    assert "training raised the mount's hijack" in out
    labels.write_text("".join(json.dumps(r) + "\n" for r in labs[1:]))  # one sample unlabelled: no verdict
    assert rp.main(["report", str(items), str(labels)]) == 1
    out = capsys.readouterr().out
    assert "{'UNLABELLED': 1}" in out and "NO VERDICT" in out and "->" not in out
