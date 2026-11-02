"""Statement coverage of bundle tasks: the part of a task's reference fix that its statement describes.

A SWE-smith bug patch can change several places while its generated statement describes one symptom, and the hidden
tests may need a change the statement never mentions. In the 2026-10-13 bundle-e6 audit, line_profiler's statement
names the `return None` but not the inverted `is_generator` test next to it, which the tests need.

The reference fix is split into units: each run of changed lines is a unit, except that a run of n removed lines
followed by n added lines becomes n one-line units. Grok labels each unit:
- described;
- not described;
- mixed, with the code as it would read with only the undescribed defect fixed.
`build_bundles.py reduce` then fixes the undescribed part in the bundle tree, so each task's bug is the one its
statement describes.

usage: python -m octopus.envs.swe.data.statement_coverage judge DATASET_DIR OUT.jsonl [BUNDLE:K ...]
           [--workers N] [--limit N]
       ... snippets DATASET_DIR RUNS.jsonl [OVERRIDES.json] [--conc N]   harbor python: statement code blocks on the
           fixed and the buggy tree (a block that fails with every fix in is broken)
       ... repair DATASET_DIR RUNS.jsonl REPAIRS.jsonl [--workers N]   Grok rewrites of the broken blocks
       ... overrides DATASET_DIR REPAIRS.jsonl OUT.json [CHECKED.jsonl]   statement texts for build_bundles reduce
The API comes from GROK_MODELS_BASE_URL / XAI_API_KEY (environment only), the model from REWRITE_MODEL (grok-4.6).
"""
import base64, collections, concurrent.futures, io, json, re, sys, tarfile, tomllib
from pathlib import Path

LABELS = ("described", "not described", "mixed")
# bug kinds whose fix lines do not correspond one to one (a rewritten function, moved lines, swapped branches): one
# unit per hunk, from its first to its last changed line, else pairing removed and added lines by position would
# match unrelated lines and a partial bake would build a function that is neither the bug nor the original
REGION_KINDS = ("lm_rewrite", "func_pm_ctrl_shuffle", "func_pm_ctrl_invert_if", "func_pm_class_shuffle_funcs")


def parse(patch: str) -> list | None:
    """[(path, [{"start", "func", "lines": [(op, text)]}])] of a git text diff; None when it is not a plain edit of
    existing text files (binary, a file created or deleted). "\\ No newline at end of file" marks are skipped: cut
    checks that the units rebuild the fix exactly."""
    if "GIT binary patch" in patch:
        return None
    files = []
    for line in patch.split("\n"):
        if line.startswith("diff --git "):
            files.append([None, []])
        elif not files:
            continue
        elif line.startswith(("new file mode", "deleted file mode", "rename from", "copy from")):
            return None
        elif line.startswith("+++ ") and not files[-1][1]:
            files[-1][0] = line[4:].removeprefix("b/")
        elif line.startswith("@@"):
            m = re.match(r"@@ -(\d+)(?:,\d+)? \+\d+(?:,\d+)? @@ ?(.*)", line)
            files[-1][1].append({"start": int(m.group(1)), "func": m.group(2).strip(), "lines": []})
        elif files[-1][1] and line[:1] in (" ", "-", "+"):
            files[-1][1][-1]["lines"].append((line[0], line[1:]))
    return [(p, hs) for p, hs in files] if files and all(p for p, _ in files) else None


def units(patch: str, kind: str = "") -> list | None:
    """The fix as units [{"id", "path", "func", "file", "hunk", "at", "minus", "plus"}]; `at` is the index of the
    unit's first removed line (its insertion point when it removes none) on the hunk's old side. For REGION_KINDS a
    unit is a hunk's span from its first to its last changed line (context lines inside it on both sides)."""
    files = parse(patch)
    if files is None:
        return None
    out = []
    for f, (path, hunks) in enumerate(files):
        for h, hunk in enumerate(hunks):
            if kind.startswith(REGION_KINDS):
                ch = [i for i, (op, _) in enumerate(hunk["lines"]) if op in "+-"]
                span = hunk["lines"][ch[0]:ch[-1] + 1]
                out.append({"id": f"U{len(out) + 1}", "path": path, "func": hunk["func"], "file": f, "hunk": h,
                            "at": sum(op in " -" for op, _ in hunk["lines"][:ch[0]]),
                            "minus": [t for op, t in span if op in " -"], "plus": [t for op, t in span if op in " +"]})
                continue
            old, run, start = 0, [], 0
            for op, text in hunk["lines"] + [(" ", "")]:
                if op in "+-":
                    if not run:
                        start = old
                    run.append((op, text))
                    old += op == "-"
                    continue
                if run:
                    minus = [t for o, t in run if o == "-"]
                    plus = [t for o, t in run if o == "+"]
                    plain = [o for o, _ in run] == ["-"] * len(minus) + ["+"] * len(plus)
                    parts = ([([m], [p], start + i) for i, (m, p) in enumerate(zip(minus, plus))]
                             if plain and len(minus) == len(plus) > 1 else [(minus, plus, start)])
                    for m, p, at in parts:
                        out.append({"id": f"U{len(out) + 1}", "path": path, "func": hunk["func"], "file": f, "hunk": h,
                                    "at": at, "minus": m, "plus": p})
                    run = []
                old += 1
    return out


def edit(texts: dict, patch: str, replace: dict, kind: str = "") -> dict:
    """{path: content} of the tree the fix applies to -> the contents with each unit u in `replace` (unit id ->
    lines) put in place of u's removed lines; raises ValueError where a hunk's old side is not in the file."""
    files, us = parse(patch), units(patch, kind)
    out = dict(texts)
    for f, (path, hunks) in enumerate(files):
        lines = out[path].split("\n")
        edits = []
        for h, hunk in enumerate(hunks):
            old = [t for op, t in hunk["lines"] if op in " -"]
            spots = [i for i in range(len(lines) - len(old) + 1) if lines[i:i + len(old)] == old]
            if not spots:
                raise ValueError(f"{path}: hunk {h + 1} not found")
            at = min(spots, key=lambda i: abs(i - (hunk["start"] - 1)))
            edits += [(at + u["at"], len(u["minus"]), replace[u["id"]]) for u in us
                      if u["file"] == f and u["hunk"] == h and u["id"] in replace]
        for pos, n, new in sorted(edits, reverse=True):
            lines[pos:pos + n] = new
        out[path] = "\n".join(lines)
    return out


def statement(task_dir: Path, k) -> str:
    """Task k's statement as the agent reads it (/tasks/<k>.md from the healthcheck's tar)."""
    cmd = tomllib.loads((task_dir / "task.toml").read_text())["environment"]["healthcheck"]["command"]
    tar = tarfile.open(fileobj=io.BytesIO(base64.b64decode(re.search(r"printf %s (\S+) \| base64 -d \| tar ", cmd).group(1))))
    return tar.extractfile(f"{k}.md").read().decode()


PROMPT = """A bug was injected into a Python repository. A developer who must fix it sees only this task statement:
<statement>
{statement}
</statement>

The reference fix (the injected buggy code -> the original correct code) is split into numbered units:
<units>
{units}
</units>

Label every unit, judging from the statement alone (not from what hidden tests might check):
- "described": someone fixing exactly the problem the statement describes has to make this change. The statement
  names the symptom, the wrong behaviour, or the code this change restores, explicitly or as its direct cause, or
  an example or expected output in the statement would still come out wrong without this change.
- "not described": the change restores behaviour the statement neither states nor implies (a separate defect), or it
  only reorders, renames or reformats code with no effect the statement mentions.
- "mixed": the unit's buggy code holds a described defect AND an undescribed one. Then also give "partial": the unit's
  code with ONLY the undescribed defect fixed, so that it still has exactly the described defect. That is the fixed
  code with only the described defect put back in: keep everything else exactly as in the fixed code. Write it as
  the complete replacement for the unit's buggy lines, with the same indentation.

Answer with one JSON object and nothing else:
{{"U1": {{"label": "described" | "not described" | "mixed", "why": "<one sentence>", "partial": "<mixed only>"}}, ...}}"""


def render(us: list) -> str:
    block = lambda ls: "\n".join(f"    {l}" for l in ls) or "    (nothing)"
    return "\n\n".join(f"{u['id']} in {u['path']}" + (f" ({u['func']})" if u["func"] else "") +
                       f":\n  buggy:\n{block(u['minus'])}\n  fixed:\n{block(u['plus'])}" for u in us)


def verdict(text: str, us: list) -> dict:
    """The labels in Grok's answer, checked: every unit labelled, a partial for each mixed unit."""
    m = re.search(r"\{.*\}", text, re.S)
    got = json.loads(m.group(0)) if m else {}
    out = {}
    for u in us:
        v = got.get(u["id"]) or {}
        if v.get("label") not in LABELS or (v["label"] == "mixed" and not isinstance(v.get("partial"), str)):
            raise ValueError(f"{u['id']}: bad label {v!r}"[:200])
        out[u["id"]] = {"label": v["label"], "why": str(v.get("why", ""))[:400],
                        **({"partial": v["partial"]} if v["label"] == "mixed" else {})}
    return out


def judge_task(ds: Path, bundle: str, k: int, instance_id: str, kind: str) -> dict:
    from octopus.envs.swe.data.acceptance_rewrite import ask
    rec = {"bundle": bundle, "k": k, "instance_id": instance_id, "kind": kind}
    us = units((ds / bundle / f"solution/fix_{k}.patch").read_text(), kind)
    if us is None:
        return rec | {"units": None, "why": "not a plain text edit: kept as it is"}
    try:
        text = ask(PROMPT.format(statement=statement(ds / bundle, k)[:8000], units=render(us)))
    except RuntimeError as exc:  # the API gave up: recorded, a rerun asks again
        return rec | {"units": us, "error": str(exc)[:300]}
    try:
        return rec | {"units": us, "verdict": verdict(text, us), "raw": text}
    except (ValueError, json.JSONDecodeError) as exc:
        return rec | {"units": us, "error": f"{type(exc).__name__}: {exc}"[:300], "raw": text}


def judge(ds: str, out: str, only: list, workers: int = 16, limit: int = 0):
    """Label every task of DATASET_DIR (its .manifest.jsonl), or the BUNDLE:K ones; resumable (OUT is appended)."""
    ds = Path(ds)
    done = {(r["bundle"], r["k"]) for r in map(json.loads, open(out)) if "error" not in r} if Path(out).exists() else set()
    todo = [(r["bundle"], t["k"], t["instance_id"], t["kind"]) for r in map(json.loads, open(f"{ds}.manifest.jsonl"))
            for t in r["tasks"] if (r["bundle"], t["k"]) not in done and (not only or f"{r['bundle']}:{t['k']}" in only)]
    todo = todo[:limit] if limit else todo
    print(len(todo), "tasks", flush=True)
    with open(out, "a") as f, concurrent.futures.ThreadPoolExecutor(workers) as pool:
        for done in concurrent.futures.as_completed([pool.submit(judge_task, ds, *t) for t in todo]):
            rec = done.result()
            f.write(json.dumps(rec) + "\n")
            f.flush()
            labels = [v["label"] for v in (rec.get("verdict") or {}).values()]
            print(rec["bundle"], rec["k"], rec.get("error") or dict((l, labels.count(l)) for l in set(labels)), flush=True)


CODE_BLOCK = re.compile(r"```(?:python|py|python3)[ \t]*\n(.*?)\n```", re.S)


def code_blocks(text: str) -> list:
    """The python code blocks of a statement, as written."""
    return [m.group(1) for m in CODE_BLOCK.finditer(text)]


SNIPPET_RUN = r"""import json, subprocess, sys
blocks = json.load(open(sys.argv[1]))  # [[k, j, code]]
out = []
for k, j, code in blocks:
    open("/tmp/snip.py", "w").write(code)
    try:
        r = subprocess.run(["/opt/miniconda3/envs/testbed/bin/python", "/tmp/snip.py"], cwd="/testbed", timeout=30,
                           stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        out.append([k, j, r.returncode, r.stdout.decode(errors="replace")[-800:]])
    except subprocess.TimeoutExpired:
        out.append([k, j, "timeout", ""])
print(json.dumps(out))
"""


async def run_snippets(d: Path, image: str, blocks: list, slot) -> dict:
    """blocks [[k, j, code]] on bundle dir d's tree with every reference fix in (fixed) and on its initial tree
    (bug): {"fixed": [[k, j, exit, output]], "bug": [...]} (or {"error"})."""
    import asyncio, tempfile
    from octopus.envs.swe.data.validate_tasks import make_env
    setup = tomllib.loads((d / "task.toml").read_text())["environment"]["healthcheck"]["command"]
    async with slot:
        env = make_env(image)
        try:
            await env.start(False)
            done = await env.exec(setup, timeout_sec=300)
            if done.return_code:
                return {"error": f"setup rc {done.return_code}: {(done.stdout + done.stderr)[-300:]}"}
            with tempfile.TemporaryDirectory() as tmp:
                Path(tmp, "blocks.json").write_text(json.dumps(blocks))
                Path(tmp, "run.py").write_text(SNIPPET_RUN)
                await env.upload_dir(tmp, "/tmp/snippets")
            await env.upload_dir(d / "solution", "/solution")
            res = {}
            for side, prep in (("fixed", "bash /solution/solve.sh"), ("bug", "true")):
                r = await env.exec(f"cd /testbed && git reset -q --hard && git clean -fdq && {prep} && "
                                   "python3 /tmp/snippets/run.py /tmp/snippets/blocks.json", timeout_sec=60 * len(blocks) + 300)
                res[side] = json.loads((r.stdout or "").strip().splitlines()[-1])
            return res
        except Exception as exc:
            return {"error": f"{type(exc).__name__}: {exc}"[:300]}
        finally:
            await env.stop(True)


async def snippets(ds: str, out: str, conc: int = 16, overrides: str = ""):
    """The python code blocks of every statement of DATASET_DIR (OVERRIDES: instance id -> statement text first), all
    of a statement's blocks as one script, run on the bundle tree with every fix in and on its initial tree ->
    OUT.jsonl {bundle, blocks: [{k, j=0, instance_id, fixed: [exit, output], bug: [exit, output]}]}. A statement whose
    code does not exit 0 with every fix in is broken: main copies it into its checks (bundle-e6 audit)."""
    import asyncio
    ds = Path(ds)
    over = json.loads(Path(overrides).read_text()) if overrides else {}
    done = {json.loads(l)["bundle"] for l in open(out)} if Path(out).exists() else set()
    sem = asyncio.Semaphore(conc)

    async def one(row):
        blocks, ids = [], {}
        for t in row["tasks"]:  # a statement's blocks run as one script: later ones often continue earlier ones
            text = over.get(t["instance_id"]) or statement(ds / row["bundle"], t["k"])
            if code_blocks(text):
                blocks.append([t["k"], 0, "\n\n".join(code_blocks(text))])
                ids[t["k"]] = t["instance_id"]
        if not blocks:
            rec = {"bundle": row["bundle"], "blocks": []}
        else:
            res = await run_snippets(ds / row["bundle"], row["image"], blocks, sem)
            rec = {"bundle": row["bundle"], "error": res["error"]} if "error" in res else {"bundle": row["bundle"], "blocks": [
                {"k": k, "j": j, "instance_id": ids[k], "fixed": [fe, fo], "bug": [be, bo]}
                for (k, j, fe, fo), (_, _, be, bo) in zip(res["fixed"], res["bug"])]}
        with open(out, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(row["bundle"], rec.get("error") or f"{sum(b['fixed'][0] != 0 for b in rec['blocks'])}/{len(rec['blocks'])} broken",
              flush=True)

    rows = [json.loads(l) for l in open(f"{ds}.manifest.jsonl")]
    await asyncio.gather(*(one(r) for r in rows if r["bundle"] not in done))


REPAIR_PROMPT = """A task statement for a Python repository holds {n} python code block(s), run in order as one script:
<statement>
{statement}
</statement>
Run with the bug fixed (the repository's correct code), that script does not exit cleanly:
<output_on_fixed_code>
{fixed}
</output_on_fixed_code>
Run on the buggy code it gives:
<output_on_buggy_code>
{bug}
</output_on_buggy_code>

Rewrite the code so that the blocks, run in order as one script, are self-contained (no files, network or tools the
script does not create itself), use only the repository's real API, exit 0 on the correct code, and still show the
problem the statement describes on the buggy code (an exception, a failed assert, or wrong printed output). Keep them
as close to the originals as you can and do not put the fixed library code into them. Answer with exactly {n} ```python
blocks in the same order (one per original block), or a single ```python block holding only NONE if the statement's
problem cannot be shown with self-contained code."""


def repair_one(ds: Path, rec: dict, block: dict) -> dict:
    """{"new": [the statement's blocks rewritten, in order]} or {"new": None} (NONE, or a wrong block count)."""
    from octopus.envs.swe.data.acceptance_rewrite import ask
    text = statement(ds / rec["bundle"], block["k"])
    n = len(code_blocks(text))
    try:
        answer = ask(REPAIR_PROMPT.format(statement=text[:6000], n=n, fixed=block["fixed"][1][-600:],
                                          bug=block["bug"][1][-600:]))
    except RuntimeError as exc:
        return {"error": str(exc)[:200]}
    new = [b.strip() for b in code_blocks(answer)]
    return {"new": new if len(new) == n and not (n == 1 and new[0].upper() == "NONE") else None}


def repair(ds: str, runs: str, out: str, workers: int = 32):
    """Grok rewrites of every broken statement's code in RUNS (snippets' output) -> OUT.jsonl {instance_id, k, new}
    (new None: no self-contained code); snippets with these as overrides then shows which ones hold."""
    ds = Path(ds)
    todo = [(rec, b) for rec in map(json.loads, open(runs)) for b in rec.get("blocks") or [] if b["fixed"][0] != 0]
    done = {r["instance_id"] for r in map(json.loads, open(out)) if "error" not in r} if Path(out).exists() else set()
    todo = [(rec, b) for rec, b in todo if b["instance_id"] not in done]
    print(len(todo), "broken statements", flush=True)
    with open(out, "a") as f, concurrent.futures.ThreadPoolExecutor(workers) as pool:
        futures = {pool.submit(repair_one, ds, rec, b): (rec, b) for rec, b in todo}
        for fut in concurrent.futures.as_completed(futures):
            rec, b = futures[fut]
            r = {"bundle": rec["bundle"], "instance_id": b["instance_id"], "k": b["k"], **fut.result()}
            f.write(json.dumps(r) + "\n")
            f.flush()


def overrides(ds: str, repairs: str, out: str, checked: str = ""):
    """Statement overrides (instance id -> text): the repaired blocks put in place, or every block removed where Grok
    answered NONE; with CHECKED (snippets' run of these overrides) a repair still broken with every fix in, or one
    that copies reference-fix code, has its blocks removed instead."""
    from octopus.envs.swe.data.build_bundles import fix_lines
    ds = Path(ds)
    bad = set()
    if checked:
        for rec in map(json.loads, open(checked)):
            bad |= {(b["instance_id"], b["j"]) for b in rec.get("blocks") or [] if b["fixed"][0] != 0}
    result = {}
    for r in map(json.loads, open(repairs)):
        if "error" in r:
            continue
        text, iid = statement(ds / r["bundle"], r["k"]), r["instance_id"]
        fix = (ds / r["bundle"] / f"solution/fix_{r['k']}.patch").read_text()
        new = r["new"]
        if new is not None and ((iid, 0) in bad or any(l in "\n".join(new) for l in fix_lines(fix))):
            new = None
        for j, m in reversed(list(enumerate(CODE_BLOCK.finditer(text)))):
            repl = "(The reproduction code that was here is omitted.)" if new is None else f"```python\n{new[j]}\n```"
            text = text[:m.start()] + repl + text[m.end():]
        result[iid] = text
    Path(out).write_text(json.dumps(result, indent=1))
    print(len(result), "statements rewritten", flush=True)


if __name__ == "__main__":
    cmd, args = sys.argv[1], sys.argv[2:]
    opt = lambda name, default: int(args[args.index(name) + 1]) if name in args else default
    plain = [a for i, a in enumerate(args) if not a.startswith("--") and (i == 0 or not args[i - 1].startswith("--"))]
    if cmd == "judge":
        judge(plain[0], plain[1], plain[2:], opt("--workers", 16), opt("--limit", 0))
    elif cmd == "snippets":  # DATASET_DIR OUT.jsonl [--conc N] [OVERRIDES.json]
        import asyncio
        asyncio.run(snippets(plain[0], plain[1], opt("--conc", 16), plain[2] if len(plain) > 2 else ""))
    elif cmd == "repair":  # DATASET_DIR RUNS.jsonl OUT.jsonl [--workers N]
        repair(plain[0], plain[1], plain[2], opt("--workers", 32))
    elif cmd == "overrides":  # DATASET_DIR REPAIRS.jsonl OUT.json [CHECKED.jsonl]
        overrides(plain[0], plain[1], plain[2], plain[3] if len(plain) > 3 else "")
