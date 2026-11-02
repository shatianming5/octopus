"""Rebuild a task pool's manifest.jsonl and splits/ from its task dirs.
usage: build_pool.py POOL_DIR          (POOL_DIR/tasks/<instance_id>/{task.toml, solution/fix.patch, pool.json})

A task dir says what the Harbor task says (image, reference fix); pool.json (written by pool_import.py when the task
entered the pool) says the rest: source_batch, f2p, bug_failing and bug_failed_n (tests failing in the nop run),
statement_covered (multi-bug: every hunk named in the statement), base_pass_rate and c2s60_pass_rate where known.
Splits: eval = 200 single-bug + 100 multi-bug tasks drawn with random.Random(0), stratified by (kind, repo) with
largest-remainder quotas; train = everything else. Deterministic for a given set of task dirs."""
import collections, json, random, re, sys, tomllib
from pathlib import Path

EVAL = {"single": 200, "multi": 100}
SYM = re.compile(r"\b(?:def|class)\s+([A-Za-z_]\w*)")


def patch_stats(text: str):
    """files, hunks, functions of a unified diff; a hunk's function is the nearest def/class above its first change
    (else the symbols of its header and changed signature lines), as in the multi-bug text screen (mb_cover.py)."""
    files, hunks, f = set(), [], None
    for l in text.splitlines():
        if l.startswith("diff --git"):
            f = l.split(" b/")[-1]; files.add(f)
        elif l.startswith("@@"):
            hunks.append({"file": f, "names": set(SYM.findall(l.split("@@", 2)[-1])), "ctx": None, "changed": False})
        elif hunks and l[:1] in " +-" and not l.startswith(("+++", "---")):
            h, m = hunks[-1], SYM.findall(l[1:])
            if l[0] == " ":
                if m and not h["changed"]: h["ctx"] = m[-1]
            else:
                h["changed"] = True
                if m: h["names"].update(m)
    return len(files), len(hunks), len({(h["file"], h["ctx"] or tuple(sorted(h["names"]))) for h in hunks})


def row(task: Path) -> dict:
    iid = task.name
    kind = re.search(r"\.([a-z_]+?)__[a-z0-9]+$", iid).group(1)
    files, hunks, functions = patch_stats((task / "solution/fix.patch").read_text(errors="replace"))
    return {"instance_id": iid, "kind": kind, "bug_type": "multi" if kind.startswith("combine_") else "single",
            "repo": iid.split(".")[0], "image": tomllib.loads((task / "task.toml").read_text())["environment"]["docker_image"],
            "files": files, "functions": functions, "hunks": hunks, **json.loads((task / "pool.json").read_text())}


def choose_eval(rows, n, rng):
    strata = collections.defaultdict(list)
    for r in rows:
        strata[(r["kind"], r["repo"])].append(r["instance_id"])
    keys = sorted(strata); exact = {k: n * len(strata[k]) / len(rows) for k in keys}
    quota = {k: int(exact[k]) for k in keys}
    for k in sorted(keys, key=lambda k: quota[k] - exact[k])[: n - sum(quota.values())]:
        quota[k] += 1
    return {t for k in keys for t in rng.sample(sorted(strata[k]), quota[k])}


def main(pool: Path):
    rows = sorted((row(t) for t in (pool / "tasks").iterdir() if (t / "task.toml").is_file()), key=lambda r: r["instance_id"])
    rng, ev = random.Random(0), set()
    for bt in ("single", "multi"):
        ev |= choose_eval([r for r in rows if r["bug_type"] == bt], EVAL[bt], rng)
    assert len(ev) == sum(EVAL.values()), len(ev)
    for r in rows:
        r["split"] = "eval" if r["instance_id"] in ev else "train"
    (pool / "manifest.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (pool / "splits").mkdir(exist_ok=True)
    for s in ("train", "eval"):
        (pool / "splits" / f"{s}.txt").write_text("".join(r["instance_id"] + "\n" for r in rows if r["split"] == s))
    c = collections.Counter((r["kind"], r["split"]) for r in rows)
    print(f"{len(rows)} tasks, eval {len(ev)}; train+eval per kind: "
          + ", ".join(f"{k} {c[k, 'train']}+{c[k, 'eval']}" for k in sorted({r['kind'] for r in rows})))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
