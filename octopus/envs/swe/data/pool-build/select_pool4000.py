"""Select a 4000-task mix: 1000 single-bug + 3000 multi-bug.

usage: select_pool4000.py DATASETS_DIR PARQUET_DIR OUT_DIR

Reads remaining swesmith-*-final dirs (skips *-unsound). Drops empty/tiny patches,
multi-bug tasks whose statement does not name every patched hunk (statement_covers),
and the longest 5% by statement+patch length. Then fills even kind quotas, and
inside each kind even difficulty terciles and even remaining-length terciles,
with a per-repo cap. Seed Random(0). If fewer than 3000 multi remain after
filters, takes all of them and shrinks single to keep 1:3 (total < 4000).
"""
from __future__ import annotations

import collections, json, math, os, random, re, sys
from pathlib import Path

TARGET = {"single": 1000, "multi": 3000}
SEED = 0
PATCH_MIN = 80
LONG_Q = 0.95
REPO_CAP_FRAC = 0.05
GENERIC = {"__init__", "base", "utils", "util", "core", "main", "common", "helpers", "api", "models", "types", "compat"}
SYM = re.compile(r"\b(?:def|class)\s+([A-Za-z_]\w*)")
KIND_RE = re.compile(r"\.([a-z_]+?)__[a-z0-9]+$")


def patch_stats(text: str):
    files, hunks, f = set(), [], None
    lines = 0
    for l in text.splitlines():
        if l.startswith(("+++", "---", "diff --git", "index ", "new file", "deleted file")):
            if l.startswith("diff --git"):
                f = l.split(" b/")[-1]
                files.add(f)
            continue
        if l.startswith("@@"):
            hunks.append({"file": f, "names": set(SYM.findall(l.split("@@", 2)[-1])), "ctx": None, "changed": False})
            continue
        if l[:1] in " +-":
            lines += 1
            if hunks:
                h, m = hunks[-1], SYM.findall(l[1:])
                if l[0] == " ":
                    if m and not h["changed"]:
                        h["ctx"] = m[-1]
                else:
                    h["changed"] = True
                    if m:
                        h["names"].update(m)
    funcs = {(h["file"], h["ctx"] or tuple(sorted(h["names"]))) for h in hunks}
    return len(files), len(hunks), len(funcs), lines, hunks


def statement_covers(stmt: str, hunks) -> bool:
    """Every patched hunk is named in the statement: its enclosing def/class, a def/class it changes, or its file stem
    (mb_cover.py's rule). A multi-bug statement that names only some hunks describes only some of the bugs, while the
    hidden tests grade all of them."""
    for h in hunks:
        stem = os.path.splitext(os.path.basename(h["file"] or ""))[0]
        names = {n for n in h["names"] | {h["ctx"], stem} if n and n not in GENERIC and len(n) >= 3}
        if not any(re.search(rf"(?<![A-Za-z0-9_]){re.escape(n)}(?![A-Za-z0-9_])", stmt) for n in names):
            return False
    return True


def tercile(values):
    xs = sorted(values)
    n = len(xs)
    if n < 3:
        return {x: 1 for x in xs}
    a, b = xs[n // 3], xs[(2 * n) // 3]
    out = {}
    for x in xs:
        out[x] = 0 if x <= a else (1 if x <= b else 2)
    return out


def even_quota(keys, n, available, leftover_order=None):
    keys = list(keys)
    if not keys or n <= 0:
        return {k: 0 for k in keys}
    cap = {k: available[k] for k in keys}
    q = {k: 0 for k in keys}
    # equal share, then largest remainder, never above cap; unused slots flow to kinds with room
    while n > 0:
        open_k = [k for k in keys if q[k] < cap[k]]
        if not open_k:
            break
        base, rem = divmod(n, len(open_k))
        if base == 0:
            order = leftover_order or sorted(open_k)
            progressed = False
            for k in order:
                if n == 0:
                    break
                if q[k] < cap[k]:
                    q[k] += 1
                    n -= 1
                    progressed = True
            if not progressed:
                break
            continue
        for k in open_k:
            take = min(base, cap[k] - q[k])
            q[k] += take
            n -= take
    return q


def nested_sample(rows, n, rng):
    """Even kind, then even (diff_t, len_t), then repo-capped draw."""
    if n <= 0 or not rows:
        return []
    by_kind = collections.defaultdict(list)
    for r in rows:
        by_kind[r["kind"]].append(r)
    kq = even_quota(sorted(by_kind), n, {k: len(v) for k, v in by_kind.items()})
    picked = []
    repo_cap = max(1, math.ceil(n * REPO_CAP_FRAC))
    repo_used = collections.Counter()
    for kind in sorted(by_kind):
        want = kq[kind]
        pool = list(by_kind[kind])
        if want <= 0:
            continue
        dmap = tercile([r["diff"] for r in pool])
        lmap = tercile([r["length"] for r in pool])
        for r in pool:
            r["_cell"] = (dmap[r["diff"]], lmap[r["length"]])
        cells = collections.defaultdict(list)
        for r in pool:
            cells[r["_cell"]].append(r)
        cq = even_quota(sorted(cells), want, {c: len(v) for c, v in cells.items()})
        kind_got = []
        deferred = []
        for cell in sorted(cells):
            grp = sorted(cells[cell], key=lambda r: r["instance_id"])
            take = cq[cell]
            rng.shuffle(grp)
            chosen, rest = [], []
            for r in grp:
                if len(chosen) < take and repo_used[r["repo"]] < repo_cap:
                    chosen.append(r)
                    repo_used[r["repo"]] += 1
                else:
                    rest.append(r)
            kind_got.extend(chosen)
            deferred.extend(rest)
        if len(kind_got) < want:
            rng.shuffle(deferred)
            for r in deferred:
                if len(kind_got) >= want:
                    break
                if repo_used[r["repo"]] < repo_cap or all(repo_used[x["repo"]] >= repo_cap for x in deferred):
                    kind_got.append(r)
                    repo_used[r["repo"]] += 1
        picked.extend(kind_got[:want])
    # fill shortfall from remaining of this bug_type
    have = {r["instance_id"] for r in picked}
    if len(picked) < n:
        rest = [r for r in rows if r["instance_id"] not in have]
        rng.shuffle(rest)
        for r in rest:
            if len(picked) >= n:
                break
            if repo_used[r["repo"]] < repo_cap * 2:
                picked.append(r)
                repo_used[r["repo"]] += 1
                have.add(r["instance_id"])
        for r in rest:
            if len(picked) >= n:
                break
            if r["instance_id"] not in have:
                picked.append(r)
    return picked[:n]


def load_parquet(pq: Path):
    import pandas as pd
    files = sorted(pq.glob("*.parquet"))
    df = pd.concat(pd.read_parquet(f, columns=["instance_id", "FAIL_TO_PASS", "problem_statement"]) for f in files)
    stmt, f2p = {}, {}
    for r in df.itertuples(index=False):
        stmt[r.instance_id] = (r.problem_statement or "").strip()
        f2p[r.instance_id] = len(r.FAIL_TO_PASS) if r.FAIL_TO_PASS is not None else 0
    return stmt, f2p


def load_finals(datasets: Path, stmt, f2p):
    rows, skipped = [], collections.Counter()
    for ds in sorted(datasets.glob("swesmith-*-final")):
        if ds.name.endswith("-unsound-final"):
            continue
        batch = ds.name[len("swesmith-"):-len("-final")]
        for t in ds.iterdir():
            if not t.is_dir():
                continue
            patch = t / "solution/fix.patch"
            if not patch.is_file():
                skipped["no_patch"] += 1
                continue
            text = patch.read_text(errors="replace")
            if len(text.encode()) < PATCH_MIN:
                skipped["tiny_patch"] += 1
                continue
            m = KIND_RE.search(t.name)
            if not m:
                skipped["bad_id"] += 1
                continue
            kind = m.group(1)
            files, hunks, funcs, plines, hunk_list = patch_stats(text)
            s = stmt.get(t.name, "")
            if not s:
                inst = t / "instruction.md"
                s = inst.read_text(errors="replace").strip() if inst.is_file() else ""
            if not s:
                skipped["no_stmt"] += 1
                continue
            if kind.startswith("combine_") and not statement_covers(s, hunk_list):
                skipped["multi_statement_uncovered"] += 1
                continue
            length = len(s) + 4 * plines
            diff = hunks + files + funcs + math.log1p(f2p.get(t.name, 0))
            rows.append({
                "instance_id": t.name, "kind": kind,
                "bug_type": "multi" if kind.startswith("combine_") else "single",
                "repo": t.name.split(".")[0], "batch": batch,
                "files": files, "hunks": hunks, "functions": funcs,
                "patch_lines": plines, "stmt_chars": len(s),
                "f2p": f2p.get(t.name, 0), "length": length, "diff": diff,
            })
    return rows, skipped


def drop_long(rows):
    if not rows:
        return rows, 0
    xs = sorted(r["length"] for r in rows)
    cut = xs[min(len(xs) - 1, int(len(xs) * LONG_Q))]
    keep = [r for r in rows if r["length"] <= cut]
    return keep, len(rows) - len(keep)


def summarize(rows):
    kinds = collections.Counter(r["kind"] for r in rows)
    repos = collections.Counter(r["repo"] for r in rows)
    lens = sorted(r["length"] for r in rows) or [0]
    diffs = sorted(r["diff"] for r in rows) or [0]
    def pct(xs, p):
        return xs[min(len(xs) - 1, int(round((len(xs) - 1) * p)))]
    return {
        "n": len(rows),
        "kinds": dict(sorted(kinds.items())),
        "n_repos": len(repos),
        "top_repos": repos.most_common(8),
        "length_p10_p50_p90_p95": [pct(lens, p) for p in (0.1, 0.5, 0.9, 0.95)],
        "diff_p10_p50_p90": [pct(diffs, p) for p in (0.1, 0.5, 0.9)],
        "hunks_mean": round(sum(r["hunks"] for r in rows) / max(len(rows), 1), 2),
        "f2p_mean": round(sum(r["f2p"] for r in rows) / max(len(rows), 1), 2),
    }


def main(datasets: Path, parquet: Path, out: Path):
    stmt, f2p = load_parquet(parquet)
    rows, skipped = load_finals(datasets, stmt, f2p)
    by_bt = {"single": [r for r in rows if r["bug_type"] == "single"],
             "multi": [r for r in rows if r["bug_type"] == "multi"]}
    dropped_long = {}
    filtered = {}
    for bt, rs in by_bt.items():
        keep, nlong = drop_long(rs)
        dropped_long[bt] = nlong
        filtered[bt] = keep

    n_multi = min(TARGET["multi"], len(filtered["multi"]))
    if n_multi < TARGET["multi"]:
        n_single = min(TARGET["single"], n_multi // 3, len(filtered["single"]))
    else:
        n_single = min(TARGET["single"], len(filtered["single"]))
    rng = random.Random(SEED)
    picked = {
        "single": nested_sample(filtered["single"], n_single, rng),
        "multi": nested_sample(filtered["multi"], n_multi, rng),
    }
    out.mkdir(parents=True, exist_ok=True)
    selected = []
    for bt in ("single", "multi"):
        ids = sorted(r["instance_id"] for r in picked[bt])
        (out / f"{bt}.txt").write_text("".join(x + "\n" for x in ids))
        selected.extend(ids)
    (out / "selected.txt").write_text("".join(x + "\n" for x in sorted(selected)))
    # every chosen multi-bug task passed statement_covers: pool_import.py's MB_MANIFEST format
    (out / "cover.json").write_text(json.dumps({r["instance_id"]: {"uncovered_hunks": 0} for r in picked["multi"]}))
    vs_inventory = {}
    for bt in ("single", "multi"):
        dmap = tercile([r["diff"] for r in filtered[bt]]) if filtered[bt] else {}
        lmap = tercile([r["length"] for r in filtered[bt]]) if filtered[bt] else {}
        vs_inventory[bt] = {
            "diff_terciles": dict(collections.Counter(dmap[r["diff"]] for r in picked[bt])),
            "len_terciles": dict(collections.Counter(lmap[r["length"]] for r in picked[bt])),
        }
    report = {
        "seed": SEED, "target": TARGET, "chosen": {bt: len(picked[bt]) for bt in picked},
        "ratio_single_multi": f"1:{len(picked['multi']) / max(len(picked['single']), 1):.3f}",
        "skipped_load": dict(skipped), "dropped_long_p95": dropped_long,
        "available_after_filters": {bt: len(filtered[bt]) for bt in filtered},
        "inventory_before_length_drop": {bt: summarize(by_bt[bt]) for bt in by_bt},
        "chosen_summary": {bt: summarize(picked[bt]) for bt in picked},
        "chosen_vs_inventory_terciles": vs_inventory,
        "short_of_3000_multi": TARGET["multi"] - len(picked["multi"]),
        "short_of_4000": 4000 - (len(picked["single"]) + len(picked["multi"])),
    }
    (out / "report.json").write_text(json.dumps(report, indent=2, default=str) + "\n")
    print(json.dumps({
        "chosen": report["chosen"], "ratio": report["ratio_single_multi"],
        "available": report["available_after_filters"],
        "dropped_long": dropped_long, "skipped": dict(skipped),
        "short_multi": report["short_of_3000_multi"], "short_total": report["short_of_4000"],
        "kinds_single": report["chosen_summary"]["single"]["kinds"],
        "kinds_multi": report["chosen_summary"]["multi"]["kinds"],
        "out": str(out),
    }, indent=2, default=str))


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))
