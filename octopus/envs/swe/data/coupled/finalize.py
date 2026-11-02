"""Track A final datasets from the samefile builds (node-94, areal python, PYTHONPATH=<code>, env.sh sourced).

finalize.py TRAIN[,TRAIN...] EVAL[,EVAL...] OUT [--plan] [--keep-split REPORT]
  bundles accepted by `accept` (NAME.acceptance-1c.json), timed by `timing` on NAME.accepted.jsonl (none and all
  within LIMIT s), then per bundle on its real healthcheck tree (golden repo clone): no reference-fix line / F2P test
  left (bb.leaks), every task's fix changes the same one .py file, each task k >= 2 collides with a task before it
  through 20 lines / one function / one class and some pair is within 20 lines (bb.coupled), that file is no task's hidden test, the package imports from /testbed (baseline). The EVAL builds' repos go to
  val / test (the split of the repos with the closest bundle counts, then the most repos on the smaller side, ties to
  the seed); at most 10 bundles per train repo and 20 per eval repo, the first in plan order. -> datasets/OUT-{train,val,test}/ + .manifest.jsonl (split field), datasets/OUT.report.json
"""
import collections, glob, json, os, random, re, shutil, statistics, subprocess, sys, tempfile, tomllib, zipfile
from pathlib import Path

import pyarrow.parquet as pq

import octopus.envs.swe.data.build_bundles as bb
from octopus.envs.swe.data.regen_tasks import N, golden_repo

LIMIT, SEED, CAP = 180, "coupled-a", {"train": 10, "eval": 20}
# left out by R1 review: a task not solvable from its statement; a repo whose code ships in the standard library
DROP = {"lincolnloop__python-qrcode.456b01d4.sf16": "task 2 cannot be solved from its statement (R1)",
        "mahmoud__glom.fb3c4e76.sf15": "task 1: the statement asks for the usage message the bug prints, the tests want {} (R1)",
        "mahmoud__glom.fb3c4e76.sf14": "task 1: the tests want exact reprs ('(M == 1) & float', '~(M) & M') that neither the "
                                       "statement nor the tree gives (R1)",
        # the tests assert an exact message that the reference fix adds and the statement does not give
        "joke2k__faker.8b401a7d.sf05": "task 1: TypeError 'The locale \"1\" must be a string.' (the statement says AttributeError)",
        "django__daphne.32ac73e1.sf00": "task 2: \"Header name 'foo' expected to be `bytes`, but got `<class 'str'>`\" and the value one",
        "facelessuser__soupsieve.a8080d97.sf08": "task 3: \"The pseudo class ':contains' is deprecated, ':-soup-contains' should be "
                                                 "used moving forward.\"",
        "pycqa__flake8.cf1542ce.sf07": "task 2: e.args == ('expected loaded plugin to be callable',)",
        "lincolnloop__python-qrcode.456b01d4.sf12": "tasks 1, 2: escape codes '\\x1b[48;5;232m', '\\x1b[1;47m' and the border "
                                                    "width, in neither the statements nor the tree",
        "dbader__schedule.82a43db1.sf01": "task 1: repr '... do [None] ...' of a job without a function, in neither the "
                                          "statement nor the tree",
        "jaraco__inflect.c079a96a.sf08": "task 2: join(..., conj='') gives one space ('apple carrot'), in neither the statement, "
                                         "the docstring, the README nor the tree (whose code gives two)",
        # full solvability audit (PR #226 judge + per-task review, coupled-solv/full/review.tsv): unsolvable or the
        # statement contradicts the tests
        "agronholm__typeguard.b6a7e438.sf25": "task 1: the too-few / keyword-only argument messages the tests match are "
                                              "in neither the statement nor the tree",
        "agronholm__typeguard.b6a7e438.sf10": "task 2: the 'is not compatible with the P protocol because' wording is in "
                                              "neither the statement nor the tree",
        "matthewwithanm__python-markdownify.6258f5c3.sf07": "task 1: escape_misc's exact per-character rules are not "
                                                            "derivable from the statement or README",
        "r1chardj0n3s__parse.30da9e4f.sf01": "task 1: the tests want NotImplementedError for an invalid group name; the "
                                             "statement says only 'a proper error'",
        "erikrose__parsimonious.0d3f5f93.sf07": "task 1: the statement calls the exact message the tests want the wrong output",
        "facelessuser__soupsieve.a8080d97.sf07": "task 2: the statement expects button2 as the default, the tests want no match",
        "facebookresearch__fvcore.a491d5b9.sf01": "task 2: the statement asks for 2/4 ops per element, the tests keep 4/5",
        "john-kurkowski__tldextract.3d1bf184.sf00": "task 1: the statement implies extract('://example.com') should parse, the "
                                                    "tests want it empty",
        "mahmoud__glom.fb3c4e76.sf17": "task 1: the statement implies S() should succeed, the tests want TypeError",
        "oauthlib__oauthlib.1fd52536.sf08": "task 2: the statement calls the space-separated header correct, the tests want ', '",
        "python-jsonschema__jsonschema.93e0caa5.sf13": "task 2: the statement wants no deprecation warning, the tests assert one",
        "spulec__freezegun.5f171db0.sf00": "task 2: the statement asks for the bug's behaviour (inject as_kwarg='')",
        "cookiecutter__cookiecutter.b4451231.sf00": "task 2: 'A valid repository for \"{}\" could not be found in the following "
                                                    "locations:\\n{}' (the tree has another wording; the statement none)",
        # the exact assert messages the fix adds are not tested; what is: 'src_image=...' for a value pformat puts on
        # several lines, in neither the statement nor the tree
        "facebookresearch__fvcore.a491d5b9.sf02": "task 2: 'BlendTransform(src_image=..., ...)' (multi-line values shown as "
                                                  "...), in neither the statement nor the tree"}
DROP_REPOS = {"hukkin__tomli.443a0c1b": "Python 3.12's tomllib is the same code (standard library, cannot be removed)",
              "pygments__pygments.27649ebb": "pip vendors pygments (its reference-fix lines are in pip/_vendor and pip's "
                                             "wheels); deleting them leaves pip unusable"}
INDEX = {}


def root_files(root: Path) -> tuple:
    """A golden ROOT's .py files by name and its archives (.whl/.zip/.egg, and every file of a cache dir: pip keeps
    wheels under hashed names), /testbed left out; once per ROOT."""
    if root not in INDEX:
        py, arc = collections.defaultdict(list), []
        for dp, dns, fns in os.walk(root):
            if dp == str(root):
                dns[:] = [x for x in dns if x not in ("testbed", "proc", "sys", "dev")]
            for fn in fns:
                if fn.endswith(".py"):
                    py[fn].append(os.path.join(dp, fn))
                elif fn.endswith((".whl", ".zip", ".egg")) or f"{os.sep}.cache{os.sep}" in dp + os.sep or dp.startswith(f"{root}/tmp"):
                    arc.append(os.path.join(dp, fn))
        INDEX[root] = py, arc
    return INDEX[root]


def outside_copies(root: Path, f: str, tasks: list) -> list:
    """Copies of the bug file (same package path) outside /testbed, as files or archive members, that the healthcheck's
    clean_globs leave and that hold a task's reference-fix line (not one already in its own tree)."""
    removed = [p for g in bb.clean_globs(bb.package(f)) for p in glob.glob(f"{root}{g}")]
    keep = lambda p: not any(p == r or p.startswith(r + "/") for r in removed)
    gold = set().union(*(bb.fix_lines(c["fix"]) - c["inherent"] for c in tasks))
    tail, (py, arc), hits = f.removeprefix("src/"), root_files(root), []
    for p in py.get(Path(f).name, []):
        if p.endswith("/" + tail) and keep(p) and gold & {l.strip() for l in open(p, errors="ignore")}:
            hits.append(p)
    for a in filter(keep, arc):
        try:
            z = zipfile.ZipFile(a)
        except (zipfile.BadZipFile, OSError):
            continue
        hits += [f"{a}:{n}" for n in z.namelist() if n.endswith("/" + tail) or n == tail
                 if gold & {l.strip() for l in z.read(n).decode(errors="ignore").splitlines()}]
    return hits
W = "/mnt/disk6/user/coupled-s1a"
DS = N / "datasets"


def trained_repos() -> set:
    """repo keys of the e8/e11 bundle training lists that e9-e14 trained on (trained_repos.txt, 73 repos incl. gunicorn,
    mistune, h11)"""
    return set(open(f"{W}/trained_repos.txt").read().split())


def scan(d: Path, row: dict, rows: dict, base: dict) -> list:
    """the bundle as shipped: its healthcheck tree and its solution/fix_k.patch (cut ones after reduce)"""
    repo = golden_repo(row["image"])
    tasks = [bb.candidate(repo, rows[t["instance_id"]]) for t in row["tasks"]]
    files = {tuple(c["bug_files"]) for c in tasks}
    if len(files) != 1 or len(next(iter(files))) != 1 or not next(iter(files))[0].endswith(".py"):
        return [f"bugs do not change one same .py file: {files}"]
    f, why = next(iter(files))[0], []
    for t, c in zip(row["tasks"], tasks):
        c["fix"] = (d / f"solution/fix_{t['k']}.patch").read_text()
        touched = set(re.findall(r"^\+\+\+ b/(.+)$", c["fix"], re.M))
        if touched != {f}:
            why.append(f"task {t['k']} fix touches {sorted(touched)}")
        if bb.change_notes(c["fix"]):
            why.append(f"task {t['k']} bug comment names the change: {bb.change_notes(c['fix'])[:2]}")
        if sum(l.startswith("+") and not l.startswith("+++") for l in c["fix"].splitlines()) >= 250:
            why.append(f"task {t['k']} bug deletes >= 250 lines")
    if any(f in c["deleted"] + c["modified"] for c in tasks) or (d / "tests/hidden_full" / f).exists():
        why.append(f"bug file {f} is a hidden test")
    if not base["import_path"].startswith("/testbed/") or base["site_copies"]:
        why.append(f"package imports from {base['import_path']}")
    setup = tomllib.loads((d / "task.toml").read_text())["environment"]["healthcheck"]["command"]
    with tempfile.TemporaryDirectory() as tmp:
        tree = Path(tmp) / "testbed"
        subprocess.run(["git", "clone", "-q", "--shared", str(repo), str(tree)], check=True)
        r = subprocess.run(["bash", "-c", setup.replace("/testbed", str(tree)).replace("/tasks", f"{tmp}/tasks")],
                           capture_output=True, text=True)
        if r.returncode:
            return why + [f"healthcheck rc {r.returncode}: {r.stderr[-200:]}"]
        src = (tree / f).read_text()
        why += [f"a fix does not apply on the shipped tree: {c['instance_id']}" for c in bb.shipped_hunks(tree, tasks)]
        for t, c in zip(row["tasks"], tasks):
            if str((row.get("cut") or {}).get(t["instance_id"], "")).startswith("cut"):  # as reduce_repo
                c["inherent"] = c["inherent"] | (bb.fix_lines(c["fix"]) & {l.strip() for l in src.splitlines()})
        if not bb.coupled(tasks):
            why.append("not coupled: tasks not connected through 20 lines / one function / one class, or no pair within 20 lines")
        why += bb.leaks(tree, tasks)
        why += [f"reference-fix lines outside /testbed: {p}" for p in outside_copies(repo.parent, f, tasks)[:3]]
    return why


def kind(w: str) -> str:
    for pat, k in (("leak", "leak"), ("nofix not 0", "a task passes with no fix"), ("interaction", "task interaction"),
                   ("all fixes not 1", "a task fails with all fixes"), ("flaky", "flaky"), ("slow", "slow (>= 60 s)"),
                   ("no verifier result", "no verifier result")):
        if pat in w:
            return k
    return "setup/container error"


def main(trains: str, evs: str, out: str, *opts: str):
    """opts: --plan (report only, to W/plan-OUT.json: nothing under datasets/ written), --keep-split REPORT (the val / test
    repos of an earlier report instead of a new split)"""
    trains, evs, plan = trains.split(","), evs.split(","), "--plan" in opts
    keep = json.loads(Path(opts[opts.index("--keep-split") + 1]).read_text())["eval_assignment"] if "--keep-split" in opts else None
    rows_by, acc, timing, base = {}, {}, {}, {}
    for name in (*trains, *evs):
        rows_by[name] = [json.loads(l) for l in open(DS / f"{name}.manifest.jsonl")]
        acc[name] = json.loads((DS / f"{name}.acceptance-1c.json").read_text())
        timing[name] = {r["bundle"]: r for r in map(json.loads, open(DS / f"{name}.accepted.jsonl.timing.jsonl"))}
        for r in rows_by[name]:
            base[r["repo"]] = json.loads((DS / f"{name}.baseline" / f"{r['repo']}.json").read_text())
    ids = {t["instance_id"] for rs in rows_by.values() for r in rs for t in r["tasks"]}
    sw = {}
    for p in sorted(glob.glob(f"{W}/swesmith-hf/*.parquet")):
        for r in pq.read_table(p, columns=["instance_id", "FAIL_TO_PASS", "PASS_TO_PASS", "problem_statement"]).to_pylist():
            if r["instance_id"] in ids:
                sw[r["instance_id"]] = r
    status, final = {}, {}
    # tasks the judge gateway refused on every try (labels/*.giveup, BUNDLE:K): not cut to their statement, left out
    refused = {l.rsplit(":", 1)[0] for g in glob.glob(f"{W}/labels/*.giveup") for l in open(g).read().split()}
    for name in (*trains, *evs):
        for r in rows_by[name]:
            b, rep = r["bundle"], acc[name].get(r["bundle"])
            if b in DROP or r["repo"] in DROP_REPOS:
                status[b] = ("dropped", [DROP.get(b) or DROP_REPOS[r["repo"]]])
                continue
            if b in refused:
                status[b] = ("judge", ["judge gateway refused a task (HTTP 502 on every try)"])
                continue
            if not rep or not rep["accepted"]:
                status[b] = ("rejected", sorted({kind(w) for w in (rep or {"why": ["not judged"]})["why"]}))
                continue
            t = timing[name].get(b, {})
            walls = [t.get(v, {}).get("wall") for v in ("none", "all")]
            if not t.get("ok") or None in walls or max(walls) > LIMIT:
                status[b] = ("timing", [f"timing ok={t.get('ok')} walls={walls} {t.get('why')}"])
                continue
            why = scan(DS / f"{name}-valid" / b, r, sw, base[r["repo"]])
            status[b] = ("scan", why) if why else ("ok", [])
            if not why:
                final[b] = (name, r, walls)
            print(json.dumps({"bundle": b, "status": status[b]}), flush=True)
    # per repo cap (train 10, val/test 20): the first surviving bundles in plan order (files take turns there)
    per = collections.defaultdict(list)
    for b in sorted(final):
        per[final[b][1]["repo"]].append(b)
    per = {k: bs[:CAP["train"] if final[bs[0]][0] in trains else CAP["eval"]] for k, bs in per.items()}
    # eval repos -> val / test: of all splits of the repos that have bundles, the one with the closest bundle counts,
    # then the most repos on the side with fewer; a tie (e.g. val and test swapped) goes to the seed
    ev_repos = sorted({r["repo"] for e in evs for r in rows_by[e]})
    have = [k for k in ev_repos if per.get(k)]
    total, scored = sum(len(per[k]) for k in have), collections.defaultdict(list)
    for mask in range(2 ** len(have)):
        val = [k for i, k in enumerate(have) if mask >> i & 1]
        nv, rv = sum(len(per[k]) for k in val), len(val)
        scored[(abs(2 * nv - total), -min(rv, len(have) - rv))].append(val)
    val = list(keep["val"]) if keep else random.Random(SEED).choice(scored[min(scored)])
    split_of = {k: "val" if k in val else "test" for k in ev_repos}
    count = {s: [(k, len(per.get(k, []))) for k in ev_repos if split_of[k] == s] for s in ("val", "test")}
    chosen = {"train": [b for k, bs in per.items() for b in bs if final[b][0] in trains]}
    for s in ("val", "test"):
        chosen[s] = [b for k, n in count[s] for b in per.get(k, [])]
    trained = trained_repos()
    report = {"limit_s": LIMIT, "eval_assignment": {s: dict(c) for s, c in count.items()}, "splits": {}}
    for s, bs in chosen.items():
        dest = DS / f"{out}-{s}"
        if not plan and dest.exists():  # a split already produced is left as it is (train when only val / test are redone)
            print(f"{dest} exists: left as it is", flush=True)
        elif not plan:
            dest.mkdir()
            with open(f"{dest}.manifest.jsonl", "w") as m:
                for b in sorted(bs):
                    name, r, walls = final[b]
                    shutil.copytree(DS / f"{name}-valid" / b, dest / b)
                    m.write(json.dumps(r | {"split": s, "timing_wall": dict(zip(("none", "all"), walls))}) + "\n")
        built = [r for t in trains for r in rows_by[t]] if s == "train" else [r for e in evs for r in rows_by[e] if split_of[r["repo"]] == s]
        reasons = collections.Counter(k for r in built if status[r["bundle"]][0] != "ok" for k in status[r["bundle"]][1])
        stage = collections.Counter(status[r["bundle"]][0] for r in built)
        walls = [final[b][2] for b in bs]
        q = lambda xs, p: sorted(xs)[min(len(xs) - 1, int(p * len(xs)))] if xs else None
        report["splits"][s] = {
            "repos": sorted({final[b][1]["repo"] for b in bs}), "bundles": len(bs), "bundle_ids": sorted(bs),
            "sizes": dict(sorted(collections.Counter(final[b][1]["n"] for b in bs).items())),
            "tasks": sum(final[b][1]["n"] for b in bs),
            "built": len(built), "stage": dict(stage), "accept_rate": round(stage["ok"] / max(1, len(built)), 3),
            "reject_reasons": dict(reasons.most_common()),
            "wall_all": {"p50": q([w[1] for w in walls], .5), "p90": q([w[1] for w in walls], .9), "max": max((w[1] for w in walls), default=None)},
            "wall_none": {"p50": q([w[0] for w in walls], .5), "p90": q([w[0] for w in walls], .9), "max": max((w[0] for w in walls), default=None)},
            "repos_in_e9_e14_training": sorted({final[b][1]["repo"] for b in bs} & trained),
            "repo_split_of": sorted({bb.split_of(final[b][1]["repo"]) for b in bs}),
            "per_repo_max": max(collections.Counter(final[b][1]["repo"] for b in bs).values(), default=0)}
    sets = {s: set(v["repos"]) for s, v in report["splits"].items()}
    report["repo_overlap"] = {f"{a}&{b}": sorted(sets[a] & sets[b]) for a, b in (("train", "val"), ("train", "test"), ("val", "test"))}
    report["scan_failures"] = {b: w for b, (st, w) in status.items() if st == "scan"}
    report["dropped"] = {b: w for b, (st, w) in status.items() if st == "dropped"}
    target = Path(W) / f"plan-{out}.json" if plan else DS / f"{out}.report.json"
    target.write_text(json.dumps(report, indent=1))
    print(json.dumps({s: {k: v[k] for k in ("bundles", "sizes", "tasks", "accept_rate")} for s, v in report["splits"].items()}))


if __name__ == "__main__":
    main(*sys.argv[1:])
