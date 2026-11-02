"""Audit of a graded SWE-smith dataset beyond oracle=1 / nop=0 (select_valid.py), against the SWE-smith dataset:
- the SWE-smith FAIL_TO_PASS tests: are they in the hidden files, not deselected as known-failing, failing on the bug
  (nop run) and passing on the reference (oracle run);
- tests that fail on the bug but are not F2P (the task would also grade something else);
- verifier has the infra restore; problem statements unique and not empty.

Pytest stdout is often a FAILURES traceback (no `FAILED nodeid` lines). Parse that block; ignore warnings.

Writes pool-build/unsound-NAME.txt (one instance id per line) for tasks whose live F2P did not fail
on the bug, failed on the reference, or whose nop run has no pytest failure evidence.

usage: audit_tasks.py PARQUET_DIR DATASET ORACLE_JOB NOP_JOB   (names under datasets/ and jobs/)"""
import collections, glob, json, os, re, sys
import pandas as pd

H = "/home/user/harbor-udocker"
B = "/home/user/pool-build"
P, D, ORACLE, NOP = sys.argv[1], f"{H}/datasets/{sys.argv[2]}", sys.argv[3], sys.argv[4]
NAME = sys.argv[2]
if NAME.startswith("swesmith-") and NAME.endswith("-final"):
    NAME = NAME[len("swesmith-"):-len("-final")]

FAILED_NODE = re.compile(r"^(?:FAILED|ERROR) (\S+?)(?: - |$)", re.M)
TB_FILE = re.compile(r"/testbed/(\S+\.py):\d+")
SUMMARY_FAIL = re.compile(r"(\d+) failed")
FAILURES_BLK = re.compile(r"={5,}\s+FAILURES\s+={5,}(.*?)(?:={5,}\s+\S|\Z)", re.S)

tasks = sorted(t for t in os.listdir(D) if os.path.isdir(f"{D}/{t}"))
df = pd.concat(pd.read_parquet(f, columns=["instance_id", "FAIL_TO_PASS", "problem_statement"]) for f in sorted(glob.glob(f"{P}/*.parquet")))
f2p = {r.instance_id: list(r.FAIL_TO_PASS) for r in df.itertuples()}
stmt = {r.instance_id: r.problem_statement for r in df.itertuples()}


def out(job):
    res = {}
    for rj in glob.glob(f"{H}/jobs/{job}/*/result.json"):
        d = json.load(open(rj))
        name = os.path.basename(d["config"]["task"]["path"])
        so = os.path.join(os.path.dirname(rj), "verifier/test-stdout.txt")
        res[name] = open(so, errors="replace").read() if os.path.exists(so) else ""
    return res


def parse_fail(txt):
    txt = txt or ""
    blk = "".join(FAILURES_BLK.findall(txt)) or ""
    nodes = set(FAILED_NODE.findall(txt))
    files = set(TB_FILE.findall(blk))
    m = SUMMARY_FAIL.findall(txt)
    nfail = int(m[-1]) if m else 0
    return nodes, files, nfail, "no tests ran" in txt, blk


def f2p_hit(item, nodes, files, blk=""):
    item = str(item)
    file_part = item.split("::", 1)[0]
    func = item.split("::")[-1].split("[")[0] if "::" in item else ""
    for n in nodes:
        if n == item or n.startswith(item + "[") or n.startswith(item + "::"):
            return True
        if n.startswith(file_part + "::") and (not func or func in n.split("::")):
            return True
    base = os.path.basename(file_part)
    file_hit = any(f == file_part or f.endswith("/" + file_part) or os.path.basename(f) == base for f in files)
    if not file_hit:
        return False
    if not func:
        return True
    return (("::" + func) in blk) or re.search(r"\b" + re.escape(func) + r"\b", blk) is not None


nop, ora = out(NOP), out(ORACLE)
flags = collections.defaultdict(list)
stats = collections.Counter()
for t in tasks:
    tests = f"{D}/{t}/tests"
    hidden = open(f"{tests}/hidden_files.txt").read().split() if os.path.exists(f"{tests}/hidden_files.txt") else []
    known = open(f"{tests}/known_failing.txt").read().split() if os.path.exists(f"{tests}/known_failing.txt") else []
    if os.path.exists(f"{tests}/test.sh") and "conftest" not in open(f"{tests}/test.sh").read():
        flags["no infra restore"].append(t)
    F = f2p.get(t, [])
    stats["f2p tests"] += len(F)
    if not F:
        flags["no F2P list in SWE-smith"].append(t)
        continue
    if any(not any(str(x).startswith(h) for h in hidden) for x in F):
        flags["F2P test outside hidden files"].append(t)
    live = [x for x in F if x not in known]
    if not live:
        flags["every F2P test deselected as known-failing"].append(t)
        continue
    if len(live) < len(F):
        flags["some F2P deselected"].append(t)
    nt, nf, nfail, nempty, nblk = parse_fail(nop.get(t, ""))
    ot, of, ofail, oempty, oblk = parse_fail(ora.get(t, ""))
    live_fail_bug = [x for x in live if f2p_hit(x, nt, nf, nblk)]
    live_fail_ref = [x for x in live if f2p_hit(x, ot, of, oblk)] if ofail > 0 or ot or of else []
    if nempty or (nfail == 0 and not nt and not nf):
        flags["nop has no pytest failure evidence"].append(t)
    elif not live_fail_bug:
        flags["live F2P did not fail on the bug"].append(t)
        if nt or nf:
            flags["bug fails only non-F2P tests"].append(t)
    if live_fail_ref:
        flags["live F2P fails on the reference"].append(t)
    stats["nop with parsed node/file"] += bool(nt or nf)
    stats["nop with summary failed>0"] += nfail > 0
    if not (stmt.get(t) or "").strip():
        flags["empty statement"].append(t)
dup = collections.Counter((stmt[t] or "").strip() for t in tasks if t in stmt)
flags["duplicate statement"] = [t for t in tasks if t in stmt and (stmt[t] or "").strip() and dup[(stmt[t] or "").strip()] > 1]
print("tasks", len(tasks), dict(stats))
for k, v in flags.items():
    if v:
        print(f"{k}: {len(v)}", v[:4])

unsound_keys = (
    "live F2P did not fail on the bug",
    "live F2P fails on the reference",
    "every F2P test deselected as known-failing",
    "no F2P list in SWE-smith",
    "F2P test outside hidden files",
    "bug fails only non-F2P tests",
    "nop has no pytest failure evidence",
    "empty statement",
)
unsound = sorted({t for k in unsound_keys for t in flags.get(k, [])})
os.makedirs(B, exist_ok=True)
open(f"{B}/unsound-{NAME}.txt", "w").write("".join(t + "\n" for t in unsound))
print(f"unsound {len(unsound)} wrote {B}/unsound-{NAME}.txt")
json.dump({k: flags[k] for k in flags if flags[k]}, open(f"{B}/unsound-{NAME}.json", "w"), indent=1)
