"""Add a finished task set to the pool: copy each task dir into POOL/tasks/ and write its pool.json with what the task
dir itself does not say (read back by build_pool.py).
usage: pool_import.py add POOL DATASET_DIR BATCH NOP_JOB_DIR [MB_MANIFEST.json]
         f2p from the SWE-smith parquet ($SWESMITH_PARQUET, default /home/user/swesmith-hf); bug_failing (test ids,
         when pytest printed its -rfE summary) and bug_failed_n (from the "N failed" line) from the nop run's verifier
         output; statement_covered from the multi-bug text-screen manifest (mb_cover.py).
       pool_import.py labels POOL LORA=FIELD ... -- ROLLOUTS.jsonl ...
         pass rate per task = mean of `passed` over that lora's rollouts, deduped by (lora, task, sample), rows with an
         "error" key excluded; e.g. base=base_pass_rate c2-s60=c2s60_pass_rate."""
import collections, glob, json, os, re, shutil, sys
from pathlib import Path

FAIL = re.compile(r"^(?:FAILED|ERROR) (\S+?)(?: - |$)", re.M)


def add(pool: Path, dataset: Path, batch: str, nop_job: Path, mb_manifest: Path | None):
    import pandas as pd
    pq = os.environ.get("SWESMITH_PARQUET", "/home/user/swesmith-hf")
    df = pd.concat(pd.read_parquet(f, columns=["instance_id", "FAIL_TO_PASS"]) for f in sorted(glob.glob(f"{pq}/*.parquet")))
    f2p = dict(zip(df.instance_id, (len(x) for x in df.FAIL_TO_PASS)))
    failing, failed_n = {}, {}
    for rj in nop_job.glob("*/result.json"):
        name = os.path.basename(json.loads(rj.read_text())["config"]["task"]["path"])
        so = rj.parent / "verifier/test-stdout.txt"
        txt = so.read_text(errors="replace") if so.exists() else ""
        failing[name] = sorted(set(FAIL.findall(txt)))  # test ids; empty when the repo's pytest config drops the -rfE summary
        summary = [l for l in txt.splitlines() if re.search(r" in [0-9.]+s", l)]  # "46 failed, 19 passed in 1.08s"
        failed_n[name] = sum(int(n) for n in re.findall(r"(\d+) (?:failed|errors?)\b", summary[-1])) if summary else None
    cover = json.loads(mb_manifest.read_text()) if mb_manifest else {}
    n = 0
    for t in sorted(p for p in dataset.iterdir() if (p / "task.toml").is_file()):
        dest = pool / "tasks" / t.name
        if dest.exists():
            raise SystemExit(f"{dest} exists")
        shutil.copytree(t, dest)
        (dest / "pool.json").write_text(json.dumps({
            "f2p": f2p[t.name], "bug_failing": failing[t.name], "bug_failed_n": failed_n[t.name], "source_batch": batch,
            "statement_covered": cover[t.name]["uncovered_hunks"] == 0 if t.name in cover else None}))
        n += 1
    print(f"{n} tasks from {dataset} -> {pool}/tasks (batch {batch})")


def labels(pool: Path, fields: dict, files: list):
    seen, acc = set(), collections.defaultdict(list)
    for f in files:
        for l in open(f):
            r = json.loads(l)
            if "task_dir" not in r or "error" in r or r.get("lora") not in fields:
                continue
            key = (r["lora"], os.path.basename(r["task_dir"]).split("--")[0], r["sample"])
            if key not in seen:
                seen.add(key); acc[(key[1], fields[r["lora"]])].append(bool(r["passed"]))
    n = 0
    for (task, field), v in acc.items():
        pj = pool / "tasks" / task / "pool.json"
        if pj.exists():
            d = json.loads(pj.read_text()); d[field] = round(sum(v) / len(v), 4); pj.write_text(json.dumps(d)); n += 1
    print(f"{n} labels written from {len(seen)} rollouts")


if __name__ == "__main__":
    if sys.argv[1] == "add":
        add(Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4], Path(sys.argv[5]), Path(sys.argv[6]) if len(sys.argv) > 6 else None)
    else:
        i = sys.argv.index("--")
        labels(Path(sys.argv[2]), dict(a.split("=") for a in sys.argv[3:i]), sys.argv[i + 1:])
