"""Held-out eval lane pool for one RL run, run on node-67 next to EVAL_DIR (results/, logs/, tasks-eval-v3, run.env,
steps.list, run/: the trial's config.yaml and episode_settings.json, which --trial copies from node-94 at each lane's
start). Every --interval seconds:
1. with --trial: lists the trial's checkpoints on node-94 and appends to steps.list every STEP N (the adapter saved after
   global step N-1, any epoch name, e.g. epoch1epochstep1globalstep49) with N % --every == 0, and --final N once its
   recover checkpoint exists (AReaL saves no adapter for a last step off saver.freq_steps: dcp_adapter.py extracts it on
   node-94 into <ckpt>/finalglobalstep<N-1>). --k4 steps are listed as N:4 (k=4: dev_eval adds samples 2-3 to the file).
2. claims pending steps newest-first (atomic mkdir results/step<N>.lockdir) and starts one lane per step (run_step.sh,
   detached) on one --hosts host: free GPUs (< FREE_MIB used, not --exclude'd, not held by a live lane of any pool:
   registry evals/lanes), as many as one wave of the step's remaining episodes needs at --par-max per server, and
   PAR = ceil(remaining / GPUs). A lane gets its own id, ports 32000+10*id and SGLang state dir, runs one step, then
   stops its servers (GPUs released). A step that failed --max-fails times is skipped.
3. with K_SMALL=k in run.env, a step is also pending until its 1-3-task bundles (symlinked into tasks-small) have k
   samples in results/step<N>-small.jsonl: the same lane runs that pass after the main one (K=0: main already done).
   steps.list token N:0 = small pass only (no main pass; --trial never re-adds a step listed as N:0). The adapter is
   chosen as for any step: N=0 is the base adapter, N>0 the --trial's adapter saved after global step N-1.
   Small-only comparison of other checkpoints under one run's eval conditions, in a fresh EVAL_DIR X:
     mkdir X && ln -s <any eval dir>/tasks-eval-v3 X/  (the 60 bundles)
     X/run.env: C=<code dir> / A=<AReaL overlay> / TOK=<tokenizer dir> / K_SMALL=8 (one per line; run_step.sh)
     X/steps.list: 0:0 40:0   (base, and the trial's step-40 adapter = globalstep39 on node-94)
     pool.py X --trial bundle-e10 --every 1000   (--every 10 would also list the trial's 10 20 30 50 ... as full steps)
   then summary.py X --base-small X/results/step0-small.jsonl (small lines only), or summary.py <the run's eval dir>
   --base ... --base-small X/results/step0-small.jsonl (its own step<N>-small vs that base).
usage: pool.py EVAL_DIR --trial bundle-e10 --final 53 --hosts node-134,node-136,node-68 [--exclude node-94,node-138:0,node-138:1]
       [--k4 30,50] [--dry-run: one pass, print, change nothing] [--stop: TERM this run's lanes, servers stop]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path

G94 = "user@10.0.1.94"
CKPTS = "/home/user/areal-exp/checkpoints/user/rq1-rl"
LOGS = "/home/user/areal-exp/logs/user/rq1-rl"  # <trial>/config.yaml (AReaL) and episode_settings.json (train.py)
DCP = "/mnt/disk1/user/areal/.venv/bin/python /home/user/octopus-sft/dcp_adapter.py"
FREE_MIB = 1000  # a GPU with more used is someone's (eval_serve.sh refuses > 4000)
SSH = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20", "-o", "StrictHostKeyChecking=accept-new"]
# Lane hosts with patched SGLang. hop: node-67 reaches the host only through node-94's reverse-tunnelled sshd.
# env: the host's eval env file (SGLANG_PY MODEL_VL CUDA_HOME); each lane copies it with its own STATE and TMPDIR.
HOSTS = {
    "node-94": dict(gh=G94, rh="/home/user/octopus-sft", serve="eval_serve.sh", env="eval-e8f-94all.env"),
    "node-138": dict(gh="user@10.0.1.138", rh="/home/user/octo-env", serve="eval_serve.sh", env="eval-e8f-138.env"),
    "node-134": dict(gh="user@10.0.2.34", rh="/mnt/disk4/user/octo-env-data", serve="eval_serve_262k.sh",
                    env="eval-e8f-134all.env"),
    "node-136": dict(hop=G94, gh="-p 2240 user@127.0.0.1", rh="/home/user/octo-env", serve="eval_serve.sh",
                    env="eval-e8f-136.env"),
    "node-68": dict(hop=G94, gh="-p 2239 user@127.0.0.1", rh="/home/user/octo-env", serve="eval_serve.sh",
                   env="eval-base-68.env"),
}


def sh(host: dict | None, cmd: str) -> str:
    """Run cmd on node-94 (host None) or a lane host; stdout, "" on failure."""
    if host is None:
        argv = SSH + [G94, cmd]
    elif host.get("hop"):
        argv = SSH + [host["hop"], " ".join(SSH) + f" {host['gh']} {shlex.quote(cmd)}"]
    else:
        argv = SSH + host["gh"].split() + [cmd]
    try:
        p = subprocess.run(argv, capture_output=True, text=True, timeout=600)
    except subprocess.TimeoutExpired:
        return ""
    return p.stdout if p.returncode == 0 else ""


def read_steps(e: Path) -> dict[int, int]:
    """steps.list tokens N or N:K -> {N: max K}."""
    out: dict[int, int] = {}
    for t in (e / "steps.list").read_text().split() if (e / "steps.list").exists() else []:
        n, _, k = t.partition(":")
        out[int(n)] = max(out.get(int(n), 0), int(k or 2))
    return out


def checkpoints(trial: str) -> tuple[dict[int, str], set[int]]:
    """{global step: saved adapter dir} and the global steps with a complete recover checkpoint, on node-94."""
    root = f"{CKPTS}/{trial}"
    ls = sh(None, f"ls -d {root}/default/*globalstep*/adapter_model.safetensors "
                  f"{root}/checkpoint_generations/generation_step*/payloads/default/.metadata 2>/dev/null; true")
    saved: dict[int, str] = {}
    for p in sorted(ls.split()):
        if m := re.search(r"globalstep(\d+)/adapter_model", p):
            saved.setdefault(int(m[1]), p.rsplit("/", 1)[0])
    return saved, {int(m[1]) for m in re.finditer(r"generation_step(\d+)/", ls)}


def watch(e: Path, a, saved: dict[int, str], gens: set[int]) -> None:
    want = {g + 1 for g in saved if (g + 1) % a.every == 0}
    if a.final:
        g = a.final - 1
        if g not in saved and g in gens and saved:  # last step off saver.freq_steps: extract it from recover checkpoint
            out = f"{CKPTS}/{a.trial}/default/finalglobalstep{g}"
            cmd = f"{DCP} {CKPTS}/{a.trial}/checkpoint_generations/generation_step{g:08d}/payloads/default {saved[max(saved)]} {out}"
            print(f"{time.strftime('%F %T')} final step {a.final}: {cmd}", flush=True)
            if a.dry_run:
                saved[g] = out
            elif "differ from" in (r := sh(None, f"{cmd} 2>&1 | tail -1")):
                saved[g] = out
            print(r if not a.dry_run else "", end="", flush=True)
        if g in saved:
            want.add(a.final)
    steps = read_steps(e); k4 = {int(x) for x in a.k4.split(",") if x}
    new = [f"{n}:4" if n in k4 else str(n) for n in sorted(want)
           if steps.get(n) != 0 and steps.get(n, 0) < (4 if n in k4 else 2)]  # N:0 = small pass only, keep it
    if new:
        print(f"{time.strftime('%F %T')} steps.list += {' '.join(new)}", flush=True)
        if not a.dry_run:
            old = (e / "steps.list").read_text().split() if (e / "steps.list").exists() else []
            (e / "steps.list").write_text(" ".join(old + new) + "\n")


def done_k(e: Path, n: int | str) -> int:
    f = e / f"results/step{n}.done"
    return int(m[1]) if f.exists() and (m := re.search(r"k=(\d+)", f.read_text())) else 2 if f.exists() else 0


def lanes(evals: Path, children: dict) -> dict[int, dict]:
    """Live lanes of every pool on this machine; a dead lane's file and its stale step claim are removed."""
    for pid, p in list(children.items()):
        if p.poll() is not None:
            del children[pid]
    live = {}
    for f in (evals / "lanes").glob("L*.json"):
        lane = json.loads(f.read_text() or "{}")
        try:
            os.kill(lane["pid"], 0); live[int(f.stem[1:])] = lane
        except (OSError, KeyError):
            if "pid" in lane:  # dead lane (killed hard: no EXIT trap): drop it and its claim
                lock = Path(lane["eval"], f"results/step{lane['step']}.lockdir")
                if lock.exists():
                    lock.rmdir()
                f.unlink()
    return live


def copy_run(e: Path, trial: str) -> None:
    """E/run: the trial's config.yaml and episode_settings.json as they are now (a resumed trainer rewrites them). A
    failed copy (sh does not retry a dropped handshake) keeps the last one; without any, run_step.sh fails the step."""
    run = {f: sh(None, f"cat {LOGS}/{trial}/{f}") for f in ("config.yaml", "episode_settings.json")}
    if not all(run.values()):
        print(f"node-94 {LOGS}/{trial}: config.yaml / episode_settings.json not read; E/run kept as it is", flush=True)
        return
    (e / "run").mkdir(exist_ok=True)
    for f, text in run.items():  # each file replaced whole: a lane starting meanwhile reads the old or the new one
        (e / "run" / f"{f}.tmp").write_text(text)
        os.replace(e / "run" / f"{f}.tmp", e / "run" / f)


def start_lane(e: Path, a, n: int, k: int, ks: int, src: str, host: str, gpus: list[int], par: int, children: dict) -> None:
    if a.trial:
        copy_run(e, a.trial)
    ld = e.parent / "lanes"; ld.mkdir(exist_ok=True)
    for i in range(100):
        try:
            fd = os.open(ld / f"L{i}.json", os.O_CREAT | os.O_EXCL | os.O_WRONLY); break
        except FileExistsError:
            continue
    h = HOSTS[host]; g = ",".join(map(str, gpus))
    env = dict(os.environ, E=str(e), N=str(n), K=str(k), KS=str(ks), SRC=src, LANE=str(i), HOST=host, HOP=h.get("hop", ""),
               GH=h["gh"], RH=h["rh"], SERVE=f"{h['rh']}/{h['serve']}", BASEENV=f"{h['rh']}/{h['env']}", GPUS=g,
               PB=str(32000 + 10 * i), PAR=str(par))
    (e / "logs").mkdir(exist_ok=True)
    p = subprocess.Popen(["bash", str(Path(__file__).with_name("run_step.sh"))], env=env, start_new_session=True,
                         stdin=subprocess.DEVNULL, stdout=open(e / f"logs/lane-L{i}-step{n}.out", "a"), stderr=subprocess.STDOUT)
    children[p.pid] = p
    os.write(fd, json.dumps(dict(pid=p.pid, eval=str(e), step=n, k=k, k_small=ks, host=host, gpus=gpus, par=par,
                                 started=time.strftime("%F %T"))).encode()); os.close(fd)
    print(f"{time.strftime('%F %T')} step {n} k={k} k_small={ks}: lane L{i} {host} GPUs {g} PAR {par} pid {p.pid}", flush=True)


def nrows(e: Path, stem: str) -> int:
    """Episodes done: rows without an error (dev_eval runs an error row's episode again)."""
    f = e / f"results/{stem}.jsonl"
    return sum("error" not in json.loads(x) for x in f.read_text().splitlines()) if f.exists() else 0


def one_pass(e: Path, a, children: dict) -> None:
    saved, gens = checkpoints(a.trial) if a.trial else ({}, set())
    if a.trial:
        watch(e, a, saved, gens)
    live = lanes(e.parent, children)
    dirs = [d for d in (e / "tasks-eval-v3").iterdir() if d.is_dir()]
    small = [d for d in dirs if len(list(d.glob("solution/fix_*.patch"))) <= 3]
    m = re.search(r"^(?:export )?K_SMALL=(\d+)", (e / "run.env").read_text(), re.M)
    k_small = int(m[1]) if m else 0
    pending = []
    for n, k in sorted(read_steps(e).items(), reverse=True):  # newest first
        failed = e / f"results/step{n}.failed"
        km = 0 if done_k(e, n) >= k else k
        ks = 0 if done_k(e, f"{n}-small") >= k_small else k_small
        if not (km or ks) or (e / f"results/step{n}.lockdir").exists() or (n and n - 1 not in saved) or (
                failed.exists() and len(failed.read_text().splitlines()) >= a.max_fails):
            continue
        rem = max(km and len(dirs) * km - nrows(e, f"step{n}"), ks and len(small) * ks - nrows(e, f"step{n}-small"))
        pending.append((n, km, ks, rem))
    if not pending:
        return
    held = {(l["host"], g) for l in live.values() for g in l["gpus"]}
    ex = [x.partition(":") for x in a.exclude.split(",") if x]
    held |= {(h, int(g)) for h, _, g in ex if g}
    free = {}
    for host in a.hosts.split(","):
        if (host, "", "") in ex:
            continue
        smi = sh(HOSTS[host], "nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits")
        free[host] = [int(i) for i, m in (l.split(", ") for l in smi.splitlines())
                      if int(m) < FREE_MIB and (host, int(i)) not in held]
    print(f"{time.strftime('%F %T')} pending {pending} free {free} lanes {len(live)}", flush=True)
    for n, k, ks, rem in pending:
        host = max(free, key=lambda h: len(free[h]), default=None)
        want = math.ceil(max(rem, 1) / a.par_max)
        take = min(want, len(free[host]) if host else 0)
        if take < min(want, a.min_gpus):
            continue
        gpus, free[host] = free[host][:take], free[host][take:]
        par = min(a.par_max, math.ceil(max(rem, 1) / take))
        if a.dry_run:
            print(f"would start step {n} k={k} k_small={ks} ({rem} episodes left) on {host} GPUs {gpus} PAR {par}", flush=True)
            continue
        (e / "results").mkdir(exist_ok=True)
        if ks:
            (e / "tasks-small").mkdir(exist_ok=True)
            for d in small:
                if not (e / "tasks-small" / d.name).exists():
                    (e / "tasks-small" / d.name).symlink_to(d)
        try:
            os.mkdir(e / f"results/step{n}.lockdir")  # the claim; run_step.sh removes it on exit
        except FileExistsError:
            continue
        start_lane(e, a, n, k, ks, "zero" if n == 0 else saved[n - 1], host, gpus, par, children)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("eval_dir")
    ap.add_argument("--trial", help="training trial name (checkpoints under CKPTS on node-94); needed for steps > 0")
    ap.add_argument("--every", type=int, default=10)
    ap.add_argument("--final", type=int, help="last STEP of the run (its adapter is extracted from the recover checkpoint)")
    ap.add_argument("--k4", default="", help="comma list of steps to evaluate with k=4")
    ap.add_argument("--hosts", default="node-134,node-136,node-68,node-138,node-94")
    ap.add_argument("--exclude", default="node-94,node-138", help="hosts or host:gpu (repeat per GPU) reserved for training")
    ap.add_argument("--par-max", type=int, default=15, help="max concurrent episodes per SGLang server")
    ap.add_argument("--min-gpus", type=int, default=2, help="do not start a lane on fewer free GPUs")
    ap.add_argument("--max-fails", type=int, default=3)
    ap.add_argument("--interval", type=int, default=300)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--stop", action="store_true")
    a = ap.parse_args()
    e = Path(a.eval_dir).resolve()
    if a.stop:
        for lane in lanes(e.parent, {}).values():
            if lane["eval"] == str(e):
                os.killpg(lane["pid"], signal.SIGTERM); print("stopped", lane)
        return
    children: dict = {}
    while True:
        one_pass(e, a, children)
        if a.dry_run:
            return
        time.sleep(a.interval)


if __name__ == "__main__":
    sys.exit(main())
