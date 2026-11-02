"""Read-tool audit (OCTOPUS_READ_TOOL, docs/cross-branch-read.md "Slice 4 run") per training step. M1-M4 stay with
steps.py; this only counts reads.

usage: python read_audit.py JOBS_DIR TRAIN_LOG [LOG ...] [--remote [USER@]HOST[:PORT]:/GLOB ...]
JOBS_DIR: the run's rl.jobs_dir. TRAIN_LOG: $STATE/train-<trial>.log: its "Train step N done" lines cut the steps, its
"[octopus] read leaves" lines give the exported leaves that carry a read, and the proxy's tracebacks end in the engine's
read errors (`RuntimeError: ...`, one line per failed request). LOG: the SGLang server logs (local and remote
serve/sglang-*g*.log), for /generate 404/503, tracebacks and mount rebuilds. --remote [USER@]HOST[:PORT]:/GLOB
(repeatable; PORT: e.g. a reverse tunnel's) counts the same patterns in a remote host's logs without copying them: one
ssh call greps them there and sends back only counts, per pattern and timestamp (`uniq -c`). Episodes go to the step whose window their result.json mtime falls in (as steps.py), log lines to the
step after their timestamp (a train-log traceback line without one: the last timestamp above it; an SGLang one: step ?).

Per step: episodes; read_thread calls (every thread's tool calls in tree.json, main's and each branch's, those a
rewind removed too (their rewinds' messages), each call id once); valid mounts and refusals by kind; mount sizes (tokens, from the result text); reads per episode; the spread of
read_thread calls over episodes, also over all steps: median, mean, the share of all read_thread calls held by the 10
busiest episodes; requests that carried a mount (requests.jsonl) and those that failed; long waits:
agent requests that took over 300 s and those that timed out (no HTTP code: the agent's 900 s OCTOPUS_HTTP_TIMEOUT), and
episodes with such a timeout; leaves with reads / leaves; and per log pattern (POST /generate answered 404 or 503,
SGLang tracebacks, the engine's read errors, and the mount patch's evicted-read abort, which since slice 6 is not a
failure but a mount rebuild: the server rebuilds the mounted rows and retries) its count."""
import collections, glob, json, re, shlex, statistics, subprocess, sys
from datetime import datetime
from pathlib import Path

args, remote = sys.argv[1:], []
while "--remote" in args:
    i = args.index("--remote")
    remote.append(args.pop(i + 1))
    args.pop(i)
jobs, train_log, logs = args[0], args[1], args[2:]
AREAL_TS = r"(\d{8}-\d\d:\d\d:\d\d\.\d+) "  # AReaL's log lines
TS = AREAL_TS + r"StatsLogger INFO: .*Train step (\d+)/"
areal_time = lambda x: datetime.strptime(x, "%Y%m%d-%H:%M:%S.%f").timestamp()  # noqa: E731
done = [(areal_time(m[1]), int(m[2])) for m in re.finditer(TS, open(train_log, errors="replace").read())]
step_of = lambda t: next((s for d, s in done if t < d), (done[-1][1] if done else 0) + 1)  # noqa: E731
TAG = re.compile(r"\[(?:[^\]\n|]+ \| )?(?:token budget left[^\]]*|context left[^\]]*|step \d+)\]\n")  # [main | ...]: OCTOPUS_READ_TOOL
KINDS = [("valid", "Mounted "), ("self", "You are "), ("own-history", "is in your own history already"),
         ("no-thread", "No thread "), ("malformed", "read_thread needs"), ("not-readable", "readable step"),
         ("not-offered", "is not available in this branch"),  # search: read_when_readable
         ("ambiguous", " is ambiguous: ")]  # search: a bare id several threads have (read_by_name)
NOTICES = re.compile(r"\A(?:\[[^\]\n]+ finished: [^\]\n]*\]\n)+")
SGLANG = {"generate 404": '"POST /generate HTTP/1.1" 404', "generate 503": '"POST /generate HTTP/1.1" 503',
          "traceback": "Traceback (most recent call last)", "mount rebuilds": "the KV of a mounted read is evicted"}
ENGINE = {"engine: read dropped": "or a cursor it mounts is gone from", "engine: other server": "a read needs both threads on one server",
          "engine: cannot hop": "cannot continue its parent's cursor", "engine: mount ignored": "ignored the mount"}
S = collections.defaultdict(lambda: collections.defaultdict(collections.Counter))  # step -> table -> counter
LONG_WAITS = ("requests > 300 s", "requests timed out", "episodes with a timeout")
sizes, per_episode = collections.defaultdict(list), collections.defaultdict(list)
spread = collections.defaultdict(list)  # step -> read_thread calls per episode


def kind(text: str) -> str:
    text = NOTICES.sub("", text)  # a result of main's can open with join-any notices: "[branch-0 finished: commit]\n"
    return next((k for k, needle in KINDS if (text.startswith(needle) if k in ("valid", "self", "no-thread", "malformed")
                                              else needle in text)), "other: " + text[:50])


for tf in glob.glob(f"{jobs}/*/*/agent/tree.json"):
    trial = Path(tf).parent.parent
    if not (trial / "result.json").exists():
        continue
    step = step_of((trial / "result.json").stat().st_mtime)
    c = S[step]["episodes"]
    c["episodes"] += 1
    calls, seen = {}, {}

    def walk(v):  # every thread's tool calls and results; a branch's inherited history repeats them: once per call id
        if isinstance(v, dict):
            if v.get("role") == "assistant":
                for call in v.get("tool_calls") or []:
                    calls[call.get("id")] = (call.get("function") or {}).get("name")
            if v.get("role") == "tool":
                seen[v.get("tool_call_id")] = TAG.sub("", str(v.get("content")), count=1)
            for x in v.values():
                walk(x)
        elif isinstance(v, list):
            for x in v:
                walk(x)
    tree = json.loads(Path(tf).read_text())
    walk(tree)
    reads = [cid for cid, name in calls.items() if name == "read_thread"]
    spread[step].append(len(reads))
    n_valid = 0
    for cid in reads:
        text = seen.get(cid, "")
        k = kind(text) if text else "no result"
        S[step]["read_thread results"][k] += 1
        S[step]["read_thread results"]["(of these, no other thread yet)"] += "There is no other thread yet" in text
        if k == "valid":
            n_valid += 1
            sizes[step].append(int(re.search(r"\((\d+) tokens", text)[1]))
    c["read_thread calls"] += len(reads)
    c["with a read_thread call"] += bool(reads)
    c["with a valid mount"] += bool(n_valid)
    per_episode[step].append(n_valid)
    rq = trial / "agent/requests.jsonl"
    timed_out = 0
    S[step]["long waits"].update(dict.fromkeys(LONG_WAITS, 0))  # fixed key order, whichever trial comes first
    for line in (rq.read_text().splitlines() if rq.exists() else []):
        row = json.loads(line)
        S[step]["long waits"]["requests > 300 s"] += (row.get("elapsed_s") or 0) > 300
        timed_out += row.get("http_code") is None and "timed out" in str(row.get("error"))
        if "mount" in row:
            S[step]["mount requests"]["sent"] += 1
            S[step]["mount requests"]["failed (http != 200)"] += row.get("http_code") != 200
    S[step]["long waits"]["requests timed out"] += timed_out
    S[step]["long waits"]["episodes with a timeout"] += bool(timed_out)

ts = 0.0
for line in open(train_log, errors="replace"):
    if m := re.search(AREAL_TS, line):
        ts = areal_time(m[1])
    if m := re.search(r"\[octopus\] read leaves (\d+) with reads (\d+)", line):  # before the step it trains in
        S[step_of(ts)]["leaves"]["exported"] += int(m[1])
        S[step_of(ts)]["leaves"]["with reads"] += int(m[2])
    for k, needle in ENGINE.items():  # the traceback's last line: once per failed request
        if line.startswith("RuntimeError: ") and needle in line:
            S[step_of(ts)]["logs"][k] += 1


def log_counts():  # (pattern, count, the start of its lines: "[YYYY-MM-DD HH:MM:SS]" or other text)
    for path in logs:
        for line in open(path, errors="replace"):
            yield from ((k, 1, line) for k, needle in SGLANG.items() if needle in line)
    for spec in remote:
        dest, glob = spec.split(":/", 1)
        host, _, port = dest.partition(":")
        glob = "/" + glob  # unquoted: the remote shell expands it
        cmd = f"ls {glob} > /dev/null || exit 2; " + "; ".join(  # the ls: a glob that matches nothing is an error
            f"echo {shlex.quote('#' + k)}; grep -ahF -e {shlex.quote(n)} {glob} | cut -c1-21 | sort | uniq -c"
            for k, n in SGLANG.items())
        r = subprocess.run(["ssh", *(["-p", port] if port else []), host, cmd], capture_output=True, text=True,
                           errors="replace")
        if r.returncode:
            sys.exit(f"--remote {spec}: {r.stderr.strip()}")
        for line in r.stdout.splitlines():
            if line.startswith("#"):
                k = line[1:]
            else:
                n, _, start = line.strip().partition(" ")
                yield k, int(n), start


for k, n, line in log_counts():
    m = re.match(r"\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)", line)
    S[step_of(datetime.strptime(m[1], "%Y-%m-%d %H:%M:%S").timestamp()) if m else "?"]["logs"][k] += n


def spread_line(eps) -> str:
    n = sorted(eps, reverse=True)
    return (f"read_thread calls per episode: median {statistics.median(n):g} mean {statistics.mean(n):.1f} "
            f"top-10 episodes {100 * sum(n[:10]) / max(sum(n), 1):.0f}% of {sum(n)}")


for step in sorted(S, key=str):
    print(f"step {step}")
    S[step]["mount requests"].update(dict.fromkeys(("sent", "failed (http != 200)"), 0))  # zeros shown too
    S[step]["long waits"].update(dict.fromkeys(LONG_WAITS, 0))
    S[step]["logs"].update(dict.fromkeys({**ENGINE, **(SGLANG if logs or remote else {})}, 0))
    for table, counter in S[step].items():
        print(f"  {table}: {dict(counter)}")
    if sizes[step]:
        xs = sorted(sizes[step])
        print(f"  mount tokens: n {len(xs)} median {xs[len(xs) // 2]} max {xs[-1]}")
    if per_episode[step]:
        print(f"  valid mounts per episode: {dict(collections.Counter(per_episode[step]))}")
    if spread.get(step):
        print(f"  {spread_line(spread[step])}")
if spread:
    print(f"all steps: {spread_line([e for eps in spread.values() for e in eps])}")
