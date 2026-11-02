"""SGLang prefix-cache thrash check of a rollout run since a given time (bundle-e13 KV fix, 2026-10-18).
usage: kv_thrash_check.py SINCE("YYYY-MM-DD HH:MM") JOBS_DIR LOG [LOG ...]
Per SGLang server log: prefix hit, median pending prefill tokens, mean queue, peak KV; JOBS_DIR's episodes: share of
replies > 30 s. PASS needs every server hit >= 95%, pending median < 4096, queue < 0.2, and slow replies <= 1%.
node-136's server logs are /home/user/rl-remote/state/serve/sglang-g*.log there (pass /nonexistent as JOBS_DIR).
Before the fix (e13 steps 2-4): node-136 servers at 15-82% hit, 120-270k pending tokens, 19% of replies > 30 s: FAIL."""
import glob, json, re, sys
from datetime import datetime
since = datetime.strptime(sys.argv[1], "%Y-%m-%d %H:%M"); jobs = sys.argv[2]
num = lambda k, s: (lambda m: float(m.group(1)) if m else None)(re.search(re.escape(k) + r": ([0-9.]+)", s))
ok = True
for p in sys.argv[3:]:
    new = cached = 0.0; pend = []; queue = []; kv = 0.0
    for line in open(p, errors="replace"):
        m = re.match(r"\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)\]", line)
        if not m or datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S") < since:
            continue
        if "Prefill batch" in line:
            new += num("#new-token", line) or 0; cached += num("#cached-token", line) or 0
            pend.append(num("#pending-token", line) or 0)
        elif "Decode batch" in line:
            queue.append(num("#queue-req", line) or 0); kv = max(kv, num("full token usage", line) or 0)
    if not pend:
        print(f"BAD {p}: no prefill since {since}"); ok = False; continue  # an idle server or a wrong SINCE proves nothing
    hit = cached / max(1, new + cached); med = sorted(pend)[len(pend) // 2]; q = sum(queue) / max(1, len(queue))
    good = hit >= 0.95 and med < 4096 and q < 0.2
    ok &= good
    print(f"{'OK ' if good else 'BAD'} {p.split('/')[-1]}: hit {hit:.1%} pending median {med:.0f} queue {q:.2f} KV peak {kv:.2f} ({len(pend)} prefills)")
n = slow = 0
for rq in glob.glob(f"{jobs}/*/*/agent/requests.jsonl"):
    for l in open(rq):
        r = json.loads(l)
        if datetime.fromtimestamp(r["t"]) >= since:
            n += 1; slow += (r.get("elapsed_s") or 0) > 30
print(f"{'OK ' if slow <= 0.01 * max(1, n) else 'BAD'} replies since {since}: {n}, > 30 s: {slow} ({slow / max(1, n):.1%})")
ok &= slow <= 0.01 * max(1, n)
print("PASS" if ok else "FAIL")
