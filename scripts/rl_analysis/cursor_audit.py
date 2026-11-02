"""Cursor-protocol acceptance over a run's OCTOPUS_CURSOR_AUDIT records (octopus/serving/cursor_audit.py) and its episodes'
requests.jsonl (docs/cursor-protocol.md "Live acceptance").

usage: python cursor_audit.py AUDIT_DIR JOBS_DIR
Prints: oracle mismatches over the cursor requests by kind (turn / fork child / rewind / control; exit 1 on any
mismatch), full sends from the client and why, the per-request cost by prompt length L (medians), the proxy->SGLang
hops that sent all ids and why, 404s, and the per-server load."""
import collections, glob, json, statistics, sys
from pathlib import Path

audit_dir, jobs_dir = sys.argv[1:3]
recs = [json.loads(x) for f in glob.glob(f"{audit_dir}/*.jsonl") for x in Path(f).read_text().splitlines() if x]
prompts = [r for r in recs if r["kind"] == "prompt"]
hops = collections.defaultdict(list)  # rid -> its /generate segments in order (a resume after a pause adds one)
for r in sorted((r for r in recs if r["kind"] == "hop"), key=lambda r: r["t"]):
    hops[r["rid"]].append(r)

rows, threads = {}, collections.defaultdict(list)  # response id -> client row; (episode, thread) -> its replies
n404 = 0
for f in glob.glob(f"{jobs_dir}/*/*/agent/requests.jsonl"):
    for x in Path(f).read_text().splitlines():
        row = json.loads(x)
        n404 += row.get("http_code") == 404
        if row.get("http_code") == 200 and row.get("id"):
            row["ep"] = f
            rows[row["id"]] = row
            threads[f, row["thread"]].append(row)
for v in threads.values():
    v.sort(key=lambda r: r["t"])


def kind(p: dict) -> str:
    row = rows.get(p["id"])
    if row is None:
        return "no client row"
    if row["thread"] == "control":
        return "control"
    if p["cursor"] is None:
        return "full"
    own = [r["id"] for r in threads[row["ep"], row["thread"]] if r["t"] < row["t"]]
    return "turn" if own and own[-1] == p["cursor"] else "rewind" if p["cursor"] in own else "fork child"


cur = [p for p in prompts if p["cursor"] is not None]
by = collections.defaultdict(lambda: [0, 0])
for p in cur:
    k = kind(p)
    by[k][0] += 1
    by[k][1] += not p["match"]
bad = sum(v[1] for v in by.values())
print(f"cursor requests {len(cur)}, oracle mismatches {bad}  (by kind: requests, mismatches)")
for k, (n, m) in sorted(by.items()):
    print(f"  {k:14s} {n:5d} {m:3d}")
for p in [p for p in cur if not p["match"]][:5]:
    print("  MISMATCH", {k: p.get(k) for k in ("id", "cursor", "L", "oracle_L", "first_diff")})

full = [p for p in prompts if p["cursor"] is None]
why = collections.Counter()
for p in full:
    row = rows.get(p["id"])
    if row is None:
        why["no client row"] += 1
    elif not (own := [r for r in threads[row["ep"], row["thread"]] if r["t"] < row["t"]]):
        why[f"first request of a thread ({'control' if row['thread'] == 'control' else 'main' if row['thread'] == 'main' else 'branch'})"] += 1
    else:  # a shorter prompt than the last reply's tokens: a rewind back before the thread's first kept reply
        cause = "rewind" if row["prompt_tokens"] < own[-1]["prompt_tokens"] + own[-1]["completion_tokens"] else "rewritten reply"
        why[f"later request ({row['thread'] if row['thread'] in ('main', 'control') else 'branch'}: {cause})"] += 1
print(f"full sends from the client {len(full)}: {dict(why)}")

B = [(0, 8192, "<8k"), (8192, 32768, "8-32k"), (32768, 131072, "32-128k"), (131072, 1 << 40, ">128k")]
med = lambda xs: round(statistics.median(xs), 2) if xs else "-"
print("\nper cursor request, medians: L | Δ tokens | client->proxy bytes cursor / full | proxy prompt build ms cursor / "
      "full render+tokenize | proxy->SGLang bytes cursor / all ids")
print("| L | n | Δ tokens | client→proxy bytes, cursor | full | prompt build ms, cursor | full | proxy→SGLang bytes, cursor | all ids |")
print("|---|---|---|---|---|---|---|---|---|")
for lo, hi, name in B:
    ps = [p for p in cur if lo <= p["L"] < hi]
    if not ps:
        continue
    sent, cf, hb, hf = [], [], [], []
    for p in ps:
        row = rows.get(p["id"])
        if row:  # bytes the client sent; full mode's = the same body with the whole history as messages
            sent.append(row["payload_bytes"])
            cf.append(row["payload_bytes"] - p["msg_bytes"] + p["full_msg_bytes"])
        if hops.get(p["rid"]):
            hb.append(hops[p["rid"]][0]["bytes"])
            hf.append(hops[p["rid"]][0]["full_bytes"])
    print(f"| {name} | {len(ps)} | {med([p['delta_tokens'] for p in ps])} | {med(sent)} | {med(cf)} | "
          f"{med([p['build_ms'] for p in ps])} | {med([p['oracle_ms'] for p in ps])} | {med(hb)} | {med(hf)} |")

first = [v[0] for v in hops.values()]
resumes = [h for v in hops.values() for h in v[1:]]
fw = collections.Counter(h["full_why"] for h in first if not h["cursor"])
print(f"\nproxy->SGLang requests {len(first)}: {sum(h['cursor'] for h in first)} with a cursor, all ids {sum(fw.values())}"
      f" {dict(fw)}; resumed segments {len(resumes)} ({sum(h['cursor'] for h in resumes)} from a cursor); "
      f"404 resends {sum(h['resend_404'] for v in hops.values() for h in v)}")
print(f"client 404s (cursor unknown to the proxy): {n404}")
srv = collections.defaultdict(lambda: [0, 0, 0, 0, 0])
for h in first:
    srv[h["server"]][0] += 1
    srv[h["server"]][1] += h["cursor"]
    srv[h["server"]][2] += h["n_input"]
    srv[h["server"]][3] += h["bytes"]
    srv[h["server"]][4] += h["full_bytes"]
print("per server: requests, from a cursor, prompt tokens, bytes sent, bytes with all ids")
for s, (n, c, t, b, f) in sorted(srv.items()):
    print(f"  {s} {n:6d} {c:6d} {t:12d} {b:12d} {f:14d}")
sys.exit(1 if bad else 0)
