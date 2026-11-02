"""steps6.py CACHE.jsonl MAIN_LOG [W=6]: per W training steps of one trial (episodes binned by finish time as in steps.py,
whose cache this reads: run steps.py first), the branch merge rate (merged / branches), abort rate (ended with abort / branches),
passed rate (committed and passed / branches) and the fork rate (forked / scored episodes)."""
import json, re, sys
from datetime import datetime
cache, log = sys.argv[1], sys.argv[2]; w = int(sys.argv[3]) if len(sys.argv) > 3 else 6
done = [datetime.strptime(m.group(1), "%Y%m%d-%H:%M:%S").timestamp()
        for m in re.finditer(r"(\d{8}-\d\d:\d\d:\d\d)\.\d+ StatsLogger INFO: Epoch \S+ Step \d+/\d+ Train step", open(log).read())]
trial = log.rstrip("/").split("/")[-2]
bins = {}
for r in map(json.loads, open(cache)):
    if f"/jobs-{trial}/" not in r["trial"] or "error" in r:
        continue
    s = next((i + 1 for i, d in enumerate(done) if r["end"] <= d), None)
    if s is not None:  # episodes of a step not yet done are left out
        bins.setdefault((s - 1) // w, []).append(r)
print(f"{trial}: steps eps fork merge abort passed (per branch)")
for k in sorted(bins):
    rs = bins[k]; nb = sum(r["nbr"] for r in rs) or 1
    print(f"  {k * w + 1:2d}-{min(k * w + w, len(done)):2d} {len(rs):5d} {sum(r['forked'] for r in rs) / len(rs):.2f} "
          f"{sum(r['merged'] for r in rs) / nb:.2f} {sum(r['nbr'] - r['commit'] - r['unclosed'] for r in rs) / nb:.2f} {sum(r['cpass'] for r in rs) / nb:.2f}")
