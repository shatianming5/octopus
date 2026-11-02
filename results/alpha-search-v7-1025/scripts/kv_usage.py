# Read-only (python3 - SINCE LOG...): per SGLang log, "Decode batch" lines after SINCE (YYYY-mm-dd HH:MM:SS): count,
# mean / median / p90 / max of full token usage, mamba usage, #running-req, #queue-req, and share of lines with full token
# usage < 0.05; then per hour (log clock) and log: mean KV and max #queue-req.
import re, sys, statistics as st
since = sys.argv[1]
pat = re.compile(r"^\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)\] Decode batch, #running-req: (\d+), #full token: \d+, full token usage: ([\d.]+), mamba num: \d+, mamba usage: ([\d.]+).*#queue-req: (\d+)")
for path in sys.argv[2:]:
    run, kv, mb, qu, hr = [], [], [], [], {}
    for line in open(path, errors="replace"):
        m = pat.match(line)
        if m and m.group(1) >= since:
            run.append(int(m.group(2))); kv.append(float(m.group(3))); mb.append(float(m.group(4))); qu.append(int(m.group(5)))
            h = hr.setdefault(m.group(1)[:13], [[], 0]); h[0].append(kv[-1]); h[1] = max(h[1], qu[-1])
    if not kv: print(path.split("/")[-1], "no lines"); continue
    q = lambda xs, p: sorted(xs)[int(p * (len(xs) - 1))]
    print(f"{path.split('/')[-1]:<28} n={len(kv):>6}  KV mean {st.mean(kv):.2f} med {st.median(kv):.2f} p90 {q(kv,.9):.2f} max {max(kv):.2f} "
          f"<5% {sum(k < .05 for k in kv)/len(kv):.0%} | mamba mean {st.mean(mb):.2f} max {max(mb):.2f} | running mean {st.mean(run):.0f} max {max(run)} | queue mean {st.mean(qu):.1f} max {max(qu)}")
    print("  hour: KV mean / queue max  " + "  ".join(f"{h[11:]}h {st.mean(v):.2f}/{q}" for h, (v, q) in sorted(hr.items())))
