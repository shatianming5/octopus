# Read-only (python3 hijack_net.py units-*.tsv): net switch of read_thread mounts = (OWN->OTHER - OTHER->OWN) / n over
# the judged reads (pre = R's last action before the read, post = its next reply), with a 95% half-width (1.96 x iid SE).
# "corrected" drops the reads whose post reply calls read_thread again (it judges a read, not R's work) and those whose
# pre action is a fork call (R's own fork, often the one that created S), as the 10-25 audit did. Split by relation.
import math, sys


def net(rs):
    v = [(p == "OWN" and q == "OTHER") - (p == "OTHER" and q == "OWN") for p, q in rs]
    n = len(v); m = sum(v) / n
    return f"n={n} net={m:+.3f} ±{1.96 * math.sqrt(sum((a - m) ** 2 for a in v) / (n - 1) / n):.3f}" if n > 1 else f"n={n}"


for f in sys.argv[1:]:
    U = [l.rstrip("\n").split("\t") for l in open(f)]
    C = [u for u in U if u[6] == "0" and u[7] == "0"]
    print(f"== {f}\n raw       {net([(u[4], u[5]) for u in U])}\n corrected {net([(u[4], u[5]) for u in C])}")
    for r in ("sib", "other", "desc", "anc"):
        s = [(u[4], u[5]) for u in C if u[3] == r]
        if s: print(f"   {r:5s} {net(s)}")
