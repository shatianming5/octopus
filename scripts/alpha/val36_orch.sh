#!/bin/bash
# after SWE step 36: Val v36 in three jobs at once (144129 part 0, 144687 part 1, 144123 part 2 if before its 07:19 TERM),
# then merge into P/val-v36.jsonl and the paired bootstrap against P/val-v0.jsonl
R=/projects/cluster/octopus/octopus; P=$R/swe/pilot/swe-read-alpha-full-1022
CK=$R/swe/areal-exp/checkpoints/user/rq1-rl/swe-read-alpha-full-1022/default
T='exec env -i HOME=$HOME USER=$USER PATH=/usr/local/bin:/usr/bin:/bin SLURM_JOB_ID=$SLURM_JOB_ID CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES bash '$R'/logs/val_part.sh'
until ls $CK/*globalstep35/adapter_model.safetensors > /dev/null 2>&1 && grep -q 'step-36 adapter\|gate runs done' $P/queue/08-swe-eps8.sh.log; do sleep 60; done
echo "$(date +%T) step-36 adapter, trainer done"
killsleep() { srun --jobid=$1 --overlap -n1 -c1 bash -c 'for p in $(ps -u user -o pid=,args= | awk "\$2==\"sleep\" && \$3==\"infinity\" {print \$1}"); do kill $p; done'; }
task[0]=$P/queue/09-val-v36-part0.sh; echo "$T 0" > ${task[0]}
task[1]=$R/search/pilot/hold-144687/queue/05-val-v36-part1.sh; echo "$T 1" > ${task[1]}; killsleep 144687
parts="0 1"
if [ $(date +%s) -lt $(date -d '2026-10-23 05:30' +%s) ]; then
  task[2]=$R/search/pilot/search-read-alpha-full-1022/queue/10-val-v36-part2.sh; echo "$T 2" > ${task[2]}; killsleep 144123; parts="0 1 2"
else  # too late for 144123: part 2 after part 0 in 144129
  task[2]=$P/queue/10-val-v36-part2.sh; echo "$T 2" > ${task[2]}
  parts="0 1 2"
fi
echo "$(date +%T) queued parts $parts"
for k in $parts; do until [ -f $P/val-v36-part$k.log ] && grep -q 'infra failures' $P/val-v36-part$k.log; do
  [ -e ${task[$k]}.failed ] && { echo "$(date +%T) part $k failed: ${task[$k]}.log"; exit 1; }; sleep 60; done; echo "$(date +%T) part $k done: $(tail -n 1 $P/val-v36-part$k.log | cut -c1-200)"; done
cat $P/val-v36-part0.jsonl $P/val-v36-part1.jsonl $P/val-v36-part2.jsonl > $P/val-v36.jsonl
python3 - $P <<'PY'
import json, random, statistics as st, sys, collections
P = sys.argv[1]
def load(n):
    d = {}
    for l in open(f"{P}/val-{n}.jsonl"):
        r = json.loads(l); d[(r["task_dir"], r["sample"])] = r
    return d
a, b = load("v0"), load("v36")
for n, d in (("v0", a), ("v36", b)):
    v = list(d.values()); ok = [r for r in v if "error" not in r]
    print(n, "n", len(v), "err", len(v) - len(ok), "y %.3f" % st.mean(r["y"] for r in ok), "passes %.3f" % st.mean(r["y"] == 1 for r in ok),
          "forked %.0f%%" % (100 * st.mean(bool(r.get("forks")) for r in ok)), "branches %.2f" % st.mean(r.get("branches") or 0 for r in ok))
by = collections.defaultdict(lambda: [[], []])
for k, r in a.items():
    if "error" not in r: by[k[0]][0].append(r["y"])
for k, r in b.items():
    if "error" not in r: by[k[0]][1].append(r["y"])
ts = sorted(t for t in by if by[t][0] and by[t][1]); d = [st.mean(by[t][1]) - st.mean(by[t][0]) for t in ts]
rng = random.Random(0); bs = sorted(st.mean(rng.choices(d, k=len(d))) for _ in range(10000))
print("bundles", len(ts), "paired diff %.3f" % st.mean(d), "95%% CI [%.3f, %.3f]" % (bs[250], bs[9749]),
      "up/down/same", sum(x > 0 for x in d), sum(x < 0 for x in d), sum(x == 0 for x in d))
PY
echo "$(date +%T) VAL36 DONE"
