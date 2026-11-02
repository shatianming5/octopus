#!/bin/bash
# curve_report.sh: the Search checkpoint curve's results (scripts/beta/tasks/search-ckpt-curve.sh), per finished queue-2 task of
# job 216715: its state, wall time (hold-216715.out run/end lines), and y with errors (v0) or scripts/alpha/paired.py vs v0 on the
# same set (the per-question mean over a task's samples: k=1 vs v0's k=2), and its infra failures by type (rows with an
# error: not scored, so in neither y nor paired). Run on a Beta login node: bash curve_report.sh
BR=/projects/cluster/octopus/beta/octopus; O=$BR/search/pilot/eval; H=$BR/logs/hold-216715.out
PY=$BR/areal/.venv/bin/python; PAIR=$BR/code-d3ca0c2/scripts/alpha/paired.py
for t in $BR/queue-2/*.sh.done $BR/queue-2/*.sh.failed; do
  [ -e "$t" ] || continue; n=$(basename "${t%.*}"); n=${n%.sh}; set=${n##*-}; lab=${n#*-}; lab=${lab%-*}
  r=$(grep -m1 "run .*/$n.sh$" $H | cut -c5-23); e=$(grep -m1 "end .*/$n.sh$" $H | cut -c5-23)
  mins=$(( ($(date -d "$e" +%s) - $(date -d "$r" +%s)) / 60 ))
  pfx=$([ $set = val ] && echo val || echo test); f=$(ls $O/$pfx-$lab-*fork-k*.jsonl 2>/dev/null | head -1)
  [ $lab = v0 ] && f=$O/$pfx-v0-fork-k2.jsonl; base=$O/$pfx-v0-fork-k2.jsonl
  echo "== $n: ${t##*.} in $mins min ($r - $e)"
  [ -f "$f" ] || { echo "   no rows ($f); log tail: $(tail -2 $t.log 2>/dev/null | tr '\n' ' ' | cut -c1-200)"; continue; }
  if [ $lab = v0 ] || [ ! -f $base ]; then $PY -c "
import json, statistics as st; r=[json.loads(l) for l in open('$f')]; ok=[x for x in r if 'error' not in x]
print('   rows', len(r), 'err', len(r)-len(ok), 'y %.3f' % st.mean(x['y'] for x in ok))"
  else $PY $PAIR $base $f | sed 's/^/   /'; fi
  $PY -c "import collections, json; e = collections.Counter(json.loads(l)['error'][:60] for l in open('$f') if '\"error\"' in l)
print('   infra failures (not in y or paired):', dict(e)) if e else None"
done
