#!/bin/bash
# search_step3_check.sh TRIAL: safety stop, on Alpha. After step 3 (+90 s): search_step_report.py, search_audit.py and
# search_audit2.py into P/audit3.txt, then a verdict by fixed criteria into P/audit3.verdict; the run continues only on an
# explicit PASS. FAIL if any of: an audit or the verdict crashed (empty verdict), assign_threads invariant errors, Traceback in
# main.log/actor.log, dropped groups (rollout.log "Workflow execution failed") > max(2, 2% of groups, = "plan group " lines +
# dropped), judge replies without a verdict (plan-group "judge_unparsed") > max(2, 1% of the 8 episodes per group), request
# error rows > 1% of rows, zero-variance groups > 85%, kl_k1 of the last step > 5e-3, a step wall > 25 min.
# FAIL -> search_stop_at.sh at the next step boundary. No step 3 within 120 min of the start -> search_stop_at.sh now.
# Exits when P/stopped exists, TJ has ended (sacct) or no longer runs 20-TRIAL.sh.
T=${1:?}; N=$(cd "$(dirname "$0")/../.." && pwd); R=$(dirname "$N"); P=$R/search/pilot/$T; read -r TJ RJ _ < "$P/jobs" || exit 1
L=$R/search/areal-exp/logs/user/rq1-rl/$T; M=$L/main.log; J=$R/search/rl/jobs-$T; QT=$R/search/pilot/hold-$TJ/queue
S=$N/scripts/alpha; t0=$(date +%s); out=$P/audit3.txt
source "$N/scripts/alpha/lib.sh"  # owner_running: sacct; an error or no answer counts as running, only a final state does not
live() { [ ! -e "$P/stopped" ] && { [ -e "$QT/20-$T.sh" ] || [ -e "$QT/20-$T.sh.running-$TJ" ]; } && owner_running "$TJ"; }  # a TERM'd TJ leaves .running-TJ
until c=$(grep -c 'Train step' "$M" 2>/dev/null); [ "${c:-0}" -ge 3 ]; do
  live || { echo "$(date +%T) $T stopped or not running: check ends"; exit 0; }
  [ $(( $(date +%s) - t0 )) -gt 7200 ] && { echo "FAIL: no step 3 in 120 min" | tee "$P/audit3.verdict"; exec bash "$S/search_stop_at.sh" "$T" 0; }
  sleep 30; done
sleep 90
JL=$P/judge-$TJ.log; [ -f "$JL" ] || JL=$P/judge-$RJ.log  # chosen after step 3, when the controller has started its own judge; else the judge in RJ (JUDGE_HOST)
JL=${JUDGE_LOG:-$JL}  # a judge outside TJ and RJ (search_launch.sh JUDGE_HOST, e.g. on Beta): its log
# shellcheck disable=SC2046  # the server logs of this trial (written since search_launch.sh), one argument each; SERVER_LOGS:
# the dirs of servers outside Alpha instead (search_launch.sh EXTRA_ROLLOUT, e.g. Beta serve_rollout.sh DIRs), word-split
if { python3 "$S/search_step_report.py" "$M" "$JL" $(find ${SERVER_LOGS:-"$R/search/rl-state-search/serve"} -name 'sglang-*.log' ! -name sglang-judge.log -newer "$P/jobs") \
     && python3 "$S/search_audit.py" "$J" "$L/rollout.log" && PYTHONPATH=$N "$R/areal/.venv/bin/python" "$S/search_audit2.py" "$J"; } > "$out" 2>&1; then
python3 - "$out" "$L" <<'PY' > "$P/audit3.verdict"
import re,sys,json
txt=open(sys.argv[1]).read(); L=sys.argv[2]; bad=[]
inv=sum(int(x) for x in re.findall(r"log (?:no requests.jsonl row|is in no leaf and no branch extends it|requests in no leaf|has no graded branch) (\d+)",txt))
if inv: bad.append(f"assign_threads invariant errors {inv}")
for f in ("main.log","actor.log"):
    n=open(f"{L}/{f}",errors="replace").read().count("Traceback")
    if n: bad.append(f"{n} Traceback in {f}")
m=re.search(r"request rows (\d+)",txt); e=re.search(r"request error rows (\d+)",txt)
if m and e and int(e.group(1))>0.01*int(m.group(1)): bad.append("request error rows >1%")
ro=open(f"{L}/rollout.log",errors="replace").read()
d=ro.count("Workflow execution failed"); tot=ro.count("plan group ")+d
if d>max(2,0.02*tot): bad.append(f"{d} of {tot} groups dropped (Workflow execution failed)")
g=[json.loads(x) for x in re.findall(r"plan group (\{.*\})",ro)]
if g and sum(1 for x in g if not x["abs_A_main"])/len(g)>0.85: bad.append("zero-variance groups >85%")
u=sum(x.get("judge_unparsed") or 0 for x in g)
if u>max(2,0.01*8*len(g)): bad.append(f"{u} judge replies without a verdict in {len(g)} groups")
rows=[l for l in txt.splitlines() if re.match(r"^\d+ \|",l)]
for l in rows:
    w=l.split("|")[1].strip()
    try:
        if float(w)>25: bad.append(f"step wall {w} min")
    except ValueError: pass
if rows and float(rows[-1].split("|")[3].split()[-1])>5e-3: bad.append("kl_k1 >5e-3")
print("FAIL: "+"; ".join(bad) if bad else "PASS")
PY
else echo "FAIL: an audit crashed (see $out)" > "$P/audit3.verdict"; fi
v=$(cat "$P/audit3.verdict"); echo "${v:-FAIL: empty verdict (the verdict script crashed, see $out)}"
[ "$v" = PASS ] || { c=$(grep -c 'Train step' "$M"); exec bash "$S/search_stop_at.sh" "$T" $((c + 1)); }
