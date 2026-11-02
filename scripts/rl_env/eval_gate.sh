#!/bin/bash
# Eval-done gate, run on node-67 (node-94 cannot reach node-67; node-67 reaches node-94). Every 120 s: once EVAL_DIR
# results/step<N>.done exist for every N in STEPS, none of them is .failed and no eval lane (driver.sh / run_step.sh) of
# EVAL_DIR runs, writes node-94:MARKER (rl_autolaunch.sh there then checks GPUs and launches the next run) and exits.
# Each new step<N>.failed is reported to node-94:LOG; lanes retry failed steps, and a failed step has no .done.
# usage: setsid nohup eval_gate.sh EVAL_DIR "10 30 50 53" MARKER LOG > /dev/null 2>&1 < /dev/null &
# e.g. eval_gate.sh /home/user/harbor-udocker/evals/bundle-e8f "10 30 50 53" /home/user/octopus-sft/e8f-evals.done
#      /home/user/octopus-sft/e9_autolaunch.out
E=$1; STEPS=$2; MARK=$3; OUT=$4; G94=user@10.0.1.94
say() { echo "$(date '+%F %T') node-67 gate: $*" | tee -a $E/logs/eval_gate.log | ssh -o BatchMode=yes -o ConnectTimeout=20 $G94 "cat >> $OUT"; }
lanes() { n=0; for p in $(pgrep -u $USER -f "[d]river.sh|[r]un_step.sh"); do { readlink /proc/$p/cwd; tr '\0' ' ' < /proc/$p/cmdline; tr '\0' ' ' < /proc/$p/environ; } 2>/dev/null | grep -q "$E" && n=$((n + 1)); done; echo $n; }
say "armed: steps $STEPS .done, none .failed, no lane of $E; poll 120 s"
seen=""
while true; do
  ok=1
  for N in $STEPS; do
    f=$E/results/step$N.failed
    if [ -f $f ]; then
      ok=0; k="$N@$(stat -c %Y $f)"
      case " $seen " in *" $k "*) ;; *) seen="$seen $k"; say "step$N.failed: $(head -c 300 $f) -- NOT launching while it stands (lanes retry it)";; esac
    fi
    [ -f $E/results/step$N.done ] || ok=0
  done
  if [ $ok = 1 ] && [ "$(lanes)" = 0 ]; then
    r=$(for N in $STEPS; do echo -n "step$N: $(cat $E/results/step$N.done); "; done)
    echo "$(date '+%F %T') $r" | ssh -o BatchMode=yes -o ConnectTimeout=20 $G94 "cat > $MARK" && { say "evals done, marker written: $r"; exit 0; }
  fi
  sleep 120
done
