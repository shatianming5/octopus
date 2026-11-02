#!/bin/bash
# test_pass_stop.sh: pass_stop.py on fake jobs/spool/checkpoint dirs, a `setsid sleep` as the trainer (~40 s).
# Usage: bash test_pass_stop.sh   (exit 1 if any check fails)
set -u +m
PS=$(cd "$(dirname "$0")" && pwd)/pass_stop.py; W=$(mktemp -d); PIDS=""; trap 'kill $PIDS 2>/dev/null; rm -rf "$W"' EXIT
fail=0
check() { if eval "$2"; then echo "ok   $1"; else echo "FAIL $1"; tail -n 3 ps.log; fail=1; fi; }
alive() { kill -0 "$1" 2>/dev/null; }
ck() {  # STEP EPOCH BATCHES: LATEST -> a generation whose dataloader is at sampler epoch EPOCH, BATCHES drawn
  python3 -I - "$@" <<'PY'
import json, os, pickle, sys
step, epoch, n = map(int, sys.argv[1:]); g = f"generation_step{step:08d}"; m = f"ck/checkpoint_generations/{g}/manifest"
os.makedirs(m, exist_ok=True)
json.dump({"global_step": step, "steps_per_epoch": 17}, open(f"{m}/step_info.json", "w"))
pickle.dump([{"_num_yielded": n, "_areal_distributed_sampler_epoch": epoch}], open(f"{m}/dataloader_info.pkl", "wb"))
json.dump({"generation": g, "global_step": step}, open("ck/LATEST.tmp", "w")); os.replace("ck/LATEST.tmp", "ck/LATEST")
PY
}
setup() { rm -rf "$W/$1"; mkdir -p "$W/$1"/{jobs,spool/{resume,ready,taken,trained}}; cd "$W/$1"; }
run() {  # a new trainer, then pass_stop.py on it
  setsid sleep 1000 > /dev/null 2>&1 & TP=$!; echo $TP > train.pid; sleep 1.1
  POLL=0.2 LAG=0.5 python3 -I "$PS" jobs spool train.pid ck state.json >> ps.log 2>&1 & SP=$!; PIDS="$PIDS $TP $SP"; sleep 0.6; }
group() { for i in $(seq 8); do mkdir "jobs/search-$1-$2:$i-0-x$i"; touch "jobs/search-$1-$2:$i-0-x$i.question.json"; done; }
end() { rm -f jobs/search-$1-$2:*.question.json; }

echo "A: fresh pass"; setup A; ck 41 0 5
for i in $(seq 16); do mkdir jobs/search-7-old$i; touch -d '-2 hours' jobs/search-7-old$i; done  # an earlier trainer's
run; group 1 a; group 2 b; end 2 b; touch spool/ready/000042~search-2~a.pt; sleep 1
check "no trigger from one group per question or an earlier trainer's dirs" "alive $TP && grep -q 'no second group' ps.log"
group 2 c; sleep 0.6; check "a second group triggers, 16 episodes open" "grep -q 'search-2 has a second group.*16 episodes' ps.log"
end 1 a; end 2 c; sleep 1.5; touch -d '+1 hour' spool/ready/000045~search-9~late.pt; ck 42 1 2; sleep 0.6
check "a pass group in ready/ holds the stop; one finished after the cut does not count" \
  "alive $TP && grep -q '1 groups of the pass not yet' ps.log"
mv spool/ready/*~a.pt spool/trained/; sleep 0.6; check "held while in trained/" "alive $TP"
echo garbage > ck/LATEST; rm spool/trained/*; sleep 0.6
check "TERM once trained/ is committed, LATEST unreadable meanwhile; pass_stop exits" "! alive $TP && ! alive $SP && grep -q 'LATEST unreadable, step 42 kept' ps.log"

echo "B: restart from a checkpoint past the epoch keeps the cut; the group is in resume/"; setup B; ck 41 0 5
run; group 3 a; end 3 a; group 3 b; touch -d '-1 min' spool/ready/000042~search-3~a.pt; sleep 0.6; end 3 b; sleep 0.6
cut=$(python3 -c "import json; print(json.load(open('state.json'))['cut'])"); kill -TERM -- -$TP; sleep 0.5
check "pass_stop exits with its trainer" "! alive $SP"
mv spool/ready/*.pt spool/resume/; ck 43 1 3; run
check "spool check with the saved cut" "grep -q 'step 43, sampler epoch 1.*spool check' ps.log && grep -q $cut state.json"
check "held while resume/ holds the group" "alive $TP"
rm spool/resume/*; sleep 0.6; check "TERM" "! alive $TP"

echo "C: restart inside the epoch drops the cut"; setup C; ck 41 0 16
echo '{"epoch": 0, "cut": 1700000000}' > state.json; run
check "counting again, cut null" "alive $TP && grep -q '\"cut\": null' state.json"; kill -TERM -- -$TP

echo "D: a pass group left, no progress for K steps"; setup D; ck 50 0 17; touch -d '-1 min' spool/taken/000049~search-4~s.pt
run; check "resumed past the drawn epoch: spool check" "grep -q 'spool check' ps.log"
ck 52 1 3; sleep 0.6; check "2 steps later: held" "alive $TP"
ck 53 1 5; sleep 0.6; check "3 steps later: TERM naming the group" "! alive $TP && grep -q '1 left for 3 steps: 000049~search-4~s.pt' ps.log"

echo "E: a backlog that shrinks each step is not stopped"; setup E; ck 50 0 17
for i in $(seq 40); do touch -d '-1 min' spool/resume/000049~search-$i~s.pt; done; run
for s in $(seq 51 56); do rm spool/resume/000049~search-$((s - 50))~s.pt; ck $s 1 2; sleep 0.5; done
check "6 steps past the cut, 34 left but shrinking: held" "alive $TP"
ck 59 1 9; sleep 0.6; check "then 3 steps without progress: TERM" "! alive $TP && grep -q '34 left for [34] steps' ps.log"

echo "F: no second group; the checkpoint that drew the epoch, then QUEUED (48) groups"; setup F; ck 41 0 15; run
group 1 a; end 1 a; for i in 1 2 3; do mkdir jobs/search-99-a:$i-0-x$i; done; ck 42 0 17; sleep 0.6  # 99: partial
for i in 4 5 6 7 8; do mkdir jobs/search-99-a:$i-0-x$i; done
for q in $(seq 100 146); do group $q a; end $q a; done; ck 43 1 2; sleep 0.6
check "47 groups since (the rest of a group started before does not count): no trigger" "! grep -q 'every prompt of the pass' ps.log && grep -q '47 groups since the epoch' ps.log"
group 147 a; sleep 0.6; check "the 48th triggers" "grep -q '48 groups started since' ps.log"
end 147 a; sleep 1; check "its episodes end, empty spool: TERM" "! alive $TP"

echo "G: episodes open at the trigger that never end"; setup G; ck 41 0 5; run
group 1 a; end 1 a; group 1 b; group 2 a; sleep 0.6; check "triggered, 16 open" "grep -q '16 episodes open' ps.log"
ck 43 0 9; sleep 0.4; end 2 a; sleep 0.4; ck 45 0 13; sleep 0.6; check "8 ended at step 43: 2 steps from it, held" "alive $TP"
ck 46 0 15; sleep 0.6; check "3 steps without progress: TERM" "! alive $TP && grep -q '8 left for 3 steps: search-1-b' ps.log"

echo "H: question.json churn (files vanish between glob and stat)"; setup F; ck 41 0 5; run
python3 -I - jobs <<'PY'
import os, sys, time
t = time.time()
while time.time() - t < 5:
    for i in range(50):
        p = f"{sys.argv[1]}/search-{i}-a-x.question.json"; open(p, "w").close(); os.unlink(p)
PY
check "pass_stop survives" "alive $SP"; kill -TERM -- -$TP
exit $fail
