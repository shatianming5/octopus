#!/bin/bash
# one oracle trial per coupled-a-test image (its first task dir) in this allocation; reward 1.0 each
C=/projects/cluster/octopus/octopus/code-dced0d2; source $C/scripts/alpha/rl_run.env; O=/projects/cluster/octopus/octopus/check/oracle-test-$SLURM_JOB_ID-$$; mkdir -p $O
for i in $(grep -h ^docker_image /projects/cluster/octopus/octopus/harbor/datasets/coupled-a-test/*/task.toml | sed -E 's/.*"(.*)"/\1/' | sort -u); do
  grep -l "$i" /projects/cluster/octopus/octopus/harbor/datasets/coupled-a-test/*/task.toml | head -1 | xargs dirname; done > $O/tasks.txt
oracle() { cd $OCTOPUS_HARBOR && PYTHONPATH=$C $OCTOPUS_HARBOR/bin/harbor run -p "$1" -a oracle -e "$OCTOPUS_HARBOR_ENV" -n 1 -o "$O/jobs" --job-name "$(basename "$1")" --environment-build-timeout-multiplier 6 -y > "$O/$(basename "$1").log" 2>&1; }
export -f oracle; export O C
xargs -P 7 -I{} bash -c 'oracle "$@"' _ {} < $O/tasks.txt
for t in $(cat $O/tasks.txt); do n=$(basename $t); echo "$(python3 -c 'import json,sys,glob; f=glob.glob(sys.argv[1]); print(((json.load(open(f[0])).get("verifier_result") or {}).get("rewards") or {}).get("reward") if f else None)' "$O/jobs/$n/*/result.json") $n"; done | tee $O/rewards.txt
ok=$(grep -c "^1.0 " $O/rewards.txt); n=$(wc -l < $O/tasks.txt)
[ "$ok" = "$n" ] && echo "oracle $ok/$n 1.0" || { echo "FAIL: oracle $ok/$n 1.0"; exit 1; }
