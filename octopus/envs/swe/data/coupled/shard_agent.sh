#!/bin/bash
# shard_agent.sh HOST WORKDIR: on node-68 / node-138, take this host's shard jobs from node-134 (shards/*/HOST.go), pull the
# bundles over the LAN, run the stage (udocker, nice 19, the job's slot count), push the result log back
# (HOST.<stage>.jsonl, HOST.done) and delete the local copies; polls node-134 once a minute until STOP appears there
H=$1; HW=$2
R=user@198.51.100.134
W=/mnt/disk6/user/coupled-s1a
DS=/home/user/harbor-udocker/datasets
source /home/user/harbor-udocker/scripts/env.sh
unset XAI_API_KEY GROK_MODELS_BASE_URL  # env.sh carries the Grok key; only judge.sh needs it
export PYTHONPATH=$HW/code
HP=/home/user/harbor-udocker/uv-tools/harbor/bin/python
while ! ssh -n -o BatchMode=yes $R "test -f $W/shards/STOP"; do
  for go in $(ssh -n -o BatchMode=yes $R "ls $W/shards/*/$H.go 2>/dev/null"); do
    d=$(dirname $go); job=$HW/jobs/$(basename $d); mkdir -p $job
    rsync -a $R:$d/$H.job.json $R:$d/$H.manifest.jsonl $job/
    read stage name out conc < <(python3 -c "import json,sys; j=json.load(open(sys.argv[1])); print(j['stage'], j['name'], j['out'] or '-', j['conc'])" $job/$H.job.json)
    python3 -c "import json,sys; [print(json.loads(l)['bundle']) for l in open(sys.argv[1])]" $job/$H.manifest.jsonl > $job/bundles.txt
    src=$name; [ $stage = refresh ] && src=$out
    mkdir -p $DS/$name $DS/$name.baseline
    rsync -a -r --files-from=$job/bundles.txt $R:/home/user/harbor-udocker/datasets/$name/ $DS/$name/
    rsync -a $R:/home/user/harbor-udocker/datasets/$name.baseline/ $DS/$name.baseline/
    case $stage in
      accept) cp $job/$H.manifest.jsonl $DS/$name.manifest.jsonl
              nice -n 19 $HP -m octopus.envs.swe.data.build_bundles accept $name $conc > $job/run.log 2>&1
              log=$DS/$name.acceptance-1c.jsonl ;;
      timing) nice -n 19 $HP -m octopus.envs.swe.data.build_bundles timing $name $job/$H.manifest.jsonl $conc > $job/run.log 2>&1
              log=$job/$H.manifest.jsonl.timing.jsonl ;;
      refresh) mkdir -p $DS/$out && rsync -a -r --files-from=$job/bundles.txt $R:/home/user/harbor-udocker/datasets/$out/ $DS/$out/
              cp $job/$H.manifest.jsonl $DS/$out.manifest.jsonl
              nice -n 19 $HP -m octopus.envs.swe.data.build_bundles refresh $name $out $conc > $job/run.log 2>&1
              log=$DS/$out.refresh.jsonl ;;
    esac
    rsync -a $log $R:$d/$H.$stage.jsonl && ssh -n -o BatchMode=yes $R "rm -f $d/$H.go; touch $d/$H.done"
    python3 - "$DS" $name $out $job <<'PY'
import shutil, sys
from pathlib import Path
ds, name, out, job = Path(sys.argv[1]), sys.argv[2], sys.argv[3], Path(sys.argv[4])
for n in {name, out} - {"-"}:  # this job's local copies only (coupled-b-* under datasets/)
    assert n.startswith("coupled-b-")
    for p in [ds / n, ds / f"{n}-valid", ds / f"{n}.baseline"]:
        shutil.rmtree(p, ignore_errors=True)
    for p in ds.glob(f"{n}.*.json*"):
        p.unlink()
    for p in [ds / f"{n}.manifest.jsonl"]:
        p.unlink(missing_ok=True)
shutil.rmtree(job)
PY
    echo "done $stage $name $out"
  done
  sleep 60
done
echo "agent stop"
