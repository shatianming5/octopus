#!/bin/bash
# Accept + scan the bundles of some repos of dataset NAME on this host, everything pulled from the build host over the
# LAN; results back as datasets/NAME.acceptance.<host>.jsonl and NAME.leaks.<host>.json (then `cooperbench.py merge NAME`
# there).   usage: cooperbench_shard.sh SRC CODE NAME CONC REPO...
#   SRC  user@<build host address reachable from here>; its /home/user/harbor-udocker has datasets/NAME,
#        NAME.manifest.full.jsonl, cooperbench/ds (the CooperBench clone), cooperbench/CODE (a pushed commit's octopus/)
#        and the blobs in /mnt/disk6/user/cooperbench/img/blobs
#   CONC containers at once (~60% of the cores);  REPO e.g. pallets_click dspy (manifest "repo" values)
set -eu
SRC=$1; CODE=$2; NAME=$3; CONC=$4; shift 4
N=/home/user/harbor-udocker; C=$N/cooperbench; D=$N/datasets; H=$(hostname)
R="rsync -a"
mkdir -p $C/img/blobs $C/ds $D/$NAME
if [ ! -x $N/uv-tools/harbor/bin/python ]; then  # bare host: the Harbor/udocker skeleton and uv's CPython
  mkdir -p $N/udocker /home/user/.local/share/uv/python $N/golden $N/containers $N/ipc $N/scripts $N/bin
  $R $SRC:$N/uv-tools $N/ && $R $SRC:$N/udocker/bin $SRC:$N/udocker/lib $N/udocker/ && $R $SRC:$N/bin/udocker $SRC:$N/bin/harbor $N/bin/
  $R $SRC:/home/user/.local/share/uv/python/cpython-3.13.13-linux-x86_64-gnu /home/user/.local/share/uv/python/
  $R $SRC:$N/scripts/env.sh $N/scripts/ && sed -i '/UDOCKER_OVERLAY\|XAI_API_KEY\|GROK_/d' $N/scripts/env.sh
fi
$R $SRC:$C/$CODE $C/ && $R $SRC:/mnt/disk6/user/cooperbench/img/blobs/ $C/img/blobs/
for r in "$@"; do $R $SRC:$C/ds/${r}_task $C/ds/; done
$R $SRC:$D/$NAME.manifest.full.jsonl $D/
pat=$(printf '"repo": "%s"|' "$@"); grep -E "${pat%|}" $D/$NAME.manifest.full.jsonl > $D/$NAME.manifest.jsonl
python3 -c "import json,sys;[print(json.loads(l)['bundle']) for l in open(sys.argv[1])]" $D/$NAME.manifest.jsonl > $C/bundles.txt
$R -r --files-from=$C/bundles.txt $SRC:$D/$NAME/ $D/$NAME/
cd $N && source scripts/env.sh && unset XAI_API_KEY GROK_MODELS_BASE_URL && export PYTHONPATH=$C/$CODE  # no Grok here
P="nice -n 10 $N/uv-tools/harbor/bin/python $PYTHONPATH/octopus/envs/swe/data/cooperbench.py"
$P fetch $C/ds $C/img 4 && cd $D && $P accept $NAME $CONC && $P scan $C/ds $NAME $CONC
$R $D/$NAME.acceptance.jsonl $SRC:$D/$NAME.acceptance.$H.jsonl
$R $D/$NAME.leaks.json $SRC:$D/$NAME.leaks.$H.json
echo SHARD_DONE
