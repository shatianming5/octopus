#!/bin/bash
# the 4 coupled-a-test SIFs Docker Hub refused (unauthenticated pull limit, 10-23): through mirror.gcr.io, plus their
# root tars, by digest: the latest tags' manifest digests on Docker Hub (equal to the mirror's; HEAD /v2/.../manifests/latest, 10-23):
#   prettytable sha256:d2a54392e1ccdc5b407084b63f5e6ae3011568a875703f1d8fcd2e029983ecd9
#   nikola      sha256:0457d5e95af2bdf812d47c5d54a5c793ed35dd5e9def1f8a795b92c10f903bf7
#   pdfplumber  sha256:eb85075e1f4d7681121842ec978b83c2859d2bed0aa0c1aaa794a857e1f95070
#   qrcode      sha256:a1c5942d72cccd58e49d1cc948dd6439b659c90260229e1e262d9b5ad73423ac
# Run inside an allocation (srun --jobid=J --overlap -c4 --mem=64G): /dev/shm trial dirs vanished on alphacpu01
export PATH=/projects/cluster/octopus/octopus/apptainer/bin:$PATH; S=/projects/cluster/octopus/octopus/sif
for bd in prettytable_1776_prettytable.ca90b055@d2a54392e1ccdc5b407084b63f5e6ae3011568a875703f1d8fcd2e029983ecd9 \
    getnikola_1776_nikola.0f4c230e@0457d5e95af2bdf812d47c5d54a5c793ed35dd5e9def1f8a795b92c10f903bf7 \
    jsvine_1776_pdfplumber.02ff4313@eb85075e1f4d7681121842ec978b83c2859d2bed0aa0c1aaa794a857e1f95070 \
    lincolnloop_1776_python-qrcode.456b01d4@a1c5942d72cccd58e49d1cc948dd6439b659c90260229e1e262d9b5ad73423ac; do
  b=${bd%@*}; d=${bd#*@}
  n=swesmith.x86_64.$b; t=$(mktemp -d /tmp/user-pull-XXXXXX)
  [ -f $S/$n.sif ] || { APPTAINER_TMPDIR=$t singularity pull --disable-cache $S/$n.sif.part docker://mirror.gcr.io/jyangballin/$n@sha256:$d > $S/logs/$n.log 2>&1 && mv $S/$n.sif.part $S/$n.sif || { echo "FAIL $n"; tail -n 2 $S/logs/$n.log; }; }
  if [ -f $S/$n.sif ] && [ ! -f $S/$n.root.tar ]; then id=$(singularity sif list $S/$n.sif | awk '/Squashfs/ {print $1; exit}')
    singularity sif dump $id $S/$n.sif > $t/fs.sqfs && unsquashfs -q -n -d $t/fs $t/fs.sqfs root > /dev/null && tar -cf $S/$n.root.tar.part -C $t/fs/root . && mv $S/$n.root.tar.part $S/$n.root.tar || echo "FAIL root.tar $n"; fi
  rm -rf $t; ls -la $S/$n.sif $S/$n.root.tar 2>&1 | awk '{print $5, $9}'
done
