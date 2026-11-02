#!/bin/bash
#SBATCH -J octopus-setup
#SBATCH -p cpu
#SBATCH -A cluster_project
#SBATCH --qos=standard
#SBATCH -c 32
#SBATCH --mem=200G
#SBATCH -t 8:00:00
# CPU-only setup of the Alpha deployment (the login node caps a user at 2 CPUs and 8 GB; compute nodes reach PyPI, HF,
# GitHub and Docker Hub), after `deploy.sh ROOT fetch` on the login node: deploy.sh install; a check that Apptainer mounts
# a SIF in place (the module's 1.1.9 has no setuid starter and no squashfuse, so each exec would unpack the image into a
# temporary sandbox: Apptainer's unprivileged install, EPEL 9's, goes to ROOT/apptainer, first on rl_run.env's PATH,
# from the login node, which has rpm2cpio: `install-unprivileged.sh -d el9 ROOT/apptainer` of apptainer's tools/; the
# login node itself cannot open /dev/fuse; then `systemd cgroups = no` in its etc/apptainer/apptainer.conf: in a Slurm
# job there is no user systemd bus and `instance start` fails on it); one SIF per coupled-a image (scripts/nvwulf/pull_sifs.sh, built in node-local /tmp); one oracle trial per image
# (its first coupled-a task dir, train or val; the reference fixes must score 1.0), summary in ROOT/check/oracle-JOB.txt.
#   sbatch -o ROOT/logs/setup-%j.log ROOT/code-SHA/scripts/alpha/setup.sbatch ROOT/code-SHA
set -uo pipefail
C=$1; source "$C/scripts/alpha/rl_run.env"
step() { echo "=== $(date +%T) $*"; }
[ -d "$CURSOR/sglang/sglang" ] || bash "$C/scripts/deploy.sh" "$OCTO_ENV" install || exit 1  # done: a rebuild would rm -rf a live overlay

step apptainer
sed -i 's/^systemd cgroups = yes/systemd cgroups = no/' "$OCTO_ENV"/apptainer/*/etc/apptainer/apptainer.conf
T=/tmp/$USER-octopus-$SLURM_JOB_ID; mkdir -p "$T"; trap 'rm -rf "$T"' EXIT
export SINGULARITY_TMPDIR=$T APPTAINER_TMPDIR=$T
runs() { out=$(singularity exec --containall "$1" true 2>&1) && ! grep -q "temporary sandbox" <<< "$out"; }  # no per-exec unpack
singularity -q pull "$T/probe.sif" docker://busybox:1.36.1 || exit 1
echo "$(command -v singularity): $(singularity --version)"
runs "$T/probe.sif" || { echo "singularity cannot run a SIF without unpacking it: $out"; exit 1; }
/usr/bin/time -f "exec of a SIF: %e s" singularity exec --containall "$T/probe.sif" true

step SIFs
for d in ${DATASETS:-coupled-a-val coupled-a-train}; do  # DATASETS: e.g. coupled-a-test
  bash "$C/scripts/nvwulf/pull_sifs.sh" "$OCTOPUS_HARBOR/datasets/$d" "$SINGULARITY_SIF_DIR" 8; cat "$SINGULARITY_SIF_DIR/failed.txt"
done

step "oracle trials (one per image)"
O=$OCTO_ENV/check/oracle-$SLURM_JOB_ID; mkdir -p "$O"
"$AREAL_PY" - "$OCTOPUS_HARBOR/datasets" ${DATASETS:-coupled-a-val coupled-a-train} > "$O/tasks.txt" <<'PY'
import re, sys
from pathlib import Path
seen = {}
for ds in sys.argv[2:]:
    for d in sorted(Path(sys.argv[1], ds).iterdir()):
        if (d / "task.toml").is_file():
            seen.setdefault(re.search(r'^docker_image\s*=\s*"([^"]+)"', (d / "task.toml").read_text(), re.M).group(1), d)
print("\n".join(map(str, seen.values())))
PY
oracle() { cd "$OCTOPUS_HARBOR" && PYTHONPATH=$C "$OCTOPUS_HARBOR/bin/harbor" run -p "$1" -a oracle -e "$OCTOPUS_HARBOR_ENV" -n 1 \
  -o "$O/jobs" --job-name "$(basename "$1")" --environment-build-timeout-multiplier 6 -y > "$O/$(basename "$1").log" 2>&1; }
export -f oracle; export O C
xargs -P 8 -I{} bash -c 'oracle "$@"' _ {} < "$O/tasks.txt"
"$AREAL_PY" - "$O" < /dev/null > "$O.txt" <<'PY'
import json, sys
from pathlib import Path
o = Path(sys.argv[1]); bad = 0
for t in Path(o, "tasks.txt").read_text().split():
    r = [json.loads(f.read_text()).get("verifier_result") or {} for f in (o / "jobs" / Path(t).name).glob("*/result.json")]
    rew = ((r[0].get("rewards") or {}).get("reward")) if r else None
    bad += rew != 1.0; print(f"{rew}\t{t}")
print(f"{bad} of {len(Path(o, 'tasks.txt').read_text().split())} oracle trials not 1.0")
PY
cat "$O.txt"; step setup done
