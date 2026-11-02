# Source this for any Harbor work on NVwulf, in the job script and (as $OCTOPUS_HARBOR/scripts/env.sh, a symlink to this
# dir) in every Harbor episode. The deployment root N holds code/ (this repo), the venvs, models/, sif/ and datasets/.
export N=$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/../../.." && pwd)
export OCTOPUS_HARBOR=$N OCTOPUS_CODE=$N/code PYTHONPATH=$N/code
export PATH="$N/harbor-venv/bin:$PATH"
export OCTOPUS_HARBOR_ENV=octopus.envs.swe.singularity:SingularityEnvironment
export SINGULARITY_SIF_DIR=$N/sif SINGULARITY_TRIALS_DIR=/dev/shm
export OCTOPUS_EPISODE_MEM=  # no user systemd in a Slurm job: the job's cgroup caps the episodes
export UDOCKER_CREATE_SLOTS=8 UDOCKER_THREADS=256 OCTOPUS_HTTP_WORKERS=256
export LITELLM_LOCAL_MODEL_COST_MAP=True  # no GitHub fetch at import
# SGLang turns on JIT DeepGEMM on sm90 (never on node-94's sm89), and the pinned deep_gemm lacks the API it imports at
# startup; this bf16 model has no FP8 GEMMs for it anyway
export SGLANG_ENABLE_JIT_DEEPGEMM=0
