# The Search full run on Alpha (trial search-read-alpha-full-1022), as queue tasks (lib.sh hold: P/queue/*.sh, run in a
# subshell of a controller) with the exact overrides used. Settings beyond search.env: BATCH=16 questions x 8 samples,
# STEPS=190 (3040 train questions, one epoch; AReaL drops a last partial batch), SAVE_EVERY=10, max_head_offpolicyness 2
# (rl_run.coupled.env's OFFPOLICY is not used here: search.env's), judge Qwen3-32B in search.sbatch (this run's record; now JUDGE_MODEL; one H200, SGLang
# --mem-fraction-static 0.85 --disable-prefill-cuda-graph, served as "judge", temperature 0 in training, 0.7 in Val),
# BM25 service (8 workers) on the OpenResearcher corpus. ROOT=/projects/cluster/octopus/octopus,
# N=ROOT/code-<sha>.
#   steps 1-5 (one job, 4 GPUs): judge GPU0, rollout GPU1-2 (EPISODES_PER_SERVER 16: 32 concurrent), training GPU3:
#     source $N/scripts/alpha/lib.sh; run_clean scripts/alpha/search.sbatch $N search-read-alpha-full-1022 BATCH=16 STEPS=190 SAVE_EVERY=10
#   steps 6-9 (plan B, two jobs): rollout on the other job's 4 GPUs (a task there, kept running with sleep):
export RL_ENV_FILE=$N/scripts/alpha/search.env INFER_GPUS=0,1,2,3 TRAIN_GPUS= SERVE_HOST=10.0.4.25
source $RL_ENV_FILE; bash $CODE/scripts/rl_env/rl_run.sh serve && exec sleep infinity
#   ... and the trainer job: judge GPU0, training GPU1-3 (FSDP world 3), 4 x 16 = 64 concurrent (a task in the
#   controller that runs the judge and the service, so both stay up):
export RL_ENV_FILE=$N/scripts/alpha/search.env C=$N BATCH=16 STEPS=190 SAVE_EVERY=10 INFER_GPUS= TRAIN_GPUS=1,2,3 EPISODES_PER_SERVER=16
export EXTRA_SERVERS=10.0.4.25:32300,10.0.4.25:32301,10.0.4.25:32302,10.0.4.25:32303  # search.env then sets OCTOPUS_KEEP_WEIGHT_UPDATES=1
source $RL_ENV_FILE; source $C/scripts/alpha/lib.sh
train_to 190 && val v190 "$(adapter 190)"
# Base Val v0 (rq1-v0, 500 validation questions, k=1): accuracy 0.360. Stopped on Alpha after step 9 (user, 10-22: run
# elsewhere); its recover checkpoint and adapters stay in ROOT/search/areal-exp.
