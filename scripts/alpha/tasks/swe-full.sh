# The SWE full run on Alpha (trial swe-read-alpha-full-1022), as queue tasks (lib.sh hold) with the exact overrides used.
# Beyond rl_run.env: BATCH=8 bundles x 8 samples, SAVE_EVERY=6, max_head_offpolicyness 2 (rl_run.coupled.env's OFFPOLICY);
# the run ends at STEPS=36 (290 train bundles, one epoch less 2). ROOT and N as in search-full.sh. History (actual):
#   steps 1-4 (14:42): STEPS=36, rollout GPU0-2 x EPISODES_PER_SERVER 11 (33 concurrent), training GPU3:
#     source $N/scripts/alpha/lib.sh; run_clean scripts/alpha/pilot.sbatch $N swe-read-alpha-full-1022 BATCH=8 STEPS=36 SAVE_EVERY=6 SPLIT_ROLLOUT=3 EPISODES_PER_SERVER=11
#   step 5 (16:36): the same with STEPS=180 EPISODES_PER_SERVER=16 (48 concurrent)
#   step 6 on (17:23): STEPS=36, training on all 4 GPUs of the episode job (FSDP world 4), rollout on 4 + 4 GPUs of two
#   other jobs, 8 x 6 = 48 concurrent (8 per server if verifier timeouts do not rise). Each rollout job runs one task,
#   kept running by sleep, from a clean environment (NODE_IP: that node's address, 10.0.4.25 and 10.0.4.27 in this run):
exec env -i HOME=$HOME USER=$USER PATH=/usr/local/bin:/usr/bin:/bin SLURM_JOB_ID=$SLURM_JOB_ID CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES bash -c 'export RL_ENV_FILE='"$N"'/scripts/alpha/rl_run.env INFER_GPUS=0,1,2,3 TRAIN_GPUS= SERVE_HOST=NODE_IP; source $RL_ENV_FILE; bash $CODE/scripts/rl_env/rl_run.sh serve && exec sleep infinity'
#   ... and the trainer job (rl_run.env then sets OCTOPUS_KEEP_WEIGHT_UPDATES=1; the addresses are this run's):
source $N/scripts/alpha/lib.sh; run_clean scripts/alpha/pilot.sbatch $N swe-read-alpha-full-1022 BATCH=8 STEPS=36 SAVE_EVERY=6 SPLIT_ROLLOUT=0 EPISODES_PER_SERVER=6 EXTRA_SERVERS=10.0.4.25:31300,10.0.4.25:31301,10.0.4.25:31302,10.0.4.25:31303,10.0.4.27:31300,10.0.4.27:31301,10.0.4.27:31302,10.0.4.27:31303
#   step 8 on (18:16): the same with EPISODES_PER_SERVER=8 (64 concurrent: step 7 waited 1043 s for rollouts); before
#   that restart the 4 servers of one rollout job were restarted: unloading rq1-v7 hung on two of them
# Base Val v0 (rq1-v0, coupled-a-val 45 x 4): solved fraction 0.403 (the pilot's, same settings).
