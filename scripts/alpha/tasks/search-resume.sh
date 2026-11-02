# The Search full trial's resume on Alpha job 145532 (10-23, user: the step-16 recover, unchanged settings), as queue tasks of
# its hold (pilot/hold-145532/queue). C=ROOT/code-d57c294: the training code of the 145532 controller (main's search outcome
# code is not mixed into the baseline run). ROOT/search/HOLD must not list 145532, or search.sbatch holds instead of training.
#
# 03 (13:40, EPISODES_PER_SERVER=16, as search.env):
#   source $C/scripts/alpha/lib.sh; run_clean scripts/alpha/search.sbatch $C search-read-alpha-full-1022 BATCH=16 STEPS=190 SAVE_EVERY=10 SPLIT_ROLLOUT=2 EPISODES_PER_SERVER=16
# 04 (restart from the step-16 recover, 8 per rollout server; needs search.env's EPISODES_PER_SERVER=${EPISODES_PER_SERVER:-16}):
#   source $C/scripts/alpha/lib.sh; run_clean scripts/alpha/search.sbatch $C search-read-alpha-full-1022 BATCH=16 STEPS=190 SAVE_EVERY=10 SPLIT_ROLLOUT=2 EPISODES_PER_SERVER=8
#
# No EXTRA_SERVERS: judge GPU0, rollout GPU1-2, training GPU3. search.sbatch resumes the trainer from the recover checkpoint
# and holds when train_to ends. That hold is nested inside the task and never returns, so a stop of the trainer alone does
# not end the task: to finish it, TERM the nested search.sbatch (its trap stops the judge and the BM25 service, then
# stop_stage), and the controller's hold moves on to the next queue file. scripts/alpha/search_epoch_stop.sh of d57c294 (JOB=145532,
# DEADLINE=epoch of 2026-10-24 05:45 EDT, TASKLOG=that task's log) stops the trainer at the step boundary after coverage
# and runs Val of that step, paired with Val v0.
