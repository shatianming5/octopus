# The SWE full trial's Test (coupled-a-test, 44 bundles, never trained on) on Alpha, 10-23: one job (144123) alone, three
# queue tasks in a row (bundles 0-14, 15-29, 30-43), each serving rq1-v0 and rq1-v36 together on its 4 GPUs and
# evaluating both at once (eval_part.sh: k=4, --parallel 6 each, Val v0's P/run settings); A = the step-36 adapter
# (globalstep35). Before: pull_test_sifs.sh and oracle_test.sh (7/7 images 1.0). Then paired.py on the merged files.
ROOT=/projects/cluster/octopus/octopus
A=$(ls -d $ROOT/swe/areal-exp/checkpoints/user/rq1-rl/swe-read-alpha-full-1022/default/*globalstep35 | head -1)
exec env -i HOME=$HOME USER=$USER PATH=/usr/local/bin:/usr/bin:/bin SLURM_JOB_ID=$SLURM_JOB_ID CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES bash $ROOT/logs/eval_part.sh 0 $ROOT/harbor/datasets/coupled-a-test test rq1-v0=$ROOT/models/adapter-v0-4b-text-r64 rq1-v36=$A
# ... the same with 1 and 2 as the first argument
