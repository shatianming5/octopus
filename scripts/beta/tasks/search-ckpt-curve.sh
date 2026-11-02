# The Search checkpoint curve on Cluster Beta (user 10-23), as queue tasks of held job 216715 (b3-36-s1-dgx-03-c17,
# scripts/beta/hold.sbatch with OCTO_QUEUE=ROOT/queue-2; ROOT=/projects/cluster/octopus/beta/octopus):
# one scripts/beta/search_eval.sh run per adapter and set (code-d3ca0c2 = main d3ca0c2), all on GB200 (same-hardware
# comparison; the curve's ext40 and v5-40 points come from Alpha's H200 matrix, see 03-06 below), fork arm, k=2 (07-20:
# k=1), judge Qwen3-32B (--disable-prefill-cuda-graph, mem 0.85, temperature 0.7), RUN = code-d3ca0c2's
# default ROOT/search/pilot/eval/run (episode settings = search-read-alpha-full-1022's; current search_eval.sh requires RUN). Sets: val (OpenResearcher validation, 500)
# and bcp (BrowseComp-Plus, 830; decrypted in memory only). Adapters (NAME, LABEL): v0 = ROOT/models/adapter-v0-4b-text-r64;
# extN = the external baseline ROOT/models/ext-search-stepN (Alpha's models, N = 10..40); v5-N = trial search-v5-beta-1023b's
# saver adapter ROOT/search/areal-exp/checkpoints/user/rq1-rl/search-v5-beta-1023b/default/epoch0epochstep{N-1}globalstep{N-1}
# (N = 10..50). Order (queue-2/NN-LABEL-SET.sh, 10-23 21:25 UTC): v0, ext40, v5-40, ext10, ext20, ext30, v5-10, v5-20,
# v5-30, v5-50; val then bcp each. Outputs: ROOT/search/pilot/eval/{val,test}-[LABEL-]vN-fork-k{2,1}.{jsonl,log}.
# GPUs 2-3 of b3-36-s1-dgx-03-c17 hold 5.2 / 4.7 GB that no process owns (leaked before this job; each still has 184 GB
# free > SGLang's 0.85 x 189): BUSY_MIB=6000 lets eval_serve.sh start there (this node only). The task makes the clean
# environment itself (SEARCH_EVAL_CLEAN=1: search_eval.sh does not re-exec). Each task, e.g. 13-v5-10-val.sh:
C=/projects/cluster/octopus/beta/octopus/code-d3ca0c2
exec env -i HOME="$HOME" USER="$USER" PATH=/usr/local/bin:/usr/bin:/bin LANG=C.UTF-8 SLURM_JOB_ID="$SLURM_JOB_ID" \
  CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" CUDA_DEVICE_ORDER=PCI_BUS_ID BUSY_MIB=6000 SEARCH_EVAL_CLEAN=1 \
  bash $C/scripts/beta/search_eval.sh $C v10 \
  /projects/cluster/octopus/beta/octopus/search/areal-exp/checkpoints/user/rq1-rl/search-v5-beta-1023b/default/epoch0epochstep9globalstep9 \
  val - 1 0 v5-10
# (v0: `... search_eval.sh $C v0 ROOT/models/adapter-v0-4b-text-r64 val - 2 0`, no LABEL; extN: NAME vN, LABEL extN.)
# Results. From scripts/beta/curve_report.sh: task wall time, y (mean over questions x samples), paired (scripts/alpha/
# paired.py vs v0 on the same set). From the eval's summary lines and rows: accuracy, forked, recall, calls/episode, episode wall:
#   01-v0-val (10-23 21:26-21:55 UTC, 29 min incl. ~8.5 min judge/service/eval-server start): 1000 rows, 0 errors, y 0.378
#   02-v0-bcp (21:55-22:52, 57 min, start ~6.7 min): 1660 rows, 0 errors, accuracy 0.201, forked 26%, recall 0.313,
#     open_recall 0.091, open_gold_recall 0.133; 21.1 calls/episode, episode wall mean 82 s
#   03-ext40-val, 04-ext40-bcp, 05-v5-40-val, 06-v5-40-bcp renamed *.sh.dup-alpha (10-23 22:09, before 02 ended): the same
#     fork k=2 evals are in Alpha's matrix (octopus/ is the same in 11ae150 and d3ca0c2, same judge and RUN settings); the
#     curve takes those two points from Alpha
#   07-20 (ext10/20/30, v5-10/20/30/50): K 2 -> 1 (user, 10-23 22:11: intermediate checkpoints at k=1; TAG ...-k1)
#   07-ext10-val (22:53-23:15, 22 min, start ~5 min): 500 rows, 0 errors, y 0.380; 18.1 calls/episode, forked 44%, episode
#     wall mean 75 s (k=1); paired vs v0 +0.002 [-0.026, +0.030] (54 up / 51 down / 395 same)
#   08-ext10-bcp (23:15-00:16, 61 min, start ~4.5 min; k=1): 828 scored, 2 infra failures (1 "a model request failed", 1
#     "harness exception: SearchEpisodeError"; not in accuracy or paired), accuracy 0.307, forked 68%,
#     recall 0.393, open_recall 0.158, open_gold_recall 0.217; 37.3 calls/episode, episode wall mean 178 s; paired vs v0
#     +0.106 [0.079, 0.132] (169 up / 73 down / 586 same)
#   09-ext20-val (00:16-00:45, 28 min, start ~8 min; k=1): 500 rows, 0 errors, y 0.444; 23.4 calls/episode, forked 78%,
#     episode wall mean 95 s; paired vs v0 +0.066 [0.038, 0.094] (76 up / 33 down / 391 same)
#   10-ext20-bcp (00:45-02:09, 83 min, start ~7.5 min; k=1): 826 scored, 4 infra failures (4 "harness exception:
#     SearchEpisodeError"; not in accuracy or paired), accuracy 0.415, forked 92%, recall 0.465, open_recall 0.221,
#     open_gold_recall 0.309; 48.2 calls/episode, episode wall mean 242 s; paired vs v0 +0.214 [0.185, 0.244]
#     (251 up / 44 down / 531 same)
#   11-ext30-val (02:09-02:38, 29 min, start ~6 min; k=1): 498 scored, 2 infra failures (2 "harness exception:
#     SearchEpisodeError"; not in y or paired), y 0.460; 28.8 calls/episode, forked 88%, episode wall mean 110 s; paired
#     vs v0 +0.080 [0.050, 0.110] (85 up / 30 down / 383 same)
#   12-ext30-bcp (10-23 22:38 - 10-24 00:12 EDT, 94 min; k=1): 828 scored, 2 infra failures (1 "a model request failed",
#     1 "harness exception: SearchEpisodeError"; not in accuracy or paired), accuracy 0.389, forked 97%, recall 0.451,
#     open_recall 0.242, open_gold_recall 0.318; 59.5 calls/episode, episode wall mean 290 s; paired vs v0 +0.188
#     [0.161, 0.217] (230 up / 46 down / 552 same)
#   v5-final (search-v5-beta-1023b step 69, .../search-v5-beta-1023b/final-step69, sha256 8818c6f1...; coordinator 10-24):
#     queued 01:54 by `scripts/beta/tasks/v5final_tasks.sh 69 ADAPTER`: ROOT/queue/13-v5final-bcp-fork-k2.sh (216338, after
#     12-search-v5's own Val) and queue-2/21-v5final-bcp-fork-k2 (done if 13 finished, else resumes 13's --out once 13 has
#     stopped), 22-v5final-val-fork-k2, 23-v5final-val-lin32, 24-v5final-bcp-lin32 (linear k=2, CAP = round mean calls
#     of the fork output's scored rows, THREAD_BUDGET default; they need the fork output's 1000 / 1660 rows and its task's
#     .done). Fork outputs: ROOT/search/pilot/eval/{val,test}-v5-final-v69-fork-k2.jsonl. Infra failures are not rerun
#     (as every arm) except when 21 resumes 13's --out after 13 was cut by 216338's end or ended .failed: search_dev_eval
#     reruns 13's rows with an error; the BCP result line says which path ran. 21 waits while 13 is queued or running
#   23-v5final-val-lin32, 24-v5final-bcp-lin32 renamed *.sh.skip-user (user, 10-23 23:10 EDT: no linear-32K arm is run); queue-2
#     would be 12 -> 13-20 -> 21 -> 22; v5-final's linear-big runs on Alpha 145533.
#   21-v5final-bcp-fork-k2, 22-v5final-val-fork-k2 renamed *.sh.skip-alpha (10-24 03:12 UTC; user, to speed up): Alpha
#     145532 resumes 13's partial BCP output (copied to Alpha's output path) after 216338 ends (10-24 07:06 UTC) and runs
#     the v5-final Val fork; 13 runs on 216338 until then, untouched; 216715 no longer writes test-v5-final-v69-fork-k2.jsonl.
#   13-v5-10-val, 14-v5-10-bcp, 15-v5-20-val, 16-v5-20-bcp renamed *.sh.skip-user (user, 10-23 23:2x EDT: the curve keeps
#     v5-30 and v5-50 only). queue-2 left: 12 (running) -> 17 -> 18 -> 19 -> 20
#   Migration (user, 10-23 23:2x EDT): octopus's Beta work moves to Alpha. 17-v5-30-val, 18-v5-30-bcp, 19-v5-50-val,
#     20-v5-50-bcp renamed *.sh.moved-alpha: Alpha 145533 runs them (results: Alpha's matrix record). 216715 ran 12 to its
#     end, then queue-2/99-cancel.sh (10-24 00:13:50 EDT): GPUs 0-1 at 19 / 1 MiB, 2-3 at the pre-existing 5249 / 4674 MiB,
#     no process on them, /dev/shm/user-eval-216715 removed (nothing else of user's there), then scancel 216715.
#     216338 runs 13-v5final-bcp-fork-k2 until its time limit (10-24 03:06 EDT); Alpha 145532 resumes that output and runs
#     the v5-final Val fork.
#   Per-request token counts for BCP prefix-reuse FLOPs (coordinator, user-approved, 10-24): search_dev_eval keeps a BCP
#     run's trials in /dev/shm/bcp-trials-*/ until the run ends; each trial's agent/requests.jsonl holds thread, times,
#     prompt / completion / cached tokens, tool names, ids, no text (checked: its keys). scripts/beta/copy_requests.sh, an
#     srun --overlap step of 216338 from 10-24 00:16 EDT to the job's end, rsyncs every 20 s only that file, path kept
#     (bcp-trials-XXXX/<job>/<trial>/agent/requests.jsonl, as the rows' trial field), to
#     ROOT/search/pilot/eval/requests-test-v5-final-v69-fork-k2/ (13's run); episodes ending in its last ~20 s may be
#     missed. 12-ext30-bcp's trials: lost (its run ended 00:12 EDT and 216715 ended before the copy existed).
