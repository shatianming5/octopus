# SWE v5 (rl.branch_credit=gate, PR #252) on Alpha, trial swe-read-alpha-v5-1023: the A/B of swe-read-alpha-full-1022
# with the same start (rq1-v0), BATCH=8 STEPS=36 SAVE_EVERY=6, offpolicyness 2, the zero-variance filter, the same
# episode settings (episode_settings.json identical); training on all 4 GPUs of 144129 (FSDP 4), rollout remote.
# N=ROOT/code-39c1180 (= main 0c63788). Queue: the task that starts a controller goes in the queue of the controller that
# holds; a stopped v5 controller then holds on ROOT/swe/pilot/swe-read-alpha-v5-1023/queue (resumes go there).
#   step 1 (02:31, Search full queue): 144687's 4 servers x 16 = 64 concurrent
#     source $N/scripts/alpha/lib.sh; run_clean scripts/alpha/pilot.sbatch $N swe-read-alpha-v5-1023 BATCH=8 STEPS=36 SAVE_EVERY=6 SPLIT_ROLLOUT=0 EPISODES_PER_SERVER=16 BRANCH_CREDIT=gate EXTRA_SERVERS=10.0.4.25:31300,10.0.4.25:31301,10.0.4.25:31302,10.0.4.25:31303
#   step 2 on (03:18, the v5 queue; switch.sh after step 1): 144687's 4 + 144123's 4 servers x 8 = 64 concurrent
#     ... EPISODES_PER_SERVER=8 ... EXTRA_SERVERS=10.0.4.25:31300-31303,10.0.4.27:31300-31303 (each port listed)
#   before 144123's 07:19 TERM (v5_switch_back.sh: the first step boundary after 06:15): back to 144687's 4 x 16
# Rollout servers: tasks/swe-serve.sh in each rollout job's own queue (NODE_IP its address).
# Val/Test of v5 (v5_eval_orch.sh): the whole Test on 145532 = alphagpu21 (48 cores, where the baseline's Test ran), the Val
# halves on 144129 and 144687 (32 cores) if they start by 12:40 and end by 14:00, else the whole Val on 145532 after the Test;
# --parallel 12 per node, the baseline's per-node load
#   switch back (v5_switch_back2.sh = v5_switch_back.sh of PR #256): step 15 done 06:20:45, trainer stopped, 144687's servers
#   restarted clean, the resume queued 06:25:42 and started 06:25:50 (02-swe-v5-4srv.sh in the v5 queue) on
#   144687's 4 servers x 16 = 64 concurrent, steps 16-18 (... EPISODES_PER_SERVER=16 ... EXTRA_SERVERS=10.0.4.25:31300-31303)
#   to 8x8 again (switch.sh, coordinator-confirmed: 145532 started on alphagpu21 07:19:51; steps 16-17 took 24 / 20 min): step 18
#   done 07:25:50, trainer stopped 07:27:25, 02b-swe-v5-8srv.sh started 07:27:56 but 10.0.4.25:31300 could not unload rq1-v18
#   (a leaked count): 144687's servers restarted clean, 02c-swe-v5-8srv.sh (the same line) trains from 07:35:35, steps 19-36 on
#   144687's + 145532's 4 servers x 8 = 64 concurrent (... EPISODES_PER_SERVER=8 ... EXTRA_SERVERS=10.0.4.25:31300-31303,10.0.4.27:31300-31303)
# results (v5_eval_orch.log; rows in ROOT/swe/pilot/swe-read-alpha-v5-1023/{val,test}-v5.jsonl): trainer done 12:18:09, Val
# halves and Test 12:21-13:16, 0 infra errors. Paired by bundle, bootstrap 95% CI (paired.py):
#   Val 45x4:  v0 0.403, baseline v36 0.524, v5 0.557; v5 - v0 +0.154 [+0.089, +0.218] (31/3/11), v5 - v36 +0.033 [-0.016, +0.084] (21/10/14)
#   Test 44x4: v0 0.477, baseline v36 0.549, v5 0.550; v5 - v0 +0.073 [+0.019, +0.129] (20/10/14), v5 - v36 +0.001 [-0.075, +0.074] (15/11/18)
#   forked / branches per episode / branch self-solved / landed: Val v0 .28/.80/.361/.590, v36 .16/.40/.097/.653, v5 .52/1.46/.382/.504;
#   Test v0 .30/.74/.427/.489, v36 .25/.61/.140/.561, v5 .47/1.23/.380/.468
#   training per 6 steps (steps6.py) fork / merge / abort / passed: v5 .27/.60/.09/.41 -> .41/.53/.09/.34,
#   baseline .24/.52/.11/.35 -> .27/.27/.20/.13; mount units over the micro-batch cap (own micro-batch, never cut): v5 13, baseline 2
