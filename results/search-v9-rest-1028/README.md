# search-v9-rest-1028: the lab-cluster run scripts of search-v9-rest (relaunch of 10-28)

search-v9-rest continues Alpha's `search-v9-1025` from global_step 41 (adapter v41) on the OpenResearcher questions
that run had not trained. The scripts in `scripts/` launched the live relaunch of 10-28 on the lab cluster. Paths, hosts
and ports are the cluster's, and the scripts contain no tokens or keys. Since the launch, review has added guards to
them: the trainer and GPU checks in `train-*.sh`, the empty-`nvidia-smi` check in `serve_f2.sh`, the port match and the
pid re-read in `watch-j.sh`, and `CU` in `serve_f2.sh`. Their header comments were also brought up to date. The commands
of the run are unchanged. The kit that moves the run to one new host, `newhost.sh` + `newhost.env`, is the section
"New host (10-29)"; "node-134 (10-30)" runs the same kit on node-134 with `newhost-134.env`, and "Trainer on node-68
(10-30)" moves the trainer to node-68 with `newhost-68.env`. "Archive on HF (10-30)" holds the state after step 43, from
which the run resumes on any host. "Together (10-31)" is the end: a collaborator trained steps 44-45 from that archive,
and the pass ended after step 45. "v45 evaluation (10-31)" evaluates the final adapter v45 on BCP (against v0) and runs
its width/depth sweep.

## Layout

| role | host | GPUs (ports) | launched by |
|---|---|---|---|
| trainer | node-138 | 1-4 (A6000), FSDP d4, `max_tokens_per_mb` 65536 | `train-138-h.sh SERVERS` |
| judge (Qwen3.5-27B) | node-134 | 0, 1, tp 2 (32500) | command below |
| rollouts (Qwen3.5-4B + LoRA) | node-134 | 2 (32302) | `serve_f2.sh STATE 2:32302` |
| rollouts | node-68 | 0-5 (32310-32315) | `serve_f2.sh STATE 0:32310 ... 5:32315` |
| search service, tunnels | node-138 | - (32600) | commands below |

`SERVERS` is `127.0.0.1:32302,127.0.0.1:32310,...,127.0.0.1:32315`: the trainer reaches the rollouts through the tunnels
from node-138, and reaches the judge directly at `198.51.100.134:32500`.

| host | `STATE` | environment of the `serve_f2.sh` launch |
|---|---|---|
| node-134 | `/mnt/disk4/user/search-v9-state/rest2` | `OV=/mnt/disk4/user/search-v9-state TMPDIR=/mnt/disk4/user/tmp` |
| node-68 | `/home/user/octo-env/search-v9-rest` | `CU=/usr/local/cuda-12.4` |

- `watch-j.sh` runs on node-138 beside the trainer. Once the trainer exits, it stops the AReaL workers, the search
  service and the tunnels on node-138, the judge and the rollout on node-134, and the rollouts on node-68. It finds the
  AReaL workers by command line (`search-v9-rest`) and sends them kill, then kill -9. Every other process comes from
  our pid files and is stopped only after its command line (for SGLang, its port) checks out. It replaced
  `watch-h.sh` when node-134's rollouts moved from GPUs 4 and 5 to GPU 2 and node-68 gained GPU 5.
- `one_pass_stop3.py` (since the 10-29 relaunch `one_pass_stop5.py`, below) runs on node-138 and decides when the run ends (see below).
- `train-67.sh` is the same trainer launch for the env staged on node-67 (GPUs 4, 5, FSDP d2). Its paths are under
  `/home/user/search-v9`, so it does not pair with `watch-j.sh` or `one_pass_stop3.py`/`5.py`, which read node-138's
  `/mnt/ssd4/user/search-v9`.

**Restarting the trainer means restarting both watchers.** `one_pass_stop3.py` (and v5) reads the trainer pid once, at its
start. `watch-j.sh` re-reads `train-rest.pid` at each 60 s poll, and a poll that finds no live trainer releases
everything at once. That includes a missing file, or a stale pid in the gap between the old trainer's exit and the new
pid being written. Any poll that lands in that gap therefore releases the new run's servers. So stop both watchers
before restarting the trainer, and start them again once the new pid is written.

## Commands not in the scripts

Judge, on node-134:

```bash
STATE=/mnt/disk4/user/search-v9-state/rest2
CUDA_VISIBLE_DEVICES=0,1 TMPDIR=/mnt/disk4/user/tmp setsid /home/user/octo-env/sglang-venv/bin/python \
  -m sglang.launch_server --model-path /mnt/disk4/user/models/Qwen3.5-27B-text --host 0.0.0.0 --port 32500 \
  --tp-size 2 --mem-fraction-static 0.85 --attention-backend triton --sampling-backend pytorch --context-length 32768 \
  --disable-prefill-cuda-graph --served-model-name judge > $STATE/judge.log 2>&1 < /dev/null & echo $! > $STATE/judge.pid
```

- `setsid` is needed: `watch-j.sh` stops the judge's whole process group, so the judge must lead its own.
- It fails without `--disable-prefill-cuda-graph` (a breakable-graph AssertionError).
- It also fails with the memory-unbalanced error when another user is on one of its two GPUs.
- `0.0.0.0:32500` is an unauthenticated endpoint, open to the whole campus LAN.

Search service and tunnels, on node-138:

```bash
D=/mnt/ssd4/user/search-v9; L=$D/logs/rest2
cd $D/runtime/code-local && setsid $D/areal-venv/.venv/bin/python -m octopus.envs.search.service serve \
  --corpus openresearcher --root /mnt/ssd4/user/search --backend bm25 --tokenizer $D/runtime/Qwen3-0.6B-tokenizer \
  --port 32600 --workers 4 > $L/search.log 2>&1 < /dev/null & echo $! > $L/search.pid
setsid ssh -N -o BatchMode=yes -o ExitOnForwardFailure=yes -o ServerAliveInterval=15 \
  -L 127.0.0.1:32302:127.0.0.1:32302 user@198.51.100.134 > $L/tunnel.log 2>&1 < /dev/null & echo $! > $L/tunnel.pid
setsid ssh -N -o BatchMode=yes -o ExitOnForwardFailure=yes -o ServerAliveInterval=15 \
  -L 127.0.0.1:32310:127.0.0.1:32310 -L 127.0.0.1:32311:127.0.0.1:32311 -L 127.0.0.1:32312:127.0.0.1:32312 \
  -L 127.0.0.1:32313:127.0.0.1:32313 -L 127.0.0.1:32314:127.0.0.1:32314 -L 127.0.0.1:32315:127.0.0.1:32315 \
  user@10.0.1.68 > $L/tunnel68.log 2>&1 < /dev/null & echo $! > $L/tunnel68.pid
```

**On node-138 everything lives under `/mnt/ssd4/user`.** The old `/mnt/SSD_2TB` symlinks were removed. So
`/home/user/search-v9-run` must point to `/mnt/ssd4/user/search-v9`, `/home/user/harbor-udocker` to
`/mnt/ssd4/user/harbor-udocker`, and `harbor-udocker/code-v2` to `/mnt/ssd4/user/search-v9/runtime/code-local`.
Otherwise every episode exits 1.

## Settings and why

- **No HiCache.** `serve_f2.sh` keeps the 10-27 throughput flags, `--mem-fraction-static 0.88` and
  `--mamba-full-memory-ratio 0.5` (a ~684K-token pool on a 4090D). It does not use `--enable-hierarchical-cache`: the read
  tool's mount patch refuses it ("cannot mount: --enable-hierarchical-cache").
- **node-134 needs `TMPDIR`.** Its root fs (`/tmp`) is full, and the JIT kernels' nvcc then fails. It also needs `OV`:
  its overlay and `code-local` live in `search-v9-state`, not in `STATE`.
- **node-68 needs CUDA 12.4.** It has no CUDA 12.9, and the JIT kernels need nvcc.
- **Admission.** `OCTOPUS_ROUTE_KV=0.5`, `OCTOPUS_ROUTE_CAP=16`, `OCTOPUS_ROUTE_WAIT=800`, `OCTOPUS_ROUTE_GAP=540`. The
  gap is PR #320: KV shows an episode only as it grows, so without the gap every restart put 5-9 episode trees on each
  server at once. With 7 routers, a server takes at most one new episode per ~77 s. Since #351 the gap is gone: the scripts
  now set `OCTOPUS_ROUTE_TREE_TOKENS=40000`, at most max(1, 0.5 x 684K / 40K) = 8 active episode trees per
  server over all routers on the host. It keeps the 10-28 load; 68000 (the time-averaged P75 tree footprint, 5 slots)
  is the conservative alternative. Calibration data: `docs/search-env.md` "Admission". The budget needs the AReaL
  overlay rebuilt with #351's `patches/areal-route-spread.patch` (its `/rl/start_session` and `/rl/end_session` hooks count a tree):
  `train-*.sh`, `search_eval_shared.sh` and `newhost.sh preflight` refuse an overlay without it.
- **Episode settings** are Alpha's `search-v9-1025` ones (its `episode_settings.json`): `OCTOPUS_CURSOR=1`,
  `OCTOPUS_READ_TOOL=1`, `OCTOPUS_REWIND_TOOL=1`, `OCTOPUS_REWIND_MAX=2`, `OCTOPUS_FORK_DEPTH=0`. Each new adapter is
  pushed to node-134 and node-68 over the LAN (`OCTOPUS_ADAPTER_PUSH`). Finished groups are spooled
  (`OCTOPUS_ROLLOUT_SPOOL`), so they survive a trainer restart.

## One pass, not step 61

The trainer config still says `total_train_epochs=4 +total_train_steps=61`. By user decision (10-27) the run instead
stops after one pass over the question list `rest2-qids.txt`, which has 281 questions. The list started as the 312
questions Alpha's run had not reached by step 41. 31 of them were skipped because they already had a complete, valid
v41 group with zero variance (all correct or all wrong), so they had already passed once.

`one_pass_stop3.py` (v5 since 10-29: see the relaunch section) counts a question as covered once it has at least 8 episode dirs since this trainer started. When
all 281 are covered and those episodes have finished, it waits for one more `weight_update_vN` (the step that trains the
tail), then 3 more minutes for its checkpoint, and then sends TERM to the trainer's process group. It also stops at most
90 min after coverage. The trainer's exit then triggers `watch-j.sh`.

## Code it depends on

The deployed code is main with these merged PRs:

- #320: `OCTOPUS_ROUTE_GAP`;
- #321: spool groups are deleted only after a recover checkpoint covers their step;
- #322: the actor keeps its batches on the host and puts one micro-batch at a time on the GPU;
- #323: Qwen3.5 decoder layers are streamed over the prefix tree, so 665K-token units fit 48 GB.

The 10-29 relaunch adds #327 (block size per micro-batch; memory: `docs/trainer-memory.md`) and #329's train-step RPC
timeout patch (below).

For `search_eval_shared.sh`, code-local also carries PR #328's `octopus/rl/proxy.py` (deployed 10-29 00:31 EDT; the
previous file is `runtime/proxy.py.bak-1029`). Training does not import it. Without it, uvicorn takes the eval's SIGTERM
for a graceful shutdown that waits on the running episodes, so a stopped eval runs on.

## 10-29 relaunch on 4 GPUs

Step 42 on 2 GPUs (k=1) ran `compute_logp` for 41 min and `ppo_update` for ~3 h 18 min. AReaL's train-step RPC had
a 2 h timeout and 3 attempts, so at 23:58 EDT the controller re-sent `ppo_update`, and the workers started the same
batch again after the first pass ended (01:16 EDT): a second update on the same data, never checkpointed. The trainer
was stopped at 01:22 EDT (watchers first), step 42 is lost, and `scripts/restart-4gpu.sh` relaunches from step 41
(`generation_step00000040`; the spool's 48 groups go back to `resume/`):

- the overlay gets `patches/areal-train-rpc-timeout.patch`: one attempt and a 24 h timeout for every train-step call;
- code-local gets #327 (`block_checkpoint.py`, `actor.py`, `config.py`): block size 4 up to 650K tree tokens per
  micro-batch, per layer above;
- `train-138-h.sh` on GPUs 1-4 (fsdp d4) with the same 7 rollout servers;
- `watch-j.sh` now waits for a running `search_eval_shared.sh` eval before it releases the rollout servers, the
  judge, the search service and the tunnels (the trainer's own GPUs are free at once);
- `one_pass_stop5.py`: v3's rule with the pass start fixed at the 2-GPU trainer's launch (10-29 09:16 CST), the
  spooled qids and the 26 prompts that process skipped for spool groups counted as covered, and episodes a stop cut
  short left out. A prompt the relaunch skips again never covers, so the run may end at total_train_steps 61.

## Spool duplicates (10-29)

The relaunches from step 41 drew the prompts after `generation_step00000040` again. A drawn call took the oldest spooled
group of any question, and a redrawn prompt whose group was spooled ran fresh beside it, so questions piled up groups:
step 43 trained 3 groups of `search-5271`. PR #334 (not deployed; it applies at the next trainer start) keeps one group
per question in `resume/`: the newest head version the resumed policy can use, then the newest file. It deletes the
others, logging each one's name, head version and mtime (`/mnt/ssd4` is 97% full, so no copy is kept). It also
gives a redrawn prompt its own group and rejects, with a log line, a call whose question's group is already out. On
the 10-29 13:23 EDT listing (33 groups, 17 qids, every duplicated qid at head 41), a restart would delete 16 groups:
search-5114 2, search-5161 1, search-5222 3, search-5271 2, search-542 3, search-6296 2, search-6806 1, search-7389 1,
search-9636 1.

## Evaluation

`search_eval_shared.sh NAME ADAPTER SET` (node-138) runs one Search eval (k=1, fork arm) of one adapter, with no GPU of
its own, against the 4B rollout servers in `SERVERS` and the judge at `JUDGE_URL`. Both are required environment
variables, as node-138 reaches them (e.g. `SERVERS=127.0.0.1:32410,127.0.0.1:32411`,
`JUDGE_URL=http://127.0.0.1:32500/v1`): the live run's servers, or the eval's own (below). The sets are BrowseComp-Plus
(`bcp`, 830 queries, BM25 service of the browsecomp-plus corpus on 32601) and Val (`val`, the 500 OpenResearcher
validation questions, BM25 service of training's openresearcher index on 32600). When no service of that corpus answers
on the port, the script starts one and stops it at exit. It follows `scripts/alpha/search_eval.sh`: the run's
`episode_settings.json` (the same six settings as Alpha's `search-v9-1025` Val), the judge at temperature 0,
`EVAL_SAMPLING`, BCP decrypted in memory only (trials in a mode-700 `/dev/shm` dir, removed at exit; only summary and
route-peak lines persist). Rows go to `/mnt/ssd4/user/search-v9/eval`.

`search_eval_queue.sh RELEASE NAME ADAPTER SET [...]` (node-138, same environment) runs several such evals one after
another and never two at once. An eval fails when it exits non-zero, or when more than `MAX_INFRA` (default 0.02) of
its rows are infra failures (search_dev_eval itself exits 0 on those). The queue stops at the first failure. When the
queue finishes, fails or gets TERM, it releases everything RELEASE lists, and a second TERM during the release is
ignored. Each line there is `HOST PIDFILE PATTERN`: HOST is `local` or an ssh target. The pid's command line must match
PATTERN (for an SGLang server, its port), and only then does its process group get TERM, then KILL after 60 s. A pid
that does not match is skipped and logged. The queue needs `SERVERS` and `JUDGE_URL` set before it does anything,
release included. At start it logs each line's pid and command line. The log is `eval/queue.log`.

- The servers load ADAPTER as `rq1-NAME`. Training unloads only `rq1-v(V-40)` when it loads `rq1-vV`, so from v42 on it
  never unloads v0 or v41. The script leaves `rq1-NAME` loaded afterwards, since an unload on a live server can hang.
- v0 is the zero-B adapter `adapter-v0-4b-text-r64` (lora_B all 0, so it serves the base model). The lab copy and
  Alpha's differ only in lora_A and the config's base path, so both serve the same base model.
- Load: at most `EVAL_PARALLEL` x servers eval episodes at once in all (default 1, so 7 on the run's servers). That is a global
  bound, not a per-server one. The eval's router admits a new episode only on a server below `EVAL_ROUTE_KV` (default 0.4,
  under training's 0.5) that holds fewer active trees, training's and the eval's, than max(1, `EVAL_ROUTE_KV` x pool /
  `EVAL_ROUTE_TREE_TOKENS`) (default 40000, training's), so it admits less than training would (no slot is reserved
  for training: between its batches the eval can take slots), and draws among the
  open servers by free slots. A new episode that has waited `OCTOPUS_ROUTE_WAIT` (800 s) takes a server under the
  cap anyway, past the KV target and the slots. Before #351 an `EVAL_ROUTE_GAP` (default 540) admitted at most 7 new
  episodes per 540 s instead (~47 an hour). The pilot ran with the earlier defaults (2, 0.5, 180); alone, it took 32310 to KV 0.99.
  Training's weight-update pauses abort and resume eval requests like its own.
- The eval needs the servers up to the end. Once the trainer exits, `watch-j.sh` stops every rollout server, so the eval's
  remaining episodes become infra-failure rows. Rerunning the same command resumes them (it runs rows with an error again),
  but only once the servers serve again.

Pilot, 10-28 23:42-23:48 EDT (`EVAL_OUT=.../eval/pilot`, `--limit 10`, BCP v0): 10 scored, 0 infra failures, accuracy
0.200, forked 20%, 43 calls/episode. Training was inside step 42's update then, so the servers carried no training
rollouts. Afterwards no tmpfs dir or service was left, and no file written since the pilot's start on node-138, node-134
or node-68 holds a 40-character slice of the 10 decrypted questions or answers.

The full BCP v0 run on the run's servers (10-29 00:38-04:21 EDT, two launches) wrote 171 of 830 rows before
training was paused and everything was released.

### Own servers, 10-29 EDT (training paused)

The user authorized this use on 10-29: node-68 GPUs 4 and 5 (whole cards), and the shared memory on node-67 ("有 30-40g
显存的能放也放进去": put servers wherever 30-40 GB is free, our use at most the free memory minus 6 GB). The queue was
launched at 14:00 EDT with three evals: BCP v0 (resuming its 171 rows), Val v41, BCP v41. The user then cut it to
BCP v0 only. `stop_after_v0.sh` (node-138 `logs/evq`) waits for `v0 bcp exit 0` in `eval/queue.log`. It then sends TERM
to the queue, whose traps stop the v41 eval it may just have started and release everything in `release.txt`. So `eval/`
may get a `launch-val-v41-fork-k1.log` or an empty `jobs-val-v41-fork-k1` dir. Neither one holds results. (On 10-29
the v0 eval never exited, so the queue was sent TERM by hand: see "BCP v0 result (10-29)" below.)

| role | host | GPUs (ports) | launched by |
|---|---|---|---|
| rollouts (Qwen3.5-4B + LoRA) | node-68 | 4, 5 (127.0.0.1:32410, 32411), whole cards | `CU=/usr/local/cuda-12.4 LORAS=rq1-v0=... serve_f2.sh STATE 4:32410 5:32411` |
| rollouts | node-67 | 7 (127.0.0.1:32412), shared with wentao | `BUDGET=36000 LORAS=rq1-v0=... serve_f2.sh STATE 7:32412` |
| judge (Qwen3.5-27B-text, tp 2) | node-67 | 4, 5 (127.0.0.1:32500), shared with user2 | `serve_judge.sh STATE MODEL 4,5` (BUDGET 30000) |
| tunnels, search services, eval | node-138 | none | commands below; the services by `search_eval_shared.sh` |

- STATE is `/home/user/octo-env/search-v9-rest` on both hosts, with `cursor-ba8072c` and `code-local` (node-67's
  copied from node-134's, identical `.py` files to node-68's). v41 is `/home/user/octo-env/models/adapter-v41-search-v9-rest`
  on both, a copy of node-138's `checkpoints/.../weight_update_v41` (same md5). The judge model on node-67 is
  `/mnt/ssd2/user/models/Qwen3.5-27B-text`, a copy of node-134's.
- Memory on the shared GPUs. `BUDGET` (MiB) makes `serve_f2.sh` and `serve_judge.sh` launch only with BUDGET + 6 GiB
  free, at `--mem-fraction-static` BUDGET / (free memory at launch). That is only an estimate. SGLang 0.5.19 applies the
  fraction to the free memory at its own distributed init (`pre_model_load_memory`), so memory the other user frees in
  between makes our server larger by the same factor. That happened to the judge: user2's jobs on GPUs 4 and 5 ended
  during its start (13:50 EDT, when they held 21.9 GB), and each rank holds 46.8 GB on a 30000 MiB budget. node-67's
  rollout holds ~38.7 GB on a 36000 MiB budget, beside wentao's 17.1 GB.
- An absolute cap is available. `KV_TOKENS` and `MAMBA_SLOTS` (both scripts) pass `--max-total-tokens` and
  `--max-mamba-cache-size` to SGLang 0.5.19. The KV pool becomes min(profiled, KV_TOKENS). The mamba slot count is used
  as given: it is taken out of the remaining memory first, and if that memory is short SGLang fails at start instead of
  shrinking it. Keep the fraction as above.
- Choosing the values: scale only the pools, not the whole footprint. Pool budget = BUDGET - weights - ~3 GB runtime;
  weights are 8.6 GB for the 4B and 25.2 GB per rank for the 27B. Then scale the logged pool sizes
  (`max_total_num_tokens`, `max_mamba_cache_size`) of a launch whose pool you know the same way:
  - node-67's rollout: 538,006 tokens and 170 mamba slots at ~38.7 GB in all, a pool of ~27 GB;
  - the judge: 315,250 tokens and 119 slots at 46.8 GB per rank, a pool of ~18.6 GB.
  At the judge's 30000 MiB budget the pool would be only ~1.8 GB, almost nothing. Not used in this queue.
- Nothing backs off by itself after the start. `share_watch.sh PIDFILE PATTERN GPUS` (on the server's host) records
  what other users hold on those GPUs at its start. It stops our server once that has grown and a GPU has under 6 GiB
  free, rechecking the pid's command line first. It only sees memory the other user already holds: if one large
  allocation of theirs fails outright, their usage never grows and it does not trigger. It was not running in this
  queue.
- Not used: node-94. another user's verl job there swung between ~10 GB and ~45 GB per GPU within a minute (sampled 10-29
  13:35 EDT), so a 26 GB server would leave it too little at its peak. The judge went to node-67 because node-68's
  `/home` returned ENOSPC for the 27B copy, even though df showed 98 GB free.
- node-138 reaches node-68 over 10.0.1.68 and node-67 over 198.51.100.67. The 27B copy went 134 -> 67 over
  10.0.2.167, and the v41 copy went 138 -> 67/68.

```bash
# node-68 (STATE): rollouts on whole cards
CU=/usr/local/cuda-12.4 LORAS=rq1-v0=/home/user/octo-env/models/adapter-v0-4b-text-r64 bash serve_f2.sh $PWD 4:32410 5:32411
# node-67 (STATE): rollout and judge on shared cards
BUDGET=36000 LORAS=rq1-v0=/home/user/octo-env/models/adapter-v0-4b-text-r64 bash serve_f2.sh $PWD 7:32412
bash serve_judge.sh $PWD /mnt/ssd2/user/models/Qwen3.5-27B-text 4,5
# node-138 (logs/evq): tunnels (pid files tunnel68.pid, tunnel67.pid), then the queue (one eval: done = released)
T="ssh -N -o BatchMode=yes -o ExitOnForwardFailure=yes -o ServerAliveInterval=15 -o ServerAliveCountMax=4"
setsid $T -L 127.0.0.1:32410:127.0.0.1:32410 -L 127.0.0.1:32411:127.0.0.1:32411 user@10.0.1.68 \
  > tunnel68.log 2>&1 < /dev/null & echo $! > tunnel68.pid
setsid $T -L 127.0.0.1:32412:127.0.0.1:32412 -L 127.0.0.1:32500:127.0.0.1:32500 user@198.51.100.67 \
  > tunnel67.log 2>&1 < /dev/null & echo $! > tunnel67.pid
SERVERS=127.0.0.1:32410,127.0.0.1:32411,127.0.0.1:32412 JUDGE_URL=http://127.0.0.1:32500/v1 \
  EVAL_PARALLEL=3 EVAL_ROUTE_KV=0.7 setsid nohup bash search_eval_queue.sh release.txt \
  v0 /home/user/octo-env/models/adapter-v0-4b-text-r64 bcp > /dev/null 2>&1 < /dev/null &
```

On 10-29 this ran with `EVAL_ROUTE_GAP=60` as well; #351 removed the gap (the eval now uses `EVAL_ROUTE_TREE_TOKENS`).

`release.txt` (node-138 `logs/evq`) has one line per process:
- node-68's two rollouts (`rollout-g4.pid`, `rollout-g5.pid`, pattern `sglang.launch_server.*--port 3241N`);
- node-67's rollout and judge (`rollout-g7.pid`, `judge.pid`, ports 32412 and 32500);
- the two tunnels (`local ... tunnel68.pid ssh -N .*32410`, `tunnel67.pid ssh -N .*32412`).

The search services need no line: `search_eval_shared.sh` stops the ones it started. The live judge's `judge.pid`
first held the pid of `serve_judge.sh`'s subshell, because `cd` and the launch shared one backgrounded line (fixed
since). On 10-29 it was rewritten by hand to the SGLang pid, 2220349.

Pool sizes: 684,223 tokens on each node-68 server and 538,006 on node-67's.

Pilot, 10-29 13:53-13:59 EDT (`EVAL_OUT=.../eval/pilot-own`, `--limit 10`, BCP v0, the settings above): 10 scored,
0 infra failures, accuracy 0.300, forked 20%. The route peak was KV 0.99 with 3 queued on 32411, and about 0.1 on the
others. Afterwards no tmpfs dir or BCP service was left. No file written since the pilot's start holds a 40-character
slice of the 10 decrypted questions or answers: 27 files checked on node-138, 127 on node-68, 51 on node-67.

The queue started at 14:00 EDT (`logs/evq/queue.pid`), with BCP v0 running the 659 remaining queries. It runs
`search_eval_queue.sh` as of 4b64c7c: without `MAX_INFRA` (a 0 exit always passes) and without ignoring TERM during the
release. Progress is in
`eval/queue.log`, `eval/launch-bcp-v0-fork-k1.log` and the row count of `eval/test-v0-fork-k1.jsonl`. Rows with an infra
failure (summary `infra failures`) can be filled in later: bring servers back up, then rerun the same eval with the same
parameters, which reruns only those rows. To stop early, send TERM to the queue pid; it also releases everything.

### BCP v0 result (10-29)

All 830 rows were written by 18:23 EDT, but the eval process then sat idle (no episode left, no summary written) and
was stopped with TERM at 18:28 EDT; the queue's trap released the servers, judge and tunnels. The rows are in
[out/bcp-v0-fork-k1.jsonl](out/bcp-v0-fork-k1.jsonl) (no question or answer text; trial paths only). The summary below
is `octopus/eval/search_dev_eval.py`'s `summary()` on those rows:

| set | adapter | mode | scored | infra failures | accuracy (Wilson 95%) | forked | recall |
|---|---|---|---|---|---|---|---|
| BCP | rq1-v0 | fork, k=1 | 826 | 4 (0.5%) | 0.263 [0.234, 0.294] | 11% | 0.305 |

- Endings: done 455 (accuracy 0.477), context 369 (all wrong: a thread that outgrows the 32K budget leaves no answer),
  step cap 1, no tool call 1. The 4 infra rows (qids 709, 450, 434, 426) are "a model request failed"; counting them all
  right or all wrong moves accuracy to 0.266 / 0.261.
- The first 171 rows ran on the training run's servers (10-29 00:38-04:21), the other 655 scored rows on our own
  servers (14:00-18:23): 0.228 vs 0.272, within noise.
- Against the earlier BCP v0 rows in [../alpha-search-v6d-1024/README.md](../alpha-search-v6d-1024/README.md)
  (fork e883240 0.248, lin32 0.246, linbig 0.260): +0.015 vs fork, unpaired 95% CI [-0.027, +0.057], i.e. consistent,
  not a reproduction. The settings differ: those runs predate #306 (only `done` scores) and #313, used fork depth 1 and
  Alpha H200s; this one fork depth unlimited (`OCTOPUS_FORK_DEPTH=0`, as Alpha's search-v9-1025 Val) on lab GPUs. No
  per-qid rows of the old runs exist, so no paired test.

## New host (10-29): everything on one 8-GPU machine

`scripts/newhost.sh` resumes search-v9-rest on one host that runs everything on localhost (no tunnels, no adapter push):
trainer, judge, rollout servers and search service. Its settings are in `scripts/newhost.env`, written for
ubuntu@198.51.100.11 (8x A100-SXM4-80GB with NVLink, 1.7 TB RAM, base dir `/home/ubuntu/octo`). The deploy agents
install the code, venvs, models, checkpoint and search data. The kit itself copies nothing and runs nothing until it is
called.

`provision_newhost.sh` installs the code and the Python environments at `newhost.env`'s paths:
- `bundle HOST [REF...]`, run from the control machine, ships origin/main (and the REFs) as a git bundle into
  `ROOT/code`. Later runs ship only the commits the host lacks.
- `venvs`, run on the host, calls `scripts/nvwulf/install_env.sh` with the HF env bundle. It builds the trainer venv
  (`AREAL_PY`; its AReaL source is unpacked and patched again, keeping the venv, when the AReaL commit or patches
  changed, and once on a host set up before `ROOT/areal/.octopus-patches` existed; `deploy.sh install` has no idle
  guard, so not while a trainer without `CURSOR` runs on that source), the SGLang venv (`SGLANG_PY`, `venvs/sglang`) and the overlay `CURSOR`, which holds `areal/areal` and `sglang/sglang` as
  real directories, rebuilt on every run. tantivy is 0.27.0, the version that built this host's
  index (below).
- `check` runs `scripts/check_env.sh` with the trainer's `PYTHONPATH=$CURSOR/areal`. It checks versions, the AReaL
  patches, `check_areal_route.py` and the SGLang patches.

`bundle` and `venvs` refuse while a trainer, an SGLang server or the watcher runs (`FORCE=1` overrides).
Compared by sha256 with node-138's `search-v9` trees on 10-29: the SGLang overlay was the same, file for file, and the
AReaL overlay differed in one comment. Tests on this 240-CPU host need `OMP_NUM_THREADS` (8-16). With torch's default
thread count, the CPU-side tests ran more than 10x slower.

| role | GPUs (ports) | memory |
|---|---|---|
| trainer, FSDP d4, `OCTOPUS_CHUNK_MAJOR=1` + `OCTOPUS_BLOCK_CHECKPOINT=1`, chunks of 32768 (needs #354: before it a short micro-batch sharing nothing ran flat beside chunk-major ones, NaN gradients; `newhost.sh` refuses older code) | 0-3 | the checkpoint's FSDP shards (torch DCP; 4 in 10-29's gen41, 3 in the archived gen42) load at any degree (`TRAIN_GPUS`; d4 -> d2 and d4 -> d3 verified, below) |
| judge Qwen3.5-27B-text, tp 1, `--mem-fraction-static` 0.88, `--max-mamba-cache-size` 100 | 4 (32500) | smoke test (default slots): 50.2 GB weights, 62 mamba slots (8.86 GiB), 165,375 KV tokens, at most 12 running requests; with 100 slots: ~76K KV tokens, 20 running (estimate) |
| rollouts Qwen3.5-4B + LoRA, one server per card, 0.88 | 5, 6, 7 (32310-32312) | 1,294,491 KV tokens each (deploy-sglang smoke test, 0.88) |
| search service, openresearcher BM25, 8 workers | - (32600) | ~2 GB of host RAM per worker |

- **One rollout server per card, not two.** A second instance would hold a second copy of the weights, and split the
  card's pool and prefix cache in two. On 4090Ds the servers were prefill-bound with KV overflowing, so the larger pool
  is worth more. Only if a pilot's `route: peak` lines show a low KV peak *with* queued requests (the scheduler, not
  the pool, is the limit) should two servers share a card.
- **Judge at tp 1: the limit is mamba slots.** SGLang caps a hybrid model's running requests at its mamba slots / 5
  (`kv_cache_configurator.py`: 3 per request, plus 2 for the overlap schedule's ping-pong buffer). So the smoke test's
  default 62 slots allowed 12 requests, against 48 concurrent episodes. `JUDGE_MAMBA_SLOTS=100` (144 MiB each) takes
  them out of the same 0.88 pool and leaves ~4.6 GiB of KV (64 KiB a token), ~3.8K tokens per running request; a
  grader prompt is ~1-2K. Check `judge.log` for the "max_running_requests is capped to" line. If judge calls still
  queue or time out (600 s), change `newhost.env` to `JUDGE_GPUS=4,5`, `ROLLOUTS="6:32310 7:32311"`, `CONCURRENT=4`
  (tp 2, two rollout servers; the admission rule below with NS = 2).
- **Rollout capacity (estimate, 未验证).** An A100 has ~2x a 4090D's memory bandwidth and 1.9x its pool (1.29M against 684K tokens). So 3 servers
  are worth about 6 of the 10-28 run's 7 4090D servers. `EXTRA_SERVERS` and `ADAPTER_PUSH` add servers on other hosts.
  Each such server runs the same overlay, has `FILEROOT` at the same path, and is released through `RELEASE` lines;
  a `ROLLOUTS` entry `SSH:GPU:PORT` does all of that itself ("Rollouts on node-68 (10-30)").
- **Admission.** Admission scales with the pool. With NS servers in all, each with a pool of POOL tokens:
  `CONCURRENT` ≈ NS x POOL / 684K groups (10-28: 7 on 7 servers of 684K), 6 here. `ROUTE_TREE_TOKENS` needs no
  change: the slots scale with each server's pool (an A100's 1.29M at KV 0.5: 16 trees). Both are starting points for
  the first steps' `route:` lines (per server trees, slots, KV, queue each minute).
- **State.** The recover checkpoint is `generation_step00000041` (`global_step` 41, i.e. 42 steps done). The rollout
  servers start on its policy, `rq1-v42` (`default/weight_update_v42`), and the trainer continues with step 43's
  update. AReaL reads it from `FILEROOT/checkpoints/<user>/rq1-rl/search-v9-rest` (`areal/utils/saver.py`, the user
  is `ubuntu` here), so the copy needs that user level.
- **Proximal logp (user decision 10-30).** Steps ≤ 42 used `recompute` (a separate `compute_logp` forward). From step
  43 the proximal logp is the training forward's (#349, `docs/trainer-memory.md` "Proximal logprobs"): the same
  gradient, except that the PPO clip no longer fires on the two forwards' bf16 disagreement (~0.02% of the tokens) and
  the rejection test reads the training forward's logp.
- **Spool.** The spool moves. A resume redraws the prompts drawn after its checkpoint, and the prompts of
  the groups already finished past it are not redrawn (`octopus/rl/spool.py` docstring). So without node-138's 33 groups,
  16 questions' step-43 prompts would be lost. On 10-29 they came from the private HF dataset
  `Tommysha/search-v9-rest-spool` (ready/ 1, taken/ 16, trained/ 16, 17.4 GB; deleted 10-30). `newhost.sh spool` now
  downloads the spool archived after step 43 (`SPOOL_REPO`'s `spool/`, "Archive on HF (10-30)") into
  `RUN/rollout-spool/search-v9-rest`, under the original names. At start, `recover(V)` (#334, V = the checkpoint's
  `global_step` + 1: 42 on 10-29, 43 for the archive) moves them to `resume/`, one per task. `spool` refuses once
  `resume/` exists, since a second download after a start would train those groups again.
- **Stop.** The run makes one pass, not 61 steps (user 10-27): it ends with the dataloader's current epoch (sampler
  epoch 0; gen41's `dataloader_info` has drawn 80 of its 17 x 16 = 272 prompts). Not every question gets one group:
  drop_last leaves 9 of the 281 out of the epoch (user 10-29), and a spooled group that a later prompt's call takes
  (`octopus/rl/spool.py`, take) skips that prompt for this epoch. The spool's questions sit at positions 6-67 of the
  epoch's order (`torch.randperm(281)` with seed 0, the sampler's; checked against `one_pass_stop5.py`'s skipped
  prompts, positions 0-33), before gen41's 80, so the first calls after the resume (positions ~80-97) take their
  groups and those calls' own questions get none, as in the lab's v5 run.
  A step count cannot mark the epoch's end: the zero-variance filter rejects ~60-68% of the groups, so a step draws
  ~2.5-3 batches, and the rollout controller cycles the dataloader into the next epoch. AReaL has no setting for it
  either: `total_train_epochs` counts `len(dataloader)` = 17 steps per epoch, and `prepare_batch` cycles the
  dataloader without limit. So `train` starts `pass_stop.py` beside the watcher (log `logs/pass-stop.log`, a line at
  least every 10 min):
  - The controller draws a prompt only to queue it and starts queued prompts first in, first out
    (`workflow_executor`: `_pending_inputs.popleft`), and one process draws each prompt of an epoch once. So once a
    question has a second group since the trainer started (more than 8 episode dirs), the next epoch has begun and
    every prompt of this one has started. A restart near the epoch's end may redraw only prompts that take their
    spooled groups (no episode dirs), so the rule also triggers once 48 groups (questions with an episode dir made
    since) have started since `pass_stop.py` first saw a checkpoint whose dataloader has drawn the whole epoch. The
    prompts drawn by then may still wait in the queue, at most AReaL's pending limit, (max_head_offpolicyness 2 + 1)
    x 16 = 48 (`staleness_manager.get_pending_limit`). The queue starts first in, first out, so 48 starts after it
    drain it.
  - It waits until the episodes open at that moment have ended (`{job}.question.json` exists while one runs), then
    until no spool group finished by then (+60 s for its save) is left in `resume/`, `ready/`, `taken/` or `trained/`
    (commit empties `trained/` after the recover checkpoint), i.e. each one has been trained and checkpointed.
  - Then it sends TERM to the trainer's process group, and the watcher releases everything. The next epoch's groups
    that finish meanwhile train too. Both waits are bounded by progress, not time: the finished groups of the pass can
    take several steps (1-3 h; node-138's step 42 alone took 1 h 52 min on 4 GPUs) to be taken, trained and
    checkpointed, and after a resume dozens of them can wait in `resume/`. A step takes the oldest results first, so
    while groups of the pass are still in the controller each step takes at least one. If what is left (open episodes,
    then spool groups) has not shrunk for 3 steps of `LATEST`, at two polls in a row (a poll between a checkpoint and
    the commit that empties `trained/` sees groups already checkpointed), it sends TERM anyway and logs what is left.
    `scripts/test_pass_stop.sh` runs these cases on fake dirs (~40 s).
  - `logs/pass-stop.json` keeps the pass's sampler epoch and that cut across a trainer restart. A restart from a
    checkpoint whose dataloader has drawn the whole epoch goes straight to the spool check: the epoch's prompts are not
    drawn again, and its finished groups are in the spool (`resume/` after recover). A restart from inside the epoch
    draws the epoch's tail again, so the rule starts over. Delete the file only for a new pass.

  `+total_train_steps=61`, the run's setting, is only a backstop. The lab's `one_pass_stop5.py` counted the 281
  questions' coverage in node-138's jobs dir, which does not move here.
- **Judge timeout.** `train` sets `OCTOPUS_JUDGE_TIMEOUT=600`, the code's default (3 attempts each). At 120 s, the tp 1
  judge (~29 tokens/s for a single stream) timed out 3 times on 1 of 90 episodes early in step 43. A judge failure is
  the episode's error, so its whole group is rejected and never trained. The only cap above it is the group's 6 h
  (`rollout.request_timeout` and `session_timeout_seconds` in `grpo.yaml`), against the judge's worst case of
  2 calls x 3 attempts x 600 s = 1 h, so a longer timeout only delays a step while the judge is slow. `search_eval_shared.sh` keeps 120 s: there a
  judge failure is an infra row, which the next run of the same eval redoes, and `MAX_INFRA` stops a queue on a slow
  judge. `train-138-h.sh` and `train-67.sh` record the lab run as it ran (120 s).
- **Public host.** `train` passes a random proxy admin key, `+rollout.agent.admin_api_key`, made once into
  `RUN/admin_api_key` (mode 600). It does not set `AREAL_ALLOW_DEFAULT_ADMIN_KEY`, which would accept the default key.
  The SGLang servers and the search service listen on 127.0.0.1, but AReaL's RPC and proxy servers bind every
  interface. `preflight` lists what listens on all interfaces; the cloud firewall should allow only ssh. The key
  reaches the trainer through its environment (`OCTOPUS_ADMIN_KEY`, read by hydra's `${oc.env:...}` in the override),
  not its command line, which every user of a shared host sees with `ps`. It also reaches AReaL's saved `config.yaml`.
- **Inputs.** `rl.search_split` must be node-138's `rest2-qids.txt` (md5 `abb8f1a6…`), because the restored dataloader
  position indexes it. The questions file must be node-138's too (md5 `dcadeb55…`). The run config is the code's
  `octopus/rl/grpo.yaml`, which differs from node-138's `octopus_rl_grpo.yaml` in `prox_logp_method: reuse_train_logp`
  (#349, below) and comments.
- **Search index (user decision 10-29).** The new host's OpenResearcher BM25 index replaces node-138's, so retrieval
  from step 43 on is not exactly the same as at steps ≤ 42. node-138's index was built on 10-16 with an older tantivy.
  The query side uses tantivy 0.27, and the old index's stems do not match its query stems: "international" is stored
  as `intern`, while a query cuts it to `internat`. The new index was built in the same environment that serves the
  queries, so index-time and query-time stemming match. Documents matching `internat`: 301 in the old index against
  1,125,123 in the new. Matching `organiz`: 76,569 against 1,259,547. A spot check of the first 30 rest2 questions
  returned identical results for 26; the other 4 changed in scores or order. Which index version Cluster's steps 1-41
  used has not been checked.

Order, on the host, from the shipped code dir (`$CODE/results/search-v9-rest-1028/scripts`):

```bash
bash newhost.sh spool       # the spool from HF (HF_HUB_DISABLE_XET=1): 10-29 node-138's 33 groups, now step 43's 10
bash newhost.sh preflight   # read only: paths, checkpoint (every DCP shard file) + v42, spool 9/1/0 (10-29: 1/16/16), md5s, #338, rpc-timeout patch,
                            #   harbor, free GPUs, what listens on all interfaces
bash newhost.sh serve       # search, judge, rollouts on rq1-v42, each unless it already answers; waits until all answer
bash newhost.sh train       # trainer, pass_stop.py (TERM after the pass), the watcher (auto-release once the trainer exits)
bash newhost.sh status
```

The watcher (`logs/watch.pid`) re-reads `train-rest.pid` every 60 s. Once the trainer has exited, it stops the AReaL
workers. Then, after any running `search_dev_eval` ends, it stops the search service it started, the judge, the rollout
servers (each pid's command line checked first, as `watch-j.sh` does) and the `RELEASE` lines. To restart only the
trainer, TERM the watcher first, then the trainer's process group, then run `train` again. `train` refuses to start
while either one runs.

## node-134 (10-30): the same kit on the lab host

The user resumed search-v9-rest on node-134 on 10-30 (the 10-29 new host was stopped mid step 43, so the recover
checkpoint is still `generation_step00000041`), steps 43..54 (one pass, `pass_stop.py`). Every command is
`TRAIN_GPUS=... ROLLOUTS=... NEWHOST_ENV=$PWD/newhost-134.env bash newhost.sh CMD` from
`ROOT/code-<sha>/results/search-v9-rest-1028/scripts` (ROOT `/home/user/search-v9-134`, a symlink to
`/mnt/disk4/user/search-v9-134` since 10-30 so that its paths also exist on node-68, below; the code a
`git archive` of the pushed commit: `octopus`, `patches`, `scripts`, `results/search-v9-rest-1028/scripts`). The GPU
lists have no default: pick them at launch from the cards that have room then (the user allows any card with room;
the caps are the `SHARED=1` rules below), e.g. `TRAIN_GPUS=3,4 ROLLOUTS="6:32310 7:32311"`. GPU 5 is used by another project of
the same account; `newhost-134.env` refuses it. The 10-30 launch (14:30 EDT): `TRAIN_GPUS=0,1,4,6` (d4, the
checkpoint's own degree) `ROLLOUTS="7:32311 2:32312 3:32313"` (shared cards), judge on node-67 GPUs 0, 1
(`JUDGE_GPUS=0,1 start_judge.sh`: GPUs 4, 5 were taken).

| role | host, GPUs (ports) |
|---|---|
| trainer, FSDP d = number of `TRAIN_GPUS` (the settings of "New host (10-29)") | node-134 `TRAIN_GPUS` |
| rollouts Qwen3.5-4B + LoRA, one server per card, all of the card's free memory but 6.5 GiB | node-134 `ROLLOUTS` (32310, ...) |
| more rollouts (optional, "Rollouts on node-68" below) | node-68 0, 1, 4 (`ROLLOUTS` entries `user@10.0.1.68:G:P`, 32320, ...) |
| judge Qwen3.5-27B-text, tp 2 (`start_judge.sh` in node-67:~/octo-env/search-v9-rest/judge-1030, `JUDGE_GPUS`, default 4,5; pid file `judge.pid` there, which `newhost-134.release` stops) | node-67 0, 1 on 10-30 (`10.0.2.167:32500`) |
| search service, openresearcher BM25, 8 workers | node-134 CPU (32600) |

- **GPUs are shared** with another user (user2), who holds ~1 GB per card and at times starts ~20 GB workers on
  0, 3, 4, 6 and 7. `newhost-134.env` sets `SHARED=1`, so `newhost.sh` leaves every card at least 6 GiB free at launch:
  - the trainer starts (`train`) only if each `TRAIN_GPUS` card has `TRAIN_MIB` + 6 GiB free (36 + 6 GiB: 35.4 GiB is
    the measured chunk-major peak on a 48 GB card in 32K chunks, docs/trainer-memory.md); otherwise it refuses and names
    the card and its free memory;
  - a rollout server starts (`serve`) only with `ROLLOUT_MIB` + 6 GiB free (12 GiB: weights and runtime, below it there
    is no KV pool), at `BUDGET` = the card's free memory - 6.5 GiB (`serve_f2.sh` reads the free memory again 5 s
    later, needs BUDGET + 6 GiB, else skips the card, so 512 MiB of slack; `--mem-fraction-static` BUDGET / free).
    SGLang applies that fraction to the free memory at its own init, so if user2 frees memory in between our server
    grows by the same factor; the headroom is then ~6 GiB minus the prefill activations (`serve_f2.sh`'s BUDGET note). `KV_TOKENS` and `MAMBA_SLOTS` exported by the caller pass
    through to `serve_f2.sh` and fix the pools instead ("Own servers, 10-29 EDT" above: how to size them).
  - `preflight` checks both rules. Nothing backs off after the start: if user2's jobs grow later, `share_watch.sh`
    is the tool (not run by `newhost.sh`). Without `SHARED` (`newhost.env`, a host of our own) a card qualifies with
    <= 4000 MiB used and a server starts at 0.88, as before.
- **Admission.** `CONCURRENT` defaults to the number of rollout servers, i.e. a whole 4090D's ~684K-token pool each.
  On shared cards the pools are smaller: after `serve`, sum `max_total_num_tokens` over `RUN/logs/rollout-g*.log` and
  pass `CONCURRENT` ≈ that sum / 684K to `train`.
- **Same `ROLLOUTS` for `serve` and `train`.** `train` routes to its `ROLLOUTS` list and the watcher it starts
  releases exactly that list, so a server that `serve` started under a different list is neither used nor stopped.
- **Inputs.** On 10-30 huggingface.co and hf-mirror.com reset the lab hosts' connections, so the checkpoint and the spool
  came from node-138 over the campus network (`ssh user@198.51.100.134`, ~60 MB/s): `LATEST`, `actor/initial_lora`,
  `generation_step00000041`, `weight_update_v41`/`v42` (each file's sha256 equals its HF LFS sha256 in
  `Tommysha/search-v9-1025`), the spool's ready/taken/trained (1/16/16; names and sizes equal the HF dataset
  `Tommysha/search-v9-rest-spool`), `rest2-qids.txt` (md5 `abb8f1a6…`). The questions file is `~/octo-env`'s (md5
  `dcadeb55…`).
- **Venvs** are node-134's `~/octo-env` ones, brought to the env bundle (`Tommysha/octopus-train`, revision `code`,
  copied from a host that reaches HF) by `install_env.sh`; PyPI through the tuna mirror (pypi.org ran at ~20 KB/s):

  ```bash
  O=/mnt/disk4/user/octo-env-data; R=/mnt/disk4/user/search-v9-134
  UV_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple TMPDIR=/mnt/disk4/user/tmp SGLANG_VENV=$O/sglang-venv \
    CURSOR=$R/cursor TANTIVY=0.27.0 bash $R/code-<sha>/scripts/nvwulf/install_env.sh $O $R/bundle $R/code-<sha> "areal sglang"
  ```
- **Search index.** node-134's own openresearcher index (10-16) is the old-tantivy one, so the run's index is built again
  with tantivy 0.27.0 into `R/search/search-scratch/openresearcher` (`R/search/datasets` -> `~/octo-env/datasets`):
  `python -m octopus.envs.search.service build --corpus openresearcher --root $R/search --threads 16`: 14,878,084
  documents in 16 min (13.7 GB RSS). Its counts match the 10-29 host's new index exactly: `international` 1,125,123
  documents, `organization` 1,259,547.
- **d4 -> d2.** The recover checkpoint is torch DCP (4 shards from fsdp d4), which reshards on load.
  `scripts/reshard_check.py` builds the real actor on 2 GPUs and loads `generation_step00000041` with its optimizer
  as AReaL's recover does. All 400 LoRA tensors and 1600 Adam state tensors, gathered from the 2 ranks, equal the
  checkpoint's own full tensors (read without a process group). The LoRA tensors equal `weight_update_v42`. Every
  Adam `step` is 42, and no moment is zero (10-30 on node-134 GPUs 3 and 4; the load takes 12 s). The first actor build
  on the host spent ~10 min compiling Triton/inductor kernels (two runs: 31 min before a kill, then 10 min on a warm
  cache). The command, set here for the archived checkpoint ("Archive on HF (10-30)": `generation_step00000042`, 3
  shards from d3, its adapter v43; on 10-30 it ran with `generation_step00000041` and v42), with `PYTHONPATH` and the
  LoRA shape as `train`'s; `--nproc-per-node` is the degree to check:

  ```bash
  R=/home/user/search-v9-134; O=/home/user/octo-env
  CK=$R/run/checkpoints/user/rq1-rl/search-v9-rest; cd $R/code-<sha>
  OCTOPUS_SGLANG_PY=$O/sglang-venv/bin/python CUDA_VISIBLE_DEVICES=3,4 PYTHONPATH=$R/cursor/areal:$PWD \
    TMPDIR=/mnt/disk4/user/tmp \
    $O/areal/.venv/bin/python -m torch.distributed.run --nproc-per-node 2 results/search-v9-rest-1028/scripts/reshard_check.py \
    $O/models/Qwen3.5-4B-text $CK/checkpoint_generations/generation_step00000042/payloads/default \
    $CK/default/weight_update_v43 actor.lora_rank=64 actor.lora_alpha=64
  ```

  `actor.lora_rank=64 actor.lora_alpha=64` are required: `grpo.yaml` has rank 32, whose LoRA shapes differ from the
  checkpoint's. Use two cards that are free at the time (never GPU 5).
- **Release.** `RELEASE` names the judge on node-67 (`10.0.2.167`, its `judge.pid`), so the watcher stops it too
  once the trainer exits; node-134 reaches node-67 over the LAN with user's key.

### Judge on another host (10-30)

The judge of the 10-30 runs ran on node-67 GPUs 0, 1 (shared with user2), with its state dir (`judge.pid`, `judge.log`)
at `/home/user/octo-env/search-v9-rest/judge-1030`, whose `judge.pid` the `RELEASE` files `newhost-134.release` and
`newhost-68.release` stop. `HOST` binds it to the host's LAN address, so node-134 reaches it at
`http://10.0.2.167:32500/v1` (node-68 through its `TUNNELS` entry). SGLang asks for no key, so anyone on the lab LAN
can then use the judge. The pools are fixed: 66 mamba slots (13 running)
and 40,000 KV tokens, measured at 34,002 MiB per card on 10-30. `BUDGET` 34140 is the 40,284 MiB then free (user2
present) minus 6 GiB; `serve_judge.sh` refuses a card with less than `BUDGET` + 6 GiB free. The launch, on node-67 (its
`start_judge.sh`, from a copy of this script):

```bash
JUDGE_GPUS=0,1   # 10-30; start_judge.sh's default 4,5
BUDGET=34140 HOST=10.0.2.167 KV_TOKENS=40000 MAMBA_SLOTS=66 bash $CODE/results/search-v9-rest-1028/scripts/serve_judge.sh \
  /home/user/octo-env/search-v9-rest/judge-1030 /mnt/ssd2/user/models/Qwen3.5-27B-text $JUDGE_GPUS
```

### Rollouts on node-68 (10-30)

The user authorized node-68 GPUs 0, 1 and 4 (10-30) as more rollout servers for this run. A `ROLLOUTS` entry
`user@10.0.1.68:GPU:PORT` runs on node-68; `newhost-134.env` refuses any other node-68 card (at the time 2, 3, 5
were not ours; later on 10-30 the user gave G0-G5 to the trainer, "Trainer on node-68 (10-30)"; G6 is aozhou's). For
such an entry `newhost.sh`
- `serve`: checks the card over ssh (`SHARED=1`: `ROLLOUT_MIB` + 6 GiB free, `BUDGET` = free - 6.5 GiB, as on node-134),
  pushes `weight_update_v(N+1)` to the same path there (`octopus/rl/adapter_push.py`), starts `serve_f2.sh` there over
  ssh with the same paths and CUDA `R_CU` (12.4), and a tunnel from node-134's `127.0.0.1:PORT` (pid file
  `logs/tunnel-PORT.pid`; SGLang binds 127.0.0.1 on node-68 too). An entry that answers through its tunnel, or whose pid
  file on node-68 names a live server, is kept;
- `train`: routes to `127.0.0.1:PORT` like the local servers, and adds `user@10.0.1.68:22` to
  `OCTOPUS_ADAPTER_PUSH`, so each weight update reaches node-68 before the servers load it;
- the watcher stops the node-68 server (pid file and command line checked there) and then its tunnel.

node-68 is then a hard dependency of the run. A tunnel that dies (e.g. ~90 s of LAN loss: `ServerAliveInterval` 30,
3 misses) drops its server from rollout until `serve` runs again (it restarts only what is down; nothing reconnects by
itself). An unreachable node-68 fails the adapter push at a weight update, which stops the trainer, and the watcher then
releases everything. Recovery: `serve`, then `train` (the recover checkpoint and the spool keep the progress).

What node-68 holds (set up 10-30): `/home/user/search-v9-134` (a real dir) with `cursor/sglang` and `code-f807e727`
copied from node-134 (sha256 of all 4,647 files, adapter included, equal to node-134's), `run/logs` (pid and log files,
`sglang-cache`) and `run/checkpoints/.../default/weight_update_v*` (pushed). `~/octo-env/sglang-venv` (sglang 0.5.19,
torch 2.8.0+cu128; `pip freeze` equals node-134's but for node-134's stale `sglang-kernel` 0.4.6 record, whose
`sgl_kernel` files 0.3.5 overwrote: every common file is identical) and `~/octo-env/models/Qwen3.5-4B` (sha256 equal
to node-134's). node-134 reaches node-68 at 10.0.1.68 with user's `id_ed25519` (10.0.2.168 has no route from
node-134). Smoke test (10-30 14:31 EDT, GPU 0, port 32390): started from node-134 over ssh at `BUDGET` 41,934 MiB, up in
~45 s; from node-134 a push of `weight_update_v41`, `/load_lora_adapter` and a chat request to `rq1-v42` and `rq1-v41`
through a tunnel answered; then stopped.

Each code deploy copies its dir to node-68 as well (`newhost.sh` runs `ROOT/code-<sha>` there):

```bash
# on node-134
R=/home/user/search-v9-134; tar -C $R -cf - code-<sha> | ssh user@10.0.1.68 "tar -C $R -xf -"
```

The trainer reads its servers once, at start: `rl.servers` makes one AReaL rollout worker per server
(`octopus/rl/train.py`), the router takes `OCTOPUS_ROUTE_SERVERS` at start (`octopus/serving/route.py`), and
`OCTOPUS_ADAPTER_PUSH` is read at import (`octopus/rl/adapter_push.py`); `live.env` changes none of them. So the node-68
servers join at a trainer restart, best just after a step's recover checkpoint (`LATEST` advanced):
1. TERM the watcher (`watch.pid`), then the trainer's process group (`train-rest.pid`); `pass_stop.py` exits with it.
2. `serve` with the old entries plus the node-68 ones, e.g. `ROLLOUTS="... user@10.0.1.68:0:32320
   user@10.0.1.68:1:32321 user@10.0.1.68:4:32322"`: the running node-134 servers are kept, node-68's start
   on rq1-v(N+1). (`preflight` would flag the cards our running servers hold and the spool count; its node-68 line is
   the one to read.)
3. `CONCURRENT` ≈ the sum of `max_total_num_tokens` (node-134's `RUN/logs/rollout-g*.log`, node-68's same path) / 684K,
   then `train` with the same `ROLLOUTS`.

## Trainer on node-68 (10-30)

`newhost-68.env` runs the trainer on node-68 GPUs 0-5 (the user authorized these six on 10-30; FSDP d = the number of
`TRAIN_GPUS`; GPU 6 is aozhou's, which the env refuses)
and nothing else there. The rollout servers stay on node-134 as remote `ROLLOUTS` entries
`user@198.51.100.134:G:P` (node-68 reaches node-134 only at that address, never its GPU 5). The search service
stays on node-134 :32600. The judge stays on node-67 `10.0.2.167:32500`, which has no route from node-68. Every
command runs on node-68 from `ROOT/code-<sha>/results/search-v9-rest-1028/scripts`:
`TRAIN_GPUS=... ROLLOUTS=... NEWHOST_ENV=$PWD/newhost-68.env bash newhost.sh CMD`.

What `newhost.sh` does for this layout:
- `TUNNELS` (`PORT:SSH:ADDR:APORT`) starts ssh tunnels at the beginning of `serve`: node-68's 127.0.0.1:32600 to
  node-134's search, and 127.0.0.1:32500 through node-134 to the node-67 judge (`JUDGE_URL` 127.0.0.1:32500). Each runs
  in a loop that reconnects 5 s after ssh exits (e.g. ~90 s of network loss, `ServerAliveInterval` 30). Without the
  loop a lost search or judge tunnel would reject every episode and stall training with no error. A live tunnel is
  kept, and the watcher stops each one (the loop and its ssh share a process group). `serve` waits until `JUDGE_URL`
  answers as well. The remote rollout tunnels do not reconnect ("Rollouts on node-68"). With `SEARCH_ROOT` empty, no local search service is started or
  checked. Both tunnels were tested from node-68 on 10-30: the judge's `/v1/models` and the search `/health` (14,878,084
  docs) answered.
- `R_TMPDIR` sets the remote servers' `TMPDIR`. node-134's root fs is full, so nvcc JIT fails in /tmp there.
- Each weight update is pushed to node-134 (`OCTOPUS_ADAPTER_PUSH` = every remote `ROLLOUTS` host). The push replaces
  every `weight_update_v*` in node-134's `CK/default` (`octopus/rl/adapter_push.py`), so node-134 keeps only the latest
  one from then on. node-68 keeps the history (`OCTOPUS_KEEP_WEIGHT_UPDATES=1`), so copy all of it over at the switch.
- `RELEASE` (`newhost-68.release`) covers node-134's search service and the node-67 judge. The judge goes through
  `judge-67`, an alias in node-68's `~/.ssh/config` (ProxyJump node-134; set up 10-30, 10.0.2.167's host keys added to
  `known_hosts`). node-67 does not yet accept node-68's key ("Permission denied"). Until
  `ssh node-68 cat .ssh/id_ed25519.pub | ssh node-67 'cat >> .ssh/authorized_keys'` is run, the watcher logs that line's
  failure and the judge must be stopped by hand.
- The GPU rules are `newhost-134.env`'s (`SHARED=1`). The trainer starts only if each card has `TRAIN_MIB` + 6 GiB
  free. A new node-134 server starts only with `ROLLOUT_MIB` + 6 GiB free, at free - 6.5 GiB.
- The watcher's wait for a running `search_dev_eval` sees node-68's processes only. An eval on node-134 that uses its
  search or servers is not waited for.
- **Host RAM.** `RANK_GIB` / `SERVER_GIB` make `preflight` and `train` refuse unless `MemAvailable` ≥ d × `RANK_GIB`
  + servers × `SERVER_GIB`. The measured values (PSS on node-134, d3, step 43 `ppo_update`, 10-30) are:

  | item | per unit | where |
  |---|---|---|
  | trainer rank | 57.6 / 62.0 / 64.0 GiB | pinned slab ∝ the rank's largest micro-batch (docs/trainer-memory.md: ~44 GiB for the 994K unit) |
  | AReaL rollout worker + proxy | 5.2 GiB per server | trainer host |
  | controller, episode agents | ~5 GiB | trainer host |
  | search service | 25 GiB PSS (46 GB RSS) | stays on node-134 |

  node-68 has ~400 GiB available (440 total, ~40 GiB other users; swap 256 GiB). The estimates are d5 + 6 servers 356
  GiB (+5) and d6 + 6 servers 420 GiB (+5). So d5 fits, and d6 is short by ~20-25 GiB at the measured peak (the check
  refuses it). Ways to fit d6, none free:
  - fewer servers (−5.2 GiB each);
  - a lower `RANK_GIB` if the peaks of steps 43-44 on d5 come out below 64. This is a gamble: a rank over the line
    pages its unpinned memory to swap or meets the OOM killer, as on node-138 (10-28);
  - a code change that shrinks the per-rank base outside the slab.

  node-68's memlock limit (55 GiB) is above the slab's ~44 GiB.
- **Disk.** node-68's `/home` has ~98 GB free and others use it too. The peak is ~50 GB:
  - 2 checkpoint generations of 10 GB while one is saved (the older is then deleted);
  - spool ≤ ~20 GB (trained/ is deleted at each checkpoint);
  - weight updates 14 × 233 MB;
  - jobs and logs ~5 GB.

  Run `df -h /home` before the switch.
- **Env on node-68** (10-30):
  - the AReaL venv's `pip freeze` equals node-134's (412 packages: torch 2.9.1+cu129, triton 3.5.1, flash_attn 2.8.3,
    fla 0.4.2) but tantivy, which only the search service uses;
  - `cursor/areal` was copied from node-134 (every file's sha256 equal, 505 files), and `cursor/sglang` and both models
    are equal too;
  - the questions file and `data/rest2-qids.txt` were copied, and their md5s match.

  The driver is the same as node-134's (615.71). The trainer's kernels need no system CUDA: torch's cu129 wheels, a
  prebuilt flash_attn, and Triton with its own ptxas 12.8. So CUDA 12.4 (`CU`) is only `preflight`'s nvcc line. The
  trainer's Triton cache (AReaL's `TRITON_CACHE_DIR` /tmp/areal-user/.cache/user/triton) is empty on node-68, so
  the first actor build compiles (10 min warm, up to 30 min cold on node-134) unless step 0 below copies it.

The switch, at a step boundary (EDT durations at ~75 MB/s node-134 → node-68; `R=/home/user/search-v9-134`,
`CK=run/checkpoints/user/rq1-rl/search-v9-rest`, `G68=user@10.0.1.68`, commands on node-134 unless stated):

0. Before the boundary, with no effect on the run:
   - deploy the code dir to both hosts: `git archive` of the pushed commit into `R/code-<sha>`, as in "node-134 (10-30)";
   - pre-sync the spool and the jobs (~14 GB, ~3.5 min): `rsync -a $R/run/rollout-spool $R/run/jobs-search-v9-rest
     $G68:$R/run/`;
   - optionally warm the Triton cache (~1 min): `rsync -a /tmp/areal-user/.cache/user/triton/
     $G68:/tmp/areal-user/.cache/user/triton/` (1.8 GB). Never copy the torchinductor cache: it embeds the
     compiling host's absolute paths, and the 10-30 node-68 start with node-134's copy died in the actor's `initialize`
     (`PermissionError: '/mnt/disk4'`);
   - authorize node-68's key on node-67 (above).
1. Boundary: `cat $R/$CK/LATEST` shows the new `global_step` (step 43 → 42, `generation_step00000042`) and
   `$R/$CK/default/weight_update_v43` exists.
2. Stop the trainer (~1 min): TERM the watcher (`run/logs/watch.pid`, command line `newhost.sh _watch`) first, so that
   it releases nothing. Then TERM the trainer's process group (`kill -TERM -- -$(cat run/logs/train-rest.pid)`). The
   AReaL workers and proxies are in it, and `pass_stop.py` exits with it. Check that no `rpc_server ... search-v9-rest`
   is left. The node-134 rollout servers, search and judge stay up.
3. Stop node-68's rollout servers G0, G1, G4 (~1 min): on node-68, `run/logs/rollout-g{0,1,4}.pid` (command line
   `sglang.launch_server.*--port 3232[012] `). Then stop their tunnels on node-134, `run/logs/tunnel-3232{0,1,2}.pid`.
4. Copy (~3 min after the pre-sync, ~6 min without):
   ```bash
   rsync -a --delete $R/run/rollout-spool/ $G68:$R/run/rollout-spool/   # ~14 GB (ready/taken/trained, resume/)
   rsync -a $R/run/jobs-search-v9-rest $G68:$R/run/                      # ~0.6 GB, pass_stop.py reads its mtimes
   rsync -a $R/$CK/ $G68:$R/$CK/       # LATEST, generation_step00000042 (~10 GB), actor/ (233 MB), weight_update_v41..43
   rsync -a $R/run/admin_api_key $G68:$R/run/; rsync -a $R/run/logs/pass-stop.json $G68:$R/run/logs/
   [ -e $R/run/live.env ] && rsync -a $R/run/live.env $G68:$R/run/   # OCTOPUS_LIVE_ENV, if one was written
   ```
   Do not copy the rest of `run/logs`: its pid files name node-134 processes.
5. Serve, on node-68 (~2 min): first `ss -ltn | grep -E ':(32600|32500) '` must print nothing (a tunnel cannot bind a
   taken port). Then run `preflight`, then `serve`, both with `ROLLOUTS` = the 3 running servers plus new
   ones on the cards the trainer freed, e.g. `"user@198.51.100.134:7:32311 user@198.51.100.134:2:32312
   user@198.51.100.134:3:32313 user@198.51.100.134:0:32314 user@198.51.100.134:1:32315
   user@198.51.100.134:4:32316"`.
   - Expected `preflight` FAILs: the spool count, and the cards our running servers hold. Every other line must be ok:
     DCP shards, host RAM, the node-134 line.
   - `serve` keeps the running servers and starts the tunnels. It pushes `weight_update_v43` to node-134 and starts the
     new servers there on rq1-v43 (~45 s to load).
6. Train, on node-68: set `CONCURRENT` ≈ the sum of `max_total_num_tokens` in node-134's `run/logs/rollout-g*.log` / 684K
   (g7, g2, g3 were 475K, 645K, 645K on 10-30). Then run `train` with the same `ROLLOUTS` and `TRAIN_GPUS=0,1,2,3,4`
   (d5).
   - Startup: the DCP checkpoint (saved at d3) reshards on load. d4 → d2 was verified, d3 → d5 was not. Then the
     actor build: 10-30 min of Triton compiling without step 0's caches. Spooled groups are reused.
   - Check: `train-rest.log` recovers global_step 42, `status` shows the trainer, the watcher and `pass_stop.py`, and
     the first weight update (v44) reaches node-134.

Recovery is as on node-134: the node-134 servers and search keep running. To restart, TERM the watcher and then the
trainer on node-68, and run `train` again.

## Archive on HF (10-30)

On 10-30 the run stopped after step 43 (`LATEST`: `generation_step00000042`, global_step 42); since 10-31 the repo
also holds steps 44-45 and `LATEST` is `generation_step00000044` ("Together (10-31)" below). Its full state is in the private
HF model repo [`octopus-org/search-v9-ckpt`](https://huggingface.co/octopus-org/search-v9-ckpt) (moved from
`Tommysha/search-v9-1025` on 10-30 so collaborators get access through the organization; HF redirects the old id), beside
Alpha's steps 1-41;
the repo's `README.md` is the step-by-step launch guide (prerequisites, downloads, env file, `preflight`/`serve`/`train`
for one host and for the lab split, first checks, stop and restart, pitfalls). The table is the 10-30 archive; 10-31 added
`generation_step00000044` (now `LATEST`), `weight_update_v44`..`v45` and `together-1031/` ("Together (10-31)"):

| repo path | what |
|---|---|
| `LATEST`, `checkpoint_generations/generation_step00000042/` | the recover checkpoint (torch DCP, 3 shards from d3, 10 GB) |
| `default/weight_update_v1`..`v43`, `actor/initial_lora/` | the adapter after each step N (`vN`), the initial LoRA |
| `spool/ready/` (9), `spool/taken/` (1), `spool/MANIFEST.tsv` | the rollout spool (5.5 GB): `newhost.sh spool` downloads it |
| `run-state/` | `pass-stop.json`, `rest2-qids.txt`, `openresearcher-questions.jsonl`, `episode_settings.json`, `config.yaml` (admin keys redacted) |

Not archived: `jobs-search-v9-rest` (rollout traces; `pass_stop.py` counts only the episode dirs made after its own
trainer started, so a resume needs none of them) and `run/admin_api_key` (`train` makes a new one). The checkpoint goes
to `FILEROOT/checkpoints/<user>/rq1-rl/search-v9-rest/`, `pass-stop.json` to `RUN/logs/`, the two question files to
`SPLIT` and `QUESTIONS`. Every file's HF sha256 equals node-68's source. The spool dataset
`Tommysha/search-v9-rest-spool` and `generation_step00000041` were deleted afterwards (user, 10-30).

## Together (10-31): steps 44-45, end of the pass

A collaborator (HF user `jrjyan`) resumed the run from the archive with `newhost.sh` at code b83277a9 on Together's
`slinky-64` (8x H200, Slurm) and pushed the result to the HF repo as commit `4b31c2dd` (10-31 03:55 EDT):
`default/weight_update_v44`, `v45`, `checkpoint_generations/generation_step00000044/` (`LATEST`, global_step 44, 2
shards from d2), `run-state/config-together.yaml`, and `together-1031/` (the launch kit for H200 + Slurm and the logs
of the trainer, `pass_stop.py`, the watcher, the run and the switch). The HF README's "Together (10-31)" section lists
the five fixes the kit needed there (no JIT DeepGEMM on sm90, `Python.h` for the SGLang venv, `ninja` on the judge's
PATH, nvcc 12.8 from the runfile, one blocking `srun` for serve + train + wait).

| step | layout | done (EDT) | task_reward | behave_approx_kl | clip_ratio | rollout / train_step |
|---|---|---|---|---|---|---|
| 44 | trainer d3 GPUs 0-2, judge GPU 3, rollouts GPUs 4-7 (CONCURRENT 14) | 01:12 | -0.218 | -4.32e-4 | 0 | 79 / 52 min |
| 45 | trainer d2 GPUs 0-1, judge GPU 3 (mem 0.5) beside a rollout, rollouts GPUs 2-7 (CONCURRENT 19) | 03:46 | -0.105 | -3.65e-4 | 0 | 54 / 41 min |

- Between the steps (01:13-01:54 EDT) the run was stopped, `generation_step00000043` (d3) was checked at d2 against
  `weight_update_v44` with `reshard_check.py` (passed), and the second layout (`newhost-tog-c.env`) was launched.
- `pass_stop.py` logged "9 groups of the pass not yet trained and checkpointed" from the relaunch until step 45's
  checkpoint, then "one pass done" (03:46 EDT) and sent TERM to the trainer; the watcher released every server and left
  all 8 GPUs at 0 MiB (03:49 EDT). The pass needed 45 steps, not the 54 estimated at 16 questions per step: with
  zero-variance groups filtered out, step 44's collection drew the rest of the 281 questions (HF README).
- A further step would start a second pass. v45's evaluation is "v45 evaluation (10-31)" below.
- Source of the table: the `StatsLogger` blocks of `together-1031/logs/train-rest-10310603.log` (step 44) and
  `train-rest.log` (step 45); times from `pass-stop.log`, `switch.log` and `watch.log` (logged in UTC).

## v45 evaluation (10-31)

The user authorized it on 10-31 (~13:30 EDT): v45 on BCP and Val, k=1, fork arm, with the protocol of "BCP v0 result
(10-29)" so that v45's BCP rows pair with v0's by qid; then (~16:00 EDT) an eval-only width/depth sweep of v45. No
training. The scripts are in [scripts/v45-eval/](scripts/v45-eval/) (node-138 `logs/evq-v45`), the rows in
[out/bcp-v45-fork-k1.jsonl](out/bcp-v45-fork-k1.jsonl), the comparison in
[scripts/bcp_compare.py](scripts/bcp_compare.py).

### Layout

| role | host | GPUs (ports) | launched by |
|---|---|---|---|
| rollouts (Qwen3.5-4B + `rq1-v45`, whole cards, 0.88) | node-138 | 0-4 A6000 (32420-32424, local) | `serve_f2.sh` with node-138's overlay (below) |
| rollouts | node-68 | 4, 5 (32410, 32411) | as "Own servers, 10-29 EDT"; lost at 21:21 EDT (below) |
| rollouts | node-134 | 6, 7 (32416, 32417) | the 10-30 kit (`cursor/sglang`, `code-9d3e21c4`), STATE `search-v9-134/eval-v45` |
| rollouts, sweep only (BUDGET 39400 MiB, free − 6 GiB under load) | node-136 | 1-4 (32431-32434) | `search-v9-state/eval-v45`, from 16:38 EDT |
| judge Qwen3.5-27B-text, tp 2 | node-67 | 0, 1 (10.0.2.167:32500) | `judge-1030/start_judge.sh` (`JUDGE_GPUS=0,1`, 40,000 KV tokens, 66 mamba slots, 34 GB per card) |
| tunnels, search services, evals | node-138 | none | `search_eval_arm.sh`, the queues |

- Adapter: `default/weight_update_v45` of the HF repo `octopus-org/search-v9-ckpt` at commit `4b31c2dd`, downloaded
  once on node-138 to `/home/user/octo-env/models/adapter-v45-search-v9-rest` and copied host to host. On every host its
  `adapter_model.safetensors` sha256 is `dcfffb0860914484c48bb0f4bd4387edc803197834e569b6a33c498de65ff927`, HF's LFS
  sha256 (`adapter_config.json`: the HF blob id).
- Network: node-138 and node-68 share the LAN 10.0.1.0/23; node-134, node-67 and node-136 sit on 10.0.2.0/24, which
  neither node-138 nor node-68 reaches (node-68's 10.0.2.168 has no route to them). So node-138's tunnels to node-68 use
  10.0.1.68, those to node-134, node-67 and node-136 the cluster addresses 198.51.100.x.
- Code: the eval runs node-138's `runtime/code-local` and `areal-overlay` with the 10-29 `search_eval_shared.sh` (copied
  from `logs/evq`), the code of the v0 rows: no file in them is newer than the v0 eval's start. That code predates #351, so
  admission is `EVAL_ROUTE_GAP` (the current `search_eval_shared.sh` refuses this overlay: no route start hook). Settings
  as on 10-29: `EVAL_PARALLEL=3 EVAL_ROUTE_KV=0.7 EVAL_ROUTE_GAP=60`. node-138's SGLang overlay equals node-68's and
  node-136's file for file (sha256 of the 3,513 `.py` files); node-134's differs in one line of `sampling_params.py` (top_k
  ≥ vocab size taken as "no truncation", no effect at the eval's sampling).
- Pilot, 13:43-14:07 EDT (`eval/pilot-v45`, `--limit 10`): 10 scored, 0 infra failures, accuracy 0.300, forked 100%.
  No tmpfs dir or BCP service was left, and no file written since its start holds a 40-character slice of the 10 decrypted
  questions or answers (`leak-v45.sh`: 32 files on node-138, 129 on node-68, 1 on node-67, 10 on node-134).

### BCP result

`python3 scripts/bcp_compare.py` (each episode's last row, as `search_dev_eval.py`'s `summary()`; v45's file has 869
rows: 830 plus 39 reruns of infra rows):

| set | adapter | mode | scored | infra failures | accuracy (Wilson 95%) | forked | branches / episode | recall |
|---|---|---|---|---|---|---|---|---|
| BCP | rq1-v0 | fork, k=1 | 826 | 4 (0.5%) | 0.263 [0.234, 0.294] | 11% | 0.41 | 0.305 |
| BCP | rq1-v45 | fork, k=1 | 824 | 6 (0.7%) | 0.496 [0.462, 0.530] | 100% | 6.41 | 0.582 |

- Paired v45 − v0 on the 820 qids scored in both: **+0.233** [+0.200, +0.267] (bootstrap over qids, 10,000 resamples,
  seed 0); up 214, down 23, same 583. One k=1 run per adapter, under different serving conditions: v0's first 171 rows
  ran on the 7 training servers during training (`EVAL_PARALLEL` 1, gap 540, training's weight-update pauses), its other
  655 on 3 own servers (9 episodes at once); v45's on 9 own servers (27 at once), its 41-episode fill on 7 (21 at once).
- v45's endings: done 570 (accuracy 0.718), context 252 (all wrong), no tool call 2. v0's: done 455 (0.477), context 369.
- Infra rows. The first run (14:09-21:19 EDT, queue `search_eval_queue.sh`, `MAX_INFRA` 0.02) had 39 infra rows of 827 by
  21:03 EDT ("a model request failed"; one traced case was a 258K-token request over SGLang's 262,144 context, a 400).
  With 2 episodes left, the queue was sent TERM by hand at 21:19 EDT (`hang_guard.sh`, armed against the 10-29 idle-tail
  hang, had not fired); its trap released every server, the judge and the tunnels. `chain_sweep.sh` then restarted them
  (judge up 21:21 EDT) and its queue reran the same eval (21:43-22:46 EDT): only the 41 episodes without a scored row (39
  infra, 2 missing), leaving 6 infra rows. The fill ran through `search_eval_arm.sh`: the 10-29 script's eval (same
  `search_dev_eval` call and settings), differing only in its tmpfs trial dir and BCP port handling. Counting them all right or all wrong moves accuracy to 0.500 / 0.493.
- node-68 GPUs 4 and 5 were taken by another user (aozhou) at 21:21 EDT, between the release and the restart.
  `serve_f2.sh` skipped them, so the fill and the sweep run on 7 lab servers.
- The fill's launch log ends with "trial dir ... was removed by another process: made again": that was the eval's own
  removal at its exit (`search_eval_arm.sh`'s watchdog then recreated it, and its cleanup removed it).

### Width/depth sweep (running)

Arms change only these episode settings of v45's (`make_arms.sh`: settings dirs `rq1-rl/run-sw-ARM` beside the run's,
as `scripts/alpha/search_ablate.sh` makes them; outputs `{test,val}-sw-ARM-v45-fork-k1.*`); 0 is no limit for both
knobs, and the default arm is the result above (10 branches, depth unlimited):

| arm | linear | w1 | w2 | w5 | w20 | d1 | d2 | d3 |
|---|---|---|---|---|---|---|---|---|
| change | `LINEAR=1` | `MAX_BRANCHES=1` | 2 | 5 | 20, `MAX_THREADS=21` | `FORK_DEPTH=1`, `MAX_BRANCHES=10` | 2, 10 | 3, 10 |

- Queues on node-138, each `sweep_queue.sh` / `sweep_queue2.sh` (a failed eval is logged and the queue goes on;
  `sweep_queue2.sh` also writes the failure's run time and last log line to `eval/sweep-failures.txt`):
  - lab BCP (7 lab servers, started 21:43 EDT by `chain_sweep.sh`): default fill, then linear, w1, w2, w5, w20;
  - node-136 (its 4 servers, BCP service on 32611): d3, d2, d1. d3 gained 6 infra rows while the judge restarted at
    21:20 EDT; a rerun fills them;
  - lab Val (same 7 servers, from 22:57 EDT): default, w20, d1, d2, d3, w5, w2, w1, linear.
- Two BCP evals now run at once on node-138, so `search_eval_arm.sh` takes its BCP service port from `EVAL_SEARCH_PORT` and
  removes only its own tmpfs trial dir (the 10-29 script removes every newer one at its exit).
- Release: the shared lab servers, tunnels and judge are not in any queue's release list (`release-lab.txt` emptied,
  node-67's `judge.pid` holds `held`). `release_shared.sh` releases the lab servers and tunnels once the lab BCP and Val
  queues have both ended, and the judge once the node-136 queue has too; the node-136 queue releases its own servers.
- Val: the lab's first Val attempt (22:47-22:51 EDT, in the lab BCP queue) failed for every arm within 30 s:
  `code-local` had no `data/search/openresearcher_split.json` (Val had never run from it). It now holds the repo's file
  (sha256 `3f034b52…`, commit 85d98fcf), copied from node-138's `runtime/code-59517af`.
- An H100 host (4×H100, another agent) ran Val arms from 10-31 afternoon until it was reclaimed (~19:35 EDT). Its rows and
  logs are in node-138 `eval/h100-1031`. Its partial rows of linear (500, 2 infra), w1 (500, 1), w2 (308, 24) and w5 (61,
  13) were copied to `eval/` (`h100_copy.sh`; provenance in [scripts/v45-eval/notes-h100.txt](scripts/v45-eval/notes-h100.txt)),
  where the lab Val queue resumes them: rows from the H100 have trial paths under `/home/ubuntu/octo/eval/`, the lab's
  under `eval/jobs-val-*`. Its default Val (347 rows, 9 infra) stays in `h100-1031` as a cross-check only. Most of its
  w2/w5 infra rows came from a rollout server on its GPU 0 capped at ~74K tokens per request, not from the model.
- No sweep result yet.
