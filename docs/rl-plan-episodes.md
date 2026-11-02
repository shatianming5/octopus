# Plan-episode RL

Free-fork training on the existing RQ1 stack (AReaL + octopus patches). One
**plan** episode is the whole tree: the main thread plus any branches, any
number, anywhere. Fork is an ordinary tool; the
start is the bare issue. The reward is the tree's outcome alone (Reward below); the branch advantage is the episode's `A_main` unless `rl.branch_credit` says otherwise (Reward);
the e9–e14 local credit (commit/abort branch matrix, token price, fork-task
credit, check validity, efficiency, closing-choice and statement terms) and the
pre-e9 branch / join episodes are at commit a6e83a9.

Train:

```
python -m octopus.rl.train --config octopus/rl/grpo.yaml scheduler.type=local \
  rl.servers=127.0.0.1:30200,... rl.plan_tasks=/path/to/plan-tasks
```

`rl.plan_tasks` is a file of task dirs or a directory of them (`rl.exclude`:
a file of those to leave out). `rl.plan_control` (the no-fork control below,
default off) is an evaluation's: training never trains a control.

## Data flow

1. `HarborEpisodeAgent` runs one Harbor trial per group sample with
   `OCTOPUS_PLAN_EPISODE=1` (`GrokHarborAgent`).
2. Chat completions go to this sample's AReaL proxy session (`export_style` /
   `chat_template_type` = `concat`). Concurrent branch requests with a shared
   prefix become **sibling leaves** of one conversation tree; each leaf
   concatenates its parent chain into **one sequence per thread** (main, each
   branch, control). See AReaL facts below. In plan episodes the history is
   sent verbatim (no shrinking of repeated tool outputs), so each thread stays
   one prefix chain (but for rewinds, below). Each completion's thread comes from `requests.jsonl`
   (response `id` -> `thread`: `main`, a branch's name, `control`; see Nested
   forks); a leaf
   with no thread or an ungraded branch rejects the group. Branch and control
   leaves inherit the forking thread's turns up to the fork as their prefix:
   every token before the thread's first request's `prompt_tokens` gets loss
   mask 0 there (trained once, in the main leaf). A rewound thread
   (`OCTOPUS_REWIND`: the next request continues from an older reply) is
   several leaves sharing the turns before the rewind point, the abandoned
   attempt and the continuation. In request order each leaf owns the thread's
   requests after the previous leaf's last one, up to its own last one; every
   token before its first owned request's `prompt_tokens` gets loss mask 0, so
   the shared turns train once, in the first leaf. Every leaf gets the thread's
   reward and advantage (a failed attempt is credited like the rest of its
   thread), spans sit in the leaf that owns their completion, and costs come
   from `requests.jsonl` rows, once per request.
   A thread's requests after its last leaf (its last request forked and only
   its branches extend it: a call cap, a deadline, a failed next request) are
   owned by the first leaf of its first branch to start then: that leaf's
   loss mask starts at their first `prompt_tokens`, and a span gives each of
   their completions its own thread's advantage. Every request's completion
   is owned by exactly one leaf, before the requests of that leaf's own
   thread, or the group is rejected.
   The same holds without a rewind when AReaL's prefix match splits a thread
   (a re-sent reply that no longer matches its record): such groups were
   rejected before and now train, each token once (e8f 1 of 497 groups, e9 2
   of 528, e10-e12 none).
   The rollout worker only has deserialized leaves (`interaction_id` and
   the tensors), so positions come from `requests.jsonl` `prompt_tokens` /
   `completion_tokens`, and the tokenizer from the dataset row's
   `tokenizer_path` (= config `tokenizer_path`).
3. The plan verifier (installed per episode on a copy of the task, removed
   afterwards) grades the final `/testbed`, then each branch patch alone
   on its fork-point snapshot (`/tmp/octopus-plan/plan.tsv`, written by
   the agent from its fork records after the episode's last tool call, so the
   policy deleting `/tmp/branches` does not matter →
   `/logs/verifier/branches.json`), then the control snapshot if present, and
   restores the final tree. A forking episode without `branches.json`, or
   with a branch count or control that does not match, is rejected.
4. `plan_group_step` sets episode reward `R`, leave-one-out `A_main` (every
   branch's `A_branch` is its episode's) and the read credit. The rollout
   worker writes them as per-token `octopus_adv`, and the actor **overwrites**
   AReaL group `adv_norm` with it for plan sequences.
5. Groups whose `octopus_adv` is 0 on every token (equal `R`, no read credit) are dropped (`has_variance`).

## Reward

\[
R(\tau)=2y-1
\]

- `y`: hidden tests on the final tree.
- `C` (recorded only): critical-path completion tokens, from main: a thread's own work in order + per fork it made
  the max critical path among that fork's branches (`requests.jsonl` `thread`,
  `group.critical_path`). A fork waits for its branches, so it adds its longest
  branch (its `tree.json` `response_id` = `join_response_id`; the critical path
  still reads older non-blocking records, whose branches overlap main's work up
  to the reply that collected each).
- `A_main`: leave-one-out over the group's G mains, no std. The whole
  tree trains on its episode's `A_main`: every branch thread (committed, aborted,
  unclosed; each leaf of a rewound thread too) takes
  it (`group.tree_advantage`), as search does. The only per-token change is the
  optional read credit (Read control below).
- Control (`rl.plan_control=true`, an evaluation's: dev_eval's config override): when the episode forks,
  restore main's first fork's snapshot, replay messages up to just before the
  forking reply and continue with the linear tool set and the linear system
  message (fork is not a tool there), in the same loop as main. It starts
  where main stood: the tool calls main had left under `OCTOPUS_MAIN_MAX_CALLS`
  (its calls before the fork count). `tree.json` `control` = `{kind, path,
  depth, response_id, calls_before, budget_before, end, budget_used}`. Graded
  (`y`, `C` = the main thread's tokens before the fork call + the control's
  tokens) for the fork − control comparison, never trained (advantage 0, loss
  mask 0 on its leaf) and in no baseline. At most one per forking episode.
  Bundles draw the fork point at random (below). Episodes with a control run
  Harbor with `--verifier-timeout-multiplier 2` (two trees graded in full).

Optional in a run's env file: `OCTOPUS_READ_CONTROL=1` (with `OCTOPUS_READ_TOOL=1`), the read counterfactual below.

Training runs without a `control` thread (main's or nested); controls and read
controls (`OCTOPUS_READ_CONTROL`) are never trained. An evaluation adds the
control with `rl.plan_control=true` after the run arguments (`octopus/eval/dev_eval.py`,
`octopus/eval/search_dev_eval.py`): the agent sets or clears `OCTOPUS_PLAN_CONTROL`
per episode from the row's `plan_control`, whatever the environment holds. The
held-out pool (`scripts/eval/run_step.sh`) adds it on both passes. A run trained
before PR #213 (its config.yaml has `rl.plan_lambda` / `plan_alpha` /
`plan_cost_unit`) fails to load here: evaluate it with its own commit.

## Fork arguments and closing

As in search, a fork carries only `tasks` (each with its acceptance check; a bundle's with its `task_id`): no
`will_pass` statement (removed; it fed only the bundle read-control score). Under `OCTOPUS_AUTO_MERGE` a branch's task
message says, as search's `branch_close` did before protocol v6 (#193), that done only closes the branch and the join
does not merge its patch; commit is what merges it (`HarborWorkspace.branch_close`).

Only a closing call hands a thread's result (#306): main's and a control's `done`, a branch's `commit`, `abort` or
`done`. A thread's first reply without a tool call, while another turn fits (turn cap, `thread_budget`, answer
reserve, serving window, shared budget), is kept and answered once with `core.REMIND_NOTE` naming that thread's
closing tools (`close with abort or commit or done now`); a second one ends the thread, unclosed. Same rule as search
([search-env.md](search-env.md), "Ending without done"); `tree.json` and the episode record count the reminders
(`reminded`).

## Nested forks

Every thread runs the same loop (`core.run_branch_continuation`):
main is the root thread (closed by `done`, capped by `OCTOPUS_MAIN_MAX_CALLS`
tool calls, with the trajectory log), a branch a thread of the same
kind (closed by commit/abort, reported to its parent). Forking is one path at
any depth (`GrokHarborAgent._thread_tool` / `_join`). A
branch's `fork` (bundle `task_id` included) starts sub-branches from its
conversation so far plus their task, each in a private copy of its tree; they
run concurrently, it waits for their join note and merges what it wants.

- `OCTOPUS_FORK_DEPTH` (default 2: main → branch → sub-branch; 0: no depth
  limit): a thread at that depth is told in its task message that forking is
  not available, and a fork call there returns that refusal. At any depth a fork
  whose branches would start with less than the answer reserve + 8192 tokens of
  the serving window (`OCTOPUS_CONTEXT_WINDOW`, 262144) left after one reply is
  refused alike, every thread's growth cap is at most that room, and a read
  whose mount would not fit is refused (`settings.window_room`; mounts
  count). The tool list stays the same, so a
  sub-branch's prompt is its parent's tokens plus its own (prefix packing).
- `OCTOPUS_MAX_THREADS` (default 16): main + open branches at any depth; a fork
  that would pass it is refused with the counts.
- Names (`core.thread_name`, from the path in the fork tree):
  `fork<k>/<branch id>` for main's k-th fork, `<parent>/fork<k>/<branch id>`
  below, e.g. `fork0/branch-1/fork0/branch-1.0`. A sub-branch's id is its
  parent's id plus its number among the parent's sub-branches
  (`core.branch_ids`: `branch-1.0`, `branch-1.1`, ..., numbered on
  across the parent's forks), so it never matches a depth-1 `branch-<i>`. That is the branch's session id
  and `requests.jsonl` `thread`; `branch_key` (`/` → `-`) is its plan.tsv and
  branches.json key, workdir and patch name. `tree.json` nests the records: a
  branch's own forks sit in its `forks`, and `core.tree_branches`
  walks them for the agent and `plan_outcome`.
- Grading: every branch at any depth gets a plan.tsv row. Its patch is its
  tree since its depth-1 ancestor's fork point (a sub-branch includes its
  parent's changes up to its fork), graded like a depth-1 branch: on that
  fork's snapshot, or on the initial tree with its named task in a bundle.
- Reward: every branch at any depth takes its episode's `A_main` (the default `rl.branch_credit=outcome`; every "takes `A_main`" above means that default). `rl.branch_credit` (a trainer setting, `plan_group_step(branch_credit=)` via the data row) is `outcome` or `gate`; `gate` applies the environment's rule, which its group step passes (`octopus/rl/group.py` `tree_advantage`): here (`code_gate`, SWE v5, #252) a branch that committed, merged (`merge` starts with `merged`), changed files (non-empty `diffstat`) and `passed` gets `max(A_main, 0)`, any other `min(A_main, 0)`; search's (`search_gate`) is in `docs/search-env.md`. The old value `search_gate` is refused like any unknown one. R is unchanged.
- Limits: every thread has its own `thread_budget` growth and call/turn cap;
  there is no shared token pool in SWE (search: `OCTOPUS_SEARCH_BUDGET`, 0 by default).

## Rewind tool (`OCTOPUS_REWIND_TOOL=1`)

Every thread (main, branches at any depth, both controls) is offered
`rewind(to_step, note)` (`core.REWIND_TOOL`, last in the tool list; search's
`agent.REWIND` describes a history-only rewind)
and run by its own loop (`run_branch_continuation`), never dispatched:

- A step is one reply with tool calls. Each tool result's tag names the
  reply's step: `[context left: X | calls left: C | step N]` (no `| calls left` when
  uncapped). Under `OCTOPUS_READ_TOOL` the thread's name
  leads the tag (`[main | context left: X | step N]`, `[branch-1 | ...]`, docs/cross-branch-read.md). A branch numbers on from its forking
  reply's step (its history ends with that reply and its results), a control
  from the step before the forking reply (`start["step"]`); the inherited
  steps are not its own,
  so `to_step` runs from that number (where the thread started) to its last
  step. Any other value is answered with an error and nothing changes.
- Tree (an environment with a workspace: SWE; search has none, its rewind
  cuts the history alone): before a step's first call that may change it
  (any call but `view`, `rewind`, `read_thread`) the loop snapshots the
  thread's tree (`SNAPSHOT`: HEAD + a stash commit, in `/testbed` or the
  branch copy), so every step's end state is known; a step without such a
  call shares the last snapshot.
- After the reply's calls and any join: the tree is restored to that step's
  snapshot, the history is cut to that step's end (its results, its join
  note, an earlier rewind note) and the note is added as a tool message of
  its own, tagged like a tool result (with thinking on or off: the latest tag
  shows the calls left now and the context left back at step N): `[runtime] Rewound
  to step N at your request: ... They edited <files>; <k> commands failed,
  last with: <error>.` + what it did to the thread's forks (branches of a
  merges after step N undone, with their patch paths; a join at step N itself
  stays) + `Your note: <note>` (search: `... the steps after it were removed
  from your history.` + `Your note: <note>`). The next step is N+1 again; a
  later rewind to N keeps this note. A closing call in the same reply wins
  (no rewind). A reply whose fork starts (accepted: `core.started`) cannot
  also call rewind (before or after the fork) or a closing call
  (`core.FORK_ALONE`: refused, not counted as calls, the fork runs; also
  without the rewind tool; every fork description says so): its branches
  extend that reply, so its thread must go on from it too, or the reply is
  in none of the thread's leaves. A fork that is its thread's last request
  anyway (a call cap or deadline right after it, a failed next request) is
  trained in the first leaf of its first branch (`assign_threads`, below).
- Calls stay spent: the rewind is a tool call, and the calls of the removed
  steps count against the thread's cap (a main-first control from a later
  fork gets what main had left then, `fork["calls_left"]`); a shared token
  pool's tokens stay spent too. The thread's context left grows back by the
  removed steps (`thread_budget` counts growth from the thread's first
  prompt), and a reply that rewinds is never clipped at the answer reserve.
  A thread in its answer reserve (or clipped) accepts rewind besides its
  closing call (its notes say so), and leaves the reserve when the growth at
  the step it went back to is below the cap less the reserve.
  `OCTOPUS_REWIND_MAX=N`: each thread runs at most N rewinds (`start["rewinds_used"]`,
  a control's from its thread's at the fork; a rewind does not give one
  back); a further one is refused (`core.REWIND_SPENT`) and the reserve
  notes stop offering rewind.
  A rewind past a fork's reply keeps the fork
  recorded from its own history (`fork["history"]`, as for `OCTOPUS_REWIND`);
  its merges are undone with the tree, and those branches' tree.json `merge`
  reads `undone: merged` (`rl.branch_credit=gate`: not merged).
- Reads (`OCTOPUS_READ_TOOL`): a rewind to the step of a read whose mount
  rode on a removed request mounts it again with the next request; a branch
  that rewinds to or before its read has no read control; under
  `OCTOPUS_READ_EVENT` its siblings' events of removed steps are dropped.
  read_thread's definition is added to main's next fork result when a rewind
  removed it.
- Plan RL: a rewound thread is several leaves (the rule above), each trained
  once with the thread's advantage. tree.json `rewinds` (main's) and a
  branch's `rewinds` record each rewind with `to_step`, the messages it
  removed and its note (evaluation counts the removed calls: search's bcp
  recall, `searched`, read_audit, and per episode `rewinds` in
  `search_dev_eval` rows).
- With `OCTOPUS_REWIND` (the runtime rule) both run: a rule rewind drops the
  tool's steps past its checkpoint, a tool rewind drops the rule's checkpoint
  if past its step (moves it past the note if at its step) and clears its
  failure count.
- Flag off: no rewind tool and no step tags (tree.json `rewinds` stays empty
  but for `OCTOPUS_REWIND`); the fork rule above applies to closing calls.

## Task bundles

A bundle task (`octopus/envs/swe/data/build_bundles.py`; its `tests/tasks.json` makes the
episode a bundle one: `octopus/envs/swe/rl.py` sets `OCTOPUS_BUNDLE` per episode) is one repo
with n bugs in one tree, statements in `/tasks/<k>.md` + `/tasks/INDEX.md`
(readable with `view`).

- Limits: no shared token budget (removed as in search, #180: one pool drawn by
  main and every branch starved forked episodes). Each tool result starts with
  `[context left: C | calls left: M]` (C: the thread's own context growth left,
  `settings.thread_budget`; M is what this thread may still call (main:
  `OCTOPUS_MAIN_MAX_CALLS` minus its own calls; a branch: `OCTOPUS_BRANCH_MAX_TURNS`
  minus its turns; no `| calls left` when uncapped). The bundle instruction
  states main's cap once. A thread whose growth passes C ends there (main:
  `main_end` `context`); out of calls, `step cap`. `tree.json` `budget` is null.
  Bundles built with the old `budget of N tokens` sentence: `build_bundles.py
  drop-budget` takes it out ([task-bundles.md](task-bundles.md)).
- Every fork task must carry `task_id` (k, required in the schema; a fork
  missing one, or naming a k without `/tasks/<k>.md`, is refused).
- Red check: before a fork's branches start, every task's acceptance check
  runs at once, each in its branch's copy of the forking thread's tree
  (`/testbed` mapped to it, 60 s), which is then reset to its fork point
  (`git reset --hard` + `git clean -fd`: ignored files are kept). A check that exits 0 (cannot detect its bug, or the task is
  already fixed), is broken (exit 127, bash/python cannot parse it) or fails
  because of an error in itself still starts its task: its branch is told the
  check is unreliable and must first write its own check that fails on the
  current code, then pass it to commit (`weak_acceptance`); the fork result
  names those tasks and why (`red_weak_note`). The red check never refuses a
  fork; a missing task id or one naming no task does (the bundle prompt says
  so, next to the usual "at least two tasks"). Older records keep
  dropped task indices (`dropped`).
  `tree.json` `red_checks`: per checked fork call
  `{thread, task_ids, exits, weak, own}`. Each branch's task
  message carries its task's `/tasks/<k>.md` statement after the forking thread's text (audit:
  [task-bundles.md](task-bundles.md#fork-acceptance-checks-bundle-e2b-audit-2026-10-11)).
- The verifier
  (`bundle_verifier_shell`) grades each branch patch alone on the initial
  tree, with only task k's ids (every task when it names none), then the final
  tree on every task: `tasks.json` `{k: 0/1}`, `reward.txt` = solved fraction,
  `branches.json` per branch `{key, task, tasks, passed}`.
- Every fork waits for its branches, main's as a branch's (search and code
  alike; there is no join tool): its result is followed by the join note.
  Fork records add `fork_time`, `join_index` (where the join note goes in the
  forking thread's history), `join_time`, `join_response_id` (the forking
  reply), and per branch `start_time` / `end_time` (wall clock) and the same
  `join_time` / `join_response_id`. Main's call cap counts only main's calls.
- Main's `commit` and `abort` close nothing in a bundle (its forks have
  joined their branches): each is refused (`tree.MAIN_CLOSE`).
- `C_wall` (recorded only: the group log, dev_eval) ends each branch at its
  tree.json `end_time`, so the tools after its final reply (a slow last test,
  commit's acceptance check) count, and a main control's `C_wall` runs main's
  last request before the forking reply to that reply's start, as in the
  forking tree. In seconds a blocking fork (every branch's own) adds nothing to
  `C_wall`: its reply's cost already runs past its branches. `plan_outcome`
  also records main's join idle `J` and post-join work `P` (`group.join_post`)
  and the bundle verifier's check validity (`checks`, `own_valid`;
  `scripts/rl_analysis/steps.py`).
- `OCTOPUS_AUTO_MERGE` (default `1` in bundle mode): at each join the runtime
  applies every committed branch patch to the forking thread's tree (main:
  `/testbed`; a nested fork: its parent branch's copy) with `git apply --3way`
  in commit order. A patch that conflicts is taken back whole (`git read-tree
  --reset -u` to the tree before it), so no conflict markers reach the tree;
  the join note says `merged` or `not applied: it conflicts in <files>; your
  files are unchanged` per patch. A patch whose only conflicts are scratch
  files it creates (at the repo root or a `test_` module) that the tree already
  has, a sibling's, is applied without them instead: `merged: except its new
  files <files> ...`. `tree.json` branch `merge` (`merged` | `merged: ...` |
  `conflict: <files>`).
- Reward: `y` = solved fraction in `R`. A branch naming task k has
  `passed` = k solved by its patch alone and `landed` = k solved in the final
  tree (the group log and the read control's score).
- Control: one per forking episode, from one accepted fork call drawn
  uniformly among all of the tree's (`random.Random(<trial name>)` over the
  calls sorted by path, so a rerun of the trial draws the same one; a refused
  fork call is no fork point). `tree.json` `control.kind`: `main-first`,
  `main-later` or `nested` (a branch's fork at any depth), `path`
  (`fork<k>` for main's k-th fork, `<branch thread>/fork<k>` below), `depth`.
  It starts where the forking thread stood just before its forking reply.
  Its tokens are its own: `C` and `y` stay the episode's.
  - Main's fork (first or later): `/testbed` back to that fork's snapshot (the
    merges of earlier joins are in it), main's messages up to the forking
    reply under the bundle's linear system message (`/tasks`,
    done only; no fork, task_id, commit or abort) and tools, main's calls left.
    The verifier grades its tree on every task between the branches and the
    final tree (`control_tasks.json`, `branches.json` `control.tasks`); its
    `y` is that solved fraction (graded, never trained); `C` = the
    tree's critical path before the forking reply + the control's tokens.
  - A branch's fork: the branch's workdir back to its tree at the fork
    (`workdir_point`, a snapshot each nested fork record now carries), its
    messages up to the forking reply with the depth-limit sentence appended to
    its task message (statement block kept) so fork is refused, its turns left
    under `OCTOPUS_BRANCH_MAX_TURNS` and prompt growth left under the branch's
    32k; closed by
    commit/abort like the branch. Its patch (since the branch's
    own base) is a plan.tsv row `control`, graded like a branch on the
    branch's task (`branch_control` `{of, passed}`: the group log's nested
    pair). It is never trained (loss mask 0, advantage 0); `A_main` and `y`
    do not see it.
  - Cost: still one control per forking episode. On bundle-e2b (1293 fork
    calls in 749 forked episodes: 58% main-first, 23% main-later, 19% nested)
    a nested control is graded on one task instead of
    the whole tree; a nested one also stops at the branch's turn cap. The
    spend should stay at or below #56's +23% generated tokens per step (not
    yet measured on a run).
- `dev_eval --tasks-dir <bundle dataset>` rows add `n`, `tasks`, `fraction`,
  `branches_per_task`, `branch_credit` and
  `wall_s`. Every plan row's `ended_with` is how main ended (`tree.json`
  `main_end`): `done`, `budget`, `step cap`, `context` (its growth past `thread_budget`), `no tool call`, `request failed`.
- Logs: per group the rollout log prints `[octopus] plan group {...}`: `fork_vs_control` per control
  kind (`main-first`, `main-later`: `[y, control y]`; `nested`: `[forking
  branch passed, control passed]`, each pair from the same fork point) and
  `fork_minus_control` (per kind, their mean difference), mean `|A|` per thread type,
  the share of bundle branches without a task, the share of committed
  passing branches that landed, the read credit (`read_adv`, `read_adv_n`) and
  `C_wall`.

## Read control (`OCTOPUS_READ_CONTROL=1`, with `OCTOPUS_READ_TOOL=1`)

This switch defaults off: `read_control()` is true only when `OCTOPUS_READ_CONTROL` strips to `"1"`. R is always the pure outcome reward 2y − 1 (leave-one-out over the mains); the default branch advantage is the episode's `A_main`, and the v5 runs set `rl.branch_credit` (Reward above). New experiments do not turn `OCTOPUS_READ_CONTROL` on. This section remains the spec of the optional eval/experiment switch.

Option-1 thread credit cannot isolate what a read is worth. In bundle-e13-read-10182126 there were about 1.8
read-carrying leaves per step. The summed push on read_thread tokens was +32 for valid calls and -1,281 for invalid ones,
and fork - control mixes forking with reading, because the control can do neither. The read control is a same-state
counterfactual for one read.

- **Pair.** At a branch's first valid read (`_read` mounted, so its next request carries the mount), the agent records
  the state just before that request: the history as sent (with the read reply, the read result text and its cursor;
  requests.jsonl rows are read up to the last complete line),
  the branch's workdir (a snapshot), the requests.jsonl rows logged so far, the episode's calls, the branch's replies
  and prompt growth so far, and its forks so far. After the episode (after the no-fork control), each recorded reader
  gets one continuation from that state with **no mount**. Its first request is the reader's mount request without
  `mount`: same cursor, same bytes but the read result, which the control sees without its step listing (its tag
  and the `Mounted ...` sentence only), so the read call is credited with listing + mount. It runs like a nested control (`_rerun_branch`, shared with it): workdir restored,
  fork and read refused, closed by commit/abort, under its reader's deadline ("Same deadline" below), with the
  reader's turns and `thread_budget()` growth left, and the budgets below ("Budgets"). It is graded like the branch on its task (plan.tsv row `readctl-<branch key>`), and its
  requests are requests.jsonl thread `readctl/<thread name>`, which the critical path, join/post and task-token sums
  leave out. tree.json `read_controls`: `{of, read_rid, read_call, read_calls (the read reply's tool calls as the server parsed them, names in order), deadline [s, 0] | null, deadline_from, cut (the deadline stopped it), forked_after, control_forked, end, budget_before, budget_used}`.
- **Same deadline on both sides** (review of #148). Each control faces the external deadline its reader faced,
  measured on its own clock from its start. Bounding a control only when its reader was cut biased A_read against
  reading: under no effect, E[A_read] = -p(1-p)(b+s), where p is the share of readers that close before main ends.
  - The deadline (`read_deadline`) exists only with a shared token pool (search's `OCTOPUS_SEARCH_BUDGET`; SWE has
    none, so its read controls run without one). It is the seconds from the reader's mount request to the moment the
    episode's shared token pool first ran out, which stops every running thread whether or not the reader had already closed (main's
    end is none: main's fork waits for every branch). It is recorded before any raise: a branch reply that empties
    the pool and is cut still records it. It is defined by the episode, never by how either side ended. tree.json
    `deadline_from` (pool) records where it came from.
  - Scoring uses the same rules on both sides. A run that stopped itself
    unclosed (a text-only reply, a failed request, `thread_budget()` growth, its turn cap) scores `-1` on either side. A
    closed run scores its commit or abort (Credit below). `cut` is recorded only.
  - The control runs after the episode, one at a time, so its server load and reply latency differ from the reader's.
    Wall-clock time is therefore not an exact match, but the deadline does not depend on either side's outcome.
- **Forks are left out of the credit.** The control's fork calls are refused (the depth-limit text) while its task text
  still allows forking. A pair where the reader called fork after its read (any fork call in its history from the
  mount request on, also in steps a later rewind removed: `read_ctl["rewound"]`), or the control called fork (`forked_after`, `control_forked`; fork calls on both sides, accepted
  or not), still runs and is recorded but carries no A_read. The reader's side reads its history from the mount
  request on. In such a pair, what forking was worth would show up as
  the read's value. A reader that rewinds to or before its read (the rewind tool) gets no read control at all
  (`_rewound`): its history from the mount request on is gone.
- **Budgets.** With a shared pool the control draws on what was left when its reader's mount request went out (the
  budget's own count then, the left that request was told). The pool's end is the common deadline above.
- **Later reads are part of the treatment.** The reader's further `read_thread` calls can mount; the control's are
  answered "read_thread is not available to the control". A_read measures reading from this point on, its first mount
  and any later ones, against no reading.
- **Retry orphans.** AReaL's `drop_retry_orphans` (grpo.yaml) groups the interactions whose input messages are
  equal. The control's first request repeats the reader's mount request with the same parent and messages, so the two
  were one retry group, and a reader whose mount request was its last (it committed right after the read) lost that
  leaf to the control's. `patches/areal-cursor.patch` adds the request's mount to the group key (rebuild the AReaL
  overlay). Without mounts, nothing changes. Test: `test_read_export.py::test_a_read_controls_first_request_is_no_retry_of_its_readers_mount_request`.
- **Main readers are excluded**: main is never re-run from a mid-episode state, and only 6 of the run's 73 valid reads
  were main's. A mount `OCTOPUS_READ_EVENT` makes never starts a read control: the control is recorded from the
  branch's own (voluntary) mount only.
- **Credit** (`group.read_values`). The environment supplies the pair's rewards through `group_step`'s
  `read_rewards` hook: `(episode, c, b) -> (r_reader, r_rc)`, each leaf on its own grade (the record's `passed`,
  `octopus/rl/group.py`'s contract: a bool or the environment's scalar, `None` for a leaf it does not grade). Code
  episodes (`reward.code_read_rewards`), the same rules on both leaves. Bundles: commit `+1` when
  its own patch solved its task (the control never merges, so the reader is judged the same way), else `-1`; abort `-1` when its own patch solved
  its task (a fix thrown away), else `0`; unclosed `-1`. A single task: commit `+1` if its patch passed, `-1.5` if it
  failed its own acceptance check (acceptance_verified, recorded as a branch's), else `-1`; abort `-1` when some branch
  of the group passed, else `0`; unclosed `-1`. **Search**
  (`search_read_rewards`): branch credit is the episode's `A_main`, so search grades a branch only as a reader: the
  read control and its reader are graded on their own result (`agent.run_trial` writes it to `verifier/answers.json`
  under their leaf keys, `reward.search_outcome` puts it in `passed`; every other branch `None`): the answer their
  `done` hands main (tree.json `summary`; search protocol v6 closes a branch with `done` only), judged against the
  question's answer like the answer itself (exact match, else the BrowseComp-Plus judge, whose prompt extracts the final
  answer from free text). A leaf that was stopped has none: incorrect. `r = 2·passed − 1`, the answer's scale, so
  `v` is `±2` or `0`. Only these two leaves per pair cost judge calls (not a pair `read_values` drops:
  `forked_after`, `control_forked`); runs without `OCTOPUS_READ_CONTROL` grade none. Known bias: the grade asks whether
  the finding answers the whole question, so a branch on a sub-question usually leaves both sides wrong (`v = 0`: a
  sparse signal), and a reader that restates an answer main could already see (the read target's finding) still
  scores `+2`. A first pilot counts the share of pairs with `v ≠ 0` and, for each `v = +2`, whether the read target
  already held the answer. If `v ≠ 0` is too rare, the upgrade is to re-run main's continuation from the join with
  the control's finding in place of the reader's and grade main's answer (the read's value to the episode). The
  `read_thread` call that mounted (`read_call`: a refused call before it in the same reply gets nothing) gets

      A_read = r_reader - r_rc

  in place of the reader's A_branch (assign_threads adds `A_read - A_branch` on `<function=read_thread>` to
  `</function>`). The reply's tool calls as the server parsed them (`read_calls`) define what the reply called; the
  format's regex only locates them (group.call_spans), and a reply whose matches are not exactly those calls (a call
  the parser took from a reply cut off inside it, a call written in prose) gets no span. Every other token keeps the episode's `A_main`.
- **The read control's tokens are not trained** (loss mask 0 on the whole leaf, advantage 0): its history says
  "Mounted ... you attend to them" while nothing is mounted, a state the policy never meets.
- **Cost** (from bundle-e13-read-10182126's 545 finished episodes, steps 1-14). 27 branch first reads, about 2 per step
  (the diagnosis estimated 2.5). Replaying each reader from its mount request to its end is 872 requests and 131k
  completion tokens: +0.57% requests and +0.47% tokens over the 152k requests and 27.9M tokens. Each control adds about
  32 replies (about 174 s of model time) to the end of its episode, since it runs after the episode, one at a time.
- Tests: `test_read_tool.py::test_a_branchs_first_read_gets_a_paired_control_without_the_mount` (scripted episode: the
  control's first request equals the reader's mount request minus the mount, its rows, workdir restore, plan.tsv row,
  then plan_outcome / plan_group_step: `read_adv` = 2.0, with every other advantage unchanged),
  `test_read_tool.py::test_a_self_stopped_reader_and_its_control_score_alike_and_the_mounting_call_is_named`,
  `test_read_tool.py::test_read_deadline_per_stop_type`,
  `test_read_tool.py::test_a_pool_ended_episode_gives_every_read_control_the_pools_deadline_and_the_pool_left`,
  `test_read_tool.py::test_a_branch_reply_that_empties_the_pool_still_sets_every_controls_deadline`,
  `test_read_tool.py::test_a_reader_main_waited_for_gets_no_main_end_deadline`,
  `test_rl_plan.py::test_read_credit_replaces_the_advantage_on_the_read_call_and_read_controls_are_not_trained`,
  `test_rl_plan.py::test_read_values_score_both_sides_alike_and_leave_out_fork_pairs` and the retry-orphan test above.
- **Deployment guard.** `rl_run.sh start` refuses `OCTOPUS_READ_CONTROL=1` unless `$CURSOR/areal`'s cache.py holds the
  mount in the retry-orphan key.

## Smoke-test checks (no GPU in this delivery)

- Train/infer logprob gap on a plan group (existing LoRA parity / step-1 check).
- A group with at least one forking episode and one no-fork episode kept by
  `has_variance`.

## AReaL facts (node-94 `/mnt/disk1/user/areal`)

1. **One sequence per thread from one proxy session.**
   `OpenAIProxyWorkflow.arun_episode` (`proxy/workflow.py:466-666`) opens one
   session per group sample. `InteractionCache.export_interactions`
   (`cache.py:564-667`) with `style='concat'` builds a parent tree from
   longest message-prefix and **returns only leaves**.
   `to_tensor_dict` (`types.py:160-281`) with `chat_template_type=='concat'`
   concatenates each leaf with its parents into one token sequence.
   Concurrent branches that share a prefix are sibling leaves, not one mixed
   sequence. Parent matching is `cache.py:454-468`. If matching fails, a
   branch becomes its own root (still one sequence; `prefix_tree` still
   dedupes). Control stays in the same session so it is another leaf; it is
   **not** a second proxy session.

2. **Advantages.** `GroupedRolloutWorkflow.arun_episode`
   (`remote_inf_engine.py:107-274`) merges samples, then octopus
   `plan_group_rewards` stamps per-sequence `octopus_adv`.
   `PPOActor._compute_advantages` (`trainer/ppo/actor.py:343-614`) runs GAE +
   `adv_norm` (`mean_level: group`). `ChunkedHeadFSDPPPOActor.compute_advantages`
   then **replaces** `advantages` where `octopus_plan` is set with the per-token
   `octopus_adv` (`octopus/rl/workflow.py` `to_tensor_dict`). `grpo_loss_fn`
   (`trainer/ppo/actor.py:1031`) reads `input_data["advantages"]`.

3. **logprobs not forwarded by stock AReaL.** Proxy `chat_completions`
   (`proxy_rollout_server.py:1031`) forwards through `_call_client_create`
   (`834-1004`), which drops kwargs not in
   `AsyncCompletionsWithReward.create` (no `logprobs` / `top_logprobs`), and
   `_build_chat_completion` sets `logprobs=None` (`client.py:945`).
   `patches/areal-logprobs.patch` keeps both kwargs, asks SGLang for
   `top_logprobs_num`, and returns `choices[0].logprobs`. No episode asks for
   them now; the patch is kept only as the base `patches/areal-cursor.patch`
   applies on (docs/cursor-protocol.md "AReaL proxy").

4. **prefix_tree fit.** `allocate_prefix_groups` (`octopus/rl/prefix_tree.py`)
   packs by **tree** tokens ≤ `max_tokens_per_mb` (262144 in `grpo.yaml`); a
   sequence (or a reader with its mount sources) over that gets a micro-batch of
   its own and is never cut. A single sequence must stay under `context_length`
   262144. A plan group of mains + branches + controls that share `s0` is split
   across micro-batches when the tree exceeds the cap; sequences that share a
   start stay together when they fit. CPU test: `tests/test_rl_plan.py::test_allocate_prefix_groups_fits_plan_main_branch_control`.
