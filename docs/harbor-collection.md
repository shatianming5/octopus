# Harbor trials

A code-environment episode is a Harbor trial: one OpenAI-compatible session
(the RL policy behind its AReaL proxy session) in the task image, scored by
`tests/test.sh` writing `/logs/verifier/reward.txt`. The agent is
`octopus.envs.swe.agent.GrokHarborAgent` (the cursor runtime, `octopus/runtime/tree.py`, on
its Harbor workspace). It offers the eight coding
tools and the Octopus runtime tools together, with `tool_choice` left on
`auto`:

- coding: `reproduce`, `rerun_reproduction`, `edit`, `shell`, `view`, `write`, `test`, `done`
- runtime: `fork`, `commit`, `abort` (`checkpoint` and `release` were removed; older traces have them);
  `rewind` with `OCTOPUS_REWIND_TOOL=1` ([rl-plan-episodes.md](rl-plan-episodes.md) "Rewind tool");
  `read_thread` with `OCTOPUS_READ_TOOL=1`, not in the tool list: its definition is appended where a read can succeed
  (a branch's task message, main's first fork result; [cross-branch-read.md](cross-branch-read.md) "The read tool")

Every tool in the list is offered from the start; whether and when to fork is the model's
choice (except Gate 0's `OCTOPUS_AUTO_FORK=1`, eval only: the environment forks one branch per bundle task before
main's first request; [cross-branch-read.md](cross-branch-read.md) "Gate 0"). Each branch should `commit` or `abort` before `done`.
In udocker trials the agent's container commands get no network off the host (#225, `OCTOPUS_NO_EGRESS=1` enforced by `udocker_nokill.c`; `octopus/envs/swe/data/README.md` Notes). Singularity trials (NVwulf, Alpha) are not covered: their agent commands keep the network.
`OCTOPUS_CHAT_URL` is the episode's AReaL proxy session. Each model turn executes every
`tool_calls` entry.
Fork continuations write the real tool name and arguments into
`trajectory.json`. It does not replay a saved step 0 and does not cap `view`.
The verifier runs `/opt/miniconda3/envs/testbed/bin/python`.

Harbor runs every trial on one asyncio loop, so nothing in the agent or the
udocker environment may block it. Model requests run on a `OCTOPUS_HTTP_WORKERS`
thread pool (default 128), and udocker calls run on `UDOCKER_THREADS` (default
128). Fork branches ask the model concurrently, while tool calls within one
trial stay serial because they share one container and one tool state.

`octopus.envs.swe.udocker` extracts each image once into a golden
container under `UDOCKER_GOLDEN_DIR` and copies it for every trial into
`UDOCKER_CONTAINERS`. On node-94 everything lives on the NVMe home disk under
`/home/user/harbor-udocker` (see its README): a golden copy takes about
10 s, where a fresh `udocker create` of the same 3.3 GB image on the busy HDD
took over 600 s. At most `UDOCKER_CREATE_SLOTS` (default 4) creates or copies
run at once. If Harbor stops a trial while it is still starting, the copy is
killed and the container removed. Workers exit on their own when the Harbor
process is gone.

History (v0.1 / lighthouse era): the seven lighthouse journals (21 cold-start samples) were kept as a separate
comparison set, not re-encoded and not mixed with Harbor rewards. The single-task path still leaves out the two holdout
instances (`amueller/word_cloud`, `prettytable/prettytable`; `HOLDOUT_IDS` in `octopus/envs/swe/taskset.py`) and every
bottlepy instance: `taskset.is_excluded`, used by `emit_tasks.py` (through `select_task_specs`) and by
`select_swesmith.py` (pool-v1). bundles-v2 does not apply it (`build_bundles.py` never calls `is_excluded`): the
word_cloud holdout is in train bundle `amueller__word_cloud.ec24191c.b01.r1`, the prettytable holdout in eval bundle
`prettytable__prettytable.ca90b055.b01`, and bottlepy bundles are in train (`data/bundles-v2/splits.json`).

`harbor datasets list` does not ship a SWE-smith split that matches these
local images, so tasks are generated from `evaluation/frozen-b7/grok-scale`
by `octopus.envs.swe.data.emit_tasks`.

Each SWE-smith image serves every instance of its repo and sits on a clean
`main`; the instance lives on branch `origin/<instance_id>` as two commits,
`Bug Patch` and `Remove F2P Tests`. A task therefore:

- runs `branch_setup` as its Harbor healthcheck (once, before the agent):
  check out that branch, then replace the git history with one commit, since
  `main` is the fix and the branch tip hides the tests;
- restores the hidden tests in `tests/test.sh`: hidden files the bug never
  touched come back wholesale from `tests/hidden_full/` (overwriting whatever the
  agent wrote there, as SWE-smith does); files that also hold the bug get only the
  removed hunks back from `tests/hidden_tests.patch`. It grades the files in
  `tests/hidden_files.txt`, deselecting tests in `tests/known_failing.txt` that
  fail on the clean reference too;
- carries the reference fix in `solution/` for Harbor's `oracle` agent.

Those files need the image's git repo, so `octopus/envs/swe/data`
(`regen_tasks.py`, `baseline_tasks.py`, `scripts/validate.sh`) writes them on
node-94 and keeps only tasks where Harbor's `oracle` scores 1 and `nop` 0.
Instances whose `Remove F2P Tests` deletes the module the bug is in, and the
hand-written oracle tasks, are left out. Rewards from runs before 2026-09-26
graded agents on bug-free `main` and are not usable.
