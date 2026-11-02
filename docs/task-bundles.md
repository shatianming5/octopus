# Task bundles (`bundles-v2`)

The bundle RL runs (trial `bundle-e1d`, from 2026-10-09) train and evaluate on task bundles, not on single tasks.
A bundle is one Harbor task: one SWE-smith Python repo with 6–16 independent single-bug instances applied to ONE
tree and a statement per task in `/tasks/<k>.md` (`/tasks/INDEX.md` lists them). There is no shared token budget
(bundles built before its removal said `You have a budget of N tokens` with N = 8,000 per task; `build_bundles.py
drop-budget BUNDLE_DIR|MANIFEST...` takes that sentence out of `instruction.md` and the `budget` key out of manifest
rows, text only: check.py and test.sh never read it; run it after the new code is deployed, on every host's copy,
NVwulf's included: see "Dropping the old budget" below). Every thread has its own context growth cap and call cap
instead. Each task is graded on its own hidden tests; the episode reward is the solved fraction.
The runtime side (limits, per-task grades, reward) is in [rl-plan-episodes.md](rl-plan-episodes.md#task-bundles);
the builder is `octopus/envs/swe/data/build_bundles.py`.

`pool-v1` ([task-pool.md](task-pool.md)) stays the single-task pool (base pass rates, single-task baselines).

This file also records two other pools. CooperBench v2 build rules and acceptance are in the section
CooperBench feature pairs (`cooperbench-v2`, 2026-10-21) below; those instructions are budget-free.
Same-file coupled bundles are in the section Same-file coupled bundles (`build --rule samefile`) below, and
the node-134 pipeline is [octopus/envs/swe/data/coupled/README.md](../octopus/envs/swe/data/coupled/README.md).

## Where

| copy | path |
|---|---|
| node-67, node-94 | `/home/user/harbor-udocker/datasets/bundles-v2-valid/` (+ `.manifest.jsonl`, `.splits.json`) |
| node-134 | `/mnt/disk4/user/harbor-udocker/datasets/bundles-v2-valid/` |
| NVwulf | `/lustre/cluster/scratch/user/octopus/datasets/bundles-v2-valid/` |
| HF `Tommysha/octopus-train`, branch `code` | `$ROOT/harbor/datasets/` (`scripts/deploy.sh ROOT fetch`; bundles-v2-valid, coupled-a-*, cooperbench-v2-*) |
| this repo | `data/bundles-v2/manifest.jsonl.gz` (one row per bundle; this copy was not run through `drop-budget` and its rows still have the `budget` key), `data/bundles-v2/splits.json` |

Acceptance evidence on node-67 and node-94, `datasets/`: `bundles-v2.acceptance-1c.json` (both rounds raw, every
variant), `bundles-v2-m*.repairs.json`, `bundles-v2.timing.json`, `bundles-v2.audit.json` (Harbor-path audit);
rejected bundles in `bundles-v2-timing-rejected/`; the size-6 v1 build in `legacy/bundles-v1-size6/`.

## What is in it

**501 bundles, 5,384 tasks, 90 repos.** Train 431 bundles / 4,656 tasks / 74 repos; eval 70 / 728 / 16. The split is
by repo (seeded, ~15% of repos eval only): no repo is in both.

| tasks per bundle | 6 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 |
|---|---|---|---|---|---|---|---|---|---|---|
| bundles | 80 | 41 | 64 | 63 | 50 | 48 | 37 | 44 | 37 | 37 |

Size rule: a size drawn from 8–16 per bundle and filled from the repo's unused eligible instances (421 bundles);
once a repo cannot fill 8, bundles of 6 (`6-fallback`, 80). Up to 10 bundles per repo (median 5); no instance is
in two bundles.

| per task | p25 | median | p75 | max | mean |
|---|---|---|---|---|---|
| bug patch lines | 4 | 8 | 20 | 486 | 17.4 |
| FAIL_TO_PASS tests | 1 | 3 | 8 | 7080 | 13.7 |
| PASS_TO_PASS tests graded | 10 | 34 | 100 | 6887 | 141.8 |

Kinds: func_basic 1814, lm_rewrite 1251, func_pm_* 2319 (ctrl_shuffle 602, remove_cond 345, ctrl_invert_if 311,
remove_assign 302, class_rm_funcs 255, remove_loop 136, class_rm_base 96, op_* 208, remove_wrapper 63, other 1).
"Hard" (above-median rank sum of patch lines and FAIL_TO_PASS count) is 43%; easy and hard alternate inside a
bundle. Verifier wall time with every fix applied: median 11 s, p75 24 s, max 270 s.

Eval repos: typeguard, chardet, schedule, nikola, textfsm, inflect, faker, pdfplumber, schema, python-qrcode,
python-pinyin, prettytable, pyasn1, pygments, safety, flashtext. Base Qwen3.5-4B on eval (k=2, 140 episodes):
solved fraction 0.127 ± 0.033 (size 6: 0.196, 8–11: 0.125, 12–16: 0.094), fork rate 38%.

## How a bundle is built

1. Candidates: SWE-smith instances of the repo with a statement and FAIL_TO_PASS, all FAIL_TO_PASS inside the
   instance's hidden test files, passing on the original commit (per-repo baseline), a statement that does not
   quote reference-fix code, and a bug patch that adds no comment naming the change (`change_notes`, from
   2026-10-20: "# Changed from self.plural to self.plural_adj"; 58 of 329 accepted same-file train bundles had one)
   and that deletes fewer than 250 lines (a whole class removed is next to unsolvable).
2. Tasks in a bundle conflict on nothing: different functions (a file without function info counts whole), no bug
   file is another task's hidden test file, FAIL_TO_PASS sets disjoint, statements distinct.
3. The tree: the original commit, every task's bug patch, the union of the tasks' "Remove F2P Tests" changes, a fresh
   git history. Every reference fix must `git apply` on it alone and all together; leak scans: no FAIL_TO_PASS test
   left, no reference-fix line the bundling introduced, no second copy of the package in site-packages.
4. Grading (`tests/check.py`): one pytest run over every task's ids, each task scored from its own ids;
   FAIL_TO_PASS must pass (skipped is not solved), PASS_TO_PASS in the task's hidden files must pass or skip. A
   crashed run falls back to per-task runs under the same rule.

## Acceptance (every pool bundle passed all of it)

- One container per bundle, two rounds of: no fix (every task 0), fix k only for each k (only k is 1), all fixes
  (every task 1), and two seeded random fix subsets (exactly the subset is 1). Both rounds identical.
- Every task's tests under 60 s; the whole verifier with all fixes under 300 s.
- Harbor-path audit: 20 sampled bundles re-run through `harbor run -a oracle` matched the single-container results.
- Independent static audit of the final pool (raw rounds, disjoint FAIL_TO_PASS, fixes present, statements,
  one check.py hash, no instance reuse, repo-level split): 0 failures.

Funnel: 679 bundles built from 101 repos → 404 passed the first round → 110 more repaired (culprit tasks replaced from
the same repo, re-accepted) → 514 → timing gate dropped 9 (4 over 300 s, 5 django channels flaky) → 4 more flaky
django bundles dropped after the audit → **501**. First-round rejects: task interaction 177, slow 47, a task passing
with no fix 31, flaky 10, leak 7, a task failing with all fixes 3.

Repos without bundles (40 of 130): too few conflict-free eligible tasks (17, e.g. icecream, tomli, addict),
FAIL_TO_PASS failing on the original commit (6: autograd, modin, gtts, pydicom, dspy, sqlglot), broken image
(pydantic, MONAI), no single-bug branches (mypy, string2string), FAIL_TO_PASS outside the hidden files (mido,
tornado), and all bundles rejected (hydra, pandas, dask, dvc, feedparser, sunpy, parso, pudo, pysnooper,
django-channels, daphne).

## Fork acceptance checks (bundle-e2b audit, 2026-10-11)

2,063 branches of training run bundle-e2b (158 bundles, all 71 repos, every ending), each acceptance check run
verbatim through the runtime's shell tool (`bash -lc` in `/testbed`, 60 s) in a fresh container of its bundle:
(a) on the initial tree, (b) with only that task's reference fix. A check is sound when (a) fails and (b) passes.
60/60 branches that ran their check before any edit saw the same pass/fail as (a).

| class | share | committed | runtime-verified commits | commits passing hidden tests |
|---|---|---|---|---|
| sound | 15.7% | 44% | 85% | 55% |
| cannot fail ((a) exits 0) | 34.3% | 29% | 99% | 35% |
| broken (not a command, or its code does not parse) | 25.6% | 27% | 5% | 32% |
| cannot pass ((a), (b) both fail) | 23.9% | 25% | 22% | 26% |

A keyword rule (`| head`, `|| true`, `; echo`, ...) finds cannot-fail checks at 86% precision but 31% recall.
Task texts: 3.2% placeholders (`"7"`, `"task 2"`), ~8% describe another task (spot check: 8 of 10 clearly wrong).

Each bundle fork's checks run on scratch copies of the forking thread's tree first (the red check). A task whose check
already exits 0, is broken (exit 127, a bash `-c` syntax error, a python `-c` / heredoc compile error) or fails
because of itself still starts (dropping such tasks cost forks in bundle-e4: a dropped task was solved 16% of the time
by main vs 37% by a started branch), but weak: its branch sees the check as unreliable, with the reason, and is told
to write its own check that fails now. Main's fork waits for its branches and returns their join note
([rl-plan-episodes.md](rl-plan-episodes.md#task-bundles)). Each branch's task message carries its
`/tasks/<k>.md` statement.

## Fork acceptance checks, bundle-e6 audit (2026-10-13)

831 runtime-checked (non-weak) checks of bundle-e6 steps 1–13 replayed in containers (`accaudit-e6` on node-67) on the
initial tree, with the task's reference fix, with every fix, and on the branch's patch alone (re-graded on the hidden
tests); the replay agrees with the recorded commit-time result for 95% of commits.

| finding | number |
|---|---|
| checks that fail on the bug but also on the reference fix | 62% (34% valid, 3% need another task's fix) |
| hidden pass when the commit-time check passed / failed, invalid checks | 50% / 55% (no signal) |
| the same, valid checks | 62% / 25% |
| "check failed, hidden passed" commits caused by an invalid check | 97% of 103 |
| "check passed, hidden failed" commits with a valid check (the statement covers part of a multi-change bug, or a weak check) | 68% of 71 |
| same task: branch cut off / hidden pass under an invalid vs a valid check | 40% vs 30% / 41% vs 49% |

Why invalid: imports of names not in the repo (19% of all checks), errors in the check's own code (17%), wrong
output or grep expectations (10%), API misuse raised inside the library (10%), wrong asserts (4%), pytest on a hidden
file (2%). Others avoid this by grading with the dataset's tests and validating generated tests against the gold patch
(SWT-Bench); Agentless's reproduction tests pass on the gold patch 44% of the time and are only a tie-breaker there.

What changed (runtime, eval-legal unless noted):

- **Checks that fail because of themselves** (`octopus/envs/swe/agent.own_failure`: pytest exit 4/5, a ModuleNotFoundError,
  or Import/Name/Type/Attribute/FileNotFound/Index/UnboundLocal raised in the check's own `<string>`/`<stdin>` line)
  start weak like broken ones, with the error line shown to main and to the branch. On the audit: 94% precision, 56%
  recall, 20 of 285 valid checks flagged.
- **A weak task's branch passes its own check to `commit` (`acceptance`)**: the runtime runs it on the branch's tree
  (must exit 0) and on a scratch copy of the fork point, the HEAD commit of the branch's private copy (must fail).
- **Commit gate**: the first commit whose runtime check fails (or whose own check also passes on the fork point), or
  a weak task's commit with no own check at all, is refused with the output; the branch fixes the code, aborts, or
  commits again as it is (the join note says so). Without the no-check case, 72 of 183 pilot commits were unverified.
- **Check validity** is recorded, not trained: the bundle verifier runs every fork check on the initial tree and with
  its task's reference fix (`tests/fixes/`, never shown to the agent) into `checks.json`; a check that fails on the
  bug and passes on the fix is valid (`plan_outcome` `checks` / `own_valid`, `scripts/rl_analysis/steps.py`). The
  e8f–e14 training terms on top of the outcome (check validity `OCTOPUS_CHECK_COEF` / `CHECK_BASE`, fork-task credit
  and its group baseline, delegate cost, the mains-only baseline, the efficiency advantage) are at commit a6e83a9.

## Statements cut to what they describe (`bundles-v3`)

A SWE-smith bug patch can change several places while its generated statement describes one symptom
(line_profiler: the statement names `return None`, the tests also need an inverted `is_generator`). Grok 4.6 labels
each unit of each task's reference fix described / not described / mixed (`statement_coverage.py judge`; units are
line pairs, or one region per hunk for rewrite and move kinds); `build_bundles.py reduce` fixes the undescribed part
in the bundle tree (a bake patch after every task's setup) so each task's bug is what its statement says; tasks with
nothing described, a hidden source module (tldextract, pinyin, patsy), or a cut that does not compile are dropped;
`refresh` re-runs the graded ids on the old and the cut tree (FAIL_TO_PASS ids that pass after the cut become
PASS_TO_PASS; a cut that makes a passing id fail drops that bundle's cut tasks). Statement code blocks that do not run
on the correct code (missing files, made-up APIs) are rewritten by Grok and kept only if they then run
(`statement_coverage.py snippets / repair / overrides`), else removed.

## Small bundles for e11 (`bundles-e11a-c`, `bundles-e11b-c`, 2026-10-17)

The e7-e10 training list (`rl/bundle-e8-tasks.txt`, 389 bundles of 4-16 tasks) has no 1-3-task bundles, so training
never shows the policy a bundle too small to fork; on held-out, 1-7-task bundles lose M2 (e10 s40: -0.29 on 1-3).
`rl/bundle-e11-tasks.txt` on node-94 is the e8 list unchanged plus:

- `build --sizes ... --only IDS --tag T` bundles of 1-7 tasks from training repos only, then the same
  reduce (judge_all labels, statements_all overrides) -> refresh -> reduce -> accept as `bundles-v3`:
  - `e11a`: the 60 labelled tasks no accepted bundle uses (never blamed by an acceptance, never dropped by a reduce,
    statement code checked), sizes 1-3;
  - `e11b`: tasks of the e8 list itself, each once, sizes drawn from 1,1,2,2,3,3,4,5,6,7, 2 per repo. Every other
    labelled training task is in an accepted bundle; new tasks would need Grok labels. A task can so be in an e8
    bundle and in one e11b bundle;
- 15 accepted `bundles-v3(f)-valid` bundles of 1-7 tasks that e7list.py left out (under 4 tasks, or a repo key
  starting with `b`: `.split(".b")` cut `gunicorn.bacbf8aa`, `h11.bed0dd4a`, `astroid.b114f6b5`,
  `cookiecutter.b4451231` and `mistune.bf54ef67` out of the list; their 8-16-task bundles stay out).

Acceptance: e11a 23/25 (one interaction; one paramiko bundle stopped after 90 min unjudged), e11b 124/131 (interaction
4, flaky 3); refresh broke no id; slowest task 28 s, slowest all-fixes run 81 s.

| bundles | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8-16 | all | 1-3 share |
|---|---|---|---|---|---|---|---|---|---|---|
| new (e11a 23, e11b 124, v3 15) | 39 | 34 | 41 | 7 | 13 | 15 | 13 | 0 | 162 (504 tasks) | 70% |
| bundle-e11-tasks.txt | 39 | 34 | 41 | 14 | 26 | 75 | 48 | 274 | 551 (4,142 tasks) | 20.7% |

399 of the 504 new tasks are also in an e8 bundle (e11b); no repo of the 16 `bundles-v3` eval repos and none of the 512
held-out task ids is in the list (`e11/verify.py` on node-94, which also loads the list through `pool_dataset` and the
bundle verifier copy). Work files: node-67 `harbor-udocker/e11/`.

## CooperBench feature pairs (`cooperbench-v2`, 2026-10-21)

CooperBench (arXiv 2601.13295; HF `CooperBench/cooperbench-dataset` at b612b1a) as 2-task bundles, built by
`octopus/envs/swe/data/cooperbench.py`: each pair of features of one PR is one bundle, `/tasks/1.md` and
`/tasks/2.md` the two `feature.md`, the tree the upstream repo at the PR's base commit from the official image
`akhatua/cooperbench-<repo>:task<id>` (Entrypoint `runner.sh` cleared; `/workspace/repo` moved to `/testbed` with a
symlink back for the editable install; `.git`, which holds the upstream PR, replaced by one commit). 9 Python repos,
24 PRs, 165 features, 560 bundles; a feature recurs in every pair of its PR, so the split is by repo (`split` in the
manifest): train Jinja, DirtyEquals, Tiktoken, Pillow, LlamaIndex, Datasets; val Outlines; test Click, DSPy.

Grading (`tests/check.py`) follows the official harness per task: the tree's test files back to the initial commit
(the harness drops test files from the graded patch), the feature's `tests.patch`, the PR's `runner.sh` test command
without its installs; solved = exit 0 and at least one test passed. What the tests would download is baked into the
image as an extra layer (tiktoken's `cl100k_base`; llama_index's `_static` caches: `cl100k_base`, nltk `stopwords`,
`punkt_tab`), and the libraries' switches are set in the image (`LITELLM_LOCAL_MODEL_COST_MAP=True` for DSPy,
whose cost-map fetch cost ~11 s per pytest start; `HF_HUB_OFFLINE=1` for Outlines; `NUM_WORKERS=1` for DSPy: 8587's
tests start a litellm proxy that forks one worker per CPU and under PRoot is not listening within their 10 s). llama_index 18813's base tests on
the `mock_pdf_bytes` / `pdf_url` fixtures download a PDF from w3.org and are deselected (as the official command does in
datasets 7309); the new tests of its features 2 and 3 use that fixture too and still need the network. llama_index
17244/18813 install a PyPI `llama-index-core` into site-packages; the healthcheck removes it and a `.pth` puts
`/testbed/llama-index-core` on the path (17070 has an editable install).
`solve.sh` applies the PR's `combined.patch`; `solution/fix_<k>.patch` are the two gold
features (together they do not always pass both: click 2800 f1+f2 conflict semantically).

Acceptance of `cooperbench-v2` (2026-10-21; the review fixes, budget-free instruction; node-136, 28 containers, pushed
commits 6df0752 / 12d67b1 (the shard restarted after overlay trials were enabled) and b3aacd9 (one bundle re-run after a
stale git lock, and the scan); one container per bundle, twice: no patch, gold A, gold B, combined; the gold of a task
must solve it, combined both, no patch neither, both rounds equal; merged by `merge` into
`cooperbench-v2.merged.acceptance.json`, no conflicts): **532 of 560** in `cooperbench-v2-valid`, all 311 Train bundles
in `cooperbench-v2-train-valid` (now `cooperbench-v2-train-valid-v1`; the current one is 302, below). The older `cooperbench-v1` acceptance ran on an unpushed snapshot and is superseded.

| repo | split | PRs | pairs | accepted | test.sh median / max (s) |
|---|---|---|---|---|---|
| Click | test | 3 | 115 | 115 | 2.6 / 10.6 |
| Jinja | train | 3 | 135 | 135 | 2.7 / 10.6 |
| DirtyEquals | train | 1 | 36 | 36 | 1.8 / 3.3 |
| Tiktoken | train | 1 | 45 | 45 | 2.0 / 12.3 |
| Pillow | train | 3 | 30 | 30 | 2.6 / 7.2 |
| DSPy | test | 4 | 55 | 55 | 39.8 / 249.4 |
| LlamaIndex | train | 3 | 39 | 39 | 15.4 / 34.2 |
| Datasets | train | 3 | 26 | 26 | 25.1 / 37.0 |
| Outlines | val | 3 | 79 | 51 | 11.4 / 26.2 (accepted ones) |

Before Step 4 (#225 blocks the network of agent commands in udocker trials; Singularity trials on NVwulf and Alpha are not covered) the Train bundles whose grading needs the network are out of
`cooperbench-v2-train-valid` (311 -> 302; the old one is `cooperbench-v2-train-valid-v1`): llama_index 18813 with
feature 2 or 3, whose new tests download a PDF from www.w3.org (`octopus/envs/swe/data/cooperbench_needs_network.txt`).
Found with `cooperbench.py offline cooperbench-v2-valid`, which grades every accepted bundle's combined patch with
`OCTOPUS_NO_EGRESS=1`, then grades the bundles that fail again with and without it (needs the network: solved with it,
not without); no Val/Test bundle needs the network. In the first pass DSPy 8394's f1-f2 hit `udocker images`' 60 s
timeout, its other 9 bundles the runner's `timeout 300` under load 25, and datasets 7309 f1-f2 a setup timeout (rc 124);
all passed with and without the marker on the re-run. 18813's `--deselect` ids were relative to the repo
root and deselected nothing; they are relative to pytest's rootdir now (`cooperbench.py commands` rewrote the existing
bundles; the 15 18813 bundles were accepted again, 15/15).

Over 180 s: DSPy 8394 f1-f2, f1-f4, f1-f5 (test.sh up to 249 s with 28 containers on node-136, 128 s with 10 on
node-134), above the planned 180 s gate; they pass acceptance and the verifier timeout is 1800 s, so they stay (user's
decision, 2026-10-21).

Rejected: Outlines 1706 (28 pairs, `cooperbench-v2.excluded.json`). Its base test file loads a llama.cpp model with
`Llama.from_pretrained`, which lists the Hub repo (`/api/models/M4-ai/TinyMistral-248M-v2-Instruct-GGUF` and its tree)
and has no offline mode. Its image uses `HF_ENDPOINT=https://hf-mirror.com`: one test run passes, but under acceptance
load hf-mirror answers `429 Too Many Requests` (`cooperbench-v2.task1706-429.log`: 120 unsolved task runs with a 429
line). DSPy 8587 failed on node-94 until `NUM_WORKERS=1` (above). A gold patch also solving the other task of its pair:
pillow 25 f1-f2, 290 f1-f3, llama_index 17244 f1-f3, datasets 3997 f2-f3.

Leak scan per PR (`cooperbench-v2.merged.leaks.json`, 24 PRs, none flagged): the tree is the upstream base commit
with one fresh commit, no other ref or `.git`; every copy of the package outside the tree (any `<pkg>/__init__.py`,
caches included) is compared line by line with the PR's added lines (>= 40 chars, absent from the base tree): 0 lines
after the healthcheck everywhere. Before it, llama_index 18813's image held 2 of them in uv's cache
(`/root/.cache/uv/archive-v0/...`, a full PyPI llama-index-core that site-packages hard-links to); the healthcheck now
removes `/root/.cache/uv` and the PyPI `llama_index/core`. Files elsewhere with >= 5 distinct added lines: none.
Same-name packages are listed, not flagged (scipy.datasets, keras.datasets for datasets).
w3.org (llama_index 18813 features 2/3's new tests download a PDF from it): reachable from node-134, node-68, node-138 and
node-136 (`curl -sI https://www.w3.org/`: 200; the PDF: 200), 2026-10-21; where it is not, those bundles score 0 even
with the gold patches.
Base Qwen3.5-4B on the 311 Train bundles (k=1, `scripts/eval/cooperbench_base/`): mean reward 0.137, at least one
task solved in 69, both in 16. Measured on `cooperbench-v1-train-valid` under the 16k shared token budget of the
protocol before #220.
`harbor run -a oracle` (node-134, code 865aece) on click 2068 f1-f2, dspy 8587 f1-f2, llama_index 18813 f2-f3 and outlines
1655 f1-f2: reward 1.0 each.

```
cooperbench.py fetch DS_DIR WORK [jobs]   # images via 2 Docker Hub mirrors (MIRRORS) -> udocker load (python3 -I)
cooperbench.py build DS_DIR NAME          # datasets/NAME + NAME.manifest.jsonl and NAME.manifest.full.jsonl
cooperbench.py accept NAME [conc]         # none / gold A / gold B / combined, twice -> NAME.acceptance.json, NAME-valid/
cooperbench.py scan DS_DIR NAME [conc]    # leak scan per PR -> NAME.leaks.json
cooperbench_shard.sh SRC CODE NAME CONC REPO...  # on another host: pull NAME's bundles of those repos, the code and the
                                          # blobs from SRC, fetch, accept, scan; results back as NAME.*.<host>.json[l]
cooperbench.py merge NAME                 # on SRC: NAME.merged.acceptance.json / .leaks.json, NAME-valid, NAME-train-valid
cooperbench.py offline NAME [conc]        # combined patch graded without network (OCTOPUS_NO_EGRESS=1) -> NAME.offline.jsonl
cooperbench.py commands NAME              # test commands in NAME's tasks.json from COMMANDS
cooperbench.py drop NAME LIST             # NAME without LIST's bundles; the old version becomes NAME-v1
```

## Same-file coupled bundles (`build --rule samefile`, 2026-10-20)

Bundles of 2-4 tasks that share code: every task's reference fix changes one and the same `.py` file, each task
added collides with a task already in the bundle (a bug hunk at most 20 lines from one of its hunks, original-commit
lines, or in the same function or class), and at least one pair of the bundle is within 20 lines (one function or
class alone is not coupling: `create` in jsonschema wraps a whole validator class). The other candidate and tree rules stay
(disjoint FAIL_TO_PASS, distinct statements, no bug file is a hidden test, patches and fixes apply, leak scan); only
"different functions" goes. `plan_samefile`: files take turns in a seeded order, one bundle per turn (size drawn
from `--sizes`, the file's next untried task first, the second task within 20 lines of it, so the pair is there), a bundle is kept only when full, up to `per_repo` bundles per
repo. A task that fails on a bundle tree goes back to the pool (its failure may come from the task it collides
with). Acceptance tests every pair of fixes for bundles of up to 4 tasks (instead of two random subsets).

Gate 0 (`scripts/eval/gate0_shard.sh`) takes all of `coupled-a-val` plus 244 of `coupled-a-train`, drawn with
`random.Random(0)` from its sorted bundle ids (289: Val 45 + Train 244). That 289 is this Gate 0 subset, not the
size of `coupled-a-train`; `coupled-a-test` never enters. The solvability audit (2026-10-21, 890 tasks) flagged 53,
kept 30 after review (23 overturned to solvable), and 12 bundles were dropped from the dataset afterwards
(`octopus/envs/swe/data/solvability.py`).

## Rebuild

```
build_bundles.py baseline NAME all            # per repo: tests passing on the original commit
build_bundles.py build PARQUET_DIR NAME all   # bundles (8–16, else 6)
build_bundles.py accept NAME [conc]           # single-container acceptance -> NAME.acceptance-1c.json, NAME-valid/
build_bundles.py repair PARQUET_DIR NAME SRC  # replace a rejected bundle's culprit tasks, then accept again
build_bundles.py timing NAME MANIFEST         # verifier wall-time gate (all fixes <= 300 s)
build_bundles.py audit NAME                   # Harbor-path sample audit
# bundles-v3: statements and bugs agree
statement_coverage.py judge DATASET LABELS.jsonl                    # Grok unit labels (resumable)
build_bundles.py reduce PARQUET_DIR NAME OUT LABELS                  # cut tasks -> datasets/OUT
build_bundles.py refresh NAME OUT                                    # graded ids, old vs cut tree
statement_coverage.py snippets / repair / overrides                  # statement code that runs
build_bundles.py reduce PARQUET_DIR NAME FINAL LABELS REFRESH bundles-v2 STATEMENTS.json, then accept FINAL
build_bundles.py refill PARQUET_DIR SRC CUT OUT [6 [8]]            # bundles under 6 tasks or rejected: refilled from the
                                                                     # repo's unused tasks, then judge/reduce/refresh/accept OUT
build_bundles.py build PARQUET_DIR NAME KEY... 2 --sizes 1,1,2,2,3,3,4,5,6,7 --only IDS --tag e11b   # e11 small bundles,
                                                                     # then reduce/refresh/reduce/accept as above
build_bundles.py build PARQUET_DIR NAME KEY... 10 --sizes 2,2,3,4 --rule samefile --tag sf   # same-file coupled bundles
```

### Dropping the old budget

Once per host holding datasets (node-94, node-136, NVwulf, ...), after the code without the shared budget is deployed
(older code fails on an instruction without it). Found by content, so it is repeatable and covers every dataset
(bundles-v2-valid, cooperbench-v1*, coupled-a-*, ...); `$N` is the harbor-udocker root, `PYTHONPATH=<code dir>`:

```
find "$N/datasets" -name instruction.md -exec grep -lZ "You have a budget of" {} + \
  | xargs -0 -r dirname -z | xargs -0 -r python -m octopus.envs.swe.data.build_bundles drop-budget
find "$N/datasets" -name '*.manifest.jsonl' -exec grep -lZ '"budget":' {} + \
  | xargs -0 -r python -m octopus.envs.swe.data.build_bundles drop-budget
grep -rl --include=instruction.md "You have a budget of" "$N/datasets" | wc -l   # 0
```

Image names: 64 repos use the local `swebench/...:transport-*` names, the rest `jyangballin/...:latest`; every
golden also has a `:latest`-keyed record (`golden/.golden/<sha1(name)[:16]>.cid`).

## Running a bundle RL run

Current SWE runs: `scripts/rl_env/rl_run.coupled.env` (README) or `scripts/alpha/` (Cluster Alpha: `rl_run.env`, `pilot.sbatch`, `tasks/swe-full.sh`, `tasks/swe-v5.sh`); the steps below are the node-94 / bundle-e9 flow.

Trainer host node-94, env file per run (`scripts/rl_env/rl_run.bundle-e9.env` was the bundle-e9 config; see commit a6e83a9); all scripts
take `RL_ENV_FILE`:

1. Launch: `rl_launch.sh TRIAL` (rl_run.sh start + watchdog), or `rl_autolaunch.sh TRIAL MARKER` to start once the
   previous run's evals are done (node-67 `eval_gate.sh` writes MARKER) and the GPUs are free, guarding the first step.
2. Watchdog: `rl_watchdog.sh` (v2) resumes the same trial after OOM, NaN, crashes, host-memory overrun, dead servers or a
   2 h stall, stops the run at the last step, and writes `$STATE/needs-main` when it gives up. `rl_wait.sh STEP` blocks
   until a step, an incident or the end; `rl_restart.sh AFTER KEY=VALUE` resumes the trial with changed settings.
3. Periodic eval: `scripts/eval/pool.py` on node-67 evaluates every saved adapter (the last step via
   `scripts/rl_analysis/dcp_adapter.py`).
4. Step metrics: `scripts/rl_analysis/steps.py JOBS_DIR MAIN_LOG CACHE` (per training step), `steps_pool.py`,
   `step_timing.py`, `fork_wall.py`.

Rollout concurrency is `EPISODES_PER_SERVER` per rollout-worker process, not per SGLang server: a group's episodes all go
to one worker. In bundle-e9 at 6, 4 of 8 workers sat at the cap with groups queued behind them while the SGLang servers
ran 2-3 requests each with queue 0, and steps 9-10 waited 892 / 640 s on rollouts. Raised to 10 after step 10: 80
concurrent episodes at CPU load ~15/32.
