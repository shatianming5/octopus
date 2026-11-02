# Search environment (`octopus/envs/search`)

Offline search episodes for Octopus fork/join: the retrieval service (`service.py`), the agent (`agent.py`), the
grade (`reward.py`), RL (`rl.py`, `scripts/rl_env/rl_run.search.env`; the runs' configs `scripts/alpha/search.env`, `scripts/beta/search.env`) and evaluation (`octopus/eval/search_dev_eval.py`, `octopus/eval/bcp.py`).
Each module's docstring has its usage.

## BrowseComp-Plus protocol

Training (OpenResearcher questions) and evaluation (BrowseComp-Plus) use the same tools and the same judge, taken from
[texttron/BrowseComp-Plus](https://github.com/texttron/BrowseComp-Plus/tree/046949032b0328319cc9a02663a759ec601d9402)
(`BCP` below, commit `0469490`).

Aligned:

| | ours | BCP |
|---|---|---|
| search | `search(query)`, a fixed top k (no `k` argument for the model); the agent asks for 10 (CLM's, `agent.SEARCH_K`), the service's default stays 5 | `--k` default 5, fixed per session: [mcp_server.py L42-L47](https://github.com/texttron/BrowseComp-Plus/blob/046949032b0328319cc9a02663a759ec601d9402/searcher/mcp_server.py#L42-L47), [tools.py L24-L34](https://github.com/texttron/BrowseComp-Plus/blob/046949032b0328319cc9a02663a759ec601d9402/searcher/tools.py#L24-L34) |
| hit text | the document's first 512 tokens of the Qwen/Qwen3-0.6B tokenizer (`encode` without special tokens, `decode` skipping them), the whole text when shorter (`service.Engine.snippet`) | [tools.py L16-L18, L36-L46](https://github.com/texttron/BrowseComp-Plus/blob/046949032b0328319cc9a02663a759ec601d9402/searcher/tools.py#L16-L46), `--snippet-max-tokens` default 512: [mcp_server.py L36-L41](https://github.com/texttron/BrowseComp-Plus/blob/046949032b0328319cc9a02663a759ec601d9402/searcher/mcp_server.py#L36-L41) |
| retriever | BM25 is the main setting (service `--backend bm25`, the default); dense Qwen3-Embedding (`--backend dense`) is the option | BM25 and Qwen3-Embedding indexes: [README L75-L79](https://github.com/texttron/BrowseComp-Plus/blob/046949032b0328319cc9a02663a759ec601d9402/README.md#L75-L79) |
| judge prompt | `reward.GRADER_TEMPLATE`, verbatim | [evaluate_run.py L17-L38](https://github.com/texttron/BrowseComp-Plus/blob/046949032b0328319cc9a02663a759ec601d9402/scripts_evaluation/evaluate_run.py#L17-L38) |
| judge parser | `reward.parse_judge`: `**correct:**`, then `**correct**:`, then `correct:` (likewise confidence, capped at 100) | [evaluate_run.py L60-L149](https://github.com/texttron/BrowseComp-Plus/blob/046949032b0328319cc9a02663a759ec601d9402/scripts_evaluation/evaluate_run.py#L60-L149) |
| judge call | top_p 0.8, top_k 20, 4096 tokens, `chat_template_kwargs: {"enable_thinking": false}`; model and temperature: CLM's (below) | [evaluate_run.py L393-L409](https://github.com/texttron/BrowseComp-Plus/blob/046949032b0328319cc9a02663a759ec601d9402/scripts_evaluation/evaluate_run.py#L393-L409), [L579-L583](https://github.com/texttron/BrowseComp-Plus/blob/046949032b0328319cc9a02663a759ec601d9402/scripts_evaluation/evaluate_run.py#L579-L583) |
| empty answer | incorrect without a judge call (`bcp`; `reward.grade` likewise) | [evaluate_run.py L523](https://github.com/texttron/BrowseComp-Plus/blob/046949032b0328319cc9a02663a759ec601d9402/scripts_evaluation/evaluate_run.py#L523) |
| recall | `recall` = evidence docs among every docid search returned in the episode (main and branches) | [evaluate_run.py L510-L514](https://github.com/texttron/BrowseComp-Plus/blob/046949032b0328319cc9a02663a759ec601d9402/scripts_evaluation/evaluate_run.py#L510-L514), [utils.py L6-L37](https://github.com/texttron/BrowseComp-Plus/blob/046949032b0328319cc9a02663a759ec601d9402/search_agent/utils.py#L6-L37) |

Still different:

- **Limits**: per-thread context growth, a branch cap and call caps (below); BCP runs an agent to its own limits.
- **`open` is kept and always on.** BCP's `get_document` is off by default
  ([mcp_server.py L48-L52](https://github.com/texttron/BrowseComp-Plus/blob/046949032b0328319cc9a02663a759ec601d9402/searcher/mcp_server.py#L48-L52))
  and returns the whole text in one call
  ([bm25_searcher.py L58-L69](https://github.com/texttron/BrowseComp-Plus/blob/046949032b0328319cc9a02663a759ec601d9402/searcher/searchers/bm25_searcher.py#L58-L69));
  our `open(docid, offset)` returns 32768-character pages (`OCTOPUS_SEARCH_OPEN_CHARS`, about CLM's 8,192 tokens at
  4 characters a token: a character estimate, not the policy tokenizer) with the next offset, so a long document does
  not fill the context at once.
- **Hit layout**: ours is text, `[i] docid=… | title | url` and the 512-token text; BCP's is a JSON list of
  `{docid, score, snippet}`. Same documents and text; we add title and url and leave out the score.
- **BM25 engine**: tantivy, BM25 k1=1.2 b=0.75 (fixed by tantivy), English stemming without stopwords, our own index
  over the corpus parquet, the query reduced to bare words. BCP uses pyserini's `LuceneSearcher` on its prebuilt
  Lucene index ([bm25_searcher.py L35-L47](https://github.com/texttron/BrowseComp-Plus/blob/046949032b0328319cc9a02663a759ec601d9402/searcher/searchers/bm25_searcher.py#L35-L47))
  with Anserini's defaults k1=0.9 b=0.4
  ([SearchCollection.java L278-L281](https://github.com/castorini/anserini/blob/c4c26663e0ae273acf9a7b2e91986e7e005286ec/src/main/java/io/anserini/search/SearchCollection.java#L278-L281)).
  Matching them needs pyserini (Java 21), which the hosts lack; rankings can differ.
- **Exact match first**: `reward.grade` (training, `dev_eval`) returns correct on a normalised exact match without a
  judge call; BCP always asks the judge. `bcp` is judge-only, as BCP. The match also needs the same numbers as
  written (sign and inner punctuation kept: `3.5` is not `35`, `-4` not `4`, `1:30` not `130`); a number written
  another way (`1,000` / `1000`) goes to the judge.
- **Ending without done**: only `done(answer)` is graded (main's; the control's and a linear run's main's likewise); a
  thread that ends any other way, a reply without a tool call too, has no answer: incorrect without a judge call
  (`reward.grade` "none"), in training and evaluation alike. A thread's (main, branch at any depth, control, read
  control) first reply without a tool call does not end it while another turn fits (its turn cap, `thread_budget`
  and the answer reserve, the serving window, the shared budget): it stays in the history and the runtime answers
  with one reminder, `You replied without a tool call. A thread's result counts only through its closing call: close
  with done now, or continue with a tool call.` (`core.REMIND_NOTE`, its closing tools from the thread's closing set);
  a second such reply ends the thread. The reminder is a runtime note (not trained, like a join note; the reply and
  the next one are trained), and with the rewind tool it is a step of its own (a rewind before it drops it; the
  thread stays reminded). `tree.json` counts them (`reminded`: main's; a branch's in its fork record; the control's
  and a read control's in theirs), the episode record as `reminded {main, branches}` (search v8: 7.4% of forked mains
  replied text after the join, y 0, and GRPO trained forking away). BCP grades the run's last message when it is text
  ([evaluate_run.py L516-L521](https://github.com/texttron/BrowseComp-Plus/blob/046949032b0328319cc9a02663a759ec601d9402/scripts_evaluation/evaluate_run.py#L516-L521)).
  A branch that ends without done hands its forking thread nothing: its join note line is only `stopped after N steps
  without done`.
- **Judge model and temperature**: Qwen3.5-27B at temperature 0 in training (`reward.grade`, `OCTOPUS_JUDGE_TEMPERATURE`)
  and evaluation (`dev_eval`, `bcp`), CLM's; BCP's is Qwen3-32B at 0.7. Served from a text-only checkpoint
  (`python -m octopus.rl.text_checkpoint`, the tool that makes the trainer's Qwen3.5-4B-text): SGLang 0.5.19
  serves the vision-language checkpoint through `qwen3_vl`, whose vision attention fails on the server's startup
  warmup (an image request) with the judge's server flags (no `--mm-attention-backend`).
- **Judge cache**: `OCTOPUS_JUDGE_CACHE` (training only: `search_dev_eval` and `bcp` unset it) keeps the first verdict
  per (the model the endpoint serves, temperature, question, answer, gold), so a repeated answer is not re-asked. The
  model is the SGLang server's `/model_info` `model_path` (`reward.judge_identity`; its `/v1/models` `root` is the
  served name), not the served name (`judge`): a judge swapped under the same name never reads the old one's verdicts.
  Only verdicts are cached.
- **Reply without a verdict**: at temperature 0 Qwen3.5-27B now and then (1 in 1221 judgments of the search v6
  pilot) writes a long reasoning that ends on `confidence: 100` alone. `reward.judge` then asks once more on the same
  conversation for the two lines only (`reward.JUDGE_FOLLOW_UP`; the same prompt again would give the same reply);
  without a verdict still, `reward.grade` grades the answer incorrect, `how` "judge_unparsed", counted in the episode
  record's `judge_unparsed` (also in `search_dev_eval` rows, and in its summary and the trainer's `[octopus] plan
  group` line when non-zero), and the episode is trained (it raised before, so AReaL dropped its whole group). `bcp`
  counts such a row among its `judge_failures`.
- **Recall scope**: the control thread's searches and opens are not counted; `open_recall` / `open_gold_recall` are
  ours (evidence / gold docs opened).

## CLM settings

The search environment follows CLM's zero-shot BrowseComp-Plus settings (user decision 10-24) as its defaults for
training and evaluation, except two:

| | ours (default) | CLM zero-shot | BCP official |
|---|---|---|---|
| hits per search | 10 (`agent.SEARCH_K`; 512-token hit text) | 10 | 5 |
| `open` per call | 32768 characters, about 8,192 tokens (paged by offset) | 8,192 tokens | off; the whole document |
| main's calls | 100 (`OCTOPUS_MAIN_MAX_CALLS`; a branch: 50 turns) | 100 | agent's own |
| policy sampling, evaluation | temperature 0.7, top_p 0.95 (`rl.EVAL_SAMPLING`, `search_dev_eval`) | 0.7, 0.95 | agent's own |
| policy sampling, training | temperature 1.0, top_p 1.0 (`gconfig`; the trainer's logprob recompute assumes it) | — | — |
| reply cap | 4,096 new tokens a call, training and evaluation (`rl.MAX_NEW_TOKENS`, whatever `gconfig.max_new_tokens`) | 4,096 | agent's own |
| judge | Qwen3.5-27B at temperature 0, training and evaluation | Qwen3.5-27B at 0 | Qwen3-32B at 0.7 |
| context cap | per-thread growth `OCTOPUS_THREAD_BUDGET` 32768 plus `OCTOPUS_ANSWER_RESERVE` 2048 (below): **differs from CLM** | CLM's own | none |
| thinking | off: **differs from CLM** | on | agent's own |

Budget at these sizes: a search result (10 hits of up to 512 tokens) is about 5,100 tokens and an `open` page about
8,000, so a thread's 32,768-token growth holds about 5 searches or 3 opens before its reserve. A reply's tool results
that would pass `OCTOPUS_THREAD_BUDGET` − `OCTOPUS_ANSWER_RESERVE` are clipped to the room left above the reserve
(`core.clip_results`: `[clipped: context nearly used up]`, then "Your context is nearly used up. Close now: call done
with what you have so far, or rewind to an earlier step to free context; any other call is refused.", the rewind
clause only with the rewind tool and a rewind left (`OCTOPUS_REWIND_MAX`); counted with the policy's tokenizer, `OCTOPUS_TOKENIZER`,
which a reserve requires, plus 40 tokens a message for its role and tag), and the thread is then in its reserve, so its
closing turn fits in the reserve. Caveat: the join note main gets after a fork is not clipped (its growth is counted
at main's next reply, as before).

The judge must be downloaded to each cluster and converted to its text-only checkpoint before a run
(`python -m octopus.rl.text_checkpoint --src models/Qwen3.5-27B --out models/Qwen3.5-27B-text`; `JUDGE_MODEL` in
`scripts/alpha/search.env`, also read by `scripts/beta/search.env`), and the retrieval service must run code that takes a request's `k` (an older service
ignores it and returns 5).

## Limits

Context-Folding-style ([2510.11967](https://arxiv.org/abs/2510.11967): a 32,768-token context per thread, up to 10
branches), set in `defaults.DEFAULTS` (dev_eval and RL; the env overrides each) except the context growth cap, which is
every thread's in coding runs too (`settings.thread_budget`). Coding runs keep the runtime's defaults when a knob
is unset: no branch cap, `OCTOPUS_FORK_DEPTH` 2,
`OCTOPUS_READ_TOOL` off, `OCTOPUS_BRANCH_MAX_TURNS` 100, main's calls uncapped.

| knob | search default | meaning |
|---|---|---|
| `OCTOPUS_SEARCH_BUDGET` | 0 | completion tokens of the whole tree; 0: no shared budget |
| `OCTOPUS_THREAD_BUDGET` | 32768 | the one default of both environments (`settings.thread_budget`): tokens a thread (main, a linear run's, every branch) may add to its context after its first request; the main-first control gets what main had left at the fork. The reply past it is dropped and the thread ends (`main_end` "context") |
| `OCTOPUS_ANSWER_RESERVE` | 2048 | the last R tokens of `OCTOPUS_THREAD_BUDGET` kept for the thread's `done` (`settings.answer_reserve`; the runtime default 0, off, is SWE's): once a reply leaves a thread less than R of its growth, every call but `done` and `rewind` (that reply's too) is refused with "Refused: your context is nearly used up (N tokens left). Close now: call done with what you have so far, or rewind to an earlier step to free context. Until you do, every other call is refused." (without the rewind tool, or once `OCTOPUS_REWIND_MAX` is spent: no ", or rewind ..."). Tool results that would pass the cap less R are clipped there and put the thread in its reserve ("CLM settings" above). The tag's "context left" stays the full cap's; a reply past the cap is still dropped ("context") |
| `OCTOPUS_MAX_BRANCHES` | 10 | branches made in one episode, every depth; a fork past it is refused with the number left |
| `OCTOPUS_FORK_DEPTH` | 1 | only main forks; D: forks nest D levels below main; 0: no depth limit (the runtime's unset or empty default is 2) |
| `OCTOPUS_CONTEXT_WINDOW` | 262144 | the serving context length (`settings.window_room`), counting a thread's mounts too (prompt_tokens leave them out): after each reply's calls a thread's growth cap is at most the window less its first prompt, its mounts and one reply (`OCTOPUS_MAX_TOKENS`); its first request's tag still shows the cap as given. A fork whose branches would start with less than `OCTOPUS_ANSWER_RESERVE` + 8192 (`tree.FORK_FLOOR`) of it is refused ("Forking is not available here: not enough context window is left at this depth; finish this task yourself."), and a read whose mount would leave less than the reserve ("X's steps A to B hold N tokens, more than your context window has left: read fewer steps. Nothing was read."), so deep threads end by context. Not counted: a join note (not clipped) and a branch's task message beyond the forking reply, which the 8192 covers |
| `OCTOPUS_READ_TOOL` | none | `read_thread` (`docs/cross-branch-read.md`): not in `defaults.DEFAULTS`, the env file sets it (`rl_run.search.env`: 1) as for code runs; dev_eval takes it from the run's `episode_settings.json` (or `--env`) |
| `OCTOPUS_MAIN_MAX_CALLS` / `OCTOPUS_BRANCH_MAX_TURNS` | 100 / 50 | main's tool calls / a branch's turns |
| `OCTOPUS_REWIND_TOOL` | none | `rewind(to_step, note)` ("Rewind" below): not in `defaults.DEFAULTS` (the runtime's default is off); the search training env file sets 1 (`scripts/alpha/search.env`, also read by `scripts/beta/search.env`), so its fork and linear runs and their evaluations (`episode_settings.json`) have it |
| `OCTOPUS_REWIND_MAX` | none | each thread's rewinds ("Rewind" below; `settings.rewind_max`; unset or 0: no cap); the search training env file sets 2 |

## Rewind

With `OCTOPUS_REWIND_TOOL=1` every search thread (main, a branch, the main-first control, a linear run's main) is
offered `rewind(to_step, note)` with search's own description (`agent.REWIND`): there is no workspace, so a rewind only
cuts the thread's history back to the end of step `to_step` (the step in each tool result's tag) and puts the note
`[runtime] Rewound to step N at your request: the steps after it were removed from your history.` + `Your note: ...`
in their place, a tool message tagged with what is left now (docs/rl-plan-episodes.md "Rewind tool").

- Context: calls stay spent, but the thread's context left grows back by the removed steps (growth counts from the
  thread's first prompt): the note's tag shows the context left at the step it went back to (all of it at the
  thread's start), only the note itself not yet counted. A reply that rewinds is never clipped at the reserve. A thread
  in its reserve (less than `OCTOPUS_ANSWER_RESERVE` left, or clipped) accepts `done` and `rewind` (its notes say
  "..., or rewind to an earlier step to free context"; the search v6 pilot's models called rewind there: main 158 of
  158 rewinds, branches 54 of 105, all refused before): after the rewind the thread leaves its reserve when the growth
  at the step it went back to is below `OCTOPUS_THREAD_BUDGET` − `OCTOPUS_ANSWER_RESERVE`, else stays in it.
- Cap: with `OCTOPUS_REWIND_MAX=N` each thread runs at most N rewinds (search-v6-1024's first steps, uncapped: 27% of
  episodes rewound, up to 26 times, 74% of all requests, no accuracy gain within a question group); the tool's
  description says "at most N times per thread", a further rewind is refused ("No rewinds left ...", nothing
  rewound), and the reserve notes drop the rewind clause once none are left. Refused rewinds do not count; a rewind
  does not give one back; a control goes on from the count its thread had at the fork.
- Forks: a reply whose fork starts cannot also call `done` or `rewind` (both refused and not counted as calls, the
  fork runs; the fork's description says so; a refused fork leaves them alone): the thread goes on from the forking
  reply, which its branches extend, so that reply is trained in the thread's own leaf. A fork that is the thread's last
  request anyway (its call cap, a failed next request) is trained in its first branch's leaf (`group.assign_threads`).
- Records: tree.json `rewinds` (main's) and each branch's `rewinds` keep the messages a rewind removed and its note.
  The gate's `searched` (`search_outcome`), bcp's recall and opened docs and read_audit count the removed calls too;
  `search_dev_eval` rows carry `rewinds` {calls, steps, note_chars} and its summary their means.

Each thread may add at most 32,768 tokens to its context; the inherited prefix is shared KV and does not count, nor do
the tokens `read_thread` mounts (prompt tokens leave out the mount's placeholders, in training and in eval alike), so
the context a thread attends to can pass its prefix plus 32,768 by what it mounted, including what the source had
mounted inside the range (a transitive read). This differs from Context-Folding, whose 32K is each thread's total
window: here a branch's window is its parent's prefix at the fork plus up to 32,768 of its own, plus its mounts. Every
tool result starts with the thread's name, its own growth left, calls left and step (`[main | context left: N | calls
left: K | step S]`; without `OCTOPUS_READ_TOOL` and the rewind tool, `[context left: N | calls left: K]`). A branch's
"calls left" is its turns left (`OCTOPUS_BRANCH_MAX_TURNS` counts replies), main's its tool calls left. The system prompt
says so in one text for main and its branches ("calls left (a branch: its turns left)"; `core.limits_note`, which the coding system prompt carries too), and states the branch limits
in effect: at most `OCTOPUS_MAX_BRANCHES` branches, that a branch cannot fork (`OCTOPUS_FORK_DEPTH` 1; D > 1: "forks
nest at most D levels below the main thread"; no limit: nothing), and
`OCTOPUS_BRANCH_MAX_TURNS` turns per branch (the control's and a linear run's prompt leave branches out). A branch starts
from its forking thread's whole history at the fork, as the control does.

Threads and their tools:

Protocol v6 (one tool list for main and its branches: search, open, done, fork; no `commit` / `abort`, a call of either
is answered as any unknown tool):

- **fork** takes one task or more (`SearchWorkspace.fork_min` 1; SWE keeps at least two). The system prompt leaves
  whether, when and how many to the model ("Fork when it helps; whether, when and how many is your choice."), within
  the limits above.
- **Main** ends with `done(answer)`, the graded answer. Every fork waits for its branches and returns their findings.
- **A branch** closes with the same `done(answer, evidence)`: `evidence` is optional (docids, short quotes), and a
  branch that found nothing says so in its answer. The join note shows each branch (by its thread name, e.g. `fork0/branch-1`) with `done: <answer>` and, given,
  `evidence: <evidence>` on the next line; tree.json's branch `summary` is that answer (`core.handed`). Under
  `OCTOPUS_READ_TOOL` it is offered `read_thread` (the roster of its siblings and every other thread it may read, at any
  depth, `CursorRuntime._others`, and the card) only where a read can succeed (`SearchWorkspace.read_when_readable`):
  in its task message when, at its fork, one of those threads already has a readable step (`CursorRuntime._readable`),
  e.g. the ended branches of an earlier fork; else in a note ("Other threads now (read_thread names): ...") before
  its first request once one has. The first fork's branches start together while main waits, so their task messages
  carry neither roster nor card, and a `read_thread` call before the offer is refused ("read_thread is not available
  in this branch. Nothing was read."; main gets the card with its first join note, as before, and again with its next
  one when a rewind removed it). A thread waiting for the fork (main, a forking branch) has its steps before it in the branch's own history,
  so no read of it could mount. Threads are named episode-unique (`SearchWorkspace.read_by_name`): the join note, a
  branch's roster, the refusals and `read_thread`'s `thread` use the thread name (`fork0/branch-1`, `main`), since every
  fork's branch ids restart at `branch-0`, and a branch's task message tells it its own ("You are fork1/branch-0, ...",
  roster "Other threads at this fork (read_thread names): ..."); a bare id is still read where one thread has it, and
  refused where several do ("branch-0 is ambiguous: fork0/branch-0, fork1/branch-0 have it; name one of them. Nothing
  was read."). Tool-result tags keep the bare id. Mounted
  tokens do not count toward a thread's `OCTOPUS_THREAD_BUDGET` growth (above; `test_search_read_thread_offered_where_a_read_can_succeed`).
- **The main-first control** (`OCTOPUS_PLAN_CONTROL`; dev_eval with the override `rl.plan_control=true`) restarts from main's forking point with
  the linear prompt and no fork tools (search, open, done), and without `read_thread`: it has no siblings to read. It is
  evaluation only: search training runs with `rl.plan_control=false`, so the policy is never trained on the control's
  prompt.

`read_thread` mounts need `OCTOPUS_CURSOR=1` and SGLang servers with the cursor and mount patches and the mount's server
config (`docs/cursor-protocol.md` "Mount", "Deployment"; RL: `CURSOR` in the env file, through the AReaL proxy;
dev_eval: `eval_serve.sh` with `CURSOR`, through the same proxy). Without
`OCTOPUS_CURSOR` (e.g. an agent sent straight to a plain SGLang chat endpoint) no reply keeps a cursor, so no step is
readable: every read is refused ("Nothing was read.", one tool call spent) and nothing is mounted. With cursors but a
server lacking the mount patch, the mounting request fails (the proxy's engine: `ignored the mount ...: it needs
patches/apply_sglang_mount.py`; SGLang without the mount config: HTTP 400) and that thread ends ("request failed").
Leave `OCTOPUS_READ_TOOL` unset (off) for such servers. Never set `OCTOPUS_READ_CONTROL` in search training: it adds read credit
(`read_adv`, a non-outcome term) and the search RL template is pure outcome. Search v5 adds branch credit, now `rl.branch_credit=gate` (v5 ran it as `search_gate`, a value since removed and refused; `scripts/beta/search_v5.sh`): `search_group_step` in `reward.py` applies search's rule, `octopus/rl/group.py` `search_gate` with `search_branch_ok` (since protocol v6: a branch that closed with `done`, a non-empty answer, and whose subtree (itself or its branches at any depth) made a search / open call after its fork keeps `A_main`, any other `min(A_main, 0)`); R stays 2y − 1.

## Sampling

Training samples the policy at temperature 1.0 (`grpo.yaml` `gconfig`; the AReaL proxy also defaults a request
without sampling fields to temperature 1.0, top_p 1.0, top_k 1e8, i.e. off). dev_eval runs training's episodes through
training's proxy (`octopus/rl/proxy.py`), with one difference: search_dev_eval's requests carry CLM's evaluation sampling,
temperature 0.7 and top_p 0.95 (`octopus/envs/search/rl.EVAL_SAMPLING`). Every search request (training and
evaluation) is capped at 4,096 new tokens (`rl.MAX_NEW_TOKENS`). Any other variant is a config override or `--env` on
the dev_eval command line, never a server default.

## Admission

Training's proxy workers and dev_eval (`octopus/rl/proxy.py`, the same AReaL engine) place each episode with
`octopus/serving/route.py` (`patches/areal-route-spread.patch`). An episode is one proxy session, one tree: all its
requests (main, branches, controls) stay on one server, as its cursors and read mounts need, and a new episode goes
where the router admits it. Follow-up requests of a running episode never wait.

With `OCTOPUS_ROUTE_KV` > 0 (`scripts/alpha/search.env`: 0.75, also read by `scripts/beta/search.env`) each router polls
every server's SGLang `GET /v1/loads?include=core` every 2 s (`token_usage`, the KV pool's used share,
`num_waiting_reqs` and `max_total_num_tokens`, the pool) and counts each server's active trees. An episode is an active
tree from its admission until its proxy session ends: AReaL's `/rl/end_session` (hooked by the patch, so an AReaL overlay
built before #351 must be rebuilt; `check_areal_route.py` and the launchers that set a tree budget check it) or the end of
dev_eval's `proxy.episode`. A tree none of whose requests started for an hour (AReaL's session timeout) counts as ended,
a backstop for a lost end. An episode that ends before its first request is admitted (still waiting, or not yet at the
router: a group's other episodes are cancelled when one fails) is never admitted, and that request fails. The router
remembers the last 4096 ended ids, and AReaL's `/rl/start_session` (also hooked) forgets one when it reuses it. Each router writes its own count per
server to a file in `/dev/shm/octopus-route-<uid>` (mode 0700, this user's) at each change and each poll. It reads every
other file there rewritten within the last 60 s at each poll and again before it admits an episode. So every router on
one host (a run's proxy workers, one per server, and an eval's) counts all their trees, and a dead router's trees drop
out. Routers match servers by address string: one server reached at its LAN address by one router and through a
tunnel's `127.0.0.1` port by another counts as two. Routers on other hosts are not counted. An eval on one host over
tunnels to servers that a trainer on another host also uses (for example search-v9-rest's node-138 eval queue beside a
node-67 trainer) sees only its own trees there, and only the KV guard keeps the two apart.

A new episode goes only to a server that

- holds fewer active trees than its slots, max(1, KV target x `max_total_num_tokens` / `OCTOPUS_ROUTE_TREE_TOKENS`),
  when that budget (tokens one tree may take, > 0) is set;
- is below the KV target and has no queued request (a guard);
- took no new episode from this router since its last poll (a poll cannot yet see what an admission adds);
- has fewer than `OCTOPUS_ROUTE_CAP` (> 0) of this router's requests in flight (training: `ROUTE_CAP` via
  `rl_run.sh`; evals: `search_eval.sh` exports 32).

Of those it draws one at random in proportion to its free slots. Without a budget it weights by free tokens below the
target, pool x (target - token_usage), so routers that cannot see each other's admissions still split episodes by
pool. Otherwise the episode waits for the next poll, a finishing request or an ending tree. A server whose poll fails
stays open (still one per poll) but is taken only when no readable server is open, then by the router's in-flight
count. After `OCTOPUS_ROUTE_WAIT` s (default 300) a waiting episode takes a server under the cap anyway, past the slots
and the target (the lowest KV when none is open): the wait spends its first request's client timeout
(`OCTOPUS_HTTP_TIMEOUT`, 900 s). Two routers that admit to one server at the same instant can put it one tree over its
slots.

Why slots. KV shows a tree only as it grows (its forks never wait), so KV alone let a start admit every waiting episode
within seconds (search-v9-rest, 10-27: 4090D servers at KV 0.9+ with 10-40 queued). #320's `OCTOPUS_ROUTE_GAP` (at
most one new episode per server per gap, per router) stopped that burst but also throttled the steady state. On
search-v9-rest (gap 540 s, 7 routers, 2239 episodes) an episode's first call took 133 s on average (P50 44 s, P90
414 s; 28 % of its wall time), while the 4090D servers ran a median of 3-5 requests, below their throughput knee
(129 tok/s per server at 4-7 running, 199 at 20-23). Their radix hit rate stayed 0.89-0.96 at any KV on 684K pools;
eviction cost only on a small pool (A6000, 270K: 0.85 to 0.62 at KV 0.8-1.0) and in the start burst. The slots bound
what fits at once, scaled by each server's pool, with no time gap, so the gap is gone.

Calibration (search-v9-rest, 2239 episodes, cursor + read + rewind, `FORK_DEPTH=0`; a tree's KV with each fork's shared
prefix counted once; an average of 7 threads per tree). Averaged over a tree's time from admission to end, the footprint
is P50 36K, P75 68K and P90 97K. Its peak is P50 65K, P75 161K and P90 228K, and the average is about half the peak. The
search-v9-rest scripts set 40000, near the time-averaged P50: at KV 0.5 that is 8 trees on a 4090D (684K), the
per-server load of the 10-28 run. 68000, the time-averaged P75, is the more conservative choice: 5 slots on a 4090D, 9 on
an A100 (1.29M) and 1 on an A6000 (270K). Peaks do not coincide across trees, and a server whose trees peak together trips the KV and queue guard.
Recalibrate from a run's log lines: lower the budget (more slots) while KV peaks stay below the target with no queue,
and raise it when queues appear.

All four (KV, TREE_TOKENS, CAP, WAIT) are host settings (`config.HOST_ENV`, not in `episode_settings.json`): a resume or
an evaluation may set or change them. An eval that shares training's servers on its host counts training's trees, and a
lower KV target gives it fewer slots, so it admits only while a server holds fewer trees than those. That does not
reserve slots for training: between two of training's batches an eval can take slots that training's next episodes then
wait for, and an eval episode past the wait bound ignores the slots and the target. Every new episode prints `[octopus] route:
new session <id> -> <server> after <s> s; trees <n>/<slots>, (KV, queue) (...)`. Every 30 polls (1 min) each router
prints `(trees, slots, KV, queue) per server`, the peak KV and queue so far, and how many episodes the wait bound
admitted. `token_usage` leaves out the evictable prefix cache, which admitted episodes take back after their tool calls.
`EPISODES_PER_SERVER` (training, groups per server) and `EVAL_PARALLEL` (`search_eval.sh`, default 32; EVAL_PARALLEL x
servers episodes in all) stay upper bounds. Without a tree budget they are also the only bound on episodes admitted
while still small (one starts at ~3K tokens and a forking one grows to ~100K). v7's fixed 16 groups per server held KV
at 0.95-0.98 with queues of up to 70 and 900 s timeouts; evals at a fixed 16 episodes per server sat at KV ~0.2, and a
fixed 55 overloaded fork-heavy BCP. Each poll is one access-log line per server per router.
