# Cross-branch read: the training-side forward

Branch B "reads" sibling A by mounting the K/V of a token range of A into B's attention layers (all of them, or the
subset `OCTOPUS_MOUNT_LAYERS` names); recurrent layers keep recurring over B's own tokens. On Qwen3.5-4B, the model of
the runs here, that is 8 full-attention layers and 24 Gated DeltaNet (GDN) layers. The same holds at inference for a
dense model and a Mamba-2 hybrid (docs/cursor-protocol.md "Mount"); the trainer's forward is checked against
SGLang's rollout log-probs on Qwen3.5-4B and, under a layer subset, against transformers on a dense model, and has
Mamba-2 hooks (`octopus/rl/model_hooks.py`) that no test compares against a reference. The offline probe
(`docs/kv-read-probe.md`, #129, run 2) chose the position rule K2. This note is how the trainer's exact
shared-prefix forward (`octopus/rl/prefix_tree.py`) computes the same thing.

## Formalism

A tree node u is a cursor `(p(u), M(u), s(u))`:

- `p(u)`: the GDN parent path, the node's ancestors. The recurrence starts from the parent's final state and the causal
  convolution sees the last 3 tokens of the parent path, as before.
- `M(u)`: the tokens visible to full attention: the ancestor path plus every segment mounted at u's start or at an
  ancestor's start. Mounted keys are those the source computed in its own sequence, at its own positions. A segment
  `[s, e)` of a source that itself mounted something at a point strictly inside it (`s < at < e`) brings that mount's
  keys too (transitively): on the server those rows sit at the source's placeholders inside the range.
- `s(u)`: the position of u's first token. Without a mount it is the parent's next position. At a mount (K2) it is
  `max(parent's next position, last position of each mounted segment + 1)`; descendants continue from there. Each tree
  token has one position id, so mounted keys are neither copied nor re-rotated.

## Mount input format

`build_tree(flat, cu, mounts)` and `tree_inputs(input_ids, cu_seq_lens, mounts)` take

```
mounts: {i: [(at, src, src_start, src_end), ...]}
```

Sequence i is the i-th sequence in `cu` order. From its token `at` on (tokens `at`, `at + 1`, ...), sequence i's
full-attention queries also see tokens `[src_start, src_end)` of sequence `src`, counted in `src`'s own tokens. Several
mounts at one `at` are allowed, and so are mounts at several points of a sequence; they accumulate. A source may itself
be a reader (chains, no cycles). The source tokens have to be in the same micro-batch. Their logprobs and loss stay with the source sequence, so they are trained once,
where they belong. `mounts=None` (or `{}`) gives the plain prefix tree, unchanged.

The AReaL export produces this per micro-batch from the rollout's reads: "Export and allocation" below.

## Implementation

- **Node split.** A mount point always starts a node. Sequences share a node only when their tokens and their mounts so
  far are the same, so a sequence with a mount and its unmounted twin diverge at the mount point.
- **Attention.** A node's gathered key list is its mounted segments (inherited ones first, each with what its source mounted inside it) followed by its path. Its
  own tokens are last, so FlashAttention's bottom-right causal alignment still holds, and every query of the node sees
  all mounted keys.
- **Positions.** K2 as above. A mount's source can come later in pre-order, so a node's first position is resolved
  recursively from its parent and its sources.
- **Gradients.** The mount is part of B's computation graph, so B's loss sends gradients into A's mounted tokens
  through their keys and values, and from there into A's earlier tokens. This follows the exact-forward principle.
- **GDN, convolution, LM head.** These follow the parent chain and are unchanged.

## Evidence

`python -m octopus.tools.mount_check --model Qwen3.5-4B-text --probes PROBES.jsonl --n 40 --backward 2` compares the
tree with kv_read_probe's K2 cache surgery on real #129 mount points. It also compares the tree without the mount
against a plain forward, and checks the backward pass. Results are in the PR.

Several segments and transitive reads (`--points mount-points.json.gz --multi 4`, `kv_read_probe.multi_cases`: per
pair of the 8 shortest #129 mount points B mounts A's segment ("one"), A's and X's ("two"), or A's segment where A
has read X's segment in its middle ("transitive"); the oracle is `Probe.k2`, cache surgery with any number of segments,
transitively; 395 scored tokens of the probe question and answer per kind; node-136, Qwen3.5-4B-text, bf16): mean
|Δ logprob| one 0.079, two 0.054, transitive 0.051 (argmax 0.982 / 0.982 / 0.985); the tree without a mount against a
plain forward on the same tokens, the kernel floor, 0.038 (0.992). These question tokens are noisier than B's replies
(#131: 0.0066), and two and transitive stay at the single mount's level. Backward on the shortest transitive case
(27.8k flat tokens): finite gradients, |grad| on X's tokens that A mounted 1.92, and exactly 0 once A's read is
dropped, so the loss reaches X only through A's segment.

## The read tool (slice 3a: rollout path)

`OCTOPUS_READ_TOOL=1` (needs `OCTOPUS_CURSOR=1`, the AReaL proxy and SGLang servers with the mount patch and its
config, `docs/cursor-protocol.md` "Mount"). Eval runs the same path: dev_eval serves training's AReaL proxy in its
own process (`octopus/rl/proxy.py`), so its mounts reach `/generate` as training's do. Until then (#191) eval sent
straight to SGLang's chat endpoint, which dropped the mount silently: every held-out read before it ran unmounted
(`results/e14-read/report.md`, caveat). Unset: request bodies, tool lists and tree.json are byte-identical to before.

**Model-visible API.** Every thread but the controls gets (in no request's tool list: its definition is appended as a
card where a read can succeed, "Append-only availability" below):

    read_thread(thread, from_step, to_step)
    "Read another thread of this episode, never your own (a branch), without copying
    its text: the readable steps among from_step to to_step are mounted into your attention from your next reply on
    and stay there. A step is a reply and the tool results after it, numbered as the step N in that thread's tool
    result tags; a branch's steps continue the numbering of the thread it forked from. A step is readable once that
    thread's next reply is back or it has ended, unless your own history holds it
    already; without from_step and to_step, its latest
    readable steps are mounted. One read_thread per reply, not with fork or rewind. For files use view."
    thread: "The other thread: a branch id such as branch-1." (SWE; search names threads, docs/search-env.md);
    from_step: "The first step
    to read (omitted: from its first step)."; to_step: "The last step to read, from_step or later (omitted: to its
    latest readable step)." (required: thread)

The name is `read_thread`, not `read`: in the first rollout with `read`, 14 of the policy's 16 calls were file reads
(`{"path": ...}`). Tool results carry the step tags of the rewind tool (`| step N]` at the end of the tag), each led
by the name of the thread whose history holds it, as `thread` takes it: `[main | context left: X | step 3]`,
`[branch-1 | context left: X | calls left: C | step 5]` (runs before the shared budget's removal had `token budget
left: B | ` before `context left`). Branches were told their name in their task message but main never was:
in bundle-e13-read-10182126 (steps 1-12) 48 of 55 self-read loops were main's, 46 of 55 starting on the thread's first
reply. Results a branch inherited keep main's name; the forking reply's own results, sent first by the branch, get the
branch's (each branch has its own copy of them); a nested control's are its branch's, main's control's main's. The
result names what was mounted, never its text: `Mounted branch-0's steps 3 to 4 (191 tokens; branch-0 has
readable steps 3 to 4): from your next reply on you attend to them; their full text is not copied here.` (the source's
prompt tokens between the two replies that bound the range, plus what the source's replies inside the range mounted
(a transitive read; prompt_tokens leave a mount's placeholders out) = M). Refusals end with `Nothing was read.` and mount
nothing. `<others>` is `Other threads: branch-0 (readable steps 3 to 4), branch-1 (no readable step yet).`, or
`There is no other thread yet: fork first.` (`There is no other readable thread.` when only waiting threads remain).
An ended thread is listed `(no readable step)`, without `yet`. A thread that waits for the
reader's fork (its forking threads) is left out of `<others>`. The bare branch ids in this section's texts are the
code environment's; search addresses threads by their names (`fork0/branch-1`), refuses an ambiguous bare id, and
refuses `read_thread` in a branch it was not offered to ("read_thread is not available in this branch. Nothing was
read."): docs/search-env.md "Threads and their tools".

- `X waits for your fork to close: none of its steps can be read. <others>` (X is such a waiting thread)
- `read_thread needs thread (another thread: a branch id) and, if given, the integers
  from_step <= to_step (for files use view).`
- `read_thread cannot share a reply with fork or rewind: call it in a reply of its own.` (the forked branches or the
  rewound history would not hold the read)
- `No thread X in this episode. <others>` A branch id that several forks gave names the latest.
- `You are X: read_thread reads another thread, never your own. <others>`
- `X has readable steps A to B (a step is readable once X's next reply is back).` / `X has no readable step yet (...)`:
  the source's own steps (after the step it started from), each once the source's reply after it has come back, and an
  ended source's last step (its reply, without the tool results no request carried). A step the reader holds (mounted
  already, transitively, also through its history's replies) is never mounted again: two mounts of one range would
  stack the same keys at the same positions (double softmax weight, no text equivalent). A range mounts from its first
  step not held up to the step before the next held one (the latest-steps read: the first such run of its 16); a range
  wholly held reads `X's step A is in your own history already (...)`. Ceiling: a step can still come twice through a
  third thread (the reader mounts S directly, then T's reply that itself mounted the same S steps). This range is the one every text names (`<others>`, success, refusals).
  An ended source, which never replies again, gets `X has readable steps A to B: it has ended and gets no more.` / `X has no
  readable step: ...`.
- `X's step A is in your own history already (X has readable steps A' to B').` (steps A.. the reader mounted already;
  A' is the first step after them)
- `X's steps A to B cannot be mounted: a reply at their edge was rewritten.` (a reply kept without its cursor)
- `One read_thread per reply: the first one of this reply is mounted with your next reply. Nothing else was read.`
- `X's steps A to B hold N tokens, more than your context window has left: read fewer steps. Nothing was read.` (the
  reader's context with its mounts and this one would leave less than the answer reserve of the serving window,
  `settings.window_room`)

**Informed reads.** In the live run no branch task message named a sibling (653/653), readable steps were learned
only from refusals, and forced mounts mostly hit 1-step ranges. Under the flag: a branch's task message ends with
`Other threads at this fork (read_thread ids): branch-1.1 (task: <its first line, 80 chars>).`
(each sibling with its task's first line, then every other thread the runtime lets it read: `CursorRuntime._others`,
any depth, e.g. an earlier fork's branches, an uncle or a cousin; facts only; main and a nested fork's forking
branches are left out: they wait for the fork, their steps before it are the branch's own history, so every read of
them was refused). A branch not offered `read_thread` at its fork (search: `read_when_readable`, nothing readable yet)
is offered it before its first request once one of those threads has a readable step: a note
`Other threads now (read_thread names): ...` and the card; its `read_thread` is refused until then.

**Mutual reads.** A reads a range of B that holds B's mount of A: A then also attends to its own steps that B mounted
(the transitive read above), a duplicate of keys A already has. Accepted as is: the served episode and the trained one
see the same mounts (train = serve), and the "held" rule stops only a direct second mount. from_step and
to_step are optional (required: thread; the description adds "without from_step and to_step, its latest readable steps
are mounted"): both omitted mounts the source's last readable steps, at most 16, through the same checks and
`_mount_range` as a range (mount and read_call alike); a read of the source's inside them is part of the range
(transitive: the reader also attends to what the source read). A mount's result then lists each
mounted step on a line of its own, `step N: tool(key arg) -> first line of its result` (path for view/edit/write,
command for shell, else the first argument; first lines, each cut to 60 chars; a step's calls joined by `; `), from the source's
history after the result's tag.

**Readable part of a range.** A reader sees no other thread's step tags, and a branch's steps continue its
parent's numbering (a branch forked at main's step 40 starts at 41), so a range was a guess: in bundle-e14-read-10191636
(steps 1-31) 103 of 1382 ranges mounted (7%), against 59% of the 634 reads without one; 186 ranges started at step 0.
A range is now cut to its readable part, `max(from_step, A)..min(to_step, B)` for readable steps A..B, an omitted end
open (from_step alone: to B; to_step alone: from A), through the same `_mount_range` checks. Refused only when none of
it is readable, with the refusal of what it covers: the source's own steps before A (`X's step a is in your own
history already` / `... was rewritten`, a = the first of them in the range), else `X has readable steps A to B` / `X
has no readable step yet`. Replayed on that run from the readable range each result named: at most 442 of the 1382
ranges would mount (339 refused ones held readable steps; a range holding a read of the source's or a rewritten edge
is still refused and is not replayed), 214 one-ended ones are no longer malformed; 467 stay refused (233 wholly in the
reader's own history, 234 wholly past the source's latest readable step), 93 found no readable step and 166 keep
another refusal (148 a refusal this rule leaves as it was, e.g. the reader itself or a read of the source's in the range;
13 still malformed; 5 a second read in a reply whose first now mounts).

**Append-only availability.** In e13-read/e14, 5,790 read_thread calls were main reading itself before any fork (99.97%
before the first fork, median at main's reply 1): the tool sat in the shared system-prompt tool list from the start.
A tool list that changes mid-episode would break the shared prefix (cursor continuation, branches sharing main's KV,
AReaL's concat), as would a mid-episode user message with thinking on (notes then go as tool results:
docs/cursor-protocol.md, "Thinking on"), so under the flag READ_TOOL is in no request's tool list; its definition is appended, as
`read_thread is available from here on:` and the paragraph of the policy's chat template that holds the tool, as the
system prompt shows a declared tool (the servers' tool shape with `strict: false`; Qwen3.5: its JSON in
`<tools>`/`</tools>` lines; `core.read_card`, from the tokenizer dir `OCTOPUS_TOKENIZER`), where a read can succeed: after the roster line of every branch's task message (search: only where a thread it may read
already has a readable step, `read_when_readable`, else in the note that offers it later), and at the end of main's
result of its first fork (the join note; later forks do not repeat it, but the next fork's does when a rewind
removed it from main's history; main's no-fork
control, which cannot read, gets its history without it). The call path needs no schema: neither qwen3_coder parser
(AReaL's XML parser, SGLang 0.5.10.post1's detector) filters by tool name, arguments arrive as strings (which
step_number accepts), and the agent dispatches by name.

read_thread is not a context-rewrite call (core `LINEAR_DROP` minus commit/abort and read_thread).
- `read_thread is not available to the control.` (a nested control; main's control does not get the tool,
  `LINEAR_DROP`)

**Flow.** Steps A..B of thread X run from X's reply of step A to X's reply after step B (each thread publishes its
step marks, `start["marks"]`); when X has ended and B is its last step, through X's reply of step B (`"ended": true`
in the mount, "to" that reply). `CursorRuntime._read` (`octopus/runtime/tree.py`, shared by the SWE and Search agents) validates and puts `{"thread": X, "from": <reply A's id>, "to":
<the next reply's id>}` on the reader's thread; its next request carries it as `"mount"` beside its cursor (and in
its requests.jsonl row). The proxy (`patches/areal-cursor.patch`, `_read_range`) checks that both are finished replies
of the session with "from" an ancestor of "to", and takes the range of "to"'s tokens from the start of "from"'s
generation prompt to that of "to"'s (`delta_render.turn_start`: the template's generation prompt's first token): the assistant turn of step A through the last message before X's next
turn, as #129's probe mounted exchanges; with `"ended"` to the end of "to"'s tokens (input + output, as its SGLang
cursor holds them). One function, `delta_render.read_range`, is this rule for the proxy and for SGLang chat
(`kv_mount.prepare_chat`); the trainer replays the proxy's recorded range. The last token of a finished reply has no KV
on the server: the mount's segment is rebuilt by the usual prefill-only rebuild (`kv_mount.generate`). The engine sends it from the reader's parent cursor with `mount` on "to"'s
SGLang cursor (one server per episode tree, slice 5; another server is an error). The mount stays part of the reader:
later requests from its cursors keep attending to it (its replies record `placeholders`, which the engine's
prompt_tokens check and 404 rule use). A rewind of the reader past its read continues from an older cursor without it.
A reply that stopped on `<|endoftext|>` cannot be hopped from (concat puts eos there), so a read in such a reply fails
the reader's next request instead of dropping the mount; slices 6-7 saw no such stop in 16k requests.

**Memory pressure (slice 6).** At the end of
a mount request the copy-based server copied the mounted rows into fresh pages so the reply's cursor was an ordinary cached
path. On main, a copy without room cached nothing past the reader's own tokens, and
the next request from that reply found placeholders to prefill: 404 `the KV of a mounted read is evicted`; the same 404
followed whenever that path was evicted later, or A's mounted range held a mount of its own and was gone; a mount
request still missing B's or A's KV after three prefills was a 503. The engine never resends all ids for a mounted
thread, so the rollout thread ended. Before slice 6 this was seen once on node-94 (#132's continuation test at
`--mem-fraction-static 0.7`).

Slice 6 measured it on node-94 (one 4090D per server: sglang-4090d + the lora/cursor/mount overlay, Qwen3.5-4B + LoRA
`adapter-v0-4b-text-r64` with rl_run.sh's flags, `--mamba-radix-cache-strategy no_buffer --disable-overlap-schedule`).
Load on each server, for 100 min: `octopus/tools/rollout_load.py`, 64 e11 bundles from `bundle-e11-tasks.txt` in one
fixed order, 8 episodes at a time (e13's episode flags, cursor mode, read_thread on; contexts reached 95-141k tokens),
plus `octopus/tools/mount_load.py` with 3 thread pairs (forced reads straight to `/generate`: roots of 8-40k tokens, M
lognormal with the slice-7 audit's median 949 and p90 9,076, capped by what the sibling has; measured median 280-660, p90
2.7-4.8k). 404/503 count requests whose last of 3 attempts (the engine's `request_retries`) failed:

| server config | KV tokens / GDN slots / max running | code | mounts | requests from mounted cursors | 404 | 503 | other errors | mount-driver latency p99 / max | requests > 900 s |
|---|---|---|---|---|---|---|---|---|---|
| (i) e13: 0.88, `--mamba-full-memory-ratio 0.45` | 705,976 / 201 / 67 | main | 252 | 650 | **145** | 0 | 0 | 26 s / 82 s | 0 |
| (ii) 0.7, ratio 0.45 | 515,347 / 146 / 48 | main | 81 | 150 | **43** | **14** | 0 | 157 s / 253 s | 0 |
| (i) | | this PR | 199 | 943 | 0 | 0 | 0 | 40 s / 50 s | 0 |
| (ii) | | this PR | 89 | 370 | 0 | 0 | 0 | 230 s / 1,064 s | **5** |

On main, 80 of 191 end-of-request copies at (i) failed, every one whose free slots were fewer than M (evictable tokens
were 100k+): result processing holds frees in a free group, so the copy's eviction gave it nothing back. A first run with
smaller mounts (6 pairs, M median 280-350) gave the same picture: (i) 536 mounts, 1,307 continuations, 270 404 + 8 503;
(ii) 178, 277, 68 + 59. On main at (ii) one of the policy's own reads ended the same way (the engine's `... or a cursor
it mounts is gone ...: a send of all 14005 input ids would drop the read`, HTTP 500 to the agent). Otherwise no
traceback, OOM or 4xx/5xx at SGLang or the proxy in any arm, beyond the 400s and timeouts noted below.

The fix (`octopus/serving/kv_mount.py`, `patches/apply_sglang_mount.py`, docs/cursor-protocol.md "Mount"): each cursor
breakpoint keeps the mount that made it, so any missing part of a mounted sequence is rebuilt exactly (B's and A's
tokens prefilled together, a mounted continuation sent again carrying its mount); there is no 404 for a mounted read and
no 503 any more. The copy releases the free group around its eviction (0 failed copies in 384 at (i)), and admission's
budget check sees A's path locked. Exactness: rebuilt rows vs resident ones after `/flush_cache`, 74 teacher-forced
tokens each: continuation 0.0075, a second mount from the mount reply 0.0033-0.0036, a mount of a source holding a mount
0.0065 mean |d logprob| (same argmax, 3 runs), the prefill-vs-decode numerics of any re-prefilled cursor.

The fixed server at (ii) still waits: near a full pool a rebuilt part is evicted again before the retry is matched. One
17-minute window (0.96-0.98 KV usage, 23 running) held 5 driver requests 958-1,064 s, and 21 episode requests (plain,
from the same server) hit the agent's 900 s timeout, failing 8 episodes. Rebuilding B and A one after the other was
worse (at (i): 2 requests > 900 s; at (ii): 10, and 43 episode timeouts); locking rebuilt paths until the retry was
admitted removed the > 900 s requests but made 24 requests at (i) wait 2-5 min (max 302 s), so it is not in the PR. In
a 150k-token pool with 150k-token threads (Qwen3.5-4B-text, mount driver only), main answered 24-27 404s per 8 minutes
and the fix made requests wait for minutes instead. An earlier variant (one-by-one rebuilds, no admission lock, an extra
debugging eviction in the copy) crashed that small server twice with `Out of memory` in `alloc_for_extend`; main (3
runs) and the variants with the admission lock (4 runs, one of them this PR) did not.

Rollout speed (the same load without the mount driver, 100 min, both servers at 0.88 and ratio 0.45): e13's flags
(extra_buffer, overlap on: 706,616 KV tokens, 201 GDN slots, max running 40) vs the mount flags (no_buffer, overlap
off): reply latency median 3.16 vs 3.90 s, p90 14.7 vs 17.6 s, p99 41.8 vs 48.0 s; 1.92M vs 1.77M generated tokens
(-7.7%); decode throughput 316 vs 296 tokens/s (-6.5%). So slice 4's arms must use the same server flags.

The untrained policy now makes valid reads (with the #137 texts): 1-4 per 100 min on each mount-capable server (all
200), and 17 on the e13-flag server, each a 400 there (`mount needs --mamba-radix-cache-strategy no_buffer`, 4 failed
episodes).

Adapter switches (review of #140): RL sends every request with the current adapter version (`rq1-v<N>`) and the
radix key holds the adapter, so after each weight update every thread misses all its KV, and a mounted thread rebuilds
B and A. `rollout_load.py ... SWITCH ADAPTER KEEP` loads the same weights as the next version every SWITCH seconds
(LORA_KEEP 10 loaded), and the episodes and `mount_load.py @<OUT_JSON>.lora` send the newest name; config (i), 100 min,
same load as above (contexts up to 99-125k). The weight update's pause (in-flight requests aborted and sent again) is
not mimicked.

| arm | switches | mounts / requests from mounted cursors | errors | mount driver p50 / p99 / max | episode replies p50 / p99 / max | > 300 s (driver, episodes) | 900 s timeouts | peak KV usage, minutes >= 0.95 |
|---|---|---|---|---|---|---|---|---|
| this PR, reads | every 10 min (9) | 219 / 1,008 | 0 | 13 / 71 / 97 s | 4.7 / 73 / 183 s | 0, 0 | 0 | 0.97, 4 |
| this PR, reads | every 5 min (19) | 181 / 836 | 0 (3 unfinished at shutdown) | 14 / 97 / >= 639 s | 6.0 / 100 / 489 s | 6 (3 done 497-576 s, 3 unfinished after 442-639 s), 6 | 0 | 0.99, 20 |
| main, no reads (`OCTOPUS_READ_TOOL=0`, no driver) | every 10 min (9) | - | 0 | - | 4.3 / 56 / 161 s | -, 0 | 0 | 0.92, 0 |

Every request in all three arms was HTTP 200 (8,135, 6,677 and 6,465 at SGLang; 6,615, 5,379 and 6,464 episode
requests), except 3 mount-driver requests of the 5-minute arm still waiting (442, 548 and 639 s) when the server was
stopped (recorded as connection errors); 118 and 309 mount rebuilds (evicted-read aborts) recovered; no traceback or
OOM; 0 locked tokens once the 10-minute arms were idle. Within 3 minutes after a switch the driver's p99 was 87 s (10 min) and 97 s (5 min).

**Record for slice 3b.** The reader's interaction keeps `metadata["mount"] = {"thread": X, "source": <"to"'s
interaction id>, "start": s, "end": e, "at": a}`: `[s, e)` in the source interaction's concat tokens (its
`input_tokens`), `a` = the index in the reader's tokens of the first token its server computed with the mount (its
parent's input + output length). In `tree_inputs`' format that is `mounts[i] = [(a, j, s, e)]` with i the reader's
sequence and j a sequence holding the source interaction's tokens. The reader's tokens (`input_tokens`, export) are
its real tokens: the placeholders exist only on the server.

**Evidence.**

- `tests/test_read_tool.py` (local): a scripted bundle episode in cursor mode: the read reply's next request alone
  carries the mount (ids of the source's replies), its requests.jsonl row too, each refusal's text, step tags; the
  direct checks of a range holding the source's own read, a rewritten edge and the control. Flag off: 12 scripted
  episodes (`tests/test_cursor_client.py`'s trees in full and cursor mode, with and without the rewind tool, and an
  async-fork tree) dump every request body, tree.json (and for the async tree requests.jsonl) byte-identical to main;
  the async tree has one timing-dependent budget number that also varies between runs of main.
- `tests/test_areal_cursor_hop.py` (node-134 AReaL venv + the overlay, CPU): the engine sends the mount from the
  parent's cursor and counts M placeholders in prompt_tokens on the mount request and on later requests from its
  cursors; a mount of a reply that holds a mount is shifted by its placeholders; a server without the mount patch fails
  loudly; a 404 on a mount request or on a mounted cursor, a mounted thread that cannot hop, and a source on another
  server are errors, never a resend of all ids; a mount request aborted before its first token goes again with its
  mount. Through the proxy (Qwen3.5 tokenizer): the range decodes to `<|im_start|>assistant ... <|im_end|>\n` with
  exactly A's replies 1-2 and the messages after them, the record as above, no placeholder in B's tokens or the
  exported leaves.
- Forced live episode (`test_live_forced_read_episode`; 2026-10-18 on node-94, GPU 6 for SGLang 0.5.19 (sglang-4090d)
  + the lora/cursor/mount overlay, Qwen3.5-4B-text, radix cache, no_buffer, overlap off, `--mem-fraction-static 0.85`;
  GPU 5 for the offline forward): the Harbor agent on the fake container through the real proxy and engine, every
  request greedy (`temperature: 0`; the agent sends none and the proxy's default is 1.0). Replies are forced as the
  scripted call's canonical tokens in the input plus the model's own next token (eos), so each is a server reply with
  its cursor. main forks; branch-0's two calls find a key; branch-1 asks for main's step 1 (`main's step 1 is in your
  own history already (main has no readable step yet).`) and branch-0's step 3, repeats the read until it mounts
  (`Mounted branch-0's steps 3 to 3 (95 tokens; branch-0 has readable steps 3 to 3)`) and replies freely; after the
  join main calls `read_thread(branch-0, 3, 4)` (M = 186) and replies freely. 12 `/generate` requests; each free
  reply's body carries its `mount` on branch-0's cursor from the reader's parent cursor; both records; the key inside
  both ranges; the next request from each mounted reply passes the prompt_tokens check. K2 both ways: branch-1 behind
  (continues at its own 3,717 > mount end 3,527), main ahead (3,618 instead of 3,489). The free replies (85 tokens) vs
  the offline K2 computation (transformers + flash-attn 2 + fla cache surgery, `kv_read_probe.Probe`): mean |d
  logprob| **0.0009** (0.0019 and 0.0003; max 0.023), argmax agreement **1.0**; 0 locked tokens; 3 runs, identical.
  (An earlier version sampled the free replies at temperature 1.0, so its argmax agreement against the sampled tokens,
  0.919, measured nothing; its log-probs on the same tokens agreed, 0.0090-0.0139.)
- #132's 37-point equivalence on the 4090D build (`tests/test_sglang_mount.py -k live`, node-94 GPU 6, 0.85): mean |d|
  0.0201, argmax 0.987, greedy-16 0.957, kernel baseline 0.050, eviction 0.0, continuation 0.0210, 0 locked: the same
  numbers as on node-134.
- Rollout only, real policy (no trainer, no optimizer step; 2026-10-18, node-134 GPU 7): the RL rollout server (Qwen3.5-4B
  + LoRA `adapter-v0-4b-text-r64` as rq1-v0, rl_run.sh's flags + no_buffer, overlap off, the overlay), the patched
  proxy app in-process on RemoteSGLangEngine, and `HarborEpisodeAgent` bundle episodes as RL runs them (e13's episode
  flags with OCTOPUS_MAIN_MAX_CALLS=40 and the since-removed OCTOPUS_CALLS_PER_TASK=10 (that run's value, not a
  current setting), cursor mode, the no-fork control on), 2 e11 bundles x
  4. With the tool named `read`: 8 episodes, 647 requests, 16 read calls (14 with a file `path`), 0 mounts. Renamed
  `read_thread`: 8 episodes, 613 requests (568 offering the tool; the controls' 45 do not), **107 read_thread calls,
  all naming main**: 103 by main itself (`main is this thread`, 90 of them one episode's loop of `read_thread(main, 0,
  0)`), 2 by branches for main's steps before their fork (`in your own history`), 1 malformed (this breakdown sums to 106; where the 107th call falls was not recorded); 0 mounts. Both runs: 0
  errors (every request HTTP 200, no traceback in the proxy log, no 4xx/5xx or traceback at SGLang, no infra failure).
  The untrained policy did not know which thread it was; the tool texts now say "never your own" and refusals name
  the reader and the other threads' readable steps (review of #134). Rerun with those texts (2026-10-18, node-94 GPU 7,
  sglang-4090d + the overlay, same flags and bundles): 8 episodes, 829 requests (686 offering the tool), **29
  read_thread calls** (counted as before, over tree.json's tool results): 25 self-reads (`You are main ...`, most
  before any fork: `There is no other thread yet: fork first.`), 4 own-history refusals (branches asking for main's
  pre-fork steps), 0 other refusals, 0 malformed, 0 valid mounts; 0 errors (every request 200, no traceback, no SGLang
  4xx/5xx). A valid read is exercised by the forced episode above.

## Export and allocation (slice 3b: training path)

Training rebuilds the rollout's DAG from the proxy's records. Without `OCTOPUS_CURSOR` (no mount possible) nothing
changes: batches, allocation and tensors are byte-identical. A cursor run's rollout workers export every row's reads
(all zeros without one: no mounts, the same tree), never asking `OCTOPUS_READ_TOOL`, the episode's own setting.

- **Proxy export** (`patches/areal-cursor.patch`, `SessionData.export_interactions` → `prefix_tree.record_reads`). The
  proxy process does not import our workflow module, and the workflow gets the leaves back as tensors and metadata only,
  so the translation runs in the export. Each concat leaf's chain is walked and every reader interaction on it
  (`metadata["mount"]`) becomes `[at, source leaf, start, end]` in the leaf's `metadata["octopus_reads"]`. `at` carries
  over unchanged, because the leaf's tokens are its chain's tokens at the same positions. A reader leaf that continues
  after the mounted request therefore keeps the read, and every later token is computed with it. The source leaf is
  the first exported leaf below the source interaction ("to"). Every leaf below it has the same tokens, reads and K2
  positions up to it. A leaf that only shares the source's token prefix but sits on another chain could hold different
  reads. A session without reads gets no metadata. Runs with the read tool need the AReaL overlay rebuilt from this
  patch (`docs/cursor-protocol.md` "Deployment").
- **Workflow tensors** (`workflow._to_tensor_dict_with_adv` → `prefix_tree.read_tensors`, flag on, every row).
  AReaL packs and splits only `[B, L]` tensors, so each row carries four long tensors. `octopus_node` holds the leaf's
  id (60 bits of its completion id) at its first token. At each read's `at`, `octopus_mount` holds the source leaf's id
  and `octopus_mount_start`/`_end` hold the range. Sequences are matched by id, never by batch row: rows get
  reordered across concatenation, dispatch and micro-batches. A reader and its source are leaves of one episode, so
  they reach the same data-parallel rank (`dp_balance` keeps an episode whole).
- **Allocation** (`actor.batch_rows` → `allocate_prefix_groups(..., mounts)`). A reader and its sources form one unit
  (union-find, so chains like "source that is itself a reader" stay together). The lexicographic walk adds a unit's
  sequences together, sizing each one exactly by its LCP with its lexicographic neighbours. The min_groups split cuts
  only between units. A unit over the cap gets its own micro-batch over the cap, the same way an over-cap single
  sequence does. It is never cut, and an `[octopus] mount unit over the micro-batch cap: N > cap tree tokens` line
  counts it. Without mounts, the walk gives the same result as before: identical to main's function on 6,400 random
  cases (caps, min_groups, divisors).
- **Actor** (`actor.micro_batch_tree`, used by the model forward hook for `ppo_update` and `compute_logp` alike). The
  packed keys go through `seq_mounts` into `tree_inputs`' mounts. If a source leaf is missing from the micro-batch
  (for example `ppo_n_minibatches > 1` splitting an episode), `seq_mounts` raises a `ValueError` naming the reader. A
  reader is never trained without its read.

Evidence:

- `tests/test_prefix_tree.py` (local CPU). A concat episode: two leaves below the source (the first one is chosen),
  a reader leaf that continues after its read, a second reader of the same range, and a reader of a range that its
  source computed with its own read. Checked: `record_reads` → `read_tensors` → a padded batch → `seq_mounts` → each
  reader's later tokens see exactly the source leaf's tree tokens `[s, e)`, the token before `at` sees none, and K2
  positions follow the shifted source. Allocation keeps the unit together where token order alone splits it (also
  under min_groups splitting). The over-cap unit gets its own micro-batch and the log line. A micro-batch missing the
  source raises. `mounts={}` gives the plain allocation.
- `tests/test_read_export.py::test_proxy_export_carries_reads_to_the_actor` (node-94 AReaL venv + overlay, CPU). The
  proxy on the fake SGLang runs the same episode shape with real `read_thread` mounts. It goes through
  export → `serialize_interactions` → `deserialize_interactions` → workflow's `to_tensor_dict` → AReaL's concat →
  `actor.batch_rows`. Every read arrives as `(at, first leaf below the source, start, end)` with the proxy's numbers,
  and the source row holds the source's tokens there. With the flag off, the batch has no read keys and every other
  tensor is equal. With `tests/test_areal_cursor_hop.py` and `tests/test_areal_cursor.py`: 20 passed, 4 live skipped.
- `test_live_read_trains_as_it_rolled_out` (2026-10-18, node-94: GPU 6 SGLang 0.5.19 sglang-4090d + the
  lora/cursor/mount overlay, Qwen3.5-4B-text, no_buffer, overlap off, greedy; GPU 5 for the offline forwards, no
  optimizer). It reruns the forced read episode as is, exports it through the real path, and splits it with the
  actor's allocation (one micro-batch, 3 leaves). The actor's micro-batch tree forward, against SGLang's rollout
  log-probs:

  | tokens | n | mean abs Δ logprob | max | argmax agreement |
  |---|---|---|---|---|
  | reader tokens (the free replies computed with a read), tree with the read | 98 | **0.0038** | 0.081 | 0.949 |
  | same tokens, read dropped from the tree | 98 | 0.0142 | 0.305 | 0.959 |
  | baseline: the model's own stop tokens of forced replies before any read | 8 | 0.0031 | 0.022 | |

  Per reader, with the read vs without it: branch-1 (behind, positions from 3,717) 0.0070 vs 0.0270, and main (ahead,
  3,618 instead of 3,489) 0.0012 vs 0.0037. The same run's offline K2 cache surgery gave 0.0048. A backward through
  the mounted tree gives finite gradients on all 426 parameter tensors (norm 26.9), with no optimizer step. The free
  replies are tool calls that do not quote the mounted text, so dropping the read moves their log-probs by only 3-4x
  the kernel noise. Every other reply is forced text with placeholder log-probs, which is why the baseline is only 8
  tokens.

## Slice 4 run

A DAG-GRPO run with the read tool on (`OCTOPUS_READ_TOOL=1`, cursor mode, SGLang mount), configured like bundle-e13
and compared on M1-M4 with a matched run without it. It ran as bundle-e13-read-10182126 (steps 1-14,
audit below) and, with the read fixes, as bundle-e14-read-10191636 (`results/e14-read/report.md`), after the slice-6 mount fix ("Memory pressure" above) reached main; the section below records its arms.

- **Arms** (the e13 env files below are deleted; see commit a6e83a9). `scripts/rl_env/rl_run.bundle-e13-read.env`: e13's flags, policy, adapter, tasks, 40 steps, layout (trainer
  node-94 3-7, rollout node-94 0-2 + node-136 0-4) and MB_TOKENS, plus `CURSOR`, `OCTOPUS_READ_TOOL=1`, the mount's server
  flags and fresh STATE, ports (31300-31302, 31800-31804), tunnel port (2244) and spool. Baseline
  `rl_run.bundle-e13-noread.env` sources it and only turns the tool off (own STATE and spool), so CODE, overlays,
  servers and flags match. On node-136 the read arm uses `rl_run.bundle-e13-read-remote136.env` and the baseline
  `rl_run.bundle-e13-noread-remote136.env` (the same plus its own STATE, so its serve keeps the read arm's server
  logs); they run one after the other.
  e13 uses `rl.plan_cost_unit=wall`, so rollout speed is in the reward: the baseline serves with the read arm's
  flags (no_buffer, overlap off). e13 itself (extra_buffer, overlap on) is a fair baseline only if slice 6 finds those
  flags leave rollout speed unchanged.
- **Deployment.** No new knob. `CURSOR` serves every SGLang server (local and REMOTE, same paths) on `$CURSOR/sglang` + CODE
  (its shims import CODE's `octopus/`; a REMOTE host needs CODE of the trainer's commit, checked at serve) and runs the trainer on `$CURSOR/areal`; `SGLANG_ARGS` stays as without the read tool (the mount works on any mamba
  strategy with overlap on, PR #172; e13-read used `--mamba-radix-cache-strategy no_buffer --disable-overlap-schedule`
  before it); the env file exports `OCTOPUS_READ_TOOL=1` (episodes
  and the rollout workers, whose tensor dicts add the read keys, inherit it).
- **Build** (from main `C` with the slice-6 fix, as `docs/cursor-protocol.md` "Deployment", which now applies the
  logprobs, cursor and route patches to AReaL and the cursor and mount patches to SGLang): on node-94 a code dir (`git
  archive` of octopus docs patches pyproject.toml scripts tests into `/home/user/harbor-udocker/code-<sha>`) and
  `D=/home/user/octopus-sft/cursor-<sha>` (areal + sglang); on node-136 the same `$D/sglang` from its own `SGLANG_PY`
  (`/home/user/octo-env/sglang-venv`, needs the code dir's patches and octopus). Check with
  `PYTHONPATH=$D/areal:$C $AREAL_PY $C/patches/check_areal_route.py` and, on node-94, `PYTHONPATH=$D/areal:$C $AREAL_PY
  -m pytest tests/test_areal_cursor_hop.py tests/test_areal_cursor.py tests/test_read_export.py`. Then set CODE and
  CURSOR in `rl_run.bundle-e13-read.env`, copy both `*-remote136.env` files under their names to
  `/home/user/rl-remote/` on node-136 and `mkdir -p` their TMPDIRs.
- **Routing** (`patches/areal-route-spread.patch`, now after the cursor patch). A request on a reply's cursor goes to
  that cursor's server; any other request goes to its proxy session's `OCTOPUS_ROUTE_SERVERS` server, and a new session
  goes to the one with the fewest of the engine's requests in flight, waiting while every server has
  `OCTOPUS_ROUTE_CAP` (rl_run.sh `ROUTE_CAP`, 0 = no cap) or more (`octopus/serving/route.py`; with
  `OCTOPUS_ROUTE_KV` only to a server below that KV usage, `docs/search-env.md` "Admission"). A session is one
  episode tree, so all its threads live on one server, as a mount needs; a mount of a source on another server is an
  error (engine). Tests: `test_route_servers_keep_each_episode_tree_on_one_server`, `tests/test_route.py`.
- **Launch.** On node-136 `RL_ENV_FILE=/home/user/rl-remote/rl_run.bundle-e13-read-remote136.env rl_run.sh tunnel`
  once (node-94's known_hosts holds node-136's key for `[127.0.0.1]:2244` since 2026-10-18), then on node-94
  `RL_ENV_FILE=<arm env> setsid nohup rl_launch.sh bundle-e13-read-<MMDDHHMM>`; the watchdog stops the run after step 40
  and releases the GPUs; `rl_run.sh untunnel` on node-136 after the last arm, with the env the tunnel was started from
  (its STATE holds tunnel.pid).
- **Trainer.** `ppo_n_minibatches: 1` (`grpo.yaml`, not overridden), so a reader and its sources reach one
  micro-batch allocation; `seq_mounts` raises otherwise. A read unit over MB_TOKENS (196,608 per rank in e13 and rl_run.coupled.env; rl_run.sh's default is 262,144) gets its own
  over-cap micro-batch, which the watchdog's 10% MB_TOKENS cut on OOM cannot shrink (3 OOM resumes, then NEEDS-MAIN).
  Unit sizes from e13's 219 finished episodes (requests.jsonl; tree tokens = main's leaf + every other thread's own
  tokens): a reader leaf + one source's own tokens p50 73,597, p99 214,587, max 259,141 (3 episodes over 196,608, none
  over 262,144); a unit as large as the whole episode tree (reads joining every thread) p50 82,868, p99 327,437, max
  350,897 (24 over 196,608, 5 over 262,144). 294,912 tokens peaked at 46.7 of 48 GB in a 3-rank step (grpo.yaml),
  and 5 ranks shard more, so GPU OOM needs reads that join most of one of the largest ~3% of trees; host memory adds
  ~0.29 MB per token on one rank (+18 GB at 259k) under MEM_MAX 400G. Not a risk while valid mounts are rare; if a
  `mount unit over the micro-batch cap` line comes with an OOM, the fix is in the allocation, not MB_TOKENS. (These
  are e13's numbers; layer streaming and block checkpointing have since lowered both: `docs/trainer-memory.md`.)
- **Reward and plan grading with reads** (audit of `octopus/envs/swe/reward.py`, `octopus/rl/workflow.py`, `octopus/runtime/tree.py`; no
  inconsistency found, so no change). Mounted tokens are prompt tokens of the reader: they count in neither the token
  budget nor its tag (completion tokens), as a branch's inherited history does not, nor in
  a thread's growth cap (`settings.thread_budget`: prompt growth), so a reader may attend to more than
  its prefix plus 32,768 tokens (`docs/search-env.md` "Limits"), transitive segments included;
  `prompt_tokens` everywhere (requests.jsonl, `assign_threads`' positions) is the reader's real tokens, without the
  server's placeholders; the read size adds the mounts inside the range (`_mount_range`). Wall cost counts a read like any reply: its request's time, including any
  prefill retry of an evicted mount. The controls never get the tool (main's: `LINEAR_DROP`; a nested one is refused);
  a main-later control continues from main's cursor and keeps a read main made before that fork, as main did. A
  read cannot share a reply with fork or rewind; a rewind past a read drops it. A read in main's post-join replies is
  post-join work (`group.join_post`'s P).
- **Audit.** `python scripts/rl_analysis/read_audit.py <jobs dir> $STATE/train-<trial>.log <local SGLang logs...>
  [--remote [USER@]HOST[:PORT]:/GLOB ...]` per training step: read_thread calls (every thread's, main's and each branch's, each call
  id once), results by kind (valid mount, self, own-history, no-thread, not-readable, malformed, other) and how many
  found no other thread yet, mount sizes, valid mounts per episode, the spread of read_thread calls over episodes per
  step and over all steps (median, mean, the share held by the 10 busiest episodes, and loops: episodes whose main
  ended at the call budget with read_thread at least half of their tool calls, with their y), requests that carried a
  mount and their failures, exported leaves with reads (the rollout worker prints `[octopus] read leaves N with reads
  R` per plan group with the flag on), the engine's read errors (the proxy's tracebacks in the train log end in one
  `RuntimeError:` line per failed request; they are in the trainer's log only), the SGLang logs' `/generate` 404/503,
  tracebacks (no timestamp: step `?`) and mount rebuilds (the evicted-read abort: since slice 6 the server rebuilds the
  rows and retries, so it is not a failure), and long waits, the failure mode left (agent requests over 300 s,
  requests that hit the agent's 900 s timeout and episodes with one) (`tests/test_read_audit.py`). A remote host's
  logs need no copy (node-136's link is too slow for them). On node-94, where the jobs dir is, node-136 is reachable only
  through its reverse tunnel (the env's REMOTE):
  `--remote 'user@127.0.0.1:2244:/home/user/rl-remote/state-e13-read/serve/sglang-g*.log'` makes one ssh call
  that greps there and sends back only counts per pattern and timestamp second (`grep | cut -c1-21 | sort | uniq -c`),
  counted per step like local lines (bundle-e13-read-10182126 at step 13: 6 mount rebuilds on node-136, 5 in step 1
  and 1 in step 4, no 404/503 or traceback). Steps
  1-10 of bundle-e13-read-10182126: 388 episodes, 5,164 read_thread calls, median 1 and mean 13.3 per episode, the 10
  busiest episodes 66% of them, 9 loops (y 0.5 once, else 0). M1-M4 stay with `steps.py` (and then
  `e11_pilot_audit.py`, since removed), unchanged.
- **Wiring pilot** (2026-10-18, `rl_run.bundle-e13-read-pilot.env` at commit a6e83a9, trial read4-pilot-10181600, code 62a087f with
  main's mount patch, no slice-6 fix): trainer node-94 GPU 7 (one rank), rollout 2 servers on node-136 GPUs 2-3, 2 bundles
  x G=4, 10 calls per task. Both servers ran the overlay (`PYTHONPATH=$CURSOR/sglang`) with no_buffer and overlap off,
  the trainer `$CURSOR/areal` with `OCTOPUS_READ_TOOL=1`, `OCTOPUS_CURSOR=1` and the two route servers; requests split
  1,000 / 1,253 between them, every `/generate` 200. Train step 1 done 18 min after launch (server start included; peak
  33.6 GiB), the watchdog stopped the run and the GPUs were free (node-94 GPU 7 1 MiB, node-136 all 1 MiB). Audit: 22
  episodes (14 before step 1; 8 cut by the stop, whose 8 in-flight requests ended `RemoteDisconnected`), 2,253 agent
  requests (1,815 offering read_thread, 2,245 HTTP 200), 216 read_thread calls: 206 self-reads (183 before any fork:
  no other thread yet), 6 own-history, 3 not readable yet, 1 malformed, **0 valid mounts**. So no mount request, no
  `/generate` 404/503, no engine read error, and 0 of the 55 exported leaves carried a read: **no mount reached the
  trainer**; the export and actor path with real mounts is covered by `test_read_export.py` and #138's live test. The
  only tracebacks came after step 1, when the stop destroyed the rollout engines (`call_destroy`).

## A0: forced-mount measurement (removed)

A0 tested whether mounted KV changes a reader's behaviour and its task outcome at all, by mounting without being asked,
randomized by episode (`OCTOPUS_READ_FORCE=<p>`; reads the policy chose are confounded with what made it read: 26 of 30
successful readers in bundle-e13-read-10182126 had first been refused). The flag, its tree.json `read_force` record,
`scripts/rl_analysis/read_force.py`, `read_force_dlogp.py` and their tests were removed with the other unused flags;
see commit a6e83a9.

## Gate 0: event-triggered sibling mounts (eval only)

Does a sibling's edit, mounted when it touches a file the branch depends on, keep the branch's patch from being taken
back at the join? Two eval arms on the same coupled bundles (several tasks in one file), both with
`OCTOPUS_AUTO_FORK=1`: (1) read tool off; (3) `OCTOPUS_READ_TOOL=1` plus `OCTOPUS_READ_EVENT=1`. Both are measurement
switches: default off, never in a training env. `OCTOPUS_READ_EVENT` adds no text, prompt, reward term or refusal;
`OCTOPUS_AUTO_FORK` adds one model-visible system-prompt sentence (below), the same in both arms. The arms are dev_eval
runs of one run's settings, each arm's switches by `--env` (e.g. `--env OCTOPUS_READ_EVENT=1`), through training's
proxy. The #171 smoke (`rl/jobs-gate0-smoke*`) ran
against SGLang's chat endpoint without the chat mount: its 18 mount requests were dropped (7 of ≥ 1,000 tokens grew
the prompt by < 0.5·M).

- **Event.** A branch E's step edits a file F (an `edit` or `write` the tool reports done), or E's commit's patch holds
  F, and a sibling B (a child of any fork of E's forking thread, still running) viewed or edited F before that.
  `shell` reads and edits are not seen; the commit catches E's shell edits. One event per (B, E's step).
- **Mount.** B's next request once the step is readable (E's next reply back) carries it, as `read_thread` would: an
  edit step alone; for a commit E's latest readable steps, at most 16, through the commit reply (E has ended). A step
  B holds already (mounted before) is not mounted again: a commit mounts the first run of steps B does not hold and
  stays queued, one mount per request, until all of them are held (each mount a record); an edit step already held is
  recorded `no mountable step`. A read of B's
  own pending goes first; one mount per request, so further events wait for B's next requests. An event whose range
  cannot be mounted (a rewritten edge in it) is dropped; a range holding a read of E's own mounts it too (transitive).
- **Record.** tree.json `read_event`: per mount (or unmounted event) reader, source, step (or `commit`), files, turn (B's replies before),
  mounted, and steps / tokens / from / to, or why not (`pending at end`: B ended or the step never became readable).
  Flag off: request bodies and tree.json are byte-identical to before.
- **Auto fork.** `OCTOPUS_AUTO_FORK=1` (both arms; eval only, SWE bundles): before any model call the environment forks
  one branch per bundle task (task order, branch-i for task i+1; `HarborWorkspace.auto_fork`), with no forking reply:
  each branch's first request is the initial one (system, task) plus its task message ("Fix task k." with no
  acceptance check, so it writes its own, verified both ways at its commit, then the task's statement), sent whole;
  main's history is the initial one plus the join note, and main may fork again. tree.json `fork_index` is null and
  each branch's `task_id` is the environment's. The system prompt's "是否 fork、何时 fork、分几个分支都由你决定。" becomes
  "本次评测开局时运行时已为每个任务各 fork 出一个分支，主线在这些分支全部收口后收尾，之后仍可再 fork。". The arms' requests then
  share system prompt, task and tools; arm 3's branch task message adds the read roster and read_thread's card, and
  main's join note the card. The auto fork has no response_id (no reply made it), so a dev_eval row's C, C_wall, J and
  P leave its branches out and read low: Gate 0 uses none of them; do not compare Gate 0 rows' cost.
- **Conflict split.** `OCTOPUS_CONFLICT_BY=1` (all arms): each file of a patch the join took back gets tree.json
  `conflict_by` `sibling` when B's version conflicts (`git merge-file`) with its fork-point version plus the patches
  merged into the forking thread since B's fork alone, else `own` (the forking thread's own edits since the fork:
  none now that every fork waits for its branches, so `own` marks a merge `git merge-file` accepts but
  `git apply --3way` refused), `unclear` when one of those patches does not apply to the fork-point version or git
  named no file.
- **Summary.** `scripts/eval/gate0.py off=A.jsonl event=C.jsonl` (dev_eval rows): the verdict is episode y of each arm
  less the first's, paired by bundle with a bundle bootstrap 95% CI, passing when its lower end is above 0; then each
  arm's episodes without a fork by how main ended (none under auto fork), and y, landed (committed patches merged /
  merged + taken back), branch task pass, branch commit rate, fork rate, y of the forked episodes, reads, mounts and
  forced mounts per episode, the conflict split, each with a 95% bootstrap CI over the bundles every arm has and the
  paired difference to the first.
- **3-arm smoke (training's request path, base model, no training run).** Containers and dev_eval on the eval host
  (node-134), SGLang on node-136 over the LAN; `$G` is the smoke's dir on the eval host, `$C` the code dir, `$CURSOR`
  the overlay built from `$C` (`docs/cursor-protocol.md` "Deployment").
  1. The settings dir, from an env file that exports the shared knobs (`OCTOPUS_CONFLICT_BY=1`,
     `CURSOR` set; no `OCTOPUS_READ_TOOL`): `RL_ENV_FILE=<env> $C/scripts/rl_env/rl_run.sh settings $G/run
     rollout.use_lora=false`. That writes `$G/run/config.yaml` and `$G/run/episode_settings.json`. `MODEL_TEXT` in
     the env file is the tokenizer dir on the eval host.
  2. Servers without an adapter on the LAN address: on node-136, `SERVE_HOST=10.0.2.36 RL_ENV_FILE=<host env>
     eval_serve.sh start 0,1,2,3` prints `10.0.2.36:30300,...` (`$SV`). The in-process proxy takes any
     `host:port` it can reach. Its engine posts `/generate` there, and `check_mount_layers` reads `/get_server_info`.
  3. The arms, each into its own file and jobs dir. Settings change only through `--env`, which may add keys:
     `D="python -m octopus.eval.dev_eval --run $G/run --servers $SV --tasks-dir <bundles> --k 1"` (with
     `PYTHONPATH=$CURSOR/areal:$C`, no `--lora`), then:
     - `$D --out $G/off.jsonl --jobs-dir $G/jobs-off`
     - `$D --env OCTOPUS_READ_TOOL=1 --out $G/read.jsonl --jobs-dir $G/jobs-read`
     - `$D --env OCTOPUS_READ_TOOL=1 --env OCTOPUS_READ_EVENT=1 --out $G/event.jsonl --jobs-dir $G/jobs-event`

     `--env KEY=` drops a key the settings have.
  4. `python $C/scripts/eval/gate0.py off=$G/off.jsonl read=$G/read.jsonl event=$G/event.jsonl`. Each row's trial is
     `<jobs dir>/<job>/<trial>/agent/{tree.json,requests.jsonl}`. The tree's `merge`, `conflict_by` and
     `read_event` fields come from the agent, as before.

  Secrets: no credential is needed. The proxy listens on 127.0.0.1. Each episode gets a per-session proxy key
  (`OCTOPUS_API_KEY`, in its Harbor command line and environment), valid for this proxy only. Any process of the same
  user, a container's agent included, can read `/proc/<pid>/environ` and `cmdline`. So do not export API keys or
  tokens in the eval shell or the env file.


## Read control (credit)

`OCTOPUS_READ_CONTROL=1` pairs each branch's first valid read with a continuation from the same state without the mount.
The read_thread call is credited with `r_reader - r_rc`. The math, skips and cost are in docs/rl-plan-episodes.md, "Read
control".
