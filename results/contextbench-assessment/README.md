# ContextBench for Octopus: evaluation and RL feasibility (draft for review, 10-24)

> Reviewed 10-24 (independent reviewer). Corrections that override the text below:
> - READY has two forms: a line `READY_FOR_NEXT_OP` in the reply (shell-less chat agents, runner default `delivery="reply"`) or `echo` for shell agents (stream.py:7-12,210-212).
> - Pull delivery (`next_op` tool) is explicitly rejected by ContextBench (docs/DESIGN.md:102-106). Preferred: reply-mode (ops stay user messages; core changes "plain-text reply ends the thread" into "ends the turn"). If next_op is used: cite DESIGN.md, MAIN only, no args, exempt from clip_results, rewritten Turn protocol, reported.
> - Log grader already requires exactly one number; the hole is "any ANSWER block for a qid counts" + substring lookup. Needle ~1/P is an honest baseline, not a hack. Rewind-specific hole: print many candidates in one turn, then rewind them away — every turn's snapshot is graded.
> - Fork helps Log only below budget (L160–L640); at P>1 it is 0 like linear. Fork is useless only because ops go to MAIN only (must hold).
> - read_thread arm invalid because mounted tokens bypass the o200k wall, not because answers are ungraded.
> - Extra integration pitfalls: answers must be in message content (not tool args); refuse done while ops remain; rewind note prefix "Your note:" breaks line-exact needle matching; raise our own thread_budget (30,720 o200k ≈ 38K Qwen tokens on kv).
> - Rewind cache: slice 7 had 41 live rewinds (token-exact, cached_tokens not reported) — reuse unmeasured; repeated rewinds to the same early step keep that target hot, so expected recompute is between a short tail and the whole prefix. Existing search eval logs with cached_tokens may answer it without GPUs.
> - CLM RL is the same task pair (OpenResearcher → BCP), not the same setting (9B full-parameter, 28K budget, thinking on, nano judge, 8×32 rollouts, efficiency advantage).
> - Needle without instructions already 97.6% (Qwen3.6-27B dev) → may discriminate poorly at low levels; per-op-rewind training cost at 24× likely 60–90× a search episode.

Sources: ContextBench repo clone `scratchpad/contextbench-src` (commit fb533c1), CLM paper text `scratchpad/clm-align/paper/clm.txt` (P:n = line), CLM repo `scratchpad/clm-align/src/clm` (4831982), our repo origin/main b67d164. Two independent fact-finding passes (ContextBench/paper; our runtime). Nothing here was run on GPUs.

## 1. The pasted claims

| Claim | Verdict | Fact |
|---|---|---|
| Fig.2: all methods GPT-5.4, zero-shot, each with a method skill | true | P:308, P:1481; skills in clm_exp/skills/<method>/<family>.md |
| context_folding skill says "Do not open a branch" | true for kv, needle, sudoku; **false for log_triage** | log_triage skill: "do the reading in a `branch`, print the ANSWER in MAIN" |
| ops arrive as new user messages; READY_FOR_NEXT_OP to get the next | true | READY must be a shell statement `echo READY_FOR_NEXT_OP` (stream.py:47,62-104) |
| SUBMIT at the end | false | the run ends after the last op; no submit needed |
| 32,768 o200k budget, 2,048 reserve, budget_wall | true | counted over message content/reasoning/tool-call args, no template/schema tokens; Qwen tokens ≈1.24× o200k on kv |
| graded from per-turn snapshots, files score 0 | true | grader reads message `content` only |
| hitting the wall still scores | partly | kv/log: questions come after all batches → 0 if the wall hits during ingestion; needle: final frame counts |
| 32 levels, 0.25×–24×, seeds 42–45 (+46–49), 256 tasks | true | top level 26.9× |
| 7 baselines' curves published, no need to re-run | **false** | no result files in repo or HF; Fig.2 is a plot only |
| CLM edits /tmp/.live_ctx/LIVE_CTX_MAIN.txt via bash | true | python heredoc offload to /tmp/ctx_offload |
| "41.8% of edit turns recompute" | **not found** anywhere in paper or repos |
| rewind(to_step=1) "0 prefix recompute, 0 GDN error" | **not supported** | exact token prefix is guaranteed (cursor = target step + Δ); KV/GDN reuse depends on SGLang's radix/mamba cache (only per-request-boundary states, LRU, shallow checkpoints evicted first); the only measured rewind ran with --disable-radix-cache (docs/cursor-protocol.md:400-406). Unmeasured; old targets likely recomputed |
| Qwen3.6-27B without instructions KV 22.3%, Log 0% | true (dev split, skill-evolution start; Fig.10/28, not Fig.8) |

## 2. Fit per task family (zero-shot, our agent)

| Family | linear (= base) | fork | rewind | needs |
|---|---|---|---|---|
| Needle (0.5×–26.9×) | ~1/P partial credit | no help (ops land in MAIN) | per op: rewind, note keeps needle lines verbatim → can approach full marks if the 4B copies exactly | rewind cap off (≈1 per op), main calls ≥ ops+rewinds |
| Sudoku (t4–t150) | ~1/P | no help | per step: note keeps the current board → feasible | same; t150 ≈ 150 rewinds |
| KV store (R130–R16000) | 0 at P>1 (questions after all SETs) | no help | not enough alone | a file/shell tool to offload batches, grep at the end |
| Log triage (L160–L12800) | 0 at P>1 | **may help** (skill reads logs in a branch, prints answers in MAIN) | with files | file/shell tool |
| read_thread | — | mounted KV is not in the text context → not graded; an arm using it is invalid | | |

Grading sees only MAIN's message contents: branch outputs count only through what the join brings to MAIN.

## 3. Integration (from our code)

Map the protocol into tool vocabulary, so core stays almost unchanged: `next_op()` tool returns the next op as its tool result (READY), `submit`/done closes; o200k context counter as a provider hook (budget 30,720 incl. the first prompt, no clipping of ops); per-turn snapshots via the mixin in `_complete_async`; a provider whose snapshot/restore are no-ops so files survive rewinds. Pitfalls: `next_op` and `rewind` in one reply would delete the op just fetched (rewind runs after the turn's tool results) → refuse like FORK_ALONE; plain-text replies end a thread today; the ContextBench role is user, ours would be tool — a declared deviation; keep thinking off.

| Stage | Work | Effort |
|---|---|---|
| (a) zero-shot Needle + Sudoku, arms linear / rewind (fork as a sanity arm) | new octopus/envs/cb/agent.py (~150–200 lines), core: generalise FORK_ALONE + context_tokens hook (~10–20), env files, fake-model test | ~300–400 lines, 1.5–2 days + eval GPUs |
| (b) KV + Log | shell/file tool in the provider, not rolled back by rewind | ~60–120 lines, 0.5–1 day |
| (c) RL | generator/grader wrapper (ContextBench's generator is procedural: `generate(seed, knobs)`), cb/rl.py, outcome, train/config generalisation, curriculum | 3–5 days + compute; training needs the user's go |

## 4. RL feasibility

- Data: unlimited and deterministic per seed; knobs per family. **Exclude seeds 42–49 at every level** (same seed → identical op-stream prefix across levels, verified 10/10 ops); training on the same generator makes ContextBench in-distribution — hold out a family or the top levels to claim generalisation.
- Reward: deterministic, cheap, partial credit in [0,1] → y directly, R = 2y−1, LOO. **Hackable as is**: kv substring match and "any answer for a key counts"; log counts accept several numbers in one answer; needle gives ~1/P for appending until the wall. Training needs a hardened grader (one answer per key/qid, exactly one number, etc.).
- Cost per episode relative to a search episode (~14 requests, ~26K tokens), unverified estimates: 1× ≈ 1× rollout / 1.6× train; 4× ≈ 3× / 6–7× (~1 h per 16×8 step); 24× ≈ 15–19× / 35–40× (impractical) → curriculum ≤ 4×, evaluate higher. Many requests per episode raise the chance a group is rejected on one failed request.
- Prior work: the CLM paper does **no** RL on ContextBench; its RL is Qwen3.5-9B GRPO on OpenResearcher, evaluated on BrowseComp-Plus (CLM 28.8 → 42.5, Summary 34.7 → 42.1) — the same setting as our Search run, so its Table 2 is the closer comparison point.
- Licences: ContextBench Apache-2.0; CLM repo CC BY-NC 4.0 (affects copying its ClmStream/ClmAgent code).
- Capability risk: 4B zero-shot compliance with a 150-step verbatim protocol is unknown; Qwen3.6-27B without guidance scored KV 22% / Log 0% (dev).

## 5. Recommendation

1. Measure first (≈1 GPU-hour): rewind cache reuse — rewind to step 1 at several depths with radix cache on, read `cached_tokens`. Decides whether "rewind is cheap" can be claimed at all.
2. Zero-shot pilot (stage a, ~2 days + a few GPU-hours): v0 and the current Search checkpoint on Needle + Sudoku, levels up to ~8×, seeds 42–45, arms linear vs rewind (cap off). Gate: rewind clearly above linear at P > 1.
3. Only if the gate passes: KV/Log with a file tool (stage b), then RL with a hardened grader and a ≤4× curriculum (stage c), after the user's go.
4. Paper framing: ContextBench would support "rewind for streaming/deep context", not fork; Log is the one family where a branch may help. Search evidence so far: rewind gave no accuracy gain on search (RL suppressed it), which is consistent with this division of labour, but the ContextBench side is untested.
