# Server-resident cursor protocol

Without a cursor every request of a thread POSTs the whole message list (`tree._complete`), and the AReaL proxy
re-renders and re-tokenizes the whole conversation to find the few new tokens (concat mode). Both costs grow with the
history L. With a cursor the server keeps the thread's token sequence; a request names it and carries only the new
messages, so request size and prompt preparation cost O(Δ). All seven slices (Slices below, #111–#116 and #121) are merged and
passed the live acceptance below; a run turns the protocol on with `CURSOR=<dir>` in its env file (Deployment), and
without it requests are sent whole as before.

## Wire format

- Request: the `/v1/chat/completions` body of today plus `"cursor": "<response id>"`, with `messages` holding only
  the messages after that response's reply (tool results, user notes; any role the template accepts). `tools`,
  `tool_choice`, `chat_template_kwargs` and sampling fields are sent as today.
- No `cursor`: today's full mode, unchanged.
- Response: unchanged. Its `id` is the new cursor (the request's prompt + this reply).
- Fork: several requests from one cursor. Rewind: a request from an older cursor. Checkpoint: any returned cursor; there
  is no separate operation.
- Unknown or evicted cursor: an explicit 4xx error naming the cursor. No silent fallback to a full render (the client
  no longer has a full list to send in that request, and a fallback would hide a broken chain).

## Semantics

A cursor's tokens are AReaL concat's parent tokens: the request's prompt ids + the output ids without the stop token
+ the reply's terminator (also when the reply was cut by length or aborted). A request from cursor C with messages Δ
has the prompt

    tokens(C) + render_delta(tokenizer, Δ, tools, chat_template_kwargs)

**Lossless** means bit-identical to the AReaL concat oracle (`areal/experimental/openai/client.py`
`_concat_prompt_token_ids_with_parent`): with `all = apply_chat_template(parent.messages + parent.output_message_list
+ Δ, tools, add_generation_prompt=True, **chat_template_kwargs)`, the prompt is `parent_tokens + all[cut:]`, cut =
right after the k-th marker in `all` (at the next marker when the terminator is empty), k = the markers up to the
parent reply's terminator where the template renders the parent's thread before one more message (`concat_start`).
So the child's sequence is the parent's tokens verbatim + the terminator + the template's own rendering of Δ after
that history. k is counted on that render, not on the parent's tokens: those keep what the model generated, which a
re-render need not show (gpt-oss always writes an analysis channel ending on the marker `<|end|>`; AReaL keeps it as
`reasoning_content`, which the gpt-oss template does not render). render_delta, AReaL's full mode, the probe and the
tests share this one count (`delta_render._cut`); when the render holds too few markers it raises. The model-visible content never changes: the client sends the same messages it
sends today, only not the history again.

**Terminators** (`delta_render.terminator`, probed once per template and chat_template_kwargs): what the template
renders after an assistant reply in the history, in place of the token the model stopped on. It is the first control
token (an added token marked special, which a model never writes as content, unlike Qwen's `<tool_call>` or
DeepSeek's tool markup) after the reply's last varying character, when it is the same before a user message, a tool
result and the generation prompt; otherwise there is none. The markers concat counts are these first control tokens.
The stop token stripped from the output is the one the generation matched: eos, pad, or a stop of the model's own (a
control token; AReaL still raises on any other last token of a stopped generation, e.g. a stop string's).

| template | model stops on (generation_config) | terminator after a plain reply / a tool call | markers | append-only pairs: synthetic / episode |
|---|---|---|---|---|
| Qwen2.5/3/3.5, Qwen3-Coder | `<\|im_end\|>`, `<\|endoftext\|>` | eos `<\|im_end\|>` / same | eos | 2/28 / 193/198: think blocks dropped before the last user query |
| Llama-3.1/3.2 | `<\|eot_id\|>`, `<\|eom_id\|>`, `<\|end_of_text\|>` | eos `<\|eot_id\|>` / same | eos | 28/28 / 198/198 |
| DeepSeek-V3/R1/V3.1 | eos | eos `<｜end▁of▁sentence｜>` / same | eos | 28/28 / 198/198 (V3) |
| Mistral v0.3 | `</s>` | eos `</s>` / same | eos | 1/10 / 97/102: the system prompt sits in the last user message |
| GLM-4.5, GLM-4.7 | `<\|user\|>`, `<\|observation\|>`, `<\|endoftext\|>` | none / none: the next message starts with its role token | `<\|user\|>`, `<\|observation\|>`, `<\|assistant\|>` | 22/28 (4.7: 2/28) / 198/198: earlier reasoning dropped; 4.7's generation prompt ends `<think>`, a reply without reasoning renders `</think>` |
| gpt-oss | `<\|return\|>`, `<\|call\|>`, `<\|endoftext\|>` | `<\|end\|>` / `<\|call\|>` | `<\|end\|>`, `<\|call\|>` | 23/28 / 198/198: a tool call's analysis channel is dropped once a plain reply follows |

Append-only pairs: (parent, request) pairs whose request's full render extends the parent request's (synthetic every
parent, episode the previous reply; thinking on, with tools). Where it does not, concat still keeps the parent's tokens
verbatim (by design: what training sees) and the tail is the template's rendering of Δ. Existing templates keep their
bytes: the probe finds eos after both kinds of reply and eos as the only marker, which is exactly the old rule
(render_delta of main and of this branch agree on every pair of the sweeps below).

SGLang's own `/v1/chat/completions` re-renders the full list, which differs from concat at a real user message: the
Qwen3.5 template drops the `<think>…</think>` block of every assistant turn before the last real user query. Concat
keeps the parent tokens verbatim, so the cursor keeps them too (by design: it is what training sees). In the recorded
bundle-e11 episode that is 4 tokens per earlier reply, 92 at main's first request after its fork.

## Thinking on: runtime notes are tool results

With thinking off nothing above depends on where a note goes. Thinking is on when `OCTOPUS_CHAT_TEMPLATE_KWARGS` is set
and does not hold `"enable_thinking": false` (`settings.thinking()`, read as the Qwen3.5 template reads it; unset is
off, as for the Grok API). It is the one switch: the agent sends it in every request, SGLang renders with it and the
AReaL proxy lets a request's `chat_template_kwargs` override `grpo.yaml`'s. A Qwen3.5 SGLang server needs the
variable set explicitly (unset, the server's template thinks while the agent sends user notes). Training sets it per
episode from the run config's `rollout.agent.chat_template_kwargs` (`octopus/rl/run_config.py` `request_kwargs` /
`episode_env`): thinking on there means changing `{enable_thinking: false}` in `grpo.yaml` (or overriding that key). With thinking on, a user message mid-episode would become
the template's last real query and SGLang's full render would drop every earlier `<think>` block: the next prompt would
not extend the last one, a branch's first prompt would not extend main's, and full-mode eval would see another history
than training. So `core.add_note` keeps the episode's first user message its only real query, and never
changes anything the thread has already sent:

| note | thinking off | thinking on |
|---|---|---|
| branch task (fork) | user message | appended to the branch's own copy of the forking reply's last tool result |
| join note | user message | appended (`\n\n`) to the pending last tool result |
| rewind note (`OCTOPUS_REWIND`, `OCTOPUS_REWIND_TOOL`) | a tool message of its own (tagged: the calls left now, the context left at the step it went back to) | a tool message of its own (the history was cut back to results already sent) |
| close reminder (`core.REMIND_NOTE`, once per thread after a reply without a tool call) | user message | a tool message of its own (tagged; the last message is the reply) |

A tool message without `tool_call_id` is accepted by SGLang (`tool_call_id` optional) and the AReaL proxy (read with
`.get`); the template renders it as one more `<tool_response>` in the user turn, and it is tagged like any tool result.
The read control's copy of a mounting read's result drops the step listing but keeps a note appended after it
(`tree.without_step_listing`). With thinking on, a fork's join note sits inside `main[join_index - 1]` (no
message of its own), so a pipeline that reads trees by that index (`tree_sft/eval_metrics.py`; the forkeval spec pipeline did
too until it was deleted in #194) takes thinking-off trees only.
`tests/test_thinking_injections.py` renders with the Qwen3.5 template.

## render_delta (`octopus/serving/delta_render.py`)

Renders `stub + Δ` (the stub ends on an assistant reply, the role every cursor ends on) with the generation prompt,
keeps the text after the stub reply's terminator (`_cut`, the count of Semantics with the stub as the thread) and
tokenizes only that. Special tokens split tokenization, so the tail's ids do not depend on the prefix. Python 3.10+, a
tokenizer object only. `terminator(tokenizer, reply, kwargs)` and `concat_start(tokenizer, history, all_tokens, tools,
kwargs)` give the other two places the same rule: the AReaL proxy (both modes) and SGLang's `put_reply` append the
terminator, AReaL's full mode cuts the re-render at concat_start (one more render of the parent's thread, text only).

`template_form(tokenizer, chat_template_kwargs)` probes the chat template once per (tokenizer, kwargs), rendering with
those kwargs (cached; ~0.1 s, on the first cursor request with them), and picks the stub and the tool_call argument form. It renders `PROBE` (system prompt, single tool calls and results,
plain replies, user turns, alternating so Mistral accepts it) and compares render_delta on every (parent reply,
request) pair, with and without tools, against AReaL concat's tail:

1. History-free templates: the constant `STUB` (a real user query, so the think-block rule is the full conversation's,
   and a plain reply). Unchanged fast path for Qwen3.5 and the other Qwen/Llama templates; also GLM-4.5/4.7.
2. Templates whose rendering of Δ reads the history: `context_stub(history)` = the thread's system message + a user
   query + a stub tool exchange if a tool result came before + the cursor's own reply. This is what the templates
   read: Mistral v0.3 puts the system prompt into the last user message and checks user/assistant alternation over
   plain replies only (a tool-call reply does not count), DeepSeek-V3/R1 render the conversation's first tool call /
   tool result differently from later ones, gpt-oss names a tool result by the call before it (and refuses one after
   a plain reply). render_delta takes `history` = the cursor's thread up to its reply (the
   AReaL proxy passes `parent.messages + parent.output_message_list`) or its stored context_stub (SGLang): the stub of
   stub + later messages + their reply is the thread's, so a cursor store keeps a few messages, not the thread.
3. No stub matches, or the oracle is undefined: `ValueError` naming the tokenizer and the failing check (for no
   match, the first mismatching probe pair of each stub). A template that renders no control token after a reply has
   nothing to mark a cursor's end in a re-render. Gemma-3 cannot render a tool role at all.
   MistralCommonBackend tokenizers (Mistral-Small-3.2, Devstral) do not parse special tokens in `encode()` and refuse
   a rendering that ends on an assistant reply.
   In SGLang, a reply whose cursor's ctx was evicted during generation is stored without ctx, so a request from it
   raises (for templates that need the history) instead of rendering from a partial history.

tool_call arguments go as dicts, as AReaL's concat path parses them, unless the template raises on dicts and renders
strings (DeepSeek-R1/V3/V3.1 concatenate them into the text): then as JSON strings (dicts from AReaL are dumped).

Evidence (`tests/test_delta_render.py`, Qwen3.5 tokenizer, also run in node-94's AReaL (3.12) and SGLang (3.10) venvs):

- Synthetic thread with every shape (parallel tool results with budget tags, reply with content / `reasoning_content` /
  `<think>` in content / XML-style args, nudge, `tool_output` user message, `<tool_response>`-wrapped user message,
  background note, join note as tool and as user, fork task): every (parent reply, request) pair, so deltas of one
  message up to most of the thread, enable_thinking on and off, tools on and off: equal to the concat tail.
- Recorded bundle-e11 episode (main + 4 branches, one nested fork; fixture `tests/fixtures/delta_render_episode.json.gz`):
  198 (parent, request) checks covering all 118 requests that have a parent equal to the concat tail, and each recorded `prompt_tokens` = parent's `prompt_tokens` +
  `completion_tokens` + `len(render_delta(Δ))`. The bundle runs send no nudge (native tools), so the nudge shape is
  synthetic only. The proxy adds `"strict": false` to each tool function; the fixture stores tools that way (it only
  touches the system block, never a delta).
- Other templates (`test_delta_matches_concat_on_other_templates`, tokenizer from `<NAME>_TOKENIZER` or the local HF
  cache, else skipped): the same synthetic every-parent and episode checks, the cursor's thread as history, with
  enable_thinking on/off and tools on/off. A prefix diff (render(STUB) as a text prefix of render(STUB + Δ)) would be
  wrong: Qwen3.5 drops the stub's empty think block once Δ has a user query, and the tail must keep the `\n` after
  the eos. Offline sweep, all four settings summed (episode every-parent = every earlier reply as the parent):

  | template | form (stub, arguments) | synthetic | episode | episode every-parent |
  |---|---|---|---|---|
  | Qwen3.5-4B, Qwen2.5-7B, Qwen3-8B, Qwen3-Coder-30B-A3B | constant, dicts | 112/112 | 792/792 | 18,344/18,344 |
  | Llama-3.2-1B, Llama-3.1-8B (first call per reply) | constant, dicts | 112/112 | 792/792 | 18,344/18,344 |
  | DeepSeek-V3.1 | constant, strings | 112/112 | 792/792 | 18,344/18,344 |
  | DeepSeek-V3, DeepSeek-R1 | context, strings | 112/112 | 792/792 | 18,344/18,344 |
  | Mistral-7B-Instruct-v0.3 (first call, 9-char ids, adjacent user turns joined) | context, dicts | 40/40 | 408/408 | 4,372/4,372 |
  | GLM-4.5, GLM-4.7 (no terminator) | constant, dicts | 112/112 | 792/792 | 18,344/18,344 |
  | gpt-oss-20b (`<\|end\|>` / `<\|call\|>`) | context, dicts | 112/112 | 792/792 | 18,344/18,344 |

  Mistral's counts are the requests its template accepts (it rejects a user message after tool results). DeepSeek-V3
  holds only for tool-call replies with `content: ""` (the recorded shape), which it renders without the calls: with
  `content: null` it renders the calls but puts no eos after the conversation's first one and one eos per call after
  that, so the eos count no longer matches the replies and AReaL concat's cut is misaligned (and AReaL's dict
  arguments crash it, as they crash R1/V3.1). R1 also passes with `content: null`.

Cost (`python -m octopus.tools.delta_render_bench <tokenizer>`, the same 2-tool-result request after L tokens):

| L (history tokens) | full render+tokenize ms | delta ms | delta tokens | full request bytes | delta request bytes |
|---|---|---|---|---|---|
| 8,073 | 20.5 | 1.36 | 166 | 35,514 | 8,004 |
| 32,227 | 79.1 | 1.23 | 166 | 129,411 | 8,004 |
| 127,716 | 358.5 | 1.44 | 166 | 501,310 | 8,004 |

## AReaL proxy (`patches/areal-cursor.patch`)

The patch also holds the engine hop below. Applies on top of `patches/areal-logprobs.patch` (`patch -p1` in an AReaL copy; CODE must be on PYTHONPATH, the client
imports `octopus.serving.delta_render`). node-94 overlay: `/home/user/octopus-sft/areal-overlay-cursor-<sha>` (the PR
names it). `AsyncCompletionsWithReward.create` takes `cursor`; the proxy passes it through like any create argument (served on
`/v1/chat/completions`, non-streaming; the responses and anthropic endpoints drop it).

- The cursor is looked up in the request's own session (`InteractionCache`, O(1)). Unknown, another session's, or a
  reply still generating: HTTP 404 `cursor <id> is not a finished reply of this session`.
- The new interaction's parent is the cursor's (no prefix matching); its `messages` = parent.messages +
  parent.output_message_list + Δ (references; export and retry-orphan grouping see today's full list).
- Prompt = parent `input_tokens + output_tokens_without_stop + terminator(reply)` + `render_delta(Δ)`: no render or
  tokenization of the history (render_delta's `history` is `parent.messages + parent.output_message_list`, list
  references; only templates that need it read its first and last message and its roles). Requests without `cursor`
  run today's code with the same terminator in place of eos and the cut at `concat_start` (eos templates: unchanged).
  `output_tokens_without_stop` strips the stop token the generation matched, also one of the model's own (GLM's
  `<|observation|>`, gpt-oss's `<|call|>`; upstream raised on any stop but eos/pad).
- `mount` (a cursor request only, from the read tool, `docs/cross-branch-read.md`): `{"from": id, "to": id, "thread":
  name}`, two finished replies of the session, "from" an ancestor of "to". The proxy turns it into a token range of
  "to"'s tokens (from "from"'s turn to "to"'s), passes it to the engine (`ModelRequest.mount`) and records it on the
  new interaction (`metadata["mount"]`). The prompt is unchanged: the reader's tokens are its own.

Evidence (`tests/test_areal_cursor.py`, node-94 AReaL venv, stub engine through `proxy_rollout_server.chat_completions`):
every prompt, every parent link and every exported concat leaf tensor equal full mode on a synthetic tree (tool loops,
3 concurrent fork children, a length-cut reply, a rewind, a control rerun) and on the recorded episode (119 replies, all
threads concurrent). One difference, by design: when a rerun of an earlier prompt returns the same reply (greedy or by
chance), full mode links the next request to the first-inserted of the equal candidates, the cursor to the rerun it
continues (`test_greedy_rerun_parent_differs`: same prompt ids, the leaf's logprobs differ).

Cost (`python tests/test_areal_cursor.py`, next request = one ~1.5k-token tool result after L tokens, median of 3,
in-process, stub engine, production processor; node-94 at load 15, cursor columns vary by ~2 ms between runs):

| L (prompt tokens) | full prompt build ms | cursor prompt build ms | full request ms | cursor request ms |
|---|---|---|---|---|
| 9,173 | 13.1 | 2.88 | 14.5 | 3.77 |
| 32,923 | 41.1 | 4.35 | 42.8 | 5.30 |
| 130,298 | 165.6 | 3.26 | 185.5 | 4.51 |

## Agent client (`OCTOPUS_CURSOR=1`)

`octopus/runtime/tree.py` and `octopus/runtime/core.py`; without the flag nothing changes. Every thread
kind (main, branches at any depth, the no-fork control) runs through `run_branch_continuation` and
`CursorRuntime._complete_async` (`octopus/runtime/tree.py`, shared by the SWE and Search agents), so one rule covers them all:

- A reply that a thread's history keeps exactly as the server sent it (content and tool_calls) keeps its response id
  under the key `cursor` (`with_cursor`). A request sends `cursor` = the last such reply's id and `messages` = the
  messages after it (`cursor_delta`, a scan back from the end to that reply); a history without one is sent whole (a
  thread's first request). The marker never reaches the server: the delta starts after the last marked reply.
- A reply the client rewrites stays unmarked: a reply whose calls the budget cut (`closing_only`: the transport then
  gives the payload no cursor). The next request continues from the cursor before it and resends the rewritten reply
  in Δ, which is what AReaL's full mode renders too (its prefix match does not take a rewritten reply as a parent).
- Bundle budget tags go on the delta's untagged tool results: the same messages and text as in full mode (every
  earlier tool result was tagged when first sent).
- Fork: a branch starts from the forking thread's history through the forking reply and its tool results (copied
  with the marker) plus its task message, so all children of a fork are requests from the forking reply's cursor,
  Δ = the fork's tool results + the task message. Nested forks the same, from the branch's forking reply.
- Rewind (OCTOPUS_REWIND) drops the messages back to the checkpoint, replies included: the next request continues from
  the last reply before it (an older cursor), Δ = what follows it + the rewind note. The model's rewind
  (OCTOPUS_REWIND_TOOL) the same: from step N's reply, Δ = step N's tool results (as tagged when first sent), the join
  note step N got (and any earlier rewind note kept there) + the note.
- Join and grace notes are messages in Δ.
- Control (OCTOPUS_PLAN_CONTROL): main's control replays main's history under another system message and tools, so its
  markers are dropped and its first request is a full send. A nested control rewrites the branch's task message:
  markers from there on are dropped, so it continues from the parent's forking reply with the branch's own messages
  (its replies included) in Δ.
- Errors: a 404 on the cursor fails the request like any other (post_chat retries only 403/429/5xx) and ends the
  thread; no fallback to a full send. A retried request is another child of the cursor (AReaL's
  `drop_retry_orphans`).
- Files: tree.json is written without the markers (`without_cursors`), as in full mode, so a run that re-sends it
  (the forkeval specs, deleted in #194) never names another run's cursor; requests.jsonl rows carry no messages (`payload_bytes` shrinks);
  trajectory.json is built from steps.

The SglangPort session path (`clone_session` patch, `resume_kv_probe`) is removed: production never bound it and SGLang 0.5.19's `/v1/chat/completions` drops `session_params`; forks are cursors now.

Evidence:

- `tests/test_cursor_client.py` (local): the agent on `tests/test_rl_plan.py`'s fake container and model, once in full
  and once in cursor mode, on a bundle tree (a call, an edit and three failures that rewind, a fork of 2 branches,
  one of which forks 2 more, joins, done; the control from main's fork or from the nested one): every cursor
  request's body equals the full-mode one, and the full-mode messages are the cursor request's messages + the server's
  reply as sent + the delta; the only full sends are each thread's first request and main's control; fork children
  share the forking reply's cursor, the rewind sends an older one; a reply the budget cuts gets no cursor; tree.json's
  main equals full mode's and holds no marker.
- `tests/test_areal_cursor.py::test_agent_cursor_equals_full` (node-94 AReaL venv, cursor overlay, stub engine, CPU):
  the same scripts through the proxy: every prompt, parent link and exported concat leaf equals full mode, for both
  controls (13 requests, 7 leaves each).
- Request bytes, one thread, each reply answered by a 1.5 kB tool result, 10 kB system + task
  (`tests/test_cursor_client.py::test_cursor_request_bytes_stay_flat_on_a_long_thread`):

| request | full bytes | cursor bytes |
|---|---|---|
| 1 | 11,911 | 11,911 |
| 2 | 13,631 | 3,419 |
| 8 | 23,951 | 3,419 |
| 16 | 37,711 | 3,419 |
| 32 | 65,231 | 3,419 |
| 48 | 92,751 | 3,419 |

## SGLang (`patches/apply_sglang_cursor.py`)

SGLang 0.5.19, both node-94 venvs: `sglang-4090d` (the RL rollout servers, `scripts/rl_env/rl_run.sh`, and the eval
servers, `eval_serve.sh`) and `sglang-0.5.19-cu128`. Idempotent, in either order with
`apply_sglang_lora_abort_release.py` (other files). Its `srt/octopus_delta_render.py` / `srt/octopus_cursor_store.py`
are shims that import `octopus/serving/delta_render.py` / `octopus/serving/cursor_store.py` from CODE (`PYTHONPATH=<overlay>:<CODE>`),
so the server runs the files the trainer imports. Deployment: "Deployment" below.

- `/v1/chat/completions`: `cursor` + the messages after its reply. Prompt ids = the cursor's tokens +
  `render_delta(tokenizer, messages, tools, chat_template_kwargs, CURSORS.ctx[cursor])`, with the messages as sent
  (`model_dump(exclude_unset=True)` of SGLang's request model) and the tools as SGLang's own template call gets them
  (`Tool.model_dump()`, filtered by `tool_choice`). Sampling, tool-call / reasoning parsing, logprobs and the response
  are unchanged. Stored: every single-choice reply, streamed or not, under its response id, as AReaL concat's parent: the
  prompt ids the scheduler got + the output ids without trailing stop tokens (eos, pad, `finish_reason.matched`; kept
  when cut by length or aborted) + the reply's terminator (of the response message), and its `ctx`: `context_stub` of (the cursor's ctx, or nothing) + the request's messages + the response message
  (a few messages, uncounted by the byte budget; read only for templates that need the history).
  The tokenizer manager now always returns the prompt ids internally (a C-level list copy); the response shows them
  only when asked.
- `/generate`: `cursor` + `input_ids` = the tokens after it. Input ids = the cursor's tokens + `input_ids`. Stored:
  every single request with `input_ids`, streamed or not, under its `meta_info.id`: its whole input ids (the cursor's
  included) + `output_ids` exactly as returned. SGLang counts the stop token in the output (`finished_len` = its
  position + 1), so an output that stopped on eos (`<|im_end|>`) ends with it and the stored tokens are AReaL concat's
  parent tokens; one cut by length or aborted has no stop token, and concat's parent is the stored tokens + eos.
  In both cases the stored tokens are a prefix of the next concat prompt and slice 5 sends `concat_prompt[len(stored):]`.
  An output that stopped on the other stop token (`<|endoftext|>`, Qwen3.5's config eos and AReaL's pad) is not a
  prefix: concat replaces it by eos, so that next request is a full send. Likewise gpt-oss's `<|return|>` (concat puts
  `<|end|>`); its `<|call|>` is its own terminator, and GLM's stop is the next message's role token, the tail's first
  token, so those stay prefixes.
- No cursor: today's path (plus the store write). Unknown or evicted cursor: HTTP 404 `cursor <id> is unknown or
  evicted` (chat: an OpenAI error body; `/generate`: `{"detail": ...}`).
- A stream is stored from its last chunk, which holds the whole output, prompt ids and finish reason; not under
  `--incremental-streaming-output` (chunks of deltas: a request from such a reply is a 404). Not stored: chat with n > 1, a chat request aborted before the scheduler took it (no prompt
  ids, nothing to continue), batch or text `/generate`. Cursor requests are text-only. The store
  lives in the HTTP process (one tokenizer worker, the default): a cursor is valid only on the server that made it.
- Store: one `array('i')` per cursor (its whole sequence), LRU, a budget of `OCTOPUS_CURSOR_BYTES` token bytes (default
  2 GiB). A lookup is one dict access and one C-level copy, and eviction frees exactly what it counts; parent-linked
  chunks would hold a thread in O(L) instead of O(n·L), but an evicted parent would stay alive under its children. The
  recorded bundle-e11 episode (121 requests) stores 6 MiB, so 2 GiB holds ~340 such episodes; a thread continues from
  its newest cursor, so old rewind points and finished threads go first.

Evidence (`tests/test_sglang_cursor.py`):

- CPU, node-94 4090d and cu128 venvs: the patch applies to an unpatched copy once, in either order with the lora patch,
  and compiles; store budget, LRU, replacement, the concat parent of a reply (eos/pad or the matched stop stripped,
  length/abort kept, the terminator appended).
- CPU, patched overlay (real `OpenAIServingChat` / `http_server.generate_request` on a stub tokenizer manager, Qwen3.5
  tokenizer): every (parent reply, request) pair of the synthetic thread sent as JSON with the tools, thinking on and
  off: prompt = stored tokens + the concat tail; a reply is stored as its concat parent and continued from; `/generate`
  stores input + output and prepends them; unknown cursors are 404s; a reply aborted in the queue (no prompt ids) is
  SGLang's 200 with finish_reason abort and is not stored.
- Live (`test_live_*`, `OCTOPUS_SGLANG_URL`; node-134 GPU 7, its `sglang-venv` 0.5.19 + the overlay, Qwen3.5-4B-text,
  the eval server flags, greedy, `--disable-radix-cache`): 9 chat requests (a tool loop, a length-cut reply and its
  continuation, a fork of two children with one continued, a rewind to an older cursor), each prompt
  (`return_prompt_token_ids`) equal to the concat oracle from the full message list and each output equal to
  `/generate` on the oracle's ids; `/generate` from a cursor, twice chained, after an eos-stopped and after a
  length-cut output, equal to the whole input ids in output ids and logprobs; unknown cursors 404 on both endpoints.

Cost (`python tests/test_sglang_cursor.py`, node-94 CPU at load 23, patched overlay; the server's request JSON parse +
`_convert_to_internal_request` for the same next request, one turn's tool results, after L tokens; median of 5; the
cursor column includes the C-level copy of the cursor's tokens):

| L (prompt tokens) | full request bytes | cursor request bytes | full prompt build ms | cursor prompt build ms |
|---|---|---|---|---|
| 11,024 | 46,521 | 7,559 | 25.6 | 1.45 |
| 34,193 | 136,232 | 7,574 | 63.5 | 1.67 |
| 126,586 | 493,390 | 8,493 | 327.6 | 3.73 |

## AReaL engine hop (`patches/areal-cursor.patch`)

The proxy's rollout engine (`RemoteInfEngine.agenerate`) sends `/generate` the cursor of the reply a prompt continues
instead of the whole prompt. It needs the slice-4 patch on every server it talks to: an unpatched SGLang drops
`cursor` and generates from the delta alone, so the engine fails loudly (`<server> ignored cursor <id>: it needs
patches/apply_sglang_cursor.py`) when a cursor hop's `meta_info.prompt_tokens` is not the prompt's length plus the
placeholders of the mounts the cursor holds (and of the request's own mount: `ignored the mount ...: it needs
patches/apply_sglang_mount.py`).

- Each `/generate` reply leaves a cursor on its server: SGLang's `meta_info.id`, holding the request's input ids + its
  output ids, i.e. the `ModelResponse`'s `input_tokens + output_tokens` (`ModelResponse.cursor` = (server, id) of the
  last segment). The client passes the interaction's parent reply (a cursor request's, or the one full mode's prefix
  match found) as `ModelRequest.parent`.
- Routing: a new rid goes to the parent's server (round robin otherwise); a rid AReaL already pinned keeps its server.
  So an episode's whole tree stays on its root's server: balancing is per episode, no longer per request (slice 6:
  watch the per-server load).
- Hop: when the parent's cursor is on that server and its tokens are a prefix of the prompt (two C-level list compares;
  text only), the body is `{"cursor": id, "input_ids": prompt[n:]}`, n = its input + output length. The tokens after
  a reply stopped on eos start with the next turn's; after one cut by length with concat's eos. Otherwise all ids:
  a thread's first request, no parent, a rid pinned elsewhere, images, a reply that stopped on `<|endoftext|>` (concat
  puts eos there), a non-concat template that re-renders.
- Interrupt/resume (pause for a weight update): the resumed segment continues from the aborted request's cursor with
  no new ids; versions stay per segment.
- The one fallback: SGLang answers 404 (LRU eviction, restart). `arequest_with_retry` now chains its last error, the
  engine logs `cursor <id> is gone from <server>: resending all <n> input ids` and sends them (AReaL holds them). A 404
  costs the request's retries first (`request_retries`, 1 s apart). Not when the request mounts or its cursor holds a
  mount (`ModelResponse.placeholders`): all ids would run without it, so that 404 is an error (`... would drop the
  read`), as is a mounted thread's request that cannot hop from its parent's cursor.
- Mount (read tool): `ModelRequest.mount` = (source reply, start, end) in the source's tokens. The engine needs a hop
  from the parent's cursor and the source's cursor on the same server (else an error), maps the range to the
  server's tokens (adding the placeholders the source holds before it, and those inside it: a transitive read) and sends `"mount": [{"cursor", "start",
  "end"}]`. The reply records `placeholders` = its parent's + `(at, M)`, at = the parent's input + output length. A
  mount request aborted before its first token goes again as the same mount request.

Evidence (`tests/test_areal_cursor_hop.py`, node-134 AReaL venv + this overlay):

- CPU, `RemoteInfEngine.agenerate` on a fake SGLang with slice 4's store: a chain after an eos stop and after a length
  cut, a fork, all on the parent's server with only the new ids; an `<|endoftext|>`-stopped parent, a rid pinned to
  another server: all ids; an evicted cursor: 404, all ids, and the next request continues from the resend; an aborted
  segment resumed from its id with no ids, versions per segment.
- Live (`test_live_cursor_hops_equal_full_ids`, node-134 GPU 7, `sglang-venv` 0.5.19 + the slice-4 overlay,
  Qwen3.5-4B (the VL model, the RL server flags without LoRA, `--disable-radix-cache`), greedy): the proxy
  (`proxy_rollout_server.chat_completions`, VL processor, qwen3_coder / qwen3 parsers, thinking off) on
  `RemoteSGLangEngine`, 12 requests: a tool loop, a length-cut reply and its continuation, a fork of two children with
  one continued, a rewind to an older reply, cursor and full-message client requests, a 1,500-token reply interrupted
  by AReaL's weight-update pause (`/pause_generation` abort, version + 1, `/continue_generation`) and continued, and a
  request whose parent's cursor is evicted. Once with every hop carrying all ids (SGLang's ids dropped: today) and
  once with cursors; the full run cuts the interrupted reply's first segment where the real abort cut it (186 tokens),
  so both see the same segments. Every reply's prompt, output ids, logprobs and versions, every parent link and every
  exported concat leaf tensor are equal. Cursor run: 14 `/generate` bodies, 2 with all ids (the root; the resend after
  the 404), the resumed segment 0 ids; 459-662 bytes per cursor hop vs 18,967-26,943 with all ids at 3-5k tokens.

| request | prompt tokens | all ids: ids sent, bytes | cursor: ids sent, bytes |
|---|---|---|---|
| r0 | 3,131 | 3,131, 18,967 | full 3,131, 18,967 |
| r1 (cursor request) | 3,181 | 3,181, 19,251 | 26, 484 |
| r2 (full message list) | 3,234 | 3,234, 19,556 | 26, 484 |
| cut (8 tokens, length) | 3,287 | 3,287, 19,859 | 26, 482 |
| cut1 | 3,322 | 3,322, 20,065 | 27, 493 |
| fa, fb (fork of r2) | 3,308 | 3,308, 19,970 | 47, 593 |
| fa1 | 3,361 | 3,361, 20,275 | 26, 484 |
| rw (rewind to r1) | 3,253 | 3,253, 19,662 | 45, 590 |
| long (interrupted) | 3,340 | 3,340, 20,145; 3,526, 20,954 | 60, 662; 0, 335 |
| long1 | 4,852 | 4,852, 26,638 | 24, 472 |
| ev (parent's cursor evicted) | 4,905 | 4,905, 26,943 | 26, 459 → 404; full 4,905, 26,943 |

The two last columns are what went on the wire (ids, bytes). The server ran with `--disable-radix-cache`, so no row
(`rw` included: a rewind sends 45 ids from r1's cursor) is evidence that a rewind reuses cached KV.

Cost (`python tests/test_areal_cursor_hop.py`, node-134 CPU at load 29: the body of the next request, a 400-token reply
and one ~1.5k-token tool result after L tokens, and the engine's cost to build it: prefix check, payload, JSON
encoding as aiohttp sends it; median of 5):

| L (prompt tokens) | full request bytes | cursor request bytes | full build ms | cursor build ms | of which prefix check ms |
|---|---|---|---|---|---|
| 8,000 | 60,667 | 11,590 | 0.42 | 0.11 | 0.02 |
| 32,000 | 241,922 | 11,609 | 1.59 | 0.19 | 0.07 |
| 128,000 | 966,540 | 11,624 | 5.82 | 0.55 | 0.28 |

## Deployment (`CURSOR=<dir>` in the run's env file)

`scripts/rl_env/rl_run.sh` with `CURSOR` set runs the trainer on the AReaL overlay `$CURSOR/areal` (instead of
`AREAL_OVERLAY`), starts every SGLang server (here and, through `on_remote`, on each REMOTE host at the same paths) with
`PYTHONPATH=$CURSOR/sglang:$CODE`, and exports `OCTOPUS_CURSOR=1`, which the episodes inherit. Unset: nothing changes.
`scripts/rl_env/eval_serve.sh` takes the same `CURSOR` for eval
servers; dev_eval takes `OCTOPUS_CURSOR` with the rest of the run's episode settings (its `episode_settings.json`) and runs
training's proxy over the servers on the same AReaL overlay (`PYTHONPATH`), so eval prompts are training's concat ones.
Build the overlay on each host from a code dir `C` (an unpatched AReaL checkout and
the `SGLANG_PY` venv, whose package carries the LoRA patches and `apply_sglang_top_k.py`: AReaL's top_k=int(1e8)
is TOP_K_ALL, not a full-vocabulary sort per decode step; an overlay built before it takes the same script on
`$D/sglang/sglang`). `patches/apply_sglang_granite_norm.py`,
upstream's granite-4.0-h final-norm fix from SGLang v0.5.22, is not applied by any install script: for granite only,
run it once by hand on the venv, `$SGLANG_PY $C/patches/apply_sglang_granite_norm.py <the venv's sglang dir>`, before
copying it (on 0.5.22+ the script only says the fix is upstream and exits 0; 0.5.20/0.5.21 exit non-zero); runs
without `CURSOR` serve from the venv itself, so the same one run covers them:

    D=/home/user/octopus-sft/cursor-<sha>; mkdir -p $D/areal $D/sglang
    cp -r /mnt/disk1/user/areal/areal $D/areal/ && (cd $D/areal && for p in logprobs cursor route-spread cpu-batch train-rpc-timeout; do patch -p1 -s < $C/patches/areal-$p.patch || exit 1; done)
    cp -r $($SGLANG_PY -c 'import os, sglang; print(os.path.dirname(sglang.__file__))') $D/sglang/ && $SGLANG_PY $C/patches/apply_sglang_cursor.py $D/sglang/sglang && $SGLANG_PY $C/patches/apply_sglang_mount.py $D/sglang/sglang

The overlay holds no copy of CODE: its `srt/octopus_{delta_render,cursor_store,mount}.py` are shims
(`sys.modules[__name__] = octopus.serving.<module>`), so the servers run CODE's `octopus/serving/delta_render.py`,
`cursor_store.py` and `kv_mount.py`, the files the AReaL proxy and the trainer import (K2 positions, terminators, the
mount). A change there reaches a server at its next restart without rebuilding the overlay; a server without CODE on
PYTHONPATH fails at startup (`No module named 'octopus'`). A remote rollout host needs `$D/sglang` and CODE (a code
dir of the same commit) at the trainer's paths. `rl_run.sh` serve refuses an overlay that still has copies (built
before the shims, or with only one patch re-run) and a REMOTE host whose CODE's three files differ from the trainer's
(md5, `CODE_SUM` from `on_remote`); `eval_serve.sh` refuses the copies. The route patch (`OCTOPUS_ROUTE_SERVERS`, which `rl_run.sh`
start sets to every server it started) and the mount patch change nothing without that variable or until a request carries `mount` (the read tool, `docs/cross-branch-read.md`). The
route patch applies after the cursor patch only (e13's overlay was built from its #135 version, without cursor). It
also ends a session's tree in the router at `/rl/end_session` and clears a reused session id's end at
`/rl/start_session` (the active-tree count, `docs/search-env.md` "Admission"): an overlay built before #351 never frees a
tree's slot. Check the AReaL overlay with
`PYTHONPATH=$D/areal:$C $AREAL_PY $C/patches/check_areal_route.py`, which checks both. The cpu-batch and
train-rpc-timeout patches change only the trainer (host-staged batches; one attempt and a 24 h timeout per train-step
call: `docs/trainer-memory.md`): dev_eval does not need them and runs on an overlay with or without them.
`OCTOPUS_CURSOR_AUDIT=<dir>` (exported in the env file) turns on the acceptance audit below; it is off in production.

## Live acceptance (`scripts/rl_env/rl_run.cursor-pilot.env`, see commit a6e83a9)

Audit (`octopus/serving/cursor_audit.py`, called from `patches/areal-cursor.patch` only under `OCTOPUS_CURSOR_AUDIT`): for
every chat request the proxy logs the prompt length L and its build time; for a cursor request it also runs AReaL's own
full-mode concat (`_concat_prompt_token_ids_with_parent` on the full message list: render + tokenize of the whole
conversation), compares the ids with the prompt actually built, and logs the delta's and the full history's message
bytes. For every `/generate` segment the engine logs the bytes sent, the bytes of the all-ids body and why a hop sent all
ids. `scripts/rl_analysis/cursor_audit.py AUDIT_DIR JOBS_DIR` joins them with the episodes' requests.jsonl (the bytes the
client actually sent, the thread) and prints the tables below; a request's kind comes from its cursor: the thread's
previous reply (turn), another thread's reply (fork child), an older reply of its thread (rewind), or the control.
Client full-mode bytes = the bytes sent − the delta's message bytes + the full history's (JSON as the proxy holds it).
The tests' CPU runs with the audit on (tests/test_areal_cursor*.py, node-94, 14 passed): 163 cursor prompts, 0 mismatches.

Pilot `cursor-pilot-10172042`, node-94 only (SGLang on GPUs 0-2, FSDP on 3-7), 2026-10-17 20:42-21:14, code-840788d,
overlay `/home/user/octopus-sft/cursor-840788d`; e11's flags, policy (Qwen3.5-4B + adapter-v0-4b-text-r64) and
bundle tasks, `train_dataset.batch_size=2` x G=8, STEPS=1. 56 episodes started (16 per batch, the rest off-policy
prefetch stopped at the end), 18 forking ones with a control; train step 1 done (58 sequences, n_total_tokens 220k),
then the watchdog stopped the run and released every GPU (nvidia-smi: 8 x 1 MiB, no compute apps). One group was
rejected by a harness PermissionError (the policy chmod-ed the container's /tmp while testing tldextract's read-only
cache), unrelated to the cursor path. e11 has no OCTOPUS_REWIND: rewinds ran live in slice 7 (below).

- Lossless: 7,643 cursor requests, **0 mismatches** with the full-mode oracle (turn 6,223, control 1,230, fork child
  183, 7 without a client row: in flight at the stop). Max L 72,509.
- Full sends from the client: 74, every one a thread's first request (56 mains, 18 controls); none later (no
  rewritten replies in these episodes). Proxy → SGLang: 7,717 requests, 7,643 from a cursor, 74 with all ids, all "no
  parent" (exactly those 74); 6 segments resumed after a weight-update pause, all from a cursor; 0 cursor 404s at the
  proxy, 0 at SGLang (no resends).
- Totals: client → proxy message bytes 7.45 MB vs 711 MB in full mode; proxy → SGLang 16.4 MB vs 1,033 MB with all
  ids; proxy prompt build 13.1 s vs 317.7 s of full render + tokenize.

Per cursor request, medians (the live proxy, under the run's load):

| L | n | Δ tokens | client→proxy bytes, cursor | full | prompt build ms, cursor | full render+tokenize | proxy→SGLang bytes, cursor | all ids |
|---|---|---|---|---|---|---|---|---|
| <8k | 725 | 167 | 7,808 | 23,984 | 1.17 | 9.8 | 1,304 | 30,597 |
| 8-32k | 4,846 | 106 | 7,591 | 76,942 | 1.36 | 31.9 | 953 | 105,108 |
| 32-128k | 2,072 | 100 | 7,545 | 168,537 | 1.78 | 68.7 | 915 | 229,787 |

Load (the slice-5 routing watch item): requests 3,669 / 2,097 / 1,951 and prompt tokens 80.6M / 53.8M / 55.7M on
servers 30200 / 30201 / 30202. Routing by parent does not cause it: AReaL's controller (`octopus/rl/train.py`) pins rollout worker k to
server k (`sglang_targets.json`), so every request of a worker went to one server before slice 5 as well; the spread is AReaL's
dispatch of episodes to workers (36 / 20 / 18 thread roots, i.e. episode mains and controls, per worker).

### Slice 7: a remote rollout host, rewinds, eval, history past 128k

The four paths slice 6 did not run, each live on 2026-10-18 (CST) with code 94013c9 and the overlay
`/home/user/octo-env/cursor-94013c9` built on each host as in "Deployment"; every GPU released afterwards (nvidia-smi:
no compute app of ours, ports and tunnel closed).

**Remote host + rewind** (`scripts/rl_env/rl_run.cursor-remote.env`, see commit a6e83a9): trial `cursor-remote-10180140`, 01:41-02:31,
trainer on node-134 (SGLang on GPU 0, FSDP on GPU 7) and REMOTE node-183 (SGLang on GPU 6, reverse tunnel `rl_run.sh
tunnel`: its sshd on node-134's 127.0.0.1:2241, its server on 127.0.0.1:30950). `CURSOR` reached node-183's `serve` through
`on_remote` (its server ran on `$CURSOR/sglang`), e11's flags and policy on 46 e11 bundles, `OCTOPUS_REWIND=2`, the audit
on; 66 episodes. No train step: plan grading rejects every group with a rewound thread (`assign_threads`: "thread main/control
split over several sequences", 11 groups: 5 main, 6 control; a rewound thread is two concat leaves in full mode too) or a rewind past a
running fork's reply (`fork_record`: "max() iterable argument is empty", 2 groups), so the run was stopped by hand once the
audit had the rewinds. Both are OCTOPUS_REWIND limitations of plan RL, independent of the cursor (the messages are the
same in both modes); slice 6 already showed training on cursor data. Both are fixed since (#122, 3725df2): a rewound
thread is several leaves, each trained once (`assign_threads`), and a fork is recorded from its own history.

- Lossless: 8,717 cursor requests, **0 mismatches** (turn 6,638, control 1,799, fork child 236, **rewind 41**, 3 without a
  client row). Max L 68,477.
- Full sends from the client: 101: each thread's first request (66 mains, 30 controls) and 5 control requests after a
  rewind back into the control's replayed history (no kept reply left before it: a full send by design).
- Proxy → SGLang: 8,818 requests, 8,717 from a cursor, 101 with all ids (all "no parent"), 0 cursor 404s at the proxy
  or SGLang. Per server: local 30200 4,178 requests (4,130 from a cursor), 10.1 MB sent vs 490.6 MB with all ids; **remote
  node-183 30950 4,640 requests (4,587 from a cursor), 10.7 MB vs 486.7 MB**; node-183's log: 4,640 `/generate` 200s, no
  404, no traceback.
- Totals: client → proxy message bytes 11.5 MB vs 662.3 MB in full mode; proxy prompt build 20.7 s vs 435.9 s of full
  render + tokenize. Per-request medians as in slice 6: 7.5-7.8 kB client bodies, 1.5-2.0 ms builds, ~1 kB hops at every L.

**Eval** (`eval_serve.sh` with `CURSOR`): node-183 GPU 3, adapter-v0-4b-text-r64 as `v0`, port 30960 tunnelled to node-134,
where `octopus.eval.dev_eval --k 2 --control` ran 4 e11 bundles (pdfminer.six b00, pyopenssl b00, flake8 b01, patsy b00)
with `OCTOPUS_CURSOR=1` and e11's eval flags, 01:55-02:16. 8 episodes, 0 infra failures (1 ended done, 7 on the call
budget; solved fraction 0.112, one forked). 1,008 requests (main 844, branch 81, control 83) straight to SGLang's chat
cursor, all HTTP 200 (no 404). Main threads: 6.9 MB sent vs 86.8 MB in full mode (the same bodies with each request's
history from tree.json); after a thread's first request median 7.7 kB vs 101 kB, max 12.7 kB vs 247 kB, at a median
prompt of 26k tokens (max 63k).

**History past 128k** (`tests/test_areal_cursor_hop.py::test_live_history_past_128k`): node-183 GPU 3, one SGLang
(Qwen3.5-4B, the RL server flags without LoRA, `--disable-radix-cache`), the proxy on `RemoteSGLangEngine` with the audit
on, 01:37-01:42 (296 s). One thread, each turn its results + a ~10k-token log note, 16 cursor requests to L = 168,705:
every prompt equals AReaL's full-mode concat, every hop after the root from a cursor.

| L | Δ tokens | client messages bytes, cursor | full | prompt build ms, cursor | full render+tokenize | /generate bytes, cursor | all ids |
|---|---|---|---|---|---|---|---|
| 13,456 | 10,301 | 18,461 | 27,584 | 11.3 | 15.5 | 45,113 | 63,880 |
| 44,501 | 10,314 | 18,525 | 83,680 | 10.5 | 43.6 | 45,166 | 198,942 |
| 85,869 | 10,314 | 18,529 | 158,646 | 12.4 | 87.2 | 45,183 | 378,895 |
| 127,277 | 10,344 | 18,549 | 233,544 | 10.9 | 140.8 | 45,293 | 559,033 |
| 148,001 | 10,341 | 18,567 | 271,013 | 11.9 | 219.3 | 45,283 | 649,177 |
| 168,705 | 10,314 | 18,516 | 308,425 | 11.2 | 257.2 | 45,161 | 739,231 |

The cursor columns track the 10k-token delta (render + tokenize of it), not L.

## Mount: attention-only read of other cursors (`patches/apply_sglang_mount.py`)

Cross-branch read, slice 2 (SGLang side). Branch B reads sibling A by mounting a token range of A's cursor into B's
8 full-attention layers (one request may mount several ranges, of several cursors); B's 24 GDN layers keep recurring over B's own tokens (Qwen3.5; a Mamba-2 hybrid's
Mamba-2 layers likewise, granite-4.0-h-350m in the live results below; a model with only
full-attention layers, dense or MoE, reads the mount in every layer; any model reads it only in the layers
`OCTOPUS_MOUNT_LAYERS` names when that is set: "Layers" below). Positions follow K2, the rule chosen by the
pre-registered offline probe (`docs/kv-read-probe.md`, run 2).

- Request (`/generate`, streaming or not): `{"cursor": B, "input_ids": delta, "mount": [{"cursor": A1, "start": s1,
  "end": e1}, {"cursor": A2, ...}, ...], ...}`: one or more segments, from one or several cursors (B's own earlier
  cursors included), `delta` not empty, `0 <= s < e <= len(tokens(A))` for each. Without `mount`: today's request.
  Errors: 400 (mount without a cursor, an empty list, a malformed entry, a range outside its cursor,
  `logprob_start_len` inside B or the mount (those rows run no logits), a server without the config below), 404 (an
  unknown mount cursor, as for the cursor).
- Chat (`/v1/chat/completions`, streaming or not; for an agent pointed at SGLang directly: evaluation no longer sends
  here, dev_eval runs training's proxy, `octopus/rl/proxy.py`; an OpenAI SDK client sends `cursor` and `mount` in
  `extra_body`): a cursor request
  takes read_thread's `"mount": {"from": id, "to": id, "thread": name}` as the AReaL proxy does and runs as the
  `/generate` mount above: A = "to", s / e = the start of the generation prompt (`delta_render.turn_start`) in
  "from"'s / "to"'s stored tokens, e = len(tokens("to")) with `"ended": true` (the source thread's last reply; "from"
  may then be "to"): `delta_render.read_range`, the proxy's `_read_range` range. One path for both endpoints: the chat cursor branch renders only
  the delta (`render_delta`), `octopus_mount.prepare_chat` turns the read into that segment and calls `prepare` (the
  cursor's tokens, the placeholders, the positions), and both chat handlers run `octopus_mount.chat`, which is
  `generate` (rebuilds included); tool-call and reasoning parsing stay SGLang's. A request from a cursor that holds a
  mount continues it (placeholders and offset kept); the reply is stored with its breakpoints. `prompt_tokens` leaves
  out the placeholders, as through the proxy; `metadata.octopus_mount` = M (non-streaming) confirms the mount, and
  the agent (`tree`) fails a mount request whose SGLang response (it carries `metadata`) lacks it (row
  `http_code` "unmounted": dev_eval's infra failure, not scored), as the AReaL engine fails a mount its server ignored
  (the proxy's responses carry no `metadata`). Errors: 400 (a mount without a cursor, "from" not before "to" in one
  thread), 404 ("from" or "to" not stored), and the `/generate` ones. Before #191, chat ignored `mount` (the field did
  not exist): every eval read with `OCTOPUS_CURSOR=1` was reported "Mounted" but ran without the mounted tokens
  (`results/e14-read/report.md`, caveat).
- Streaming (both endpoints): the scheduler aborts an attempt for missing rows before it runs anything, so the
  rebuild and the retry (below) come before the request's first output, and a stream carries only the attempt that
  runs: the same text, tokens and log-probs as the non-streaming request. Each attempt gets a new rid unless the
  client set one, which every attempt keeps (the response id is the cursor key); the request carries its attempt's
  rid from the start, so a stream whose client leaves aborts the attempt that runs, and an aborted stream attempt is
  closed before the retry. A running request that SGLang retracts
  under memory pressure is re-matched, and if a segment was evicted meanwhile it aborts after some of its output was
  sent: a non-streaming request is then sent again whole (the client sees only the retry), a stream ends with that
  abort as an error (SGLang's own stream error for a 400 abort), never with text of two attempts. A streamed reply is
  a cursor like any other (with its breakpoints), so a streaming-only client continues it and mounts its range.
- Semantics: the request runs as the sequence `tokens(B) + M1 placeholders + M2 placeholders + ... + delta` (Mi = ei -
  si). Its prefix rows are B's KV (radix cache) and then, in request order, each segment's KV pages for its cursor's
  tokens [si, ei) (radix cache, matched by those tokens up to ei: the KV computed in that cursor's context). Every token
  the request computes (delta in prefill, every decode step) attends to B + every segment + its own predecessors in the
  attention layers; the GDN layers start from B's state at B's last token and never run the mounted tokens. The mount
  stays part of B: the reply's cursor is `tokens(B) + placeholders + delta + output`, and later requests from it (with
  or without a new mount) keep attending to the mounted rows. So the mount is an attention-only segment inserted into
  B's context after `tokens(B)`, not a one-request view.
- Transitive: a range of a cursor that itself continues a mount may hold that cursor's placeholders (C mounts a range
  of A that holds A's read of X). Its rows are A's rows as A computed them, which include X's rows at the
  placeholders, so C attends to A's tokens of the range and to the X segment A read there. A range starting or ending
  at a placeholder run leaves it out (the placeholders sit before the token at that index; `start`/`end` count them).
- Positions (K2): mounted keys keep their cursor's positions (no copy at mount time, no rotation). The request's tokens
  continue at `max(p_B, last mounted position + 1)` over all segments, p_B = B's next position, a segment's last
  mounted position being its cursor's position of token `e - 1` (its own offset included); with B and the cursors free
  of earlier mounts that is `max(len(B), e1, e2, ...)`. A transitive segment ends on a token of its own cursor, whose
  position is already past the rows it read. The cursor store keeps per cursor its position breakpoints `((index,
  offset, mount), ...)` (token i at `i + offset` of the last breakpoint at or before i): the mount request's cursor
  gets B's breakpoints + `(len(B) + M1 + M2 + ..., offset, mount)`, and every later request from B's cursors carries the
  last offset to the scheduler (`octopus_pos` in `custom_params`), which adds it to the request's positions (and M-RoPE
  positions) in prefill and decode.
- Layers (`OCTOPUS_MOUNT_LAYERS`, one setting for inference and training): unset, every attention layer reads the
  mounted rows (the above; on Qwen3.5 the 8 full-attention layers, the GDN layers are not attention layers). A
  comma-separated list of decoder layer ids (e.g. `3,7,11,...,35`: every 4th of a 36-layer transformer) mounts into
  those attention layers only; every other attention layer attends to the request's own rows alone (B's, its own
  delta and outputs; the mounted rows invisible), positions unchanged (K2). The server reads it at launch (scheduler
  process env; ids that are not attention layers of the model stop the start) and reports it
  (`/get_server_info` `internal_states[].octopus_mount_layers`, null for all); the trainer's prefix tree reads the same
  variable (`octopus/rl/prefix_tree.py`: the layers outside the list attend to each node's own path), and
  `octopus/rl/train.py` refuses to start when any server in `rl.servers` reports another value; `rl_run.sh` passes
  the trainer's value to the REMOTE servers. Inference, triton backend only (a subset with another attention backend
  is a 400 at `serve`, as is a subset with `--enable-deterministic-inference` or `--dcp-size` > 1):
  `ForwardBatch.octopus_hidden` (a dataclass field: the runners pass `dataclasses.replace` copies) holds each request's placeholder rows when a batch has any, `TritonAttnBackend.init_forward_metadata` builds
  a second KV view without them (`layer_view`: the prefix rows in extend, all rows in decode) and `forward_extend` /
  `forward_decode` hand it to the layers outside the list (`layer_kv`); such a batch runs eager (no prefill or decode
  CUDA graph: the second view has no graph buffers). Sliding-window models stay refused: a window over a row with
  placeholders counts them, and SWA pages outside a source's window are freed, so its mounted rows would be gone in
  those layers anyway.
- Placeholders: one negative id per segment, unique for the server's life. They never reach the model (they sit in the
  prefix); they make B's continuation a radix-cache path no other request can match, and in the tree they mark the
  entries that hold another path's pages (below). The response's `prompt_tokens` counts them (the engine hop adds the
  M of every mount its cursor holds, see "AReaL engine hop").
- Lifecycle (scheduler, zero-copy): after the request's prefix match (`Req.init_next_round_input`), if B's KV reaches
  exactly `len(tokens(B))` every segment's pages are appended to the prefix, all or none (a queued request is re-matched
  every round and keeps no match from an earlier round); the tree owns them (`cache_protected_len` covers them), so the
  request never frees them. Admission's budget check (`PrefillAdder.add_one_req`, which runs with the request's prefix
  locked) runs with the segments' paths locked too, so their pages never count as evictable for it; when the scheduler
  takes the request (`PrefillAdder._req_inc_lock_ref`) each segment's path is locked (FULL pages only; its mamba state
  is not read). While the row holds them nothing of it is inserted into the tree (`cache_unfinished_req`). At the end
  (`cache_finished_req`, which finish, abort and retraction all pass) the row is cached as one path as SGLang caches
  any row, and every segment is unlocked: nothing stays locked once a request ends. Its entries at the placeholders
  are the segments' own pages, now shared (`lend`): each such page counts one more holder (a scheduler-side map from
  page slot to extra holders), and the source nodes holding those rows are marked (the mark survives node splits:
  `redistribute_on_node_split`). An evicted node that is marked or has placeholder keys frees only the pages no
  other node holds and takes one holder off the rest (`FullComponent.evict_component` → `owned`; one device read of
  that node's slots), so a page is freed exactly once, by whichever of its holders leaves the tree last, in any
  order: a source may go before its continuation, a continuation of a continuation (transitive) holds three nodes'
  pages, and LoRA versions (SGLang's `extra_key` namespaces, where a version switch rebuilds a mounted cursor with the
  same placeholders, review of #185) are independent. The tree counts a shared page in every node holding it, so
  each added holder is taken off the evictable count once and an evicted node counts only what it frees: available +
  evictable + protected stays equal to the pool, at idle the whole pool is admissible (no locks), and while a holder
  is locked by a running request evictable is lower by its shared pages (the admission budget is never over-stated).
  Nothing is copied and no page is allocated for mounted rows. A range may not end inside a mount of its source's or
  between two of them (400): that would split the source's placeholder runs in the tree.
- Not resident: B's KV short of its last token (always after a generated reply: the last output token has no KV yet)
  or holding placeholders that are gone, a segment's pages gone, or a later request from B's cursors whose prefill
  would run placeholders (the cache kept the continuation only in part, or the path was evicted since). The scheduler
  aborts the request before it runs anything (the abort message names which) and the HTTP side rebuilds what is
  missing, then sends it again: each cursor breakpoint keeps the mount that made it (where it sits, each segment's
  start, its cursor's tokens up to its end and that cursor's breakpoints; shared by the thread's later cursors), so the
  tokens before the request's last mount and every segment's cursor tokens are prefilled together (`max_new_tokens:
  0`, same LoRA; they snapshot the recurrent state at their end: "Config"): the ones without a mount as one batch
  request, so the scheduler receives them together (one message) and splits them into prefill batches by their
  lengths and the free pool (chunked-prefill and max-prefill budgets), no longer by arrival timing: on an idle server
  the same rebuild gets the same batches (sent one by one, overlap off took B and A in one batch or in two by arrival,
  other prefill numerics and so another greedy reply for the same request on either endpoint; see the chat evidence
  below). Under load other requests share those batches, and with `--enable-dp-attention` SGLang sends a batch's items
  one by one. One holding a mount (a second read in the same thread, a read from a continuation of a read, a
  transitive read) goes as a request that carries its own last mount, still on its own, its batch by arrival; it
  rebuilds what it
  needs the same way (a source holding a read of its own is rebuilt with that read); a request from a mounted cursor
  goes again carrying its last mount, so the scheduler re-mounts the segments' pages and recomputes the rows after
  them. A rebuild has the same tokens, positions and mounts as the first computation; its numerics are within bf16
  prefill-split precision of the resident rows (prefill in other splits, and prefill instead of decode for earlier
  outputs, as for any re-prefilled cursor: rebuilt vs resident 0.0075 / 0.0033 / 0.0065 below). There is no retry cap and no 503: a retry fails again
  only if what was just rebuilt (the most recently used path) is evicted before the retry is matched; rebuilds stop
  when the client has disconnected (an agent timeout or abort) or the request is aborted (`/abort_request` by rid or
  all, e.g. AReaL's `pause_generation` at a weight update; SGLang alone would drop it, the request is in no
  `rid_to_state` between attempts), which then answers as SGLang does for a request aborted in its waiting queue
  (`finish_reason` abort, no output). When the pool cannot hold B, A and the request at
  once, it waits, rebuilding, until running requests free memory (the waits cited here are the pre-#185 copy-based
  run, not this zero-copy lend path: slice 6, e13, no_buffer, overlap off, 8-16-task bundles, trees about 95-141k
  tokens; none at e13's config, up to 17 minutes at `--mem-fraction-static 0.7`, docs/cross-branch-read.md
  "Memory pressure").
- Config: radix cache on and page size 1 (the mount splices the segments' rows at token granularity); any mamba strategy and
  overlap scheduling on or off, so a mount server runs today's RL flags (auto: extra_buffer + overlap on a GDN model).
  A recurrent state is needed at B's last token only, and a rebuild provides it on demand: a prefill-only request
  (`max_new_tokens: 0`, every rebuild) snapshots its live state at its end (`cache_finished_req`: copied into the
  request's next track slot and tracked at its KV length, so SGLang caches the whole sequence with that state).
  no_buffer caches a state at every sequence end anyway; extra_buffer keeps a finished request only up to its last
  tracked state (state-chunk boundaries in prefill: 64 tokens for GDN, 256 for mamba2; every
  `mamba_track_interval` = 256 tokens in decode) and frees the KV past it, so on extra_buffer a mount usually starts
  with the B/A rebuild (B's tail past that boundary, A's when e lies past A's), as after any generated reply on
  no_buffer. The snapshot needs the live state at the end, which overlap would step one token further: the request
  finishes after its prefill, but that is known only once the batch's results are processed, after the next decode
  batch (which would carry it once more) is launched; so prefill-only requests stay out of that decode batch
  (`Scheduler.get_next_batch_to_run`, as SGLang does for all-prefill-only batches). Ordering under overlap is SGLang's
  own: every batch is launched after the scheduler stream's work (`forward_stream.wait_stream(schedule_stream)`), and
  result processing, which caches the row and lends or unlocks the segments on the scheduler stream, is fenced behind
  the batch in flight's reads of the shared buffers (`_apply_war_barrier`); no KV is written at the end of a request
  any more, and a shared page is freed only when the last node holding it is evicted. Every
  model kind in scope uses `UnifiedRadixCache` (FULL component, plus MAMBA for a recurrent state; its locks skip it);
  the zero-copy path moves no KV, so it is the same for MHA/GQA, MLA and fp4/fp8 pools. Positions: the offset goes to the 1-D positions and, on an M-RoPE model, to the M-RoPE positions, in
  prefill and decode; CUDA-graph decode copies both into its buffers. Each scheduler decides at startup
  (`Scheduler` init, `serve`) whether it can mount, from its own objects: a model runner with an attention window (any
  model that has one: hybrid or pure SWA, chunked attention) or an attention layer with its own window (models that set
  it only per layer; the window hides mounted rows and the cache frees SWA pages outside it) or a radix cache other than `UnifiedRadixCache` (LMCache, FlexKV, the C++ tree, a custom backend)
  cannot, and neither can a server with features whose own cache paths shared pages are not checked against
  (`--enable-hierarchical-cache`: host copies and load-back; `--enable-streaming-session`,
  `--enable-session-radix-cache`: their own insert and lock paths; `--speculative-algorithm`: draft pools and EAGLE's
  bigram keys; `--enable-unified-memory`, `--enable-hisparse`: other pools and allocators; `--disaggregation-mode`: KV
  transfer). A mount request there aborts before any match with the reason,
  which `/generate` answers with 400.
  Every launch path (data parallel, several tokenizer workers) runs that scheduler init.

SGLang internals touched (`# octopus-mount`): `Req.init_next_round_input` (mount after the prefix match, abort if it
cannot), `PrefillAdder.add_one_req` (the segments locked for the budget check), `PrefillAdder._req_inc_lock_ref` (lock
them), `UnifiedRadixCache.cache_finished_req` (at its start a prefill-only request's state snapshot; after the insert
`lend`) and `cache_unfinished_req` (no insert while the row holds the segments' pages), `FullComponent.evict_component`
(`owned`: an evicted node frees and counts only the pages no other node holds) and `redistribute_on_node_split` (the
shared mark), `ForwardBatch.init_new` (position offset; under a layer subset each request's mounted rows, the
`octopus_hidden` field), `TritonAttnBackend.init_forward_metadata` / `forward_extend` / `forward_decode` (the second KV
view), `ModelRunner._forward_raw` (a batch with that view runs eager), `Scheduler` init (`serve`: can it mount),
`Scheduler.get_internal_state` (`stats`: `octopus_mount_layers`, the layer setting; `octopus_full_protected`, the locked token count, 0 when idle;
`octopus_shared_holds`, the extra holders of shared pages; `octopus_available`, free slots: leak and page accounting
checks), the
`mount` fields of `GenerateReqInput` and `ChatCompletionRequest`, the `/generate` handler and `OpenAIServingChat` (the
cursor branch gives the delta; `prepare_chat`, then `chat` (`generate`) in both handlers; the reply stored with its
breakpoints). The request's mount data reaches the scheduler in
`sampling_params.custom_params` (`octopus_*` keys, plain JSON), so the tokenizer manager and the request structs are
unchanged. Logic: `octopus/serving/kv_mount.py` (imported through the shim `srt/octopus_mount.py`). Applies after
`apply_sglang_cursor.py`, in either order with `apply_sglang_lora_abort_release.py`.

Evidence (`tests/test_sglang_mount.py`):

- CPU (node-134 `sglang-venv` 0.5.19; the needles also apply to node-94's `sglang-4090d` copy): the patch refuses to run
  before the cursor patch, applies once in every order with the cursor and lora patches, each second run a no-op, and
  compiles; K2 positions for peer behind / ahead, breakpoints kept by the next cursor, a mount of a cursor with its own
  offset; every 400/404 case; the `/generate` handler on a stub tokenizer manager: B + placeholders + delta with the
  mount in `custom_params`, the not-resident aborts answered by a B prefill then an A prefill (prefill-only, with
  their offsets) and a retry, the stored breakpoints (with their mounts), a continuation of the mount reply whose rows
  are gone rebuilt (again carrying the mount, B and A prefilled), a mount from it whose prefix lost its mounted rows
  (that prefix rebuilt with its own mount first), a two-segment request (one segment B1's own rows after its
  placeholders) rebuilding every source, a plain request unchanged; A locked for the budget check
  (`test_check_lock_holds_the_mounted_path_for_the_budget_check`); segments mounted all or none, the tree owning their
  rows (`test_failed_rematch_leaves_no_stale_mount_hit`); zero copy on SGLang's own `UnifiedRadixCache` (FULL, a CPU
  allocator, `test_zero_copy_shares_mounted_pages_and_frees_each_once`): a two-segment continuation allocates
  slots for its own tokens only, a transitive one (C mounting B's range over B's placeholders) borrows A's, X's and
  B's slots, nothing stays locked once a request ends (idle: available + evictable = pool), a rebuild re-mounting
  into its old path adds no holder, eviction frees each of the 33 slots exactly once and never a slot a node still
  holds, available + evictable + protected = total after every step; the cursor store drops
  breakpoints with the cursor. The existing `tests/test_sglang_cursor.py` passes unchanged on the overlay with both
  patches.
- Live (node-134 GPU 4, `sglang-venv` + lora/cursor/mount overlay, Qwen3.5-4B-text base, triton attention, radix
  cache, no_buffer, decode CUDA graphs on, greedy): the #129 run-2 mount points (`prep` in the test rebuilds them on
  node-94 from `run2/results-base.jsonl`: the first 20 peer-ahead and 20 peer-behind in id order, 37 with a scored
  probe; p_B 9.6k-51k tokens, M 88-3,056). Per mount point B and A are prefill-only cursors, then B's probe
  continuation is sent with the mount; compared with #129's offline K2 (transformers, flash-attn 2 + fla, bf16):

| | n | mean abs d logprob (answer tokens) | max | argmax agreement (offline greedy steps) | greedy 16 tokens equal |
|---|---|---|---|---|---|
| mount (K2) vs offline K2 | 37 mount points, 94 probes, 325 tokens | 0.020 | 0.67 | 0.987 (1,504 steps) | 0.957 |
| same, after the review fixes | first 10 points, 30 probes, 110 tokens | 0.015 | | 0.988 (480 steps) | 0.967 |
| no mount vs offline none (kernel baseline) | 37 points | 0.050 | 0.42 | 0.983 (1,504 steps) | |

  Tolerance: mean |d| <= 0.03 and argmax agreement >= 0.98 against the offline K2. Revised in review (the first
  version asserted max(0.02, no-mount baseline) after seeing 0.0201): the reviewer measured 0.0139 with 128-token
  chunks and found the per-point error uncorrelated with mount length or peer distance, so 0.03 is a fixed bar with
  room for prefill-split numerics (the same no-mount tokens prefilled in other chunks move by 0.01-0.41 nats on
  low-probability answers).
- Eviction path (first 5 mount points): `/flush_cache` evicts B and A, the same mounted request then aborts twice
  (B's KV, then A's pages), prefills both and gives log-probs bit-identical (max |d| 0.0, asserted <= 1e-3) to the
  resident run; the same with `--chunked-prefill-size 32` (3 points, 0.0).
- Stale match (review finding 1): with `--max-total-tokens 40000` and the admission estimate raised, a mount request
  queued beside a long decode (admission refuses it each round after matching B and A) finds B evicted by that decode
  and aborts; its one-token stand-in no longer carries the earlier round's match. The previous commit crashed the
  scheduler on this scenario (`pool memory leak detected`: the stand-in copied and leaked M slots); now the request
  recovers through the prefills (200) with 0 locked tokens after. Unit test: `test_failed_rematch_leaves_no_stale_mount_hit`.
- Continuation: from the mount reply's cursor, the next tokens' log-probs equal one mounted request over reply + next
  tokens (3 mount points, 7 tokens: mean |d| 0.021, max 0.069, same argmax; prefill-split numerics as above). After
  `/flush_cache` (slice 6, node-94 sglang-4090d) the rows are rebuilt: the continuation, a second mount from the mount
  reply (a prefix holding a mount) and a mount of its own rows after its placeholders (a source holding a mount) give
  the resident log-probs (74 teacher-forced tokens each: mean |d| 0.0075, 0.0033-0.0044, 0.0065 over 4 runs; same
  argmax), 0 locked tokens after.
- No mount unchanged: `test_live_chat_cursor_tree` and `test_live_generate_cursor_equals_full` pass on the patched
  server (cursor outputs and logprobs equal to full sends).
- Memory: after every live test (one run of both: ~330 mount requests, 30 back to back from one cursor) `octopus_full_protected` is 0
  at idle, SGLang's idle pool check reports no leak (available + evictable + protected = total), no traceback; the
  cursor store gains exactly one entry per client request (the internal prefills are not stored) and drops a cursor's
  breakpoints with it.
- Dense model (`test_live_dense_mount_equals_hf`; node-94 GPU 0 server / GPU 1 transformers, sglang-4090d + this overlay,
  Qwen3-4B-Base: 36 full-attention layers, no M-RoPE; triton, overlap off, no mamba flag, decode CUDA graphs on, every
  decode batch ran a graph; text: `tests/fixtures/mount_text.txt.gz`): peer ahead (offset +500) and peer behind (-600),
  B 1.5k/3k tokens, M 600, 64 answer tokens each, against transformers (bf16, sdpa) on the exact sequence (B's KV, A's
  rows [s, e) in A's own context, the request's tokens at the K2 positions). Mount 0.033 mean |d| (argmax 0.969),
  CUDA-graph decode 0.010 (1.0), rebuilt after `/flush_cache` 0.028 (0.984); the server's own error without a mount is
  0.034 (0.977), so the mount adds nothing measurable; the reference without the offset is 1.29 away (argmax 0.56). That
  kernel error is above the Qwen3.5 bar of 0.03 (0.039-0.046 on other texts), so the dense bar is the no-mount error +
  0.01 nats / - 0.02 argmax. Rebuilt rows are not bit-identical to the resident ones on this model: the rebuild
  prefills B and A in one batch (other prefill splits; the same tokens prefilled in one request instead of two differed
  by up to 0.22 in a run on another text). Repeated runs on one server give identical numbers. 0 locked tokens after.
- After this change, Qwen3.5 unchanged (node-136, sglang-venv + this overlay, `--chunked-prefill-size 4096
  --disable-prefill-cuda-graph`): the 37-point check gives the numbers above exactly (0.0201, max 0.67, argmax 0.987,
  greedy 0.957, no-mount baseline 0.050 / 0.983, eviction 0.0), continuation 0.021 / 0.069, rebuilt 0.0075 / 0.0044 /
  0.0065. Kinds covered by inspection only: MoE with MHA/GQA attention (same pool and cache as dense), MLA (pool's
  `move_kv_cache`), mamba hybrids (MAMBA component, `mambaish_config`; a Mamba-2 hybrid has
  since been checked live: granite-4.0-h-350m, `test_live_hybrid_mount_equals_hf`, below).
- Overlap scheduling and extra_buffer (node-136 `sglang-venv`, node-94 `sglang-4090d`, overlays built from the change;
  every decode batch ran a CUDA graph). Qwen3.5-4B-text, `--mamba-radix-cache-strategy extra_buffer`, overlap on: the
  37-point check gives mount vs offline K2 0.0187 mean |d| (max 0.67, argmax 0.985, greedy-16 0.926), no-mount
  baseline 0.049 / 0.982, eviction [0.0] x 5, continuation 0.041 / 0.092, rebuilt 0.0069 / 0.0070 / 0.0051, 0 locked;
  the numbers move from the no_buffer ones because extra_buffer caches other states, so later prefills split
  differently. The same server code with no_buffer, overlap off gives the recorded numbers exactly (0.0201, 0.666,
  0.9867, 0.957; eviction 0.0; continuation 0.021 / 0.069; rebuilt 0.0075 / 0.0033 / 0.0065). Dense Qwen3-4B-Base with
  overlap on (`test_live_dense_mount_equals_hf`): mount 0.033 / 0.969, decode 0.010 / 1.0, rebuilt 0.028 / 0.984,
  no-mount 0.034 / 0.977, as with overlap off.
- Second hybrid family, granite-4.0-h-350m (mamba2 + attention without positions; `test_live_hybrid_mount_equals_hf`,
  4 cases, 256 teacher-forced tokens): vs transformers fp32 (token-by-token after B's cache with A's rows spliced
  into its attention layers) mount 0.450, no mount 0.465, decode 0.234, rebuilt 0.453 mean |d|; B's continuation from
  its end state vs one prefill 0.054 (argmax 0.965). extra_buffer + overlap and no_buffer + overlap off give
  identical numbers (rebuilt 0.4530 vs 0.4539). The large base error is SGLang's granite against transformers' on
  any plain request (0.4 nats, 0.87 on the first token, not the mount; `apply_sglang_granite_norm.py` fixes it); argmax
  agreement (0.82 mount, 0.84 without)
  is printed, not asserted.
- Throughput (Qwen3.5-4B-text, rl_run.sh's flags without LoRA, `--mamba-full-memory-ratio 0.45`; 32 concurrent
  requests, 1,024 random prompt tokens, 512 sampled tokens with ignore_eos, second of two rounds): 1,623 tokens/s
  without the overlay, 1,622 with it (the same flags now serve mounts), 1,554 with the old mount flags (no_buffer,
  overlap off; -4.3%). At 64 concurrent requests extra_buffer's state slots (5 per request) cap the batch at 41
  running requests against no_buffer's 68 (3 per request): 1,335 vs 1,819 tokens/s, a property of the strategy, not
  of the mount.
- Several segments, zero copy, transitive reads (node-136, `sglang-venv` + an overlay built from this change;
  Qwen3.5-4B-text with the RL flags (extra_buffer, overlap on) and with no_buffer, overlap off). The 37-point check
  gives mount vs offline K2 0.0192 mean |d| (max 0.67, argmax 0.985, greedy-16 0.936), no-mount baseline 0.050 /
  0.983, eviction [0.0] x 5, continuation 0.041 / 0.092, rebuilt 0.0069 / 0.0070 / 0.0051 (extra_buffer; the recorded
  0.0187 / 0.985 moves by prefill splits: with no copies the pool evicts less); no_buffer gives the recorded numbers
  (0.0201, 0.666, 0.987, 0.957; eviction 0.0; continuation 0.021 / 0.069; rebuilt 0.0075 / 0.0033 / 0.0065).
  `test_live_multi_and_transitive_mounts_equal_offline_k2` (`prep-multi`: `kv_read_probe.multi_cases` on the 8 shortest
  mount points, references from `Probe.k2`, cache surgery with any number of segments, transitively; 395 scored
  question and answer tokens per kind, the same on both servers): one segment 0.057 (argmax 0.980), two segments 0.065
  (0.977), transitive 0.053 / 0.050 (0.985); after `/flush_cache` the request rebuilds B, A (with its own read of X)
  and X: 0.046 / 0.063 / 0.057 against the reference, and sent again on the rebuilt rows it repeats them exactly (0.0
  x 12). These question tokens are noisier than the probes' answers: the trainer's tree without a mount against a plain
  forward already differs by 0.038 on them. Page accounting: the source's own read (a prefill-only mount request)
  allocates exactly its own tokens (1,429 / 838 / 159 / 213) and lends all its mounted rows (1,941 / 2,360 / 703 /
  847: `octopus_shared_holds`), nothing is copied; at idle nothing is locked (`octopus_full_protected` 0 after every
  case and test); after `/flush_cache` nothing is locked or shared and the pool is whole (551,728 free); SGLang's idle pool check passed throughout. Dense Qwen3-4B-Base (node-94, overlap on): mount
  0.033 / 0.969, decode 0.010 / 1.0, rebuilt 0.028 / 0.984, no mount 0.034 / 0.977, as recorded. granite-4.0-h-350m
  (both strategies): mount 0.450, no mount 0.465, decode 0.234, rebuilt 0.454 / 0.455, snapshot 0.053, as recorded.
- Layer subset (`OCTOPUS_MOUNT_LAYERS`). Unset is today's behaviour: Qwen3.5-4B-text on node-136 gives main's (909811a)
  output line for line, 37-point check, continuation, rebuilt, multi-segment and transitive, with the RL flags
  (extra_buffer, overlap: 0.0192 / 0.985, continuation 0.042 / 0.092) and with no_buffer, overlap off (0.0201 / 0.987);
  granite (both strategies: mount, no mount, decode and snapshot the same; rebuilt 0.4539 / 0.4556, where main's own
  runs give 0.4521-0.4546: B and A were then rebuilt as separate concurrent requests, batched by arrival) and dense Qwen3-4B-Base (node-94, line for line: mount
  0.033 / 0.969, decode 0.010 / 1.0, rebuilt 0.028 / 0.984, no mount 0.034 / 0.977) as recorded. Every 4th layer of Qwen3-4B-Base (`3,7,...,35`; node-94, same flags,
  decode CUDA graphs on, overlap on): `test_live_dense_mount_equals_hf` against transformers with A's rows in those
  layers only (`dense_logits`, a mask per layer): mount 0.043 / 0.961, decode 0.014 / 0.969, rebuilt 0.040 / 0.961,
  no mount 0.034 / 0.977 (bar: + 0.01 / - 0.02); the reference with the mount in every layer is 1.01 away (argmax
  0.625), so the server reads the mount in those layers only. The trainer (`test_tree_dense_subset_mount_equals_hf`,
  areal venv, flash-attn 2, bf16) against the same reference: answer log-probs 0.038 (its own error without a mount
  0.033), gradient of their sum per input embedding, relative error B 0.15, answer 0.09, A 0.13 (without a mount B 0.26,
  answer 0.15); against the every-layer reference 1.01 and 2.95 / 2.41 / 1.21. `train.check_mount_layers` passes on
  equal settings and exits naming both values on a server with the subset and a trainer without it, and the
  reverse. 0 locked tokens after every test. Without a subset, A's bf16 gradient in the peer-behind case is
  ill-conditioned, not wrong: equivalent bf16 computations of the same function scatter along one direction on a few A
  tokens (their errors |cos| ≥ 0.96). Relative error vs fp32: tree (flash-attn 2) 1.47 (norm 41 vs 20), dense
  reference with rows A, B, x 0.99, per-node memory-efficient sdpa on the tree 0.96, fp32-internal math attention
  0.79, the original reference 0.54 on the opposite side (norm 19). The fp32 tree and reference differ by 4e-4 on A
  (1e-3 on its first mounted tokens) vs ≤ 6e-5 elsewhere: the same amplification at fp32 rounding. Over 8 cases the
  tree is not systematically worse (a2: 0.53 vs 0.26–1.90). Not a tree defect; an A-gradient bar needs the spread of
  equivalent bf16 algorithms or an fp32 reference, so this check runs under a subset.
- Chat and streaming (node-94 `sglang-4090d` + an overlay built from the change, Qwen3.5-4B-text, extra_buffer,
  overlap on, `--chunked-prefill-size 4096`; `test_live_chat_mount_equals_generate_and_streams`): a chat request with
  read_thread's mount (branch B reading branch A's turns) gives /generate's output ids and log-probs on the same
  sequence (the segment the chat mount names; max |d| 0.0), resident and after `/flush_cache` (both rebuild); a chat
  continuation of its reply gives them exactly after a flush (0.0) and, resident, the same ids at mean |d| 0.013 (the
  second request matches the first's cached path deeper and prefills from another split). On the same cache state as the non-streamed reply, streamed chat gives the same text
  and log-probs (SGLang's chat stream has no chunk, so no log-prob, for the stop token, as for a request without a
  cursor) and streamed /generate the same output ids and log-probs: the resident mount read and, after a flush, the
  mount read and the continuation (each rebuilding before its first chunk); 0 locked tokens after.
  On no_buffer + overlap off (e13/e14's server flags; node-136 `sglang-venv`, an overlay from 951b409, chunked 4096)
  the same test failed 2 of 3 runs before the batched rebuild: after `/flush_cache` chat and /generate differed from
  the first output token (log-prob -0.166874 vs -0.166513) and greedy tokens diverged later. Not the endpoint: 30
  requests each from an empty cache, chat and /generate alternating, gave two results, by how the scheduler took B's
  (1,132 tokens) and A's (1,146) rebuilds: in two prefill batches (chat 14, /generate 9; first log-prob -0.161646) or
  in one (chat 1, /generate 6; -0.176518). extra_buffer + overlap on took them in one batch 30 of 30. With the rebuilds
  in one message: one batch 30 of 30 on both configs, the same first log-prob and 35 tokens on both (-0.195133), and
  the test passes 4 of 4 runs on no_buffer and 2 of 2 on extra_buffer. -0.195133 vs the earlier one-batch -0.176518 is
  another episode, not the fix: the episode's earlier greedy replies (same lengths: a 1,232-token read prompt in every
  run) depend on what
  earlier runs left in the cache, as the resident mount read (no rebuild) shows across runs before the fix (first
  log-prob -0.167123 / -0.166513 / -0.171836); with the fix both values occur (test runs 1 and 2-3 on no_buffer, each
  with chat = /generate). The other live mount tests pass on both configs: eviction 0.0 (the first request after a
  flush vs its repeat: self-consistency), rebuilt vs resident 0.0075 / 0.0033 / 0.0065 on no_buffer, 0.0116 / 0.0031 /
  0.0059 on extra_buffer (prefill-split numerics; the second and third cases rebuild a sequence holding a mount, which
  still goes on its own).
  `test_live_openai_sdk_fork_and_read`: the openai package (3.19.1) drives a main turn, a fork of two branches, two
  more steps on A and B's read of A's turns (`extra_body` cursor and `{"from", "to", "thread"}` mount: 130 tokens
  mounted, confirmed in `metadata`) through chat only, and the streamed continuation of the read's reply equals the
  non-streamed one (each from an empty cache). `test_live_openai_sdk_streaming_only`: every request streamed (each
  chunk id the cursor), the agent forks, continues streamed replies and mounts the turns of streamed branch A (72
  tokens), then continues the read; streamed `/generate` replies are continued too. The 37-point check gives main's numbers on the same GPU exactly
  (0.0192, max 0.666, argmax 0.985, greedy-16 0.936; no-mount 0.0499 / 0.983; eviction [0.0] x 5; continuation
  0.0416 / 0.0922; rebuilt 0.0116 / 0.0031 / 0.0059; 0 locked), and `test_live_chat_cursor_tree` passes.

## Slices

1. This renderer, its exactness tests and benchmark, this spec.
2. AReaL proxy: cursor requests (store the concat parent per response id, prompt = parent tokens + render_delta).
3. Agent client: sends only deltas; fork, control and rewind as requests from cursors.
4. SGLang: a cursor store for `/v1/chat/completions` and `/generate`.
5. AReaL → SGLang: the rollout engine passes the cursor instead of the full token ids.
6. node-94 end-to-end acceptance.
7. Live: a remote rollout host, rewinds, the eval path and a history past 128k.
