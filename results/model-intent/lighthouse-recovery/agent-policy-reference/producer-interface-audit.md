# Producer/interface audit: SWE-Gym/OpenHands-7B-Agent

**status:** completed (bounded public-source research; provenance limits below)  
**as of:** 2026-09-16  

## Decision-bearing conclusion

**The evidence does not support "the actor cannot solve this task" or "512 tokens explains everything." It supports a concrete executable-interface mismatch, interacting with a modest-capability actor, before task-solving ability has been meaningfully exercised in the newly reported episode.** The producer's `execute_bash` really executes shell commands; the local tool with that name accepts only a restricted parsed command language. Identical XML syntax is not an equivalent action space.

The parent's supplied corrected ACI-notice episode has zero length caps, 40 normally stopped, complete, unfenced `execute_bash` calls rejected by `command_action`, 38 `ls -R` requests (one for `lighthouse-core`, 37 `cd /testbed && ls -R .`), two `find ... -not -path` requests, feedback in 39/39 next capsules, and 36 raw repeats. No edit/test/finish or successful issue-audit source read occurred. **These are parent-provided observations, not independently re-audited here.** Against verified producer code, they establish that this episode stalls at command execution/recovery, not that a fully functioning agent has exhausted its reasoning ability on the issue. Ordinary shell syntax being rejected is a real protocol difference; repeated failure to adapt after feedback is also a real weakness of this actor under the local protocol. Neither cancels the other.

## Verified producer configuration and differences

All OpenHands code citations below use the assigned commit **e644a2ca45c3623b27a7e6c169e3d479f0a87fbc**. Official SWE-Gym reproduction documentation is pinned to observed current commit **b681068ca20628c6987b7416cc4cf03f06b77ba5**. The paper is final arXiv **v2, 2025-06-06**; v1 was checked and the pertinent 7B table values agree.

| Surface | Verified first-party evidence | Comparison / limit |
| --- | --- | --- |
| Pinned weights/card | Revision `97054e70501d79c27499518af5ce613be6278bcb` exists. Its API file manifest contains `.gitattributes`, `config.json`, and four weight shards, **no README/model card or tokenizer files**. Both pinned and main README endpoints returned 404. [M1] | There is no model card here from which to verify a complete checkpoint-specific recipe. This is not a failure to find the weights. |
| Architecture/context | Pinned config: `Qwen2ForCausalLM`, `max_position_embeddings=32768`, `torch_dtype=bfloat16`, EOS 151645, BOS 151643, vocabulary 152064, `use_sliding_window=false`. [M2] | Supports the stated BF16 and text-only architecture. `sliding_window=131072` is also present but disabled; it is **not** evidence for a 128K deployed context. |
| Serving | Official instructions use an OpenAI-compatible **SGLang** endpoint and an explicitly separate base-model tokenizer path. Serving script passes `--context-length`; its default is **32768**. The worked example is 32B, not a separately published, pinned 7B deployment command. [D1, D2] | A separate tokenizer is not inherently wrong. The paper identifies Qwen2.5-Coder-Instruct as the base family, but the exact local C03 tokenizer/chat-template identity cannot be certified from these sources. No tokenizer bytes were inspected locally. |
| Per-completion output | `LLMConfig.max_output_tokens` starts `None`; `LLM.init_model_info()` supplies **4096** if neither explicit config nor recognized model metadata supplies another limit. Explicit config wins; recognized metadata can replace the fallback. The resolved value is sent as API `max_tokens`. [C1, C2] | **512 is eight times smaller than the released 4096-token fallback**, a substantial per-action reduction. However, the official example omits the setting and the actual 7B benchmark `config.toml`/model-info response is not released in the inspected materials. Therefore **the exact cap used for the paper's 7B run is unverified**, not established as 4096 (and not 8192). |
| Context enforcement | Paper evaluation: **100 interaction turns or the base LM's 32K context, whichever comes first**. Official serving default and training maximum are 32768. [P1, D2] | Local context 16384 is half that nominal window; its journal cap 8192 is an additional policy not in the producer recipe. `max_input_tokens`'s generic 4096 fallback must not be confused with server context: its own docstring says the parameter is currently unused. [C1, C2] |
| Evaluation iteration budget | Dedicated Verified rollout explicitly sets **100**, dataset test split, limit **500**, default **one run**. Lite explicitly sets **100**, limit **300**, default one run. They pass these to `run_infer.sh`, which passes `--max-iterations`. [C3, C4, C5] | Local 50 proposals is a smaller ceiling, though proposal and interaction counters are not necessarily one-to-one. **50** in `rollout-swe-train-full.sh` is a training-trajectory collection setting, not the published test rollout budget. Generic README prose saying 30 is stale relative to the script. [C6, C7] |
| Sampling | Paper defaults to **temperature 0** unless specified otherwise; the code defaults temperature 0, top-p 1. [P2, C1] | Local temperature 0 agrees. No published evaluation seed 17 was found; the training YAML seed 42 is not an evaluation seed. |
| Action space | Evaluation enables Bash + finish + string-replace editor, disables standalone Jupyter action and browser. The editor still executes internally via Jupyter/agent-skills. [C8, C9] | The producer is not a bounded no-shell command adapter. Absence of web browsing locally is **not** a demonstrated deviation from this benchmark configuration. |
| View format | `str_replace_editor.view` advertises **`cat -n` numbered files**; directory views list non-hidden entries two levels deep. Optional `view_range` is 1-indexed inclusive and supports `[start, -1]`; absent range means whole file, subject to clipping. Long editor output uses `<response clipped>`. [C9] | Local raw 6000-character views differ from this interface. Shell `cat` remains a separate raw-view route in the producer, so it would be wrong to claim all producer file text is numbered. |
| Observation size/feedback | Generic observation-content limit is **30000 characters** before the added truncation marker/exit suffix. CodeAct truncates the middle, retaining head and tail, then adds command exit status; Bash observations include the executed command, output and interpreter details. Errors get an explicit processing-error suffix. [C1, C10, C11, C12] | Local 6000-character raw views are 1/5 of this generic character limit, but not necessarily 1/5 of usable information. The editor package has its own clipping; its exact clipping threshold is not verified here. Its implementation is imported from `openhands_aci.editor`, not implemented in this fork. [C13] |

### Actual shell availability: decisive for the new parent evidence

The chain is explicit, not inferred from a tool name:

1. `execute_bash` becomes `CmdRunAction(**arguments)`. [C9, lines 454-464]
2. The runtime creates a persistent `BashSession`; its `run` method delegates directly to `bash_session.run(action)`. [C14]
3. `BashSession` starts a pexpect shell process via `su <username>`, initializes shell state, and sends commands with `self.shell.sendline(command)`. Its action runner splits shell input and calls `_execute_bash`; it does not restrict the command to local `command_action`'s grammar. [C12]

Consequently, `ls -R`, shell conjunctions such as `cd ... && ls ...`, and ordinary `find` predicates are part of the producer's shell execution surface. Execution success still depends on paths, installed utilities, permissions, and timeouts; this is not a claim that the exact Lighthouse commands have been executed here. The producer's tool description also explicitly documents background processes, redirection, interactive continuation, and interrupts. [C9, lines 28-47]

**The reported rejections are not model-format errors and are not evidence of a producer shell limitation.** They show that the actor emitted commands appropriate to its original action space but outside the local one. Since these forms were not advertised locally, the subsequent repetition also measures failure to follow/adapt to the new restricted interface. Generic feedback visibility disproves "the model never saw any error" for these transitions; it does not prove the feedback conveyed a usable replacement strategy.

### Stop tokens, conversion, natural messages, finish, and prompts

- For non-native-function models, CodeAct uses mocked function calling. The system suffix describes `<function=...>` and `<parameter=...>` markup, allows reasoning **before** the call, and asks for one function call with no suffix. `STOP_WORDS = ['</function']`. The default also inserts a substantial worked interaction before the first task and a one-call reminder after it. [C15, C16]
- The response converter **repairs a missing final function delimiter** when exactly one opening function tag is present, then searches with regex. Parameter values still need recognizable closed parameter tags and required parameters are validated. It is not a strict full-message XML/fence validator. The LLM wrapper converts the response without first branching on `finish_reason=length`. Thus a local blanket incomplete/fence/length rejection is stricter in behavior; this is **not** a recommendation to execute incomplete commands. [C16, C17]
- A response with no recognized function call is retained as an assistant message, then becomes `MessageAction(wait_for_response=True)`. It is not automatically classified as a invalid no-tool turn. [C16, C9]
- Evaluation's exact `codeact_user_response` asks the actor to continue, to send its answer before finishing when solved, and never to ask for human help; after at least two user messages it adds that `finish` may be used to give up. `finish` itself is described as ending when complete **or unable to proceed**, and directly maps to `AgentFinishAction`. [C18, C9]
- Therefore, a stock `finish` event is **not** proof of a correct patch. Conversely, a natural no-call message need not mean the actor has failed. The parent's stated exact callback for naturally stopped no-call messages addresses one mismatch, but does not make the remaining tool execution or failure gates stock-equivalent.
- The stock SWE-bench task prompt embeds `<uploaded_files>` and `<pr_description>`, identifies a **Python** repository, asks for minimal non-test changes, and explicitly requests a reproduction script, running it before and after the fix, and considering edge cases. It also permits long thinking. The system identifies the agent as OpenHands. [C8, C19] The local initial XML instructions **already require tests before finish**, per the dispatch: "forgot to tell it to test" is not a supported root-cause explanation.

### History/condenser: important source-versus-paper distinction

CodeAct builds messages from `list(state.history)`, preserves action/observation relationships, combines consecutive same-role non-tool messages, and reconstructs tool calls/results. Non-function conversion changes tool feedback into user messages prefixed `EXECUTION RESULT of [tool]:`. This is neither a deduplicated recent journal nor a prefix capsule. [C20, C16]

There is **no LLM summarization condenser in this CodeAct call path**. However, the pinned controller catches `ContextWindowExceededError`, cuts approximately the oldest half of the event history, preserves relevant action/observation pairing and the first user message, and retries. [C21] Therefore "the producer always preserves its entire history" would also be false.

Two publication/provenance discrepancies must remain visible: the paper calls its scaffold **CodeActAgent 2.1** and describes stopping at 32K; the assigned fork identifies CodeActAgent as **2.2** and contains the overflow-history retry. [P2, C15, C21] The official reproduction guide itself warns that using a different OpenHands version is expected to produce notable performance differences. [D1, lines 5-10] The fork is the best supplied released implementation evidence, but not a byte-for-byte archived 7B benchmark run.

## Baseline ability, training, and documented limitations

The final paper's Table 3 reports the following **OpenHands, fine-tuned Qwen2.5-Coder-Instruct 7B** results, not the verifier-scaled 32B headline:

| Benchmark | Denominator | Zero-shot resolve | Fine-tuned resolve | Fine-tuned empty patch | Fine-tuned stuck in loop | Fine-tuned mean turns |
| --- | --- | --- | --- | --- | --- | --- |
| SWE-bench Lite | 300 instances | 1.0% (+/- 1.0) | **10.0% (+/- 2.4)** | 29.7% | 31.0% | 22.2 |
| SWE-bench Verified | 500 instances | 1.8% (+/- 1.1) | **10.6% (+/- 2.1)** | 33.8% | 21.0% | 35.3 |

Source: [P3]. Uncertainty terms are copied as reported; their precise statistical interpretation for this table was not established in the inspected text, so they are not presented as a seed-to-seed distribution or a per-task confidence interval. The study's v1 explicitly defines "stuck in loop" as repeating the exact same action for the last three turns. That definition is not automatically identical to the parent's local loop metric. [P6]

- Training uses **491 successful off-policy agent trajectories** from GPT-4o-2024-08-06 and Claude-3.5-Sonnet-20241022 in SWE-Gym, whose 2438 tasks are Python-repository tasks. It is not supervised training on arbitrary alternative tool grammars or image-grounded Lighthouse work. Mean successful training trajectory is about 19 turns / 19K tokens. [P2, P4]
- Paper training: full-parameter finetuning, learning rate 1e-4, maximum five epochs, global batch eight, maximum sequence 32768, covering 7B/14B/32B. The released worked policy YAML is 32B, uses OpenAI-style `messages`, `train_on_input: False`, and a Qwen tokenizer. Its 7B-related header comments are generic boilerplate, not a 7B-specific recipe. [P1, D3]
- **32.0% Verified / 26.0% Lite are not this 7B actor's single-attempt performance.** They are headline results with learned verifiers and inference-time candidate selection. The 32B actor alone is 20.6% Verified / 15.3% Lite in Table 3. [P3, P4]
- Even on its intended scaffold, the 7B succeeds on only about one-tenth of the aggregate benchmark and exhibits substantial empty-patch and loop rates. This gives credible evidence of limited robustness, not a guarantee on any particular problem. It also prevents treating repetition as proof that only the local harness can be at fault.
- More iterations do not automatically improve the model: the paper's 32B self-improvement experiment adds 868 on-policy successful trajectories but decreases Lite resolve from 15.3% to 8.7%. This is evidence against assuming monotonic improvement from iteration, **not** proof that this local fixed actor underwent that training or suffered the same cause. [P5]
- Repeating a fixed greedy actor against similar histories is not equivalent to the paper's independent complete candidates at temperature 0.5 ranked by a trained verifier. Fifty within-episode proposals are not best-of-50 solutions. With the checkpoint fixed, weeks of scaffold iteration do not themselves update model weights. [P4; fixed-checkpoint premise from dispatch]

**Task transfer:** the supplied task type is multimodal Lighthouse, while these training/evaluation sources describe Python repositories and a text causal LM. If image input was omitted, that is a potential loss of task information and a domain-transfer issue, not proof that an image contained the decisive answer. No task images, issue text, hidden hints, gold solution, or linked PR were inspected. The published 10.6% average is not a calibrated success probability for this task.

**No official positive-control performance evidence under the local protocol was found.** The sources do not evaluate this pinned actor with C03 tokenizer identity, 512-token proposals, 16K context / 8K journal, raw 6000-character views, bounded no-shell adapters, capsules/deduplication, or V4 gates. Selected legacy loop reproductions are not an entire published-scaffold benchmark baseline. Public benchmark performance also does not certify exact checkpoint-to-table provenance in the absence of its model card and archived run config.

The local ban on web access is consistent with the paper's disabled browser. Useful Git-history availability in the producer's task containers was not established; a real shell alone does not prove that historical repository content was available or used. Neither restriction is assigned a demonstrated causal role here.

## Priority-ranked causal assessment

1. **Executable action-space mismatch plus poor recovery: highest-confidence immediate explanation for the supplied corrected episode.** Forty complete tool calls never executed; repeated ordinary shell operations are valid for the producer but unsupported locally. The resulting no-source-read/no-edit trajectory cannot establish downstream task-solving incapacity. The 36 repeats after visible feedback simultaneously demonstrate poor adaptation to the restriction.
2. **Modest intrinsic policy robustness and domain transfer: strongly supported general vulnerability, not an identified task-specific impossibility.** Published 7B failures/loops are common even in-distribution. Moving from Python/text SWE-bench to a multimodal Lighthouse task and a different action language plausibly worsens results. The relative contribution remains unmeasured.
3. **Budget/history/observation constraints: substantial plausible contributors elsewhere, not sufficient for this zero-cap episode.** 512 versus the 4096 fallback, 16K versus documented 32K, 50 versus 100, smaller raw views and deduplicated history can suppress long edits, investigation and recovery. Raising only the output cap would not cause the 40 already-complete unsupported commands to execute.
4. **Parsing/finish/prompt differences: narrower contributors, not a universal account.** Stock closing-tag repair, natural-message handling, permissive give-up finish and worked tool examples differ from strict local gates. But this latest episode contains complete unfenced calls and receives feedback, so neither missing closing tags nor invisible errors explains its stall. Existing test instructions rule out a simple missing-reminder account.

**Unsupported claims:** definitive inability on task 10176; definitive solvability if stock were used; tokenizer corruption merely because it is separate; omitted images as the proven decisive cause; "more tokens will fix it"; "valid XML means the interface works"; "a gate pass means quality"; or attributing every historical failure to this one command restriction.

## One smallest high-information validation recommendation (not executed)

**One matched counterfactual continuation from the saved state immediately before the first complete rejected shell call**, in an independently authorized disposable task environment: compare the already observed bounded-adapter branch with a branch using the pinned producer Bash execution semantics. Hold the actor, tokenizer, generation settings (including 512), prompt/history state, task assets, remaining budget, and non-shell tools fixed. Do not broaden this into a model/temperature/prompt sweep or production protocol change.

Predeclare two separate outcomes: (a) whether the same call obtains real repository evidence and the actor escapes the rejection loop; (b) whether the full continuation produces a genuinely resolved task under the existing authorized black-box quality criterion. An executable command, source read, edit, test invocation, or parser pass counts only toward (a), **never** as task quality. Improvement would identify the shell restriction's causal contribution for this trajectory; successful resolution would refute inability for that condition. Failure would **not** establish intrinsic incapacity because other protocol/domain differences remain. This design is smaller and more diagnostic for the concrete new observation than increasing 512 tokens or arbitrarily rerunning all settings. No validation was launched here.

## Sources

[M1]: https://huggingface.co/api/models/SWE-Gym/OpenHands-7B-Agent/revision/97054e70501d79c27499518af5ce613be6278bcb
[M2]: https://huggingface.co/SWE-Gym/OpenHands-7B-Agent/raw/97054e70501d79c27499518af5ce613be6278bcb/config.json
[D1]: https://github.com/SWE-Gym/SWE-Gym/blob/b681068ca20628c6987b7416cc4cf03f06b77ba5/docs/OpenHands.md#L5-L112
[D2]: https://github.com/SWE-Gym/SWE-Gym/blob/b681068ca20628c6987b7416cc4cf03f06b77ba5/scripts/serving/serve_sglang.py#L36-L71
[D3]: https://github.com/SWE-Gym/SWE-Gym/blob/b681068ca20628c6987b7416cc4cf03f06b77ba5/scripts/training/openhands/configs/policy/1116-sonnet-4o-491i-32k-qwen25_coder_32b_full-lr1e-4.yaml#L20-L88
[C1]: https://github.com/SWE-Gym/OpenHands/blob/e644a2ca45c3623b27a7e6c169e3d479f0a87fbc/openhands/core/config/llm_config.py#L31-L80
[C2]: https://github.com/SWE-Gym/OpenHands/blob/e644a2ca45c3623b27a7e6c169e3d479f0a87fbc/openhands/llm/llm.py#L135-L145
[C3]: https://github.com/SWE-Gym/OpenHands/blob/e644a2ca45c3623b27a7e6c169e3d479f0a87fbc/scripts/rollout-eval-verified.sh#L3-L27
[C4]: https://github.com/SWE-Gym/OpenHands/blob/e644a2ca45c3623b27a7e6c169e3d479f0a87fbc/scripts/rollout-eval-lite.sh#L3-L27
[C5]: https://github.com/SWE-Gym/OpenHands/blob/e644a2ca45c3623b27a7e6c169e3d479f0a87fbc/evaluation/swe_bench/scripts/run_infer.sh#L6-L98
[C6]: https://github.com/SWE-Gym/OpenHands/blob/e644a2ca45c3623b27a7e6c169e3d479f0a87fbc/scripts/rollout-swe-train-full.sh#L21-L28
[C7]: https://github.com/SWE-Gym/OpenHands/blob/e644a2ca45c3623b27a7e6c169e3d479f0a87fbc/evaluation/swe_bench/README.md#L29-L51
[C8]: https://github.com/SWE-Gym/OpenHands/blob/e644a2ca45c3623b27a7e6c169e3d479f0a87fbc/evaluation/swe_bench/run_infer.py#L74-L165
[C9]: https://github.com/SWE-Gym/OpenHands/blob/e644a2ca45c3623b27a7e6c169e3d479f0a87fbc/openhands/agenthub/codeact_agent/function_calling.py#L28-L523
[C10]: https://github.com/SWE-Gym/OpenHands/blob/e644a2ca45c3623b27a7e6c169e3d479f0a87fbc/openhands/agenthub/codeact_agent/codeact_agent.py#L229-L295
[C11]: https://github.com/SWE-Gym/OpenHands/blob/e644a2ca45c3623b27a7e6c169e3d479f0a87fbc/openhands/events/serialization/event.py#L119-L130
[C12]: https://github.com/SWE-Gym/OpenHands/blob/e644a2ca45c3623b27a7e6c169e3d479f0a87fbc/openhands/runtime/utils/bash.py#L69-L330
[C13]: https://github.com/SWE-Gym/OpenHands/blob/e644a2ca45c3623b27a7e6c169e3d479f0a87fbc/openhands/runtime/plugins/agent_skills/file_editor/__init__.py#L1-L8
[C14]: https://github.com/SWE-Gym/OpenHands/blob/e644a2ca45c3623b27a7e6c169e3d479f0a87fbc/openhands/runtime/action_execution_server.py#L100-L173
[C15]: https://github.com/SWE-Gym/OpenHands/blob/e644a2ca45c3623b27a7e6c169e3d479f0a87fbc/openhands/agenthub/codeact_agent/codeact_agent.py#L43-L98
[C16]: https://github.com/SWE-Gym/OpenHands/blob/e644a2ca45c3623b27a7e6c169e3d479f0a87fbc/openhands/llm/fn_call_converter.py#L22-L751
[C17]: https://github.com/SWE-Gym/OpenHands/blob/e644a2ca45c3623b27a7e6c169e3d479f0a87fbc/openhands/llm/llm.py#L183-L232
[C18]: https://github.com/SWE-Gym/OpenHands/blob/e644a2ca45c3623b27a7e6c169e3d479f0a87fbc/evaluation/utils/shared.py#L118-L165
[C19]: https://github.com/SWE-Gym/OpenHands/blob/e644a2ca45c3623b27a7e6c169e3d479f0a87fbc/openhands/agenthub/codeact_agent/prompts/system_prompt.j2#L1-L6
[C20]: https://github.com/SWE-Gym/OpenHands/blob/e644a2ca45c3623b27a7e6c169e3d479f0a87fbc/openhands/agenthub/codeact_agent/codeact_agent.py#L372-L469
[C21]: https://github.com/SWE-Gym/OpenHands/blob/e644a2ca45c3623b27a7e6c169e3d479f0a87fbc/openhands/controller/agent_controller.py#L499-L507
[P1]: https://arxiv.org/html/2412.21139v2#A2.SS2
[P2]: https://arxiv.org/html/2412.21139v2#S4.SS2
[P3]: https://arxiv.org/html/2412.21139v2#S4.T3
[P4]: https://arxiv.org/html/2412.21139v2
[P5]: https://arxiv.org/html/2412.21139v2#S4.SS2.p6
[P6]: https://arxiv.org/html/2412.21139v1#S3.SS2.p3

Additional exact anchors for multi-location claims:

- Output/input fallback and metadata precedence: https://github.com/SWE-Gym/OpenHands/blob/e644a2ca45c3623b27a7e6c169e3d479f0a87fbc/openhands/llm/llm.py#L340-L365
- History-window preservation algorithm: https://github.com/SWE-Gym/OpenHands/blob/e644a2ca45c3623b27a7e6c169e3d479f0a87fbc/openhands/controller/agent_controller.py#L799-L880
- Missing stop-delimiter repair: https://github.com/SWE-Gym/OpenHands/blob/e644a2ca45c3623b27a7e6c169e3d479f0a87fbc/openhands/llm/fn_call_converter.py#L551-L558
- Tool-result-to-user conversion: https://github.com/SWE-Gym/OpenHands/blob/e644a2ca45c3623b27a7e6c169e3d479f0a87fbc/openhands/llm/fn_call_converter.py#L451-L467
- No-call preservation: https://github.com/SWE-Gym/OpenHands/blob/e644a2ca45c3623b27a7e6c169e3d479f0a87fbc/openhands/llm/fn_call_converter.py#L740-L745
- Finish and no-call action mapping: https://github.com/SWE-Gym/OpenHands/blob/e644a2ca45c3623b27a7e6c169e3d479f0a87fbc/openhands/agenthub/codeact_agent/function_calling.py#L421-L428 and https://github.com/SWE-Gym/OpenHands/blob/e644a2ca45c3623b27a7e6c169e3d479f0a87fbc/openhands/agenthub/codeact_agent/function_calling.py#L472-L503
- Actual shell command send: https://github.com/SWE-Gym/OpenHands/blob/e644a2ca45c3623b27a7e6c169e3d479f0a87fbc/openhands/runtime/utils/bash.py#L165-L174
