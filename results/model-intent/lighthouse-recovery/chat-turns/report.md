# Native conversation history

**Official result: `resolved=false`, zero patch bytes.** Native chronological
turns changed the action sequence but did not solve the task:12 proposals /
3distinct,12 physical submissions comprising1 search,1 mismatching edit and
10 successful repeated reads. There were11 nonempty inspection calls but only
2unique results. No known-failed action was physically resubmitted.

The model first used an import line returned from gather-runner.js as the old
text for an edit to stack-collector.js. The exact tool correctly found0matches.
It then read the complete target file ten times without making another edit.
Thus better conversation structure did not by itself produce grounded repair.

The only protocol delta was chat_turns_v1. Previous raw model decisions are
assistant messages; external observations follow in tool_response/user turns
with literal multiline output, not escaped code inside a sorted JSON journal.
Current memory and action budget follow the latest result. The real context
compiler places its semantic capsule inside the initial user turn. Whole
oldest turn pairs are dropped, never half messages. Legacy rendering is intact.
No new task information or action rewriting was introduced.

All relevant role-order and truncation tests passed, including the actual
prefix/capsule compilation seam. Later turns dropped3,4,5old journal entries
under the unchanged budget. Costs:66,160 prompt /20,816 prefill /1,445 decode
tokens,59.537155engine seconds and102 wrapper seconds, below1200. All compute
and task containers were released. Removing the newly added rationale
requirement is a separate subsequent ablation, not a replay of this episode.
