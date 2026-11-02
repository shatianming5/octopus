# Same-size coding checkpoint reference

**Official result: `resolved=false`, zero patch bytes.** The pinned public
Qwen2.5-Coder-7B-Instruct checkpoint made12 identical root-content searches for
`stack-collector.js`. All12 tool results were empty **and limited**, not
exhaustive negative searches. No source read, edit or test occurred.

This exposed a repository-tool defect, not evidence that coding pretraining
cannot help. The recursive walker spent its10000-entry budget on ignored
dependencies before reaching relevant tracked source. A separate read-only
check of the exact query in the unchanged task image found three tracked
reference files. No discovered paths were manually injected into model input.
The follow-up repository-search treatment changes that generic traversal only.

All11 inference files,15,242,788,143bytes, match the official immutable HF
revision's Git/LFS digests. Direct HF failed with connection reset; mirror API
access returned403. Official Qwen ModelScope supplied matching tensor metadata
and partial bytes, but its bounded transfer timed out. Following the user's
proxy suggestion, the fixed-revision hf-mirror.com file route passed a206range
and matching-prefix check. Byte-range resumption preserved existing downloads;
all four completed tensor SHA256 values match the official HF LFS values.

Active download jobs totaled2027.608315seconds, including failures, within the
explicit pre-inference transport-only amendment to5400seconds. The GPU episode
budget stayed1200seconds and its count stayed one. Logical /home and canonical
/mnt model paths were verified with samefile, not treated as different weights.
Tokenization assets are byte-equal except tokenizer-config model_max_length.
Checkpoint generation defaults differ in repetition penalty, but the actual
direct SamplingParams constructor/update retains1.0 for both; no pure
pretraining-held-out or old-adapter recovery-score claim is made.

The episode used12 native model calls,31,991 prompt /9,271 prefill /1,241 decode
tokens and ran131 wrapper seconds. Its model identity was correctly recorded
as the coding checkpoint. Original official tests really ran: subsequent log
classification found no EAGAIN/worker-retry failure in the evaluator. Full
actions, costs and denominator details are in metrics.json. All GPU jobs and
task containers were released; reusable model assets remain remote.
