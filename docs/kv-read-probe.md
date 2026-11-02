# Cross-branch KV read probe (attention-only mount)

Decided whether to build cross-branch KV read in SGLang; it was built afterwards (SGLang mount #132, trainer forward
#131, read_thread tool #134: [cursor-protocol.md](cursor-protocol.md) "Mount", [cross-branch-read.md](cross-branch-read.md)). Code: `octopus/tools/kv_read_probe.py`, test
`tests/test_kv_read_probe.py`, probe set `octopus/tools/kv_read_probe_set.jsonl`. Other architectures and external
methods (text, summary, CacheBlend-style replay, Hogwild!/KVCOMM, KVComm) on the same probe set: [baselines/README.md](../baselines/README.md). H2b for Search, on the built SGLang mount
(read_thread's real mount points, a judge's hijack labels): [search-read-hijack.md](search-read-hijack.md).

## Question

Qwen3.5-4B has 32 layers: 24 Gated DeltaNet (GDN) and 8 full attention (`full_attention_interval` 4), interleaved
M-RoPE `[11, 11, 10]` on a quarter of each 256-dim head. Sibling branch B "reads" sibling A by mounting the K/V of A's
selected turns into its 8 full-attention layers only; its GDN layers keep recurring over B's own tokens.

- **H2a (retrieval).** With the attention-only mount, B retrieves facts that exist only in A's turns, close to giving B
  those turns as text.
- **H2b (no control leakage).** The mount does not hijack B: B keeps doing its own task (contrast: text read, and the
  full replay where B's GDN also runs over A's tokens).
- **H3 (positions).** Which position rule for mounted pages works. K1 keeps A's positions (in the slice-7 run the
  sibling is ahead of B in 42% of B's requests, so mounted keys can sit after B's query); K2 continues B at
  max(mounted) + 1; K3 relocates the block to the positions just before B's next token (keys re-rotated by the offset,
  Hogwild!-style).

Prior: arXiv 2609.04434 (split-prefill on Qwen3.5-4B): kv-only keeps 0.89 of key-value retrieval, recurrent-only 0.00;
behaviour (language, persona) follows the recurrent state, kv-only ~0.01. KVComm (2510.03346, App. M) and Hogwild!
(2504.06261) shift positions of reused caches.

## Method (fixed before the run)

**Data.** Real sibling pairs from recorded bundle RL episodes (`tree.json` main + fork branches, `requests.jsonl`
per-thread request times and prompt lengths): the slice-7 acceptance run on node-134
(`/mnt/disk4/user/harbor-udocker/rl/jobs-cursor-remote-10180140`, 34 trees with forks) and 80 of the 235 forked
e11 trees on node-94 (`jobs-bundle-e11`, sampled with seed 0, CPU reads only, copied over the LAN through node-67).

**Mount points.** For an ordered pair (A, B) of branches of one fork, a mount point is B's j-th request (j >= 1). A's
turns available then are the messages A's latest request sent at or before that time carried (timing from
`requests.jsonl`). The mount is A's last <= 4 completed exchanges (assistant call + tool results) within 12,000
characters, never A's task message. B's context is rendered with the chat template exactly as the policy saw it: the
token count of our rendering must equal the request's recorded `prompt_tokens` (else the mount point is dropped).
Up to 2 mount points per pair (B's latest valid request and the middle one, then neighbours), up to 4 per tree; pairs
shuffled with seed 0. "Peer ahead" = A's last mounted position >= B's next position.

*Deviation (commit `b36181f`, before run 1):* the exact `prompt_tokens` match above was relaxed to a constant offset
per thread. Our rendering of the shared prefix is a constant number of tokens shorter than the server's, so a mount
point is kept when B's offset at that request equals B's offset at its first request (and the same for A at the
request whose messages are mounted). Kept offsets: e11 -90 to -508, slice-7 -110 to -354 tokens. The offset is the
same for every condition of a mount point, so paired comparisons are unaffected; absolute positions are shifted by it
relative to what the policy saw.

**Retrieval probes.** A line of A's mounted tool output; the answer is its last identifier-like string (a path, dotted
name, test id or a token with a digit or underscore), absent from B's rendered context, as is the line up to it; the
cue (the line before the answer, >= 8 non-blank characters) occurs once in the mount. Up to 4 per mount point. After
the mount, B gets a user message: "A sibling branch (branch-N) works on another task in parallel. One line of its tool
output begins exactly like this: ```<cue>``` What text comes next on that line? Reply with only that text." Scored:
teacher-forced mean log-prob per answer token, and greedy exact match (every answer token is the argmax given the
answer so far, i.e. greedy decoding starts with the answer).

**Leakage probes** (one per mount point). B's recorded next reply after the mount point (<= 256 tokens): per-token
NLL minus the none condition's, and mean next-token KL(p_none || p_cond) over the reply. Behaviour: greedy generation
of B's next reply (<= 96 tokens, to `</tool_call>`); **hijack** = it contains an A-only string (identifiers in A's task
and mount, and A's branch id, absent from B's context and recorded reply) and no B-only string (identifiers in B's
task, context and reply absent from A's context and mount); also whether it calls the same tool as B's recorded reply.

**Conditions.** none; text (A's mounted turns, chat markers stripped, as one user message headed "Read-only log of
sibling branch N"); replay (A's mounted tokens spliced into B's context: every layer, GDN included, recomputes them
from B's state, the "no private GDN" contrast); K1, K2, K3 (attention-only mount). Text and replay continue B at its
natural positions; replay's positions equal K3's.

**Mechanism.** transformers 5.15's Qwen3.5 hybrid cache continues a multi-token input from a stored state (GDN conv
context + recurrent state as the fla kernel's initial state; `octopus/rl/gdn_varlen.py` routes the delta rule to
fla). Per mount point: prefill B, prefill A (to the end of the mount), copy B's cache and append A's mounted K/V rows
of each full-attention layer (K3: keys re-rotated by `p_B - a0`), then continue with the probe. This is the SGLang
operation itself (append KV pages, leave the GDN state), so no prefix-tree forward is needed;
`octopus/rl/prefix_tree.py` stays the training-side DAG forward (mounts: `docs/cross-branch-read.md`).

**Models.** Qwen3.5-4B-text base, and base + the RL-trained e11 adapter `bundle-e11-step40` (LoRA r64, merged).

**Sanity checks (must pass before results count).**
- S1: cache continuation (prefill B, then continue with B's reply) equals one plain forward over the concatenated
  tokens: mean |delta log-prob| <= 0.02 nats and argmax agreement >= 0.98 on the reply tokens (first 4 mount points).
- S2: the text condition through the cache equals the plain forward on the concatenated text (same thresholds).
- S3: the mounted token ids decode exactly to A's selected turns and every probe's cue + answer is in them, its answer
  not in B's context (asserted per mount point).
- S4: keys of the mount block continued at shifted positions vs A's keys re-rotated by the shift, first
  full-attention layer: relative error <= 0.02 (bf16). Plus a CPU test of the rotation identity on Qwen3.5's RoPE.

**Statistics.** Means with 95% bootstrap CIs resampling trees (clusters; 2,000 replicates, seed 0). Paired
differences are computed per probe / mount point, then clustered the same way. Reported per model x condition x
group (all / peer ahead / peer behind).

## Pre-registered decision rules

Per model. A rule "meets the retrieval bar" in a group when the paired CI of `EM(K) - 0.8 EM(text)` has its lower
bound >= -0.05 and the CI of `EM(K) - EM(none)` excludes 0.

- **H3.** A position rule *works* if it meets the retrieval bar in both the peer-ahead and the peer-behind group.
  K1 is additionally judged by `EM(K1) - EM(K3)` in the peer-ahead group: a CI entirely below 0 means K1 fails when
  the peer is ahead. The *chosen rule* is the working rule with the highest pooled EM (no working rule: the rule with
  the highest pooled EM, and H3 is "not supported").
- **H2a.** Supported if the chosen rule meets the retrieval bar pooled; not supported if the CI of
  `EM(K) - EM(none)` includes 0 or the point estimate `EM(K) < 0.5 EM(text)`; mixed otherwise.
- **H2b.** On the chosen rule. Supported if all of: the CI of `KL(K) - KL(text)` lies below 0; the CI upper bound of
  the reply `NLL(K) - NLL(none)` is <= 0.10 nats/token; the CI upper bound of `hijack(K) - hijack(none)` is <= 0.05.
  Not supported if `KL(K) - KL(text)` has its CI above 0, or the NLL increase's CI lower bound is > 0.10, or the
  hijack increase's CI lower bound is > 0.05. Mixed otherwise.
- **Across models.** A hypothesis is supported (or not) overall when both models agree; mixed when they disagree.
- **Implication for SGLang.** Build the attention-only mount if H2a and H2b are supported for the e11 adapter, using
  the chosen position rule; if H2a holds only for some position rule, that rule is the one to implement.

## Run 1 (2026-10-18 05:16-05:55): invalid for text and replay

**Invalid for the text and replay conditions** (review of PR #129, F1): the condition loop continued text and replay
at `p_B`, so the question and B's reply overlapped the positions of the inserted block instead of following it. The
none, K1, K2 and K3 conditions used their intended positions. S2 did not catch it because `sanity()` continued at
`p_B + len(text)`, a different path from the measured one (F2). Every text- or replay-based quantity below (EM(text),
the retrieval bar, KL(K) - KL(text), text/replay rows) is therefore invalid. Kept as a dated record; run 2 replaces it.

Run: node-94, 2026-10-18 05:16-05:55, code at commit `b36181f` (the copy on node-94 was byte-identical), 8 shards
(`--shard i/4`; base on GPUs 0-3, base + `bundle-e11-step40` on GPUs 4-7), all exited 0, every one of the 366 mount
points finished for both models. The probe file on node-94 is `kv_read_probe_set.jsonl` with the 104 slice-7
`agent_dir` prefixes pointed at the local copy (`/home/user/kv-read-probe/data/s7/`); nothing else differs. Raw
results: `/home/user/kv-read-probe/results-{base,e11}.jsonl`, reports `report-{base,e11}.md` (output of
`python -m octopus.tools.kv_read_probe report`). CIs below are 95% tree-cluster bootstraps (2,000 replicates, seed 0).

### Sanity checks (all pass; S2 checked a different path, see above)

| check | threshold | base | e11 |
|---|---|---|---|
| S1 cache continuation vs plain forward (16 mount points: first 4 of each shard) | mean abs delta <= 0.02, argmax >= 0.98 | 0.0001-0.0108, argmax >= 0.991 | 0.0001-0.0116, argmax >= 0.992 |
| S2 text read through the cache vs plain forward (same 16) | same | 0.0016-0.0173, argmax >= 0.984 | 0.0008-0.0122, argmax >= 0.989 |
| S3 mounted ids decode to A's turns; cue + answer in the mount, answer not in B's context | asserted | 366/366 | 366/366 |
| S4 K3 re-rotated keys vs keys continued at the shifted position (4 shards) | rel. err <= 0.02 | 0.0082-0.0094 | 0.0082-0.0093 |

### Tables

Peer ahead = A's last mounted position >= B's next position (52% of mount points). EM - none equals EM (none scores
0 everywhere), so it is omitted. In the peer-behind group K1 and K2 are the same computation (max(p_B, a0 + M) = p_B),
hence identical rows.

**Base** (all: 366 mount points / 1122 probes / 99 trees; ahead: 190 / 647 / 85; behind: 176 / 475 / 82)

| condition | group | EM | answer log-prob/token | reply NLL - none | KL(none, cond) | hijack | same tool as recorded |
|---|---|---|---|---|---|---|---|
| none | all | 0.000 [0.000, 0.000] | -7.635 [-7.939, -7.317] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.003 [0.000, 0.008] | 0.738 [0.687, 0.787] |
| none | ahead | 0.000 [0.000, 0.000] | -7.767 [-8.112, -7.427] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.005 [0.000, 0.017] | 0.726 [0.658, 0.793] |
| none | behind | 0.000 [0.000, 0.000] | -7.455 [-7.972, -6.906] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.750 [0.674, 0.825] |
| text | all | 0.005 [0.001, 0.011] | -4.272 [-4.511, -4.036] | 0.157 [0.145, 0.169] | 0.160 [0.149, 0.171] | 0.500 [0.434, 0.564] | 0.552 [0.487, 0.616] |
| text | ahead | 0.008 [0.000, 0.018] | -4.329 [-4.593, -4.066] | 0.154 [0.139, 0.171] | 0.157 [0.143, 0.173] | 0.553 [0.474, 0.635] | 0.537 [0.449, 0.624] |
| text | behind | 0.002 [0.000, 0.007] | -4.194 [-4.605, -3.749] | 0.159 [0.143, 0.176] | 0.164 [0.148, 0.179] | 0.443 [0.355, 0.533] | 0.568 [0.477, 0.655] |
| replay | all | 0.001 [0.000, 0.003] | -4.575 [-4.829, -4.324] | 0.157 [0.145, 0.168] | 0.162 [0.152, 0.173] | 0.527 [0.465, 0.588] | 0.560 [0.496, 0.623] |
| replay | ahead | 0.002 [0.000, 0.005] | -4.629 [-4.901, -4.347] | 0.155 [0.140, 0.171] | 0.159 [0.145, 0.174] | 0.589 [0.503, 0.674] | 0.563 [0.473, 0.649] |
| replay | behind | 0.000 [0.000, 0.000] | -4.502 [-4.943, -4.043] | 0.159 [0.143, 0.175] | 0.166 [0.151, 0.181] | 0.460 [0.383, 0.542] | 0.557 [0.468, 0.645] |
| K1 | all | 0.001 [0.000, 0.003] | -4.871 [-5.125, -4.624] | 0.004 [0.002, 0.005] | 0.006 [0.005, 0.007] | 0.022 [0.008, 0.037] | 0.749 [0.696, 0.797] |
| K1 | ahead | 0.000 [0.000, 0.000] | -5.044 [-5.328, -4.758] | 0.004 [0.002, 0.006] | 0.008 [0.006, 0.009] | 0.032 [0.010, 0.057] | 0.763 [0.701, 0.825] |
| K1 | behind | 0.002 [0.000, 0.007] | -4.634 [-5.065, -4.159] | 0.003 [0.002, 0.005] | 0.004 [0.003, 0.005] | 0.011 [0.000, 0.029] | 0.733 [0.656, 0.809] |
| K2 | all | 0.002 [0.000, 0.005] | -4.527 [-4.793, -4.285] | 0.008 [0.006, 0.010] | 0.010 [0.008, 0.012] | 0.038 [0.022, 0.059] | 0.749 [0.697, 0.797] |
| K2 | ahead | 0.002 [0.000, 0.005] | -4.449 [-4.719, -4.170] | 0.012 [0.008, 0.016] | 0.016 [0.013, 0.020] | 0.063 [0.031, 0.102] | 0.763 [0.700, 0.826] |
| K2 | behind | 0.002 [0.000, 0.007] | -4.634 [-5.065, -4.159] | 0.003 [0.002, 0.005] | 0.004 [0.003, 0.005] | 0.011 [0.000, 0.029] | 0.733 [0.656, 0.809] |
| K3 | all | 0.002 [0.000, 0.005] | -4.447 [-4.702, -4.214] | 0.016 [0.013, 0.019] | 0.019 [0.017, 0.022] | 0.077 [0.049, 0.106] | 0.762 [0.709, 0.811] |
| K3 | ahead | 0.002 [0.000, 0.005] | -4.410 [-4.688, -4.132] | 0.017 [0.013, 0.021] | 0.022 [0.018, 0.025] | 0.084 [0.050, 0.120] | 0.779 [0.718, 0.839] |
| K3 | behind | 0.002 [0.000, 0.007] | -4.496 [-4.911, -4.045] | 0.016 [0.011, 0.020] | 0.017 [0.013, 0.021] | 0.068 [0.033, 0.108] | 0.744 [0.665, 0.819] |

**e11 adapter** (all: 366 mount points / 1122 probes / 99 trees; ahead: 190 / 647 / 85; behind: 176 / 475 / 82)

| condition | group | EM | answer log-prob/token | reply NLL - none | KL(none, cond) | hijack | same tool as recorded |
|---|---|---|---|---|---|---|---|
| none | all | 0.000 [0.000, 0.000] | -7.484 [-7.785, -7.175] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.005 [0.000, 0.014] | 0.768 [0.713, 0.818] |
| none | ahead | 0.000 [0.000, 0.000] | -7.629 [-7.974, -7.289] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.011 [0.000, 0.027] | 0.768 [0.698, 0.833] |
| none | behind | 0.000 [0.000, 0.000] | -7.286 [-7.779, -6.742] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.767 [0.696, 0.835] |
| text | all | 0.007 [0.002, 0.014] | -4.085 [-4.320, -3.852] | 0.152 [0.141, 0.164] | 0.152 [0.141, 0.162] | 0.525 [0.462, 0.585] | 0.563 [0.503, 0.625] |
| text | ahead | 0.011 [0.002, 0.023] | -4.122 [-4.382, -3.861] | 0.149 [0.134, 0.166] | 0.149 [0.135, 0.164] | 0.547 [0.465, 0.626] | 0.568 [0.484, 0.651] |
| text | behind | 0.002 [0.000, 0.007] | -4.035 [-4.436, -3.594] | 0.156 [0.140, 0.171] | 0.155 [0.140, 0.169] | 0.500 [0.417, 0.587] | 0.557 [0.474, 0.642] |
| replay | all | 0.004 [0.000, 0.008] | -4.429 [-4.672, -4.185] | 0.153 [0.142, 0.164] | 0.153 [0.143, 0.163] | 0.522 [0.455, 0.586] | 0.533 [0.474, 0.589] |
| replay | ahead | 0.006 [0.000, 0.015] | -4.486 [-4.752, -4.213] | 0.150 [0.135, 0.165] | 0.150 [0.137, 0.165] | 0.589 [0.505, 0.670] | 0.532 [0.453, 0.610] |
| replay | behind | 0.000 [0.000, 0.000] | -4.352 [-4.764, -3.908] | 0.156 [0.140, 0.171] | 0.157 [0.143, 0.171] | 0.449 [0.365, 0.537] | 0.534 [0.452, 0.619] |
| K1 | all | 0.002 [0.000, 0.005] | -4.654 [-4.895, -4.420] | 0.003 [0.001, 0.004] | 0.005 [0.004, 0.006] | 0.025 [0.011, 0.041] | 0.765 [0.712, 0.815] |
| K1 | ahead | 0.002 [0.000, 0.005] | -4.866 [-5.138, -4.589] | 0.003 [0.001, 0.005] | 0.007 [0.006, 0.009] | 0.037 [0.014, 0.064] | 0.795 [0.730, 0.856] |
| K1 | behind | 0.002 [0.000, 0.007] | -4.367 [-4.779, -3.919] | 0.002 [0.001, 0.004] | 0.003 [0.002, 0.005] | 0.011 [0.000, 0.029] | 0.733 [0.659, 0.804] |
| K2 | all | 0.002 [0.000, 0.005] | -4.325 [-4.583, -4.093] | 0.007 [0.005, 0.009] | 0.009 [0.008, 0.012] | 0.027 [0.011, 0.047] | 0.773 [0.726, 0.819] |
| K2 | ahead | 0.002 [0.000, 0.005] | -4.295 [-4.556, -4.017] | 0.011 [0.008, 0.015] | 0.015 [0.012, 0.019] | 0.042 [0.015, 0.077] | 0.811 [0.751, 0.864] |
| K2 | behind | 0.002 [0.000, 0.007] | -4.367 [-4.779, -3.919] | 0.002 [0.001, 0.004] | 0.003 [0.002, 0.005] | 0.011 [0.000, 0.029] | 0.733 [0.659, 0.804] |
| K3 | all | 0.004 [0.001, 0.007] | -4.269 [-4.513, -4.042] | 0.015 [0.012, 0.018] | 0.018 [0.015, 0.021] | 0.071 [0.043, 0.101] | 0.768 [0.719, 0.814] |
| K3 | ahead | 0.005 [0.000, 0.010] | -4.258 [-4.530, -3.981] | 0.016 [0.012, 0.020] | 0.020 [0.017, 0.024] | 0.084 [0.047, 0.128] | 0.784 [0.719, 0.843] |
| K3 | behind | 0.002 [0.000, 0.007] | -4.285 [-4.690, -3.842] | 0.014 [0.010, 0.019] | 0.015 [0.012, 0.019] | 0.057 [0.023, 0.096] | 0.750 [0.679, 0.820] |

### Pre-registered rules applied

Retrieval bar (`EM(K) - 0.8 EM(text)` CI lower bound >= -0.05 and `EM(K) - EM(none)` CI excluding 0):

| model | rule | all | ahead | behind |
|---|---|---|---|---|
| base | K1 | no (EM - none 0.001 [0.000, 0.003]) | no (0.000 [0.000, 0.000]) | no (0.002 [0.000, 0.007]) |
| base | K2 | no (0.002 [0.000, 0.005]) | no (0.002 [0.000, 0.005]) | no (0.002 [0.000, 0.007]) |
| base | K3 | no (0.002 [0.000, 0.005]) | no (0.002 [0.000, 0.005]) | no (0.002 [0.000, 0.007]) |
| e11 | K1 | no (0.002 [0.000, 0.005]) | no (0.002 [0.000, 0.005]) | no (0.002 [0.000, 0.007]) |
| e11 | K2 | no (0.002 [0.000, 0.005]) | no (0.002 [0.000, 0.005]) | no (0.002 [0.000, 0.007]) |
| e11 | K3 | **yes** (0.004 [0.001, 0.007]; vs 0.8 text -0.002 [-0.008, 0.003]) | no (0.005 [0.000, 0.010]) | no (0.002 [0.000, 0.007]) |

The `0.8 EM(text)` half of the bar holds everywhere (all lower bounds >= -0.017); every failure is `EM(K) - EM(none)`.
`EM(K1) - EM(K3)`, peer ahead: base -0.002 [-0.005, 0.000], e11 -0.003 [-0.010, 0.002], not entirely below 0.

- **H3: not supported** (both models). No rule meets the bar in both groups. Chosen rule (highest pooled EM): base
  K2 (0.0018, tied with K3; the rules name no tie-break, and K3 gives the same verdicts below), e11 K3 (0.0036).
- **H2a: mixed** (base not supported: K2 `EM - none` CI includes 0; e11 supported: K3 meets the bar pooled, on 4
  exact matches out of 1,122 probes).
- **H2b: mixed** (both models mixed). KL lies below the text read's (base K2 -0.150 [-0.161, -0.139], e11 K3 -0.134
  [-0.144, -0.123]) and the reply NLL increase is far under 0.10 (base 0.008 [0.006, 0.010], e11 0.015 [0.012,
  0.018]), but the hijack increase's upper bound exceeds 0.05 (base 0.036 [0.019, 0.057], e11 0.066 [0.039, 0.093]).
  K1 would pass every H2b criterion (hijack - none: base 0.019 [0.008, 0.034], e11 0.019 [0.008, 0.033]), but K1 is
  not the chosen rule.
- **Implication by the rules:** H2a and H2b are not both supported for e11, so the pre-registered rule does not
  license building the SGLang attention-only mount on this run.

### Run 1 conclusion

Superseded by run 2. Besides F1/F2 (above), the retrieval probe failed its positive control: the text read got 0.5%
(base) and 0.7% (e11) exact match, 0/415 and 2/415 on answers of at most 2 tokens, so every EM bar decision rested on
0-4 hits among 1,122 probes. The cause is the answer slot: after the question the policy starts its reply with
`<tool_call>` (first answer token at -7 to -9 nats), while once the answer has started the text read copies it. Run 1
therefore decides nothing about H2a/H3, and its H2b verdict used the overlapped text read as the KL reference.

## Run 2 pre-registration (written before run 2)

Same probe set, models, conditions, statistics and decision rules as above, with these changes:

1. **Positions (F1).** Text continues B at `p_B + len(text)`, replay at `p_B + M` (its positions equal K3's), as the
   method specifies.
2. **S2 (F2).** S2 continues the text condition at exactly the `start` the condition loop uses, so it checks the
   measured path against the plain forward at natural positions. S1, S3, S4 unchanged.
3. **Completion-style answer slot (F3).** B's reply is forced to start with the cue line: the probe input is the
   question, B's generation prompt, then "```\n" + the cue without trailing spaces. The answer tokens are the tokens of
   the joint encoding of that prefix + the rest of the line up to the answer's end, after the prefix's tokens; a probe
   whose prefix tokens are not a prefix of the joint encoding is dropped and counted. EM = the greedy continuation
   (16 tokens, saved per probe) decodes to text starting with the answer tokens' text. Answer log-prob is
   teacher-forced over the same answer tokens.
4. **Positive-control gate**, per model, pooled, same tree bootstrap: `EM(text) - EM(none) >= 0.20` with CI lower
   bound `>= 0.10`. Gate fails: H2a and H3 are "uninformative" for that model and there is no build decision. Gate
   passes: the H2a retrieval bar, H3 and H2b rules above apply unchanged.
5. **Fallback if the EM gate fails** (computed in the same run): 4-way ranking by total answer log-prob against 3
   distractors of the same token length, drawn (seeded by probe) from the answers of the same tree's other probes and
   aligned with the same prefix; probes with fewer than 3 such distractors are left out of the ranking and counted.
   Gate: `Acc(text) - Acc(none) >= 0.25` with CI lower bound `>= 0.15`. If it passes, the H2a/H3 rules are applied with
   Acc in place of EM, and the `EM(K) - EM(none)` "excludes 0" test becomes `Acc(K) - Acc(none)` excluding 0. If it
   also fails, H2a/H3 are "uninformative".
6. **Exploratory, not used for any verdict:** paired answer log-prob/token (all answer tokens, and without the first
   one), printed by `report`.

## Run 2 (2026-10-18 06:35-07:32)

Run: node-94, code at commit `0d4e97d` (the copy on node-94 was byte-identical), the same probe file and 8-shard layout
as run 1. All shards exited 0, and all 366 mount points finished for both models. A first launch at 06:34 stopped at
once, before writing any rows: the e11 `agent_dir`s are relative (`../data/e11/...`) and the code copy had moved one
directory down, so a `data` symlink was added and the same launcher re-run. Raw results:
`/home/user/kv-read-probe/run2/results-{base,e11}.jsonl`, reports `run2/report-{base,e11}.md`.

**Probes.** 947 of the 1,122 retrieval probes scored; 175 (15.6%, the same probes for both models) were dropped
because the reply prefix's tokens were not a prefix of the joint encoding. 368 probes had 3 equal-length distractors.

### Sanity checks (all pass)

| check | threshold | base | e11 |
|---|---|---|---|
| S1 cache continuation vs plain forward (16 mount points) | mean abs delta <= 0.02, argmax >= 0.98 | 0.0001-0.0108, argmax >= 0.991 | 0.0001-0.0116, argmax >= 0.992 |
| S2 text condition at the loop's `start` vs plain forward (same 16) | same | 0.0016-0.0173, argmax >= 0.984 | 0.0008-0.0122, argmax >= 0.989 |
| S3 mount ids, cue + answer in the mount, answer not in B's context | asserted | 366/366 | 366/366 |
| S4 K3 re-rotation rel. err (4 shards) | <= 0.02 | 0.0082-0.0094 | 0.0082-0.0093 |

S1 and S2 equal run 1's values: the none path did not change, and run 1's `sanity()` already continued the text at
`p_B + len(text)`, the position the condition loop now uses.

### Positive-control gate: passed (both models)

`EM(text) - EM(none)`, pooled: base 0.731 [0.678, 0.776], e11 0.730 [0.678, 0.775] (bar: >= 0.20, lower bound >= 0.10).
EM decides; the ranking fallback is not used (it would also pass: `Acc(text) - Acc(none)` 0.364 [0.306, 0.426] and
0.370 [0.308, 0.432]). Rank accuracy is far above chance even for none (0.63): same-tree distractors are partly
distinguishable without A's turns.

### Tables

Peer ahead = 52% of mount points. In the peer-behind group K1 and K2 are the same computation.

**Base** (mount points / probes / probes ranked / trees: all 366/947/368/99; ahead 190/532/231/85; behind 176/415/137/82)

| condition | group | EM | rank acc | answer log-prob/token | reply NLL - none | KL(none, cond) | hijack | same tool as recorded |
|---|---|---|---|---|---|---|---|---|
| none | all | 0.030 [0.018, 0.044] | 0.630 [0.571, 0.687] | -4.005 [-4.241, -3.759] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.003 [0.000, 0.008] | 0.738 [0.687, 0.787] |
| none | ahead | 0.030 [0.016, 0.047] | 0.615 [0.549, 0.682] | -4.088 [-4.378, -3.818] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.005 [0.000, 0.017] | 0.726 [0.658, 0.793] |
| none | behind | 0.029 [0.013, 0.049] | 0.657 [0.558, 0.764] | -3.897 [-4.297, -3.497] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.750 [0.674, 0.825] |
| text | all | 0.760 [0.708, 0.807] | 0.995 [0.983, 1.000] | -0.288 [-0.347, -0.235] | 0.202 [0.187, 0.218] | 0.206 [0.192, 0.220] | 0.579 [0.516, 0.640] | 0.560 [0.496, 0.622] |
| text | ahead | 0.784 [0.735, 0.829] | 0.991 [0.972, 1.000] | -0.272 [-0.332, -0.220] | 0.200 [0.180, 0.223] | 0.204 [0.185, 0.224] | 0.626 [0.546, 0.706] | 0.563 [0.479, 0.650] |
| text | behind | 0.730 [0.650, 0.809] | 1.000 [1.000, 1.000] | -0.308 [-0.413, -0.219] | 0.203 [0.183, 0.225] | 0.208 [0.188, 0.229] | 0.528 [0.443, 0.614] | 0.557 [0.476, 0.639] |
| replay | all | 0.720 [0.663, 0.773] | 0.995 [0.983, 1.000] | -0.355 [-0.425, -0.295] | 0.202 [0.187, 0.217] | 0.207 [0.194, 0.221] | 0.560 [0.495, 0.624] | 0.549 [0.486, 0.612] |
| replay | ahead | 0.748 [0.690, 0.799] | 0.991 [0.972, 1.000] | -0.350 [-0.431, -0.281] | 0.199 [0.181, 0.219] | 0.204 [0.186, 0.223] | 0.621 [0.543, 0.700] | 0.542 [0.457, 0.628] |
| replay | behind | 0.684 [0.603, 0.765] | 1.000 [1.000, 1.000] | -0.361 [-0.466, -0.270] | 0.204 [0.183, 0.225] | 0.211 [0.191, 0.231] | 0.494 [0.400, 0.589] | 0.557 [0.466, 0.649] |
| K1 | all | 0.531 [0.475, 0.589] | 0.946 [0.909, 0.979] | -0.802 [-0.940, -0.680] | 0.004 [0.002, 0.005] | 0.006 [0.005, 0.007] | 0.022 [0.008, 0.037] | 0.749 [0.696, 0.797] |
| K1 | ahead | 0.459 [0.395, 0.525] | 0.918 [0.866, 0.966] | -1.034 [-1.244, -0.835] | 0.004 [0.002, 0.006] | 0.008 [0.006, 0.009] | 0.032 [0.010, 0.057] | 0.763 [0.701, 0.825] |
| K1 | behind | 0.624 [0.542, 0.705] | 0.993 [0.975, 1.000] | -0.504 [-0.636, -0.387] | 0.003 [0.002, 0.005] | 0.004 [0.003, 0.005] | 0.011 [0.000, 0.029] | 0.733 [0.656, 0.809] |
| K2 | all | 0.627 [0.574, 0.676] | 0.973 [0.950, 0.992] | -0.572 [-0.670, -0.480] | 0.008 [0.006, 0.010] | 0.010 [0.008, 0.012] | 0.038 [0.022, 0.059] | 0.749 [0.697, 0.797] |
| K2 | ahead | 0.630 [0.575, 0.688] | 0.961 [0.929, 0.990] | -0.624 [-0.763, -0.494] | 0.012 [0.008, 0.016] | 0.016 [0.013, 0.020] | 0.063 [0.031, 0.102] | 0.763 [0.700, 0.826] |
| K2 | behind | 0.624 [0.542, 0.705] | 0.993 [0.975, 1.000] | -0.504 [-0.636, -0.387] | 0.003 [0.002, 0.005] | 0.004 [0.003, 0.005] | 0.011 [0.000, 0.029] | 0.733 [0.656, 0.809] |
| K3 | all | 0.608 [0.556, 0.656] | 0.973 [0.950, 0.992] | -0.597 [-0.701, -0.502] | 0.016 [0.013, 0.019] | 0.019 [0.017, 0.022] | 0.077 [0.049, 0.106] | 0.762 [0.709, 0.811] |
| K3 | ahead | 0.624 [0.571, 0.679] | 0.961 [0.929, 0.990] | -0.624 [-0.763, -0.494] | 0.017 [0.013, 0.021] | 0.022 [0.018, 0.025] | 0.084 [0.050, 0.120] | 0.779 [0.718, 0.839] |
| K3 | behind | 0.588 [0.504, 0.670] | 0.993 [0.975, 1.000] | -0.563 [-0.711, -0.431] | 0.016 [0.011, 0.020] | 0.017 [0.013, 0.021] | 0.068 [0.033, 0.108] | 0.744 [0.665, 0.819] |

**e11 adapter** (mount points / probes / probes ranked / trees: all 366/947/368/99; ahead 190/532/231/85; behind 176/415/137/82)

| condition | group | EM | rank acc | answer log-prob/token | reply NLL - none | KL(none, cond) | hijack | same tool as recorded |
|---|---|---|---|---|---|---|---|---|
| none | all | 0.031 [0.018, 0.045] | 0.625 [0.561, 0.686] | -4.022 [-4.261, -3.776] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.005 [0.000, 0.014] | 0.768 [0.713, 0.818] |
| none | ahead | 0.032 [0.017, 0.050] | 0.606 [0.541, 0.675] | -4.106 [-4.395, -3.839] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.011 [0.000, 0.027] | 0.768 [0.698, 0.833] |
| none | behind | 0.029 [0.013, 0.049] | 0.657 [0.554, 0.766] | -3.914 [-4.322, -3.509] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.767 [0.696, 0.835] |
| text | all | 0.760 [0.710, 0.805] | 0.995 [0.983, 1.000] | -0.293 [-0.350, -0.242] | 0.196 [0.181, 0.211] | 0.194 [0.181, 0.208] | 0.579 [0.516, 0.643] | 0.574 [0.513, 0.632] |
| text | ahead | 0.786 [0.738, 0.831] | 0.991 [0.972, 1.000] | -0.284 [-0.346, -0.230] | 0.194 [0.174, 0.216] | 0.193 [0.175, 0.213] | 0.616 [0.534, 0.694] | 0.600 [0.520, 0.677] |
| text | behind | 0.728 [0.648, 0.806] | 1.000 [1.000, 1.000] | -0.305 [-0.403, -0.220] | 0.198 [0.178, 0.218] | 0.196 [0.178, 0.215] | 0.540 [0.453, 0.629] | 0.545 [0.465, 0.626] |
| replay | all | 0.715 [0.659, 0.765] | 0.992 [0.981, 1.000] | -0.360 [-0.426, -0.302] | 0.195 [0.180, 0.209] | 0.195 [0.182, 0.208] | 0.552 [0.489, 0.610] | 0.566 [0.503, 0.625] |
| replay | ahead | 0.731 [0.675, 0.784] | 0.991 [0.978, 1.000] | -0.364 [-0.444, -0.294] | 0.192 [0.173, 0.211] | 0.192 [0.175, 0.211] | 0.616 [0.532, 0.695] | 0.579 [0.495, 0.656] |
| replay | behind | 0.694 [0.614, 0.774] | 0.993 [0.975, 1.000] | -0.354 [-0.454, -0.266] | 0.198 [0.178, 0.217] | 0.198 [0.179, 0.216] | 0.483 [0.395, 0.574] | 0.551 [0.466, 0.634] |
| K1 | all | 0.521 [0.465, 0.576] | 0.943 [0.905, 0.977] | -0.814 [-0.950, -0.688] | 0.003 [0.001, 0.004] | 0.005 [0.004, 0.006] | 0.025 [0.011, 0.041] | 0.765 [0.712, 0.815] |
| K1 | ahead | 0.449 [0.386, 0.514] | 0.913 [0.859, 0.963] | -1.050 [-1.265, -0.846] | 0.003 [0.001, 0.005] | 0.007 [0.006, 0.009] | 0.037 [0.014, 0.064] | 0.795 [0.730, 0.856] |
| K1 | behind | 0.612 [0.536, 0.688] | 0.993 [0.975, 1.000] | -0.512 [-0.638, -0.397] | 0.002 [0.001, 0.004] | 0.003 [0.002, 0.005] | 0.011 [0.000, 0.029] | 0.733 [0.659, 0.804] |
| K2 | all | 0.612 [0.560, 0.660] | 0.965 [0.936, 0.990] | -0.588 [-0.690, -0.494] | 0.007 [0.005, 0.009] | 0.009 [0.008, 0.012] | 0.027 [0.011, 0.047] | 0.773 [0.726, 0.819] |
| K2 | ahead | 0.613 [0.554, 0.671] | 0.948 [0.907, 0.983] | -0.647 [-0.786, -0.511] | 0.011 [0.008, 0.015] | 0.015 [0.012, 0.019] | 0.042 [0.015, 0.077] | 0.811 [0.751, 0.864] |
| K2 | behind | 0.612 [0.536, 0.688] | 0.993 [0.975, 1.000] | -0.512 [-0.638, -0.397] | 0.002 [0.001, 0.004] | 0.003 [0.002, 0.005] | 0.011 [0.000, 0.029] | 0.733 [0.659, 0.804] |
| K3 | all | 0.601 [0.549, 0.650] | 0.967 [0.939, 0.992] | -0.607 [-0.712, -0.512] | 0.015 [0.012, 0.018] | 0.018 [0.015, 0.021] | 0.071 [0.043, 0.101] | 0.768 [0.719, 0.814] |
| K3 | ahead | 0.611 [0.553, 0.671] | 0.952 [0.912, 0.986] | -0.646 [-0.788, -0.508] | 0.016 [0.012, 0.020] | 0.020 [0.017, 0.024] | 0.084 [0.047, 0.128] | 0.784 [0.719, 0.843] |
| K3 | behind | 0.588 [0.510, 0.667] | 0.993 [0.975, 1.000] | -0.558 [-0.695, -0.435] | 0.014 [0.010, 0.019] | 0.015 [0.012, 0.019] | 0.057 [0.023, 0.096] | 0.750 [0.679, 0.820] |

### Pre-registered rules applied (EM)

Retrieval bar: `EM(K) - 0.8 EM(text)` CI lower bound >= -0.05, and `EM(K) - EM(none)` CI excludes 0. The second part
holds in every cell (EM(K) - EM(none) >= 0.41, lower bounds >= 0.35), so the table shows `EM(K) - 0.8 EM(text)`.

| model | rule | all | ahead | behind |
|---|---|---|---|---|
| base | K1 | no: -0.077 [-0.122, -0.032] | no: -0.168 [-0.227, -0.113] | yes: 0.040 [-0.009, 0.088] |
| base | K2 | yes: 0.019 [-0.018, 0.054] | yes: 0.003 [-0.041, 0.048] | yes: 0.040 [-0.009, 0.088] |
| base | K3 | yes: -0.000 [-0.034, 0.032] | yes: -0.003 [-0.041, 0.038] | yes: 0.004 [-0.047, 0.053] |
| e11 | K1 | no: -0.088 [-0.132, -0.044] | no: -0.179 [-0.235, -0.124] | yes: 0.030 [-0.023, 0.082] |
| e11 | K2 | yes: 0.004 [-0.034, 0.040] | no: -0.016 [-0.062, 0.032] | yes: 0.030 [-0.023, 0.082] |
| e11 | K3 | yes: -0.007 [-0.043, 0.026] | no: -0.018 [-0.060, 0.029] | yes: 0.006 [-0.042, 0.053] |

`EM(K1) - EM(K3)`, peer ahead: base -0.165 [-0.211, -0.124], e11 -0.162 [-0.208, -0.120]: entirely below 0, so K1
fails when the peer is ahead (both models).

- **H3: mixed.** Base: supported, K2 and K3 work in both groups. e11: not supported. K2 and K3 miss the ahead bar
  narrowly (lower bounds -0.062 and -0.060 against -0.05), so no rule works there. Chosen rule (highest pooled EM)
  is K2 in both models: base 0.627 vs K3 0.608 and K1 0.531; e11 0.612 vs K3 0.601 and K1 0.521.
- **H2a: supported** (both models). K2 meets the bar pooled. EM: base K2 0.627 vs text 0.760 and none 0.030; e11
  0.612 vs 0.760 and 0.031.
- **H2b: mixed.**
  - e11: supported. On K2, `KL(K2) - KL(text)` = -0.185 [-0.198, -0.171], reply NLL increase 0.007 [0.005, 0.009],
    hijack increase 0.022 [0.008, 0.039].
  - Base: mixed. KL -0.196 [-0.210, -0.181] and NLL 0.008 [0.006, 0.010] pass, but the hijack increase is
    0.036 [0.019, 0.057], with its upper bound above 0.05.
  - For contrast, text and replay raise reply NLL by 0.20, and hijack B's next action in 55-58% of mount points.
- **Implication by the rules:** H2a and H2b are both supported for the e11 adapter, so the rule says build the SGLang
  attention-only mount with the chosen position rule, K2. K2 mounts A's pages at A's positions, and B continues at
  max(its next position, last mounted position + 1). K2 and K3 are close. Paired EM `K2 - K3` is about 0 when the
  peer is ahead (base 0.006 [-0.006, 0.018], e11 0.002 [-0.013, 0.017]). When it is behind, where K2 keeps A's
  positions, K2 is ahead by 0.036 [0.007, 0.067] (base) and 0.024 [-0.002, 0.050] (e11). K1 (no position change)
  is ruled out when the peer is ahead.

**Exploratory, not pre-registered and not used for any verdict: teacher-forced answer log-prob** (paired per probe,
nats/token; all answer tokens | without the first answer token). Pooled:

| model | text - none | replay - none | K1 - none | K2 - none | K3 - none | K1 - K3 | K2 - K3 |
|---|---|---|---|---|---|---|---|
| base | 3.72 [3.47, 3.95] \| 2.62 [2.37, 2.88] | 3.65 [3.40, 3.89] \| 2.61 [2.36, 2.87] | 3.20 [2.94, 3.47] \| 2.37 [2.11, 2.63] | 3.43 [3.18, 3.68] \| 2.49 [2.24, 2.75] | 3.41 [3.15, 3.66] \| 2.48 [2.23, 2.73] | -0.21 [-0.28, -0.14] \| -0.11 [-0.17, -0.06] | 0.03 [0.01, 0.04] \| 0.02 [0.00, 0.03] |
| e11 | 3.73 [3.48, 3.97] \| 2.66 [2.40, 2.92] | 3.66 [3.41, 3.90] \| 2.64 [2.38, 2.90] | 3.21 [2.94, 3.47] \| 2.39 [2.13, 2.66] | 3.44 [3.18, 3.68] \| 2.52 [2.26, 2.78] | 3.42 [3.15, 3.66] \| 2.51 [2.25, 2.77] | -0.21 [-0.28, -0.14] \| -0.11 [-0.17, -0.07] | 0.02 [0.01, 0.03] \| 0.01 [0.00, 0.02] |

By group, K1 - K3 is -0.41 / -0.40 (base / e11) when the peer is ahead and +0.06 / +0.05 when it is behind. Per-group
lines are in `run2/report-*.md`.

## SGLang implementation

K2 is implemented on SGLang `/generate` as a mount on cursor requests (`patches/apply_sglang_mount.py`,
docs/cursor-protocol.md "Mount"). It is this probe's operation on server pages: B's KV + A's KV rows for the mounted
tokens in the attention layers, B's GDN state, B continuing at max(p_B, a0 + M). On 37 of these mount points (base
model) the server's teacher-forced answer log-probs match this probe's K2 run to mean |d| 0.020 (argmax 0.987); the
same server without a mount matches the none run to 0.050, so the difference is the SGLang vs transformers kernel gap.
The mount stays in B's context for B's later requests (an inserted attention-only segment), which this probe did not
test (one mount, one request).

## Limitations

- Zero-shot: neither model was trained to read mounted KV or to answer the probe question; e11 differs from base only
  by an RL adapter trained without any read.
- Offline: transformers cache surgery on recorded contexts, not SGLang pages; one mount per request, no repeated
  reads, no decode-time effects on later turns.
- Probe construction: facts are identifier tails of single tool-output lines, cued by the rest of the line, and run 2
  forces B's reply to start with the cue line, so they test copy-style lookup, not use of A's findings. The 15.6% of
  probes whose answer is not token-aligned with the prefix are dropped (the same for every condition and model).
  Hijack is a string heuristic (A-only identifiers, no B-only identifier) over a 96-token greedy reply.
- Run 2's answer slot was redesigned after run 1 (review F3), but it was pre-registered before run 2 and run 2's
  rules are run 1's rules plus the gate.
- Sample: 99 trees from two runs (slice-7 acceptance: 104 mount points; e11: 262); up to 4 mount points per tree;
  the ahead/behind split is by position, not by task progress.
- The answer log-prob analysis was chosen after seeing run 1's EM floor; it stays exploratory.
