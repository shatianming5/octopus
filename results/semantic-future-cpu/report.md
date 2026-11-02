# Semantic future-only: CPU delivery

**CPU implementation accepted; GPU performance remains unverified.**
This slice is based on `d7b730d2d92235e69a62d472c4330ae62441aac3`.
Its implementation and evidence are the enclosing delivery commit; no old
policy/run artifact is overwritten.

| Evidence | Result |
|---|---|
| Real vLLM 0.28.0 CPU acceptance suite | 107 passed; 9 GPU cases excluded. See `native-cpu-junit.xml`. |
| Public interfaces | Typed output intent reaches `run_brain`, `BranchTask`, `commit`, backend capture and input consumer binding. |
| Native NONE gate | APC, gated cost-greedy and semantic-soft acquire zero NONE leases; legacy cost-greedy acquires one. |
| Retention-only union | Legacy 8 block-seconds; APC/gate/soft zero. Same native request-owner geometry. |
| Retention-owner union | Legacy 16 block-seconds, distinct from exclusive/retention-only accounting. |
| Shared owner | Dropping the first owner frees zero blocks; its exclusive block-time is zero. |
| Restricted V2a | 1,087 initially missing APC blocks; optimistic model minimum zero. Per-task missing blocks: 448, 639, 0, 0. |
| V2b | `v2b-comparison.json` prepared; no GPU execution, calibration or provisioning. |

Raw deterministic CPU evidence: [`cpu-evidence.json`](cpu-evidence.json).
V1 uses the real native KV manager with a 32-usable-block pool and injected
0/1/2-second lifecycle tape. Costs are synthetic fixture values. It does not
construct a model/LLMEngine or generate model output. The shared-owner case
also uses native refcounts and records actual free-count deltas.

V2a reads only the existing schema-3 APC design trace
`results/semantic-control-repair-gpu/traces/real_tool-tight_burst-101-apc.json`.
Actual adoption, not probe maxima, identifies initial missing prefix blocks.
Logical prefix sharing follows capture parentage; capture physical IDs are
not reused as persistent cache identity. The oracle is an optimistic
whole-checkpoint retention model with fixed APC access order, unit block
costs, no background contention and per-task reset. Historical timestamps
only order offline events; no online treatment receives them as horizons.
The 17,392-token ceiling is not identified causal GPU work or a latency bound.
Already preserved prefixes in tasks 2/3 offer no initial-miss benefit.

## Validation

Executed in isolated source deployment
`/mnt/disk2/user/octopus-semantic-future-cpu-4007b0b5`, using the existing
`/mnt/disk2/user/octopus-contention/.venv/bin/python` (vLLM 0.28.0):

```bash
export CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONHASHSEED=17 PYTHONPATH=src
python -m pytest -q -m 'not gpu' --disable-warnings \
  --junitxml=results/semantic-future-cpu/native-cpu-junit.xml
python -m benchmarks.semantic_cpu \
  --apc-trace results/semantic-control-repair-gpu/traces/real_tool-tight_burst-101-apc.json \
  --output results/semantic-future-cpu/cpu-evidence.json
```

The local lightweight venv lacks torch/vLLM; native validation was not
replaced with mocks or counted as locally passing. The native CPU fixtures
replace only model execution / scheduler construction, not the real
KVCacheManager, allocation, ownership, backend binding, or retention provider.
The owner-audit follow-up (`owner-audit-junit.xml`, 9 passing tests) is a subset,
not additional independent evidence. The final suite also covers NONE while
an accepted native request still owns blocks and measurement beginning with
an already-captured live checkpoint.

## Go / no-go scope

**Go:** CPU contract/resource invariants and a prepared GPU feasibility
diagnostic. **Not established:** generated task correctness on the new
policy, foreground E2E improvement, background SLO, or queue-aware increment.

The frozen target is all tasks correct, no background failures, background
P95 degradation point and one-sided 95% upper bound both <=5%, foreground
mean E2E improvement >=10% versus APC, and independent paired-block evidence
of improvement versus both APC and queue-aware. A single-block reporter
cannot claim this target even when point metrics pass.

Remaining work requires separate GPU authorization, then freezing held-out
blocks/seeds, rotated method order and run budget before observing outcomes.
Run the prepared future-only GPU lifecycle case and the same-config
APC/queue-aware/future-only/semantic-naive-pin comparison, report independent
paired effects, and stop rather than sweep on no-go. Seed101 remains a design
fixture, not held-out evidence. V3 predictors/controllers are not implemented.
