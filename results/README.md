# Results

One directory per experiment or delivery. Most have a `README.md` or `report.md` next to their raw records; the seven
without one hold raw records described by the commit that added them. Directories keep their original names because
many are linked from docs, scripts and other reports. Grouped by topic:

## Search RL

| Directory | Contents |
|---|---|
| [`search-v9-rest-1028`](search-v9-rest-1028) | v9 continued to step 45 (lab cluster, then Together); v45 evaluation on BrowseComp-Plus and Validation (the paper's BrowseComp-Plus result), eval and sweep scripts |
| [`alpha-search-v8-v9-1025`](alpha-search-v8-v9-1025) | Search v8 and v9 on Cluster Alpha |
| [`alpha-search-v7-1025`](alpha-search-v7-1025) | Search v7 (unlimited fork depth, stopped early) |
| [`alpha-search-v6d-1024`](alpha-search-v6d-1024) | Search v6d (fork depth 1, 55 steps) |
| [`beta-search-v5`](beta-search-v5) | Search v5 on Cluster Beta (aarch64 GB200) |
| [`alpha-search-read-1022`](alpha-search-read-1022) | Run record of the `read_thread` search run (concurrency timeline, rejection semantics, coverage) |

## SWE RL

| Directory | Contents |
|---|---|
| [`bundle-rl`](bundle-rl) | Task-bundle GRPO runs on Qwen3.5-4B: per-run training and held-out eval records |
| [`e14-read`](e14-read) | RL with the `read_thread` interface |
| [`coupled-plan`](coupled-plan) | Pre-build checks for the coupled SWE datasets |
| [`harbor-phase2`](harbor-phase2) | Harbor phase-2 SWE trials: trajectories, requests, verifier output |

## Trainer systems

| Directory | Contents |
|---|---|
| [`chunk-major`](chunk-major) | Chunk-major prefix-tree execution measured on real micro-batches |
| [`block-checkpoint`](block-checkpoint) | Block activation-checkpointing benchmark (script and log) |
| [`dp-balance`](dp-balance) | Data-parallel balance of a training step, replayed on the CPU |

## KV serving and scheduling (September, before the RL work)

| Directory | Contents |
|---|---|
| [`v0.1`](v0.1) | First-run records of the single-GPU protocol |
| [`backend-conformance`](backend-conformance) | SGLang backend conformance runs (routing, HiCache, Mooncake ABI, inline) |
| [`controller-admission`](controller-admission) | Controller admission rounds r1–r3 |
| [`checkpoint-admission`](checkpoint-admission) | Checkpoint admission on real GPUs, 81 traces, go/no-go |
| [`retention-contention`](retention-contention), [`retention-contention-interleaved`](retention-contention-interleaved) | Checkpoint retention under real tool waits and competing inference |
| [`selective-cpu`](selective-cpu), [`selective-final-json-cpu`](selective-final-json-cpu), [`selective-gpu`](selective-gpu), [`selective-gpu-json-safe`](selective-gpu-json-safe), [`selective-gpu-output-contract`](selective-gpu-output-contract), [`selective-output-contract-cpu`](selective-output-contract-cpu) | Selective (B5) acceptance epochs |
| [`selector-fastpath`](selector-fastpath) | Selector fast-path GPU diagnostic |
| [`semantic-control-repair`](semantic-control-repair), [`semantic-control-repair-gpu`](semantic-control-repair-gpu) | Semantic retention controls (CPU evidence, GPU confirmation) |
| [`semantic-future-cpu`](semantic-future-cpu), [`semantic-future-gpu`](semantic-future-gpu), [`semantic-future-main-revalidation`](semantic-future-main-revalidation) | Semantic future-only retention studies |
| [`b6-semantic-kv`](b6-semantic-kv) | B6 semantic KV campaign; the raw evidence archive is a release asset |
| [`b7-online-horizon`](b7-online-horizon) | B7 online-horizon GPU campaign |
| [`model-intent`](model-intent) | Model-intent native KV path and Lighthouse recovery |
| [`qasper-kv`](qasper-kv) | QASPER native KV reuse (E1, E2) |
| [`p1-real-agent`](p1-real-agent) | Real-issue pilot comparing projected APC and B6 semantic KV with a real agent |

## Benchmark scouting

| Directory | Contents |
|---|---|
| [`benchmark-discovery`](benchmark-discovery) | Workload discovery and prefix audit across candidate benchmarks |
| [`contextbench-assessment`](contextbench-assessment) | ContextBench evaluation and RL feasibility |
