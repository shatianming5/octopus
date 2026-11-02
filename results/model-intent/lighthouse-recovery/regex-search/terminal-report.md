# Terminal report: lighthouse-regex-search-e3e25abd

`QUALITY_FAIL`: the one frozen explicit-regex episode completed its original
official evaluation, but `official_resolved=false`; the sole `FAIL_TO_PASS`
test remained failed. The solver completed 12 actions at the action limit with
no edit, test, done action, or patch.

Metrics: 12 raw proposals / 4 distinct; 12 physical submissions / 4 distinct;
0 physical failures and 0 effective failed repeats; 6 nonempty outputs / 2
byte-exact unique outputs. Regex mode was enabled, but all 6 searches returned
empty output, so no nonempty regex-search evidence was obtained. Native work:
17,723 prefill tokens, 1,485 decode tokens, 0 evictions, concurrency 1;
77.815 s GPU-engine, 80.766 s solver E2E, 112 s terminal elapsed.

Cleanup is complete: the selected GPU is at 2 MiB / 0% with no selected-device
compute process, and the local `octopus-lighthouse-` namespace is empty. Random
container names were not persisted and could not be recovered after removal.
No active `nvidia-smi` sample was captured before this short run ended; the
completed native request and engine records are retained as actual execution
evidence. This negative result closes only this episode.
