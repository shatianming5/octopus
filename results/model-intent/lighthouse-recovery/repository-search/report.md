# Repository-aware root search

**Official result: `resolved=false`, zero patch bytes.** The same coding model
still made12 identical searches. Unlike the previous treatment, all12 returned
the actual three tracked-source references without exhausting the search
budget. These are12 nonempty inspection calls but only **one unique result**,
not12 independent evidence acquisitions.

Only root search file discovery changed. In a Git root, the tool uses cached
and nonignored untracked paths; explicit files, explicit subdirectories and
non-Git fixtures retain their prior scope. Git-index errors are explicit.
A real10005ignored-file regression was red before the fix and green afterward.
The exact saved model query was also executed through the changed tool on the
unchanged task image and returned three references, without a supplied hint.

The repaired tool removes the actual evidence-starvation defect, but it does
not solve the model's repeated successful-inspection loop. No failure was
fabricated to classify these valid searches as errors. There were no reads,
edits or tests, and no failed-action rejections in the episode.

The original evaluator completed two tests with one failed and one passed.
There were no EAGAIN or worker-retry failures in that evaluation. Thus this is
an actual quality failure, not the later developer-tool concurrency defect.

Costs were38,305 prompt /10,321 prefill /1,263 decode tokens,58.325295engine
seconds and96 wrapper seconds, within the1200-second cap. Active utilization
was not sampled; terminal GPU was4MiB/0%, without the original job or task
container. See metrics.json for complete counts and timing.
