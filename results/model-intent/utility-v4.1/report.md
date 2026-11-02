# Utility v4 inline-intent mechanism final report

**Terminal scope result: `MECHANISM_PASS` for v4.1.** This passes the frozen
authored mechanism workload only. It does not establish natural multi-branch
contention or generalized serving utility; that conditional continuation
remains a parent-owned later pilot.

## V4.0: duplicated-carrier failure

V4.0 asked the model to emit a separate leading 0/1 carrier before the real
normal object. Runtime streaming and native retention worked, and the normal
body dispatched the correct follow-up in 4/4 seen smoke episodes with 4/4 exact
answers. The redundant carrier was nevertheless constant 1, agreeing with the
body in only 2/4 episodes. V4.0 therefore ended `SCIENTIFIC_NO_GO`, and its
held-out six-trace matrix was not launched.

That evidence remains unchanged under `results/model-intent/utility-v4/`.

## V4.1: consume the genuine normal action

Before any held-out outcome, v4.1 removed the carrier entirely. The normal
schema remained the exact original bare object with first required property
`dispatch_index`, integer enum `[0, 1]`. Runtime streams that genuine action
digit and applies it before the closing brace. The normal prompt changed only by
removing the obsolete carrier-match sentence; semantic task text, held-out
cohorts and hashes, seeds, method order, weights, candidate prefixes, capacity,
pressure, arrivals, accounting, evaluator, and gates were unchanged.

Tokenizer freeze established zero marginal guidance tokens and zero envelope
tokens. Standard JSON exposes the digit at output-bearing step 6 before the
seventh-token closing brace; compact JSON exposes it at step 5 before token 6.
The fixed pressure step was therefore 6. The longest normal prompt plus output
budget fell from 501 to 481 tokens. The unchanged normal schema compiled with
XGrammar, and the affected native test passed all 12 cases across four Brain
seams and three policies.

The rerun of the same seen four-episode integration cohort passed: hint observed
and applied 4/4, hint/body identity 4/4, correct dispatch 4/4, and exact quality
4/4. This was an integration diagnostic, not a held-out result or tuning
surface.

## Untouched held-out results

All six traces completed on the same physical GPU. Every gate passed
independently on both seeds.

| Seed | Selection | Quality APC / structural / inline | Gross continuation saving | Total native prefill reduction | Net oracle saving | FG change vs structural | BG P95 vs APC |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 8101 | 12/12 | 12/12 / 12/12 / 12/12 | 100.00% | 10.278% | 100.262% | -7.304% | -0.461% |
| 8202 | 12/12 | 12/12 / 12/12 / 12/12 | 100.00% | 10.272% | 100.262% | -7.654% | +1.671% |

Background prefill and decode did not increase. Both seeds also passed the
absolute quality threshold, zero paired quality regression, at least 11/12
selection, at least 80% gross saving, at least 10% total prefill reduction, at
least 70% net oracle saving, non-increased foreground mean, and background P95
within APC plus 5%.

The net oracle fraction slightly exceeds 100% because inline also scheduled 24
fewer normal-prefill tokens than structural in each trace; all native committed
work remains included.

## Instrumentation repairs and resources

V4.0 preserved one invalid 24-second attempt where the harness counted prompt
prefill scheduler calls as output steps; it was repaired without changing the
protocol. V4.1 preserved one scorer-only terminal failure after all traces had
completed: the scorer still used v4.0's four-token readiness limit instead of
the pre-frozen v4.1 limit of six. The scorer now reads the versioned freeze, and
the existing traces were rescored without rerunning inference.

V4.0 used 51 runner wall-seconds and v4.1 used 231, for 282 seconds total against
the authorized 1800-second budget. After retrieval, the owned tmux session and
processes were absent; the selected GPU
`GPU-3e940a5a-a24d-b615-dc69-df141ced7af0` reported 4 MiB used and 0%
utilization.

Frozen files, raw episode traces, environments, logs, terminal records, scorer
failure evidence, derived score, and cleanup evidence are under
`results/model-intent/utility-v4.1/`.
