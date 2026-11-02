# Analysis B (pasted by the user 10-25 ~15:00 EDT; produced by another tool on tommy-dev, partly from the files in scratchpad/branch-weight/)

## Part 1

Root cause claimed: `tree_advantage` (octopus/rl/group.py) violates "per-episode credit conservation". GRPO keeps sum_i A_i = 0 over G=8 episodes only if each episode counts once. Currently every branch takes the full A_branch = A_main whatever N is.
Measured on v7: mean tokens per branch is almost constant across depth (~600-1000):
- md=1 (1.6 branches): 544 tokens per branch, total branch weight = 1.6 x A_main
- md=3 (7.1 branches): 716, total = 7.1 x A_main
- md=4 (9.4 branches): 633, total = 9.4 x A_main
Proposed fix: A_branch = A_main / N_branches (one line). Claimed: zero hyperparameters; N=1 unchanged; branches together weigh ~650 tokens regardless of count/depth.
Table on v7 steps 1-6 with PR #306 applied (main_end != done -> y=0):

| md | nb | T_br/nb | sum(A_br*fork_br) full | sum((A/nb)*fork_br) | sum(A*tok) full | sum(A*tok_eff) /N |
|---|---|---|---|---|---|---|
| 1 | 1.6 | 544 | -4.6 | +0.3 | +17,619 | +8,934 |
| 2 | 4.5 | 651 | +0.5 | +3.5 | +27,288 | +16,337 |
| 3 | 7.1 | 716 | +42.8 | +8.4 | +1,891 | +7,080 |
| 4 | 9.4 | 633 | +55.1 | +6.1 | +47,191 | +8,492 |

Claim: #306 + A/N together cure both "no tool call" and branch explosion.

## Part 2

Extracted v7 (search-v7-1025, depth=0, 5,071 episodes) and v6d (search-v6d-1024, depth=1, 15,151 episodes): requests.jsonl (main + branch completion tokens), tree.json (nb, md, fork calls in main/branches), answers.json; LOO advantage per octopus/rl/group.py.

### I. v7 steps 1-6 (main no-tool-call 6-13%), 897 mixed-group episodes, by max depth md

| md | n (%) | y | mean_adv | sum(adv) | mean_tok | forks/ep | sum(adv*tok) | sum(adv*fork_br) |
|---|---|---|---|---|---|---|---|---|
| 0 | 598 (66.7%) | 0.500 | -0.0920 | -55.0 | 758 | 0.09 | -77,579 | 0.0 |
| 1 | 132 (14.7%) | 0.629 | +0.1540 | +20.3 | 1,519 | 1.14 | +33,047 | +0.9 |
| 2 | 81 (9.0%) | 0.617 | +0.2098 | +17.0 | 4,018 | 3.33 | +26,027 | +14.1 |
| 3 | 49 (5.5%) | 0.633 | +0.2501 | +12.3 | 6,003 | 5.94 | +41,220 | +73.6 |
| 4 | 16 (1.8%) | 0.500 | +0.2452 | +3.9 | 7,114 | 9.12 | +29,455 | +35.5 |
| >=5 | 21 (2.3%) | 0.524 | +0.1020 | +2.1 | 6,203 | 10.38 | +28,200 | +2.9 |

Claims: md>=3 = 9.6% of episodes (86), sum(adv) +18.3, but 62.6% of the POSITIVE token-weighted gradient (+98,875 / +157,949). 88.2% (+112.0 / +127.0) of the positive branch-fork push comes from md>=3. Steps 7-11 tokens per md 3-4 episode 12,305-15,260 (6-10x md=1).

### II. v7 vs v6d, episodes whose branches called fork (forkcalls[1] > 0), mixed groups

| run | branch called fork | n | y | mean_adv | mean_tok | sum(adv*fork_br) |
|---|---|---|---|---|---|---|
| v7 | >0 | 658 | 0.503 | +0.0753 | 9,943 | +103.0 |
| v7 | =0 | 1,227 | 0.548 | -0.0404 | 1,289 | 0.0 |
| v6d | >0 | 1,192 | 0.448 | -0.0232 | 9,086 | -133.1 |
| v6d | =0 | 3,133 | 0.559 | +0.0088 | 2,632 | 0.0 |

Claim: in v6d a branch's fork is refused (depth 1), wastes a step, lowers accuracy -> negative feedback; in v7 it is allowed and forked episodes (0.461) beat md=0 (0.412) in-group -> positive.

### III. v7 steps 1-6 with PR #306 applied (main_end != "done" -> y_new = 0)

| md | n | y_new | mean_adv_new | mean_tok | sum(adv_new*tok) | sum(adv_new*fork_br) |
|---|---|---|---|---|---|---|
| 0 | 831 | 0.576 | -0.0208 | 645 | -51,745 | 0.0 |
| 1 | 166 | 0.548 | -0.0450 | 1,304 | +17,619 | -4.6 |
| 2 | 89 | 0.551 | +0.1073 | 3,819 | +27,288 | +0.5 |
| 3 | 50 | 0.500 | +0.1627 | 5,918 | +1,891 | +42.8 |
| 4 | 18 | 0.667 | +0.3884 | 6,729 | +47,191 | +55.1 |

Claim: even with #306, md 3-4 (5.8%, 68 episodes) give +97.9 branch-fork push and +49,082 token-weighted positive gradient.

### Conclusion of B
1. #306 fixes main "no tool call" (42% of md>=1 episodes got y=1 via no tool call).
2. With FORK_DEPTH=0 the branch-fork positive feedback remains; either keep depth=1 or normalise branch advantage / loss weight by branch count.
