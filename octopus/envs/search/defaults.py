"""The search episode's per-question defaults (docs/search-env.md "Limits"), apart from agent.py so that the evaluation
side (octopus/eval/search_dev_eval.py, under AReaL's python, which has no harbor) reads the same values as the
episode (agent.run_trial: the env overrides each)."""

# per question (dev_eval and RL): no shared token budget, at most 10 branches, only main forks, the last 2048 tokens of
# each thread's growth kept for its done, main's calls 100 (CLM's), a branch's turns 50 (20 ran out, 10-24)
DEFAULTS = {"OCTOPUS_MAIN_MAX_CALLS": "100",
            "OCTOPUS_BRANCH_MAX_TURNS": "50", "OCTOPUS_SEARCH_BUDGET": "0", "OCTOPUS_MAX_BRANCHES": "10",
            "OCTOPUS_FORK_DEPTH": "1", "OCTOPUS_ANSWER_RESERVE": "2048"}
