"""rl.branch_credit (group.tree_advantage): "outcome" gives every branch its episode's A_main; "gate" applies the
environment's rule, chosen by its group step: code (plan_group_step, code_gate) gives an ok branch (committed, merged,
non-empty diff, passed) max(A_main, 0) and any other min(A_main, 0); search (search_group_step, search_gate; protocol
v6) gives an ok branch (closed with done, a non-empty answer, made its own search/open call) A_main and any other
min(A_main, 0); main's A never changes; a record without the rule's fields raises, as does gate without a rule and the
old value search_gate; search_outcome records each branch's summary and own search/open calls."""
import json

import pytest

from octopus.envs.search.reward import search_group_step, search_outcome
from octopus.envs.swe.reward import plan_group_step
from octopus.rl.group import code_gate, search_gate, tree_advantage

OK = {"ended_with": "commit", "merge": "merged", "diffstat": "1 file changed", "passed": True}
SEARCH_OK = {"ended_with": "done", "summary": "Mira Okonkwo", "searched": 2}


@pytest.mark.parametrize("a", [0.75, -0.75])
@pytest.mark.parametrize("branch", [OK, {**OK, "passed": False}, {**OK, "merge": None}, {**OK, "diffstat": ""},
                                    {**OK, "ended_with": "abort"}])
def test_gate_sides(a, branch):
    eps = [{"A_main": a, "branches": [dict(branch)]}]
    tree_advantage(eps, "outcome")
    assert eps[0]["branches"][0]["A_branch"] == a
    tree_advantage(eps, "gate", code_gate)
    ok = branch is OK
    assert eps[0]["branches"][0]["A_branch"] == (max(a, 0.0) if ok else min(a, 0.0)) and eps[0]["A_main"] == a


@pytest.mark.parametrize("a", [0.75, -0.75])
@pytest.mark.parametrize("branch", [SEARCH_OK, {**SEARCH_OK, "summary": " "}, {**SEARCH_OK, "summary": None},
                                    {**SEARCH_OK, "searched": 0}, {**SEARCH_OK, "ended_with": "context"}])
def test_search_gate_sides(a, branch):
    eps = [{"A_main": a, "branches": [dict(branch)]}]
    tree_advantage(eps, "gate", search_gate)
    assert eps[0]["branches"][0]["A_branch"] == (a if branch is SEARCH_OK else min(a, 0.0)) and eps[0]["A_main"] == a


def test_gate_takes_the_environments_rule():
    """One setting, gate: each environment's group step applies its own rule (code: an ok branch max(A_main, 0); search:
    an ok branch A_main); a branch of the other environment lacks the rule's fields and raises."""
    code = [{"y": True, "branches": [dict(OK)]}, {"y": False, "branches": [dict(OK)]}]
    plan_group_step(code, branch_credit="gate")
    assert [e["branches"][0]["A_branch"] for e in code] == [2.0, 0.0]  # A_main ±2: an ok branch keeps max(A, 0)
    search = [{"y": True, "branches": [dict(SEARCH_OK)]}, {"y": False, "branches": [dict(SEARCH_OK)]}]
    search_group_step(search, branch_credit="gate")
    assert [e["branches"][0]["A_branch"] for e in search] == [2.0, -2.0]  # an ok branch keeps A_main
    with pytest.raises(ValueError):
        search_group_step([{"y": True, "branches": [dict(OK)]}], branch_credit="gate")  # a code branch
    with pytest.raises(ValueError):
        plan_group_step([{"y": True, "branches": [dict(SEARCH_OK)]}], branch_credit="gate")  # a search branch


def test_gates_need_their_fields_a_rule_and_a_known_value():
    with pytest.raises(ValueError):
        tree_advantage([{"A_main": 1.0, "branches": [{"ended_with": "commit", "passed": True}]}], "gate", code_gate)
    with pytest.raises(ValueError):
        tree_advantage([{"A_main": 1.0, "branches": [dict(OK)]}], "gate")  # no environment's rule
    for old in ("search_gate", "other"):  # no alias for the old name
        with pytest.raises(ValueError):
            tree_advantage([], old)
        with pytest.raises(ValueError):
            search_group_step([], branch_credit=old)
    [episode] = search_group_step([{"y": True, "branches": [{**SEARCH_OK, "searched": 0}]}], branch_credit="gate")
    assert episode["A_main"] == 0.0 and episode["branches"][0]["A_branch"] == 0.0


def _call(name: str) -> dict:
    return {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": name, "arguments": "{}"}}]}


def test_search_outcome_counts_a_branchs_own_calls(tmp_path):
    """tree.json branch messages start with the forking thread's history (main's search here): only the calls after
    the fork's prefix_len are the branch's. The tree has no rewinds (one written before they were recorded)."""
    prefix = [{"role": "system", "content": ""}, _call("search"), {"role": "tool", "content": ""}, _call("fork"),
              {"role": "tool", "content": ""}]
    def branch(bid, summary, *calls):
        return {"branch_id": bid, "ended_with": "done", "summary": summary,
                "messages": prefix + [{"role": "user", "content": "task"}] + [_call(c) for c in calls]}
    (tmp_path / "agent").mkdir()
    (tmp_path / "verifier").mkdir()
    (tmp_path / "agent/tree.json").write_text(json.dumps({"forks": [{"prefix_len": len(prefix), "branches": [
        branch("branch-0", "d1", "search", "open", "done"), branch("branch-1", "d2", "read_thread", "done")]}]}))
    (tmp_path / "verifier/answers.json").write_text(json.dumps({"main": {"correct": True}}))
    out = search_outcome(tmp_path, [])
    assert [(b["summary"], b["searched"]) for b in out["branches"]] == [("d1", 2), ("d2", 0)]
