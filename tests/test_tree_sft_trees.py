from octopus.tools.tree_sft.trees import fork_trees


def _asst(name):
    return {"role": "assistant", "content": "", "tool_calls": [{"id": name, "function": {"name": name, "arguments": "{}"}}]}


def test_fork_trees_split_root_join_and_branches():
    main = [
        {"role": "system", "content": "s"}, {"role": "user", "content": "bug"},
        _asst("view"), {"role": "tool", "content": "files"},              # exploration (root, trained in the first tree)
        _asst("checkpoint"), {"role": "tool", "content": "ok"},           # prefix_len = 4 -> join starts here
        _asst("fork"), {"role": "tool", "content": "forked"},
        {"role": "user", "content": "branches finished"},
        _asst("edit"), {"role": "tool", "content": "edited"},
        _asst("checkpoint"), {"role": "tool", "content": "ok"},           # second fork's freeze at 11
        _asst("fork"), {"role": "tool", "content": "forked"},
    ]
    branch = lambda task: {"branch_id": task, "task": task, "messages": [{"role": "user", "content": task}, _asst("view")], "ended_with": "view"}
    log = {"tools": [], "main": main, "forks": [
        {"prefix_len": 4, "fork_index": 6, "join_index": 8, "branches": [branch("a"), branch("b")]},
        {"prefix_len": 11, "fork_index": 13, "join_index": 15, "branches": [branch("c")]},  # one branch: skipped
    ]}
    trees = fork_trees(log)
    assert len(trees) == 1
    tree = trees[0]
    assert tree["root"] == main[:4]
    join, *branches = tree["children"]
    assert join["kind"] == "join" and join["messages"] == main[4:11]  # stops at the next fork's freeze
    assert [b["branch_id"] for b in branches] == ["a", "b"]
    assert tree["train_root"]


def test_only_first_tree_trains_its_root():
    main = [{"role": "user", "content": "bug"}, _asst("view"), {"role": "tool", "content": "x"}, _asst("fork"), {"role": "tool", "content": "y"},
            _asst("edit"), {"role": "tool", "content": "z"}, _asst("fork"), {"role": "tool", "content": "w"}, _asst("done")]
    branch = lambda task: {"branch_id": task, "task": task, "messages": [{"role": "user", "content": task}, _asst("view")]}
    log = {"tools": [], "main": main, "forks": [
        {"prefix_len": 3, "fork_index": 3, "join_index": 5, "branches": [branch("a"), branch("b")]},
        {"prefix_len": 7, "fork_index": 7, "join_index": 9, "branches": [branch("c"), branch("d")]},
    ]}
    first, second = fork_trees(log)
    assert first["train_root"] and not second["train_root"]  # second root = main[:7], already trained by the first tree


def test_flat_rows_put_root_labels_in_the_join_row_only():
    from octopus.tools.tree_sft.encode import IGNORE, flat_rows

    enc = {"root_ids": [1, 2], "root_labels": [IGNORE, 2], "children": [
        {"kind": "join", "ids": [3], "labels": [3]}, {"kind": "branch", "ids": [4], "labels": [4]},
    ]}
    join, branch = flat_rows(enc)
    assert join["labels"] == [IGNORE, 2, 3] and branch["labels"] == [IGNORE, IGNORE, 4]
