from octopus.envs.swe.data.solvability import label, test_source as source_of  # renamed: pytest would collect it

A = {"class": "a"}
FOUND = {"hits": {"needle": ["m.py:1: needle"]}}


def test_label():
    assert label([A]) == "solvable"
    assert label([A, {"class": "b"}]) == "unsolvable"  # a lookup the tree does not answer
    assert label([{"class": "b"} | FOUND]) == "solvable"
    assert label([{"class": "c", "kind": "missing"} | FOUND]) == "solvable"
    assert label([{"class": "c", "kind": "ambiguous"}]) == "uncertain"
    assert label([{"class": "c", "kind": "ambiguous"} | FOUND]) == "solvable"
    assert label([{"class": "c", "kind": "contradicts"} | FOUND]) == "contradicts"  # a hit does not resolve it
    # the worst requirement wins: contradicts > unsolvable > uncertain > solvable
    assert label([{"class": "c", "kind": "ambiguous"}, {"class": "b"}]) == "unsolvable"
    assert label([{"class": "b"}, {"class": "c", "kind": "contradicts"}, A]) == "contradicts"


TESTS = '''import pytest

LIMIT = 3


def helper(x):
    return x * LIMIT


def unused():
    pass


class TestThing:
    def setUp(self):
        self.v = 1

    def test_method(self):
        assert helper(self.v) == 3


@pytest.mark.parametrize("n", [1, 2])
def test_param(n):
    assert helper(n) > 0
'''


def test_test_source(tmp_path):
    (tmp_path / "tests/hidden_full").mkdir(parents=True)
    (tmp_path / "tests/hidden_full/test_m.py").write_text(TESTS)
    src = source_of(tmp_path, ["test_m.py::TestThing::test_method", "test_m.py::test_param[1]",
                                 "test_m.py::test_param[2]", "test_m.py::test_gone", "other.md::other.md"])
    assert "def helper(x):" in src  # a top-level helper the test names
    assert "def unused" not in src and "def setUp(self):" in src
    assert src.count("def test_param(n):") == 1 and '@pytest.mark.parametrize("n", [1, 2])' in src
    assert "test_m.py::test_gone: not found" in src and "other.md: not in the hidden test files" in src
