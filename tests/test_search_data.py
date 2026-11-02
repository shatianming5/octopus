"""Search-env data: the stratified split is deterministic and proportional; BrowseComp-Plus decrypts with the README canary."""
import base64, collections, hashlib, json

import pytest

pa = pytest.importorskip("pyarrow")
import pyarrow.parquet as pq

from octopus.envs.search import data

CANARY = "BENCHMARK DATA SHOULD NEVER APPEAR AS PLAIN TEXT ONLINE. canary GUID browsecomp:12345678-0000-4000-8000-00000000abcd"


def encrypt(text):
    key = hashlib.sha256(CANARY.encode()).digest()
    return base64.b64encode(bytes(b ^ key[i % len(key)] for i, b in enumerate(text.encode()))).decode()


def test_split_and_loaders(tmp_path):
    rates = {q: (q * 7919 % 17) / 16 for q in range(6102)}
    split = data.make_split(rates)
    assert split == data.make_split(dict(reversed(rates.items())))  # deterministic, order-free
    val, train = set(split["validation"]), set(split["train"])
    assert (len(val), len(train), len(val & train)) == (500, 3040, 0)
    by = collections.Counter(rates.values())
    for name, n in (("validation", 500), ("train", 3040)):
        got = collections.Counter(rates[q] for q in split[name])
        assert all(abs(got[s] - n * c / 6102) <= 1.5 for s, c in by.items())  # stratified by pass_rate

    gold = tmp_path / "gold"
    (gold / "data").mkdir(parents=True)
    pq.write_table(pa.table({"query_id": [1, 2, 3], "query": ["q1", "q2", "q3"],
                             "answer": ["a1", r"\boxed{\text{superheated vapor}}", r"$$\boxed{0.3 \text{ kg}}$$"],
                             "pass_rate": [0.0, 0.5, 1.0]}), gold / "data" / "train-0.parquet")
    (tmp_path / "split.json").write_text(json.dumps({"train": [3, 1], "validation": [2]}))
    assert [r["answer"] for r in data.openresearcher_split("train", gold, tmp_path / "split.json")] == ["0.3 kg", "a1"]
    assert data.openresearcher_split("validation", gold, tmp_path / "split.json")[0]["answer"] == "superheated vapor"
    for kept in (r"\boxed{a} and \boxed{b}", r"x = \frac{1}{2}", "$99 USD"):
        assert data.unbox(kept) == kept
    assert data.unbox(r"\boxed{\frac{x}{2}}") == r"\frac{x}{2}"

    bcp = tmp_path / "bcp"
    (bcp / "data").mkdir(parents=True)
    (bcp / "README.md").write_text(f"# card\n\nexample: CANARY = \"{CANARY}\"\n---\nCanary:\n\n{CANARY}")
    doc = lambda d: {"docid": encrypt(d), "text": encrypt("body"), "url": encrypt("u")}
    pq.write_table(pa.table({"query_id": ["769"], "query": [encrypt("Who wrote it? ünï")], "answer": [encrypt("Ann")],
                             "evidence_docs": [[doc("11"), doc("12")]], "gold_docs": [[doc("12")]],
                             "negative_docs": [[doc("99")]]}), bcp / "data" / "test-0.parquet")
    assert data.browsecomp_plus(bcp) == {"769": {"query_id": "769", "query": "Who wrote it? ünï", "answer": "Ann",
                                                 "evidence_docids": ["11", "12"], "gold_docids": ["12"]}}
