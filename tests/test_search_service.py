"""Retrieval service on a tiny corpus: bm25 and dense backends answer /search, /open and /health per the contract:
BrowseComp-Plus's search output (the top k, 5 by default, each hit the document's first 512 Qwen3 tokens)."""
import json, pickle, threading, urllib.error, urllib.request

import pytest

pa = pytest.importorskip("pyarrow")
pytest.importorskip("tantivy")
np = pytest.importorskip("numpy")
import pyarrow.parquet as pq

from octopus.envs.search import service

DOCS = {"d1": "---\ntitle: Octopus facts\ndate: 2020\n---\nThe octopus has three hearts and blue blood.",
        "d2": "Volcanoes of Iceland\n\n" + "filler words here " * 300 + "Eyjafjallajokull erupted in 2010 near Iceland.",
        "d3": "Cooking pasta: boil salted water, add the pasta, stir.",
        **{f"v{i}": f"Volcano note {i}: " + "lava cools into rock. " * 200 for i in range(5)}}


def call(port, path, body=None):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", json.dumps(body).encode() if body else None)
    try:
        with urllib.request.urlopen(req) as r:
            return r.status, json.load(r)
    except urllib.error.HTTPError as e:
        return e.code, json.load(e)


def test_service(tmp_path):
    (tmp_path / "pq").mkdir()
    pq.write_table(pa.table({"docid": list(DOCS), "text": list(DOCS.values()), "url": [f"https://x/{d}" for d in DOCS]}),
                   tmp_path / "pq" / "train-0.parquet")
    assert service.build(tmp_path / "pq", tmp_path / "store", threads=1) == 8

    bm25 = service.Engine(tmp_path / "store", "bm25", "toy")
    server = service.make_server(bm25, "127.0.0.1", 0)
    port = server.server_address[1]
    threading.Thread(target=server.serve_forever, daemon=True).start()
    assert call(port, "/health") == (200, {"corpus": "toy", "n_docs": 8, "backend": "bm25"})
    code, out = call(port, "/search", {"query": "When did Eyjafjallajokull erupt? (Iceland) AND -", "k": 2})
    top = out["results"][0]
    assert code == 200 and len(out["results"]) == 2  # only d2 and d1 ("and") match
    assert len(call(port, "/search", {"query": "volcano note"})[1]["results"]) == 5  # default k = 5 of 6 matches
    assert len(call(port, "/search", {"query": "volcano note", "k": 6})[1]["results"]) == 6  # the agent's k (10)
    assert top["docid"] == "d2" and top["title"] == "Volcanoes of Iceland" and top["score"] > 0
    tokens = bm25.tok.encode(DOCS["d2"], add_special_tokens=False)
    assert len(tokens) > 512 and top["snippet"] == bm25.tok.decode(tokens[:512], skip_special_tokens=True)
    assert top["snippet"].startswith("Volcanoes of Iceland") and "Eyjafjallajokull" not in top["snippet"]
    for long in ("filler words here " * 2000, "=" * 40000):  # past the 16k-char prefix; very long tokens
        whole = bm25.tok.encode(long, add_special_tokens=False)[:512]
        assert len(long) > service.PREFIX_CHARS
        assert bm25.snippet(long) == bm25.tok.decode(whole, skip_special_tokens=True)
    assert call(port, "/search", {"query": "octopus hearts"})[1]["results"][0] | {"score": 0} == {
        "docid": "d1", "title": "Octopus facts", "url": "https://x/d1", "snippet": DOCS["d1"], "score": 0}  # whole text
    q = {"query": "Iceland volcano"}
    assert call(port, "/search", q) == call(port, "/search", q)  # deterministic
    assert call(port, "/search", {"query": "?!"}) == (200, {"results": []})
    assert call(port, "/search", {"query": "snake_case octopus"})[0] == 200  # underscores must not become position-needing phrase queries
    assert call(port, "/open", {"docid": "d3", "offset": 15, "length": 4}) == (200, {
        "docid": "d3", "title": "Cooking pasta: boil salted water, add the pasta, stir.", "url": "https://x/d3",
        "text": "boil", "offset": 15, "total_length": len(DOCS["d3"])})
    assert call(port, "/open", {"docid": "nope"})[0] == 404 and call(port, "/search", {"k": 1})[0] == 400
    server.shutdown()

    pytest.importorskip("faiss")
    vecs = np.eye(3, 8, dtype=np.float16) * 3  # unnormalised fp16, as the published pickles may be
    pickle.dump((vecs[:2], ["d1", "d2"]), open(tmp_path / "corpus.shard1_of_2.pkl", "wb"))
    pickle.dump((vecs[2:], ["d3"]), open(tmp_path / "corpus.shard2_of_2.pkl", "wb"))
    dense = service.Engine(tmp_path / "store", "dense", "toy", str(tmp_path / "corpus.*.pkl"),
                           encode=lambda q: np.array([[0.6, 0, 0.8, 0, 0, 0, 0, 0]], dtype=np.float32))
    hits = dense.search("pasta water")
    assert [h["docid"] for h in hits] == ["d3", "d1", "d2"] and abs(hits[0]["score"] - 0.8) < 1e-3
    assert hits[0]["snippet"] == DOCS["d3"] and dense.health()["backend"] == "dense"
