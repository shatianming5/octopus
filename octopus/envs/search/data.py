"""Questions for the search environment, read from the downloaded HF parquet at runtime (only ids live in the repo).

- OpenResearcher (train/validation): OpenResearcher/OpenResearcher-Corpus-Gold-Doc, 6,102 questions with pass_rate.
  `python -m octopus.envs.search.data split <Gold-Doc dir>` writes the deterministic ids-only split (seed 0, stratified
  by pass_rate: 500 validation, then 3,040 train from the rest) to data/search/openresearcher_split.json.
  `python -m octopus.envs.search.data export <Gold-Doc dir> <out dir>` writes the RL workflow's inputs (part B,
  rl.search_questions / rl.search_split) on the lab host: openresearcher-questions.jsonl ({qid, question, answer}),
  train-qids.txt and validation-qids.txt.
- BrowseComp-Plus (test): Tevatron/browsecomp-plus, 830 queries whose fields are XOR-obfuscated with the canary in the
  dataset README. `browsecomp_plus()` decrypts in memory only; never write its output into the repo or committed logs.

Needs pyarrow.
"""
import argparse, base64, collections, glob, hashlib, json, os, random, re

SPLIT = os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "search", "openresearcher_split.json")


def _parquet(dataset_dir, columns):
    import pyarrow.parquet as pq
    return [r for f in sorted(glob.glob(os.path.join(dataset_dir, "data", "*.parquet")))
            for r in pq.read_table(f, columns=columns).to_pylist()]


def unbox(answer):
    """`\\boxed{x}` (alone, or inside $ / $$) -> x with `\\text{y}` -> y; any other answer unchanged. A third of the
    Gold-Doc answers are LaTeX-wrapped this way, so a bare response could never exact-match them."""
    m = re.fullmatch(r"\s*(\$*)\s*\\boxed\{(.*)\}\s*\1\s*", answer, re.S)
    if not m:
        return answer
    depth = 0
    for c in m[2]:
        depth += (c == "{") - (c == "}")
        if depth < 0:  # the \boxed brace closes before the end: `\boxed{a} and \boxed{b}`
            return answer
    if depth:
        return answer
    return re.sub(r" {2,}", " ", re.sub(r"\\text\{([^{}]*)\}", r"\1", m[2])).strip()  # `0.3 \text{ kg}` -> `0.3 kg`


def openresearcher(dataset_dir):
    """query_id -> {query_id, query, answer, pass_rate} for every Gold-Doc question, answers unboxed."""
    return {r["query_id"]: r | {"answer": unbox(r["answer"])}
            for r in _parquet(dataset_dir, ["query_id", "query", "answer", "pass_rate"])}


def allocate(sizes, n):
    """Largest-remainder split of n over strata proportional to sizes (ties -> smaller stratum key)."""
    total = sum(sizes.values())
    quota = {s: n * c / total for s, c in sizes.items()}
    out = {s: int(q) for s, q in quota.items()}
    for s in sorted(sizes, key=lambda s: (out[s] - quota[s], s))[:n - sum(out.values())]:
        out[s] += 1
    return out


def make_split(pass_rate, n_val=500, n_train=3040, seed=0):
    """{query_id: pass_rate} -> {"validation": ids, "train": ids}, both stratified by pass_rate in 1/16 bins (most
    rates are k/16; a few questions were sampled 15 or 7 times and land in the nearest bin)."""
    rng, strata = random.Random(seed), collections.defaultdict(list)
    for qid in sorted(pass_rate):
        strata[round(pass_rate[qid] * 16)].append(qid)
    for s in sorted(strata):
        rng.shuffle(strata[s])
    val = allocate({s: len(v) for s, v in strata.items()}, n_val)
    rest = {s: v[val[s]:] for s, v in strata.items()}
    train = allocate({s: len(v) for s, v in rest.items()}, n_train)
    return {"validation": sorted(q for s, v in strata.items() for q in v[:val[s]]),
            "train": sorted(q for s, v in rest.items() for q in v[:train[s]])}


def openresearcher_split(name, dataset_dir, split_path=SPLIT):
    """The questions of split `name` ("train" | "validation") in split-file order."""
    rows = openresearcher(dataset_dir)
    return [rows[q] for q in json.load(open(split_path))[name]]


def _decrypt(text, canary):
    data = base64.b64decode(text)
    key = hashlib.sha256(canary.encode()).digest()
    return bytes(b ^ key[i % len(key)] for i, b in enumerate(data)).decode()


def canary(dataset_dir):
    """The canary line at the end of the dataset README; it is the XOR password."""
    return re.findall(r"BENCHMARK DATA SHOULD NEVER APPEAR[^\n]*?browsecomp:[0-9a-f-]+",
                      open(os.path.join(dataset_dir, "README.md")).read())[-1]


def browsecomp_plus(dataset_dir):
    """query_id -> {query_id, query, answer, evidence_docids, gold_docids}, decrypted in memory."""
    pw, out = canary(dataset_dir), {}
    for r in _parquet(dataset_dir, ["query_id", "query", "answer", "evidence_docs", "gold_docs"]):
        out[r["query_id"]] = {"query_id": r["query_id"], "query": _decrypt(r["query"], pw),
                              "answer": _decrypt(r["answer"], pw),
                              "evidence_docids": [_decrypt(d["docid"], pw) for d in r["evidence_docs"]],
                              "gold_docids": [_decrypt(d["docid"], pw) for d in r["gold_docs"]]}
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["split", "export"])
    ap.add_argument("gold_doc_dir", help="downloaded OpenResearcher__OpenResearcher-Corpus-Gold-Doc")
    ap.add_argument("out", nargs="?", default=SPLIT, help="split: the split file; export: a directory")
    a = ap.parse_args()
    rows = openresearcher(a.gold_doc_dir)
    if a.cmd == "split":
        split = make_split({q: r["pass_rate"] for q, r in rows.items()})
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        json.dump({"seed": 0, "stratified_by": "round(pass_rate * 16)"} | split, open(a.out, "w"))
        print({k: len(v) for k, v in split.items()})
    else:
        os.makedirs(a.out, exist_ok=True)
        with open(os.path.join(a.out, "openresearcher-questions.jsonl"), "w") as f:
            f.writelines(json.dumps({"qid": q, "question": r["query"], "answer": r["answer"]}) + "\n"
                         for q, r in sorted(rows.items()))
        for name, ids in json.load(open(SPLIT)).items():
            if isinstance(ids, list):
                open(os.path.join(a.out, f"{name}-qids.txt"), "w").write("\n".join(map(str, ids)) + "\n")
        print(len(rows), "questions ->", a.out)
