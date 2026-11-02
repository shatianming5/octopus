"""Offline retrieval service for the search environment: one HTTP JSON service per corpus.

API (see the search-env contract):
  POST /search {"query", "k"=5}                    -> {"results": [{"docid", "title", "url", "snippet", "score"}]}
  POST /open   {"docid", "offset"=0, "length"=6000} -> {"docid", "title", "url", "text", "offset", "total_length"}
  GET  /health                                     -> {"corpus", "n_docs", "backend"}
Same query -> same results. As BrowseComp-Plus's search tool (searcher/tools.py, its defaults k=5 and
--snippet-max-tokens 512): the top K = 5 hits (a request's k: the search agent asks for 10), snippet = the
document's first 512 tokens of the Qwen/Qwen3-0.6B tokenizer (--tokenizer: that tokenizer, or a local copy of it),
the whole text when it is shorter.

Corpora (downloaded to <root>/datasets/<org>__<name>, built files in <root>/search-scratch/<corpus>/):
  browsecomp-plus  Tevatron/browsecomp-plus-corpus, 100,195 docs (3.3 GB text); dense: browsecomp-plus-indexes
  openresearcher   OpenResearcher/OpenResearcher-Corpus, 14.9M docs (48.6 GB text); dense: OpenResearcher-Indexes
Build once (CPU), then serve:
  python -m octopus.envs.search.service build --corpus browsecomp-plus
  python -m octopus.envs.search.service serve --corpus browsecomp-plus --backend bm25 --port 8000 --workers 8  # main
  OMP_NUM_THREADS=1 python -m octopus.envs.search.service serve --corpus browsecomp-plus --backend dense \
      --dense qwen3-embedding-0.6b --encoder /path/Qwen3-Embedding-0.6B --device cpu --port 8001
  (openresearcher: same commands with --corpus openresearcher; its only dense index is qwen3-embedding-8b.)

Backends and resources:
- Store: docs.arrow, an uncompressed Arrow IPC file memory-mapped at startup (text stays in the page cache, not the
  heap); a docid -> row dict costs ~0.1 GB for BrowseComp-Plus and ~2 GB / ~30 s for OpenResearcher.
- bm25: tantivy (Rust Lucene-like engine, a 5 MB wheel; BM25 k1=1.2 b=0.75 fixed by tantivy, English stemming, no
  stopwords, on-disk mmap index, multi-threaded build). Chosen over pyserini (needs Java 21; node-134 has 11) and bm25s
  (in-RAM sparse matrix and single-threaded Python tokenisation, impractical for 14.9M docs). BrowseComp-Plus's own
  BM25 is pyserini's LuceneSearcher with its defaults (k1=0.9 b=0.4, Lucene English analyzer): same model, other
  parameters, so scores and ties differ. BrowseComp-Plus on node-134 (busy, load ~40 on 32 cores): build 2 min on 16
  threads; /search k=5 p50 80 ms, p95 335 ms with the earlier query-window snippet. The 512-token cut tokenises a
  16k-char prefix of each hit (same first 512 tokens as the whole text, which BrowseComp-Plus tokenises): ~18 ms per
  hit on one core of the dev host, against ~38 ms whole for a 33k-char doc (BrowseComp-Plus's mean) and ~300 ms for
  200k chars; --workers N processes share the port; /open ~1 ms.
- dense: FAISS flat inner product with fp16 storage (IndexScalarQuantizer QT_fp16: lossless for OpenResearcher's fp16
  vectors, ~1e-3 rounding of BrowseComp-Plus's fp32 ones, half the RAM of IndexFlatIP) over the L2-normalised
  published (vectors, docids) pickles; the query is encoded with the matching Qwen3-Embedding model (last-token
  pooling, the BrowseComp-Plus query instruction) on --device.
  BrowseComp-Plus 0.6B (1024-d): 0.2 GB, loads in 1 s; FAISS search 15 ms with OMP_NUM_THREADS=1 (200 ms with 32
  OpenMP threads on the busy host), /search p50 24 ms excluding the encoder. OpenResearcher 8B (4096-d): the index
  alone is 122 GB of RAM, and loading its single 122 GB pickle peaks at ~250-370 GB, so it fits only a big-RAM host
  (node-134: 503 GB); a flat CPU scan of 122 GB is memory-bandwidth bound (estimate ~1 s per query, many OpenMP
  threads, unmeasured) and concurrent queries do not batch. For RL throughput on OpenResearcher use bm25, or put the
  vectors on GPUs (not built here). Run dense with --workers 1: every worker loads its own index.
Deps: pyarrow, tantivy, transformers (the snippet tokenizer); dense adds faiss-cpu, torch.
"""
import argparse, glob, json, os, pickle, re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

CORPORA = {"browsecomp-plus": ("Tevatron__browsecomp-plus-corpus", "Tevatron__browsecomp-plus-indexes/{dense}/*.pkl"),
           "openresearcher": ("OpenResearcher__OpenResearcher-Corpus", "OpenResearcher__OpenResearcher-Indexes/{dense}/*.pkl")}
QUERY_PREFIX = "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:"
K, SNIPPET_TOKENS, PREFIX_CHARS = 5, 512, 16384


def title_of(text):
    """The front-matter title if the page has one (BrowseComp-Plus docs start with ---\\ntitle: ...), else line 1."""
    m = re.match(r"---\n(?:.*\n)*?title: *(.*)\n", text[:2000])
    return (m.group(1) if m else next((l for l in text.splitlines() if l.strip()), "")).strip()[:200]


def build(parquet_dir, out_dir, threads=16):
    """parquet (docid, text, url) -> out_dir/docs.arrow (docid, url, title, text) + out_dir/bm25 (tantivy)."""
    import pyarrow as pa, pyarrow.parquet as pq, tantivy
    os.makedirs(os.path.join(out_dir, "bm25"))
    sb = tantivy.SchemaBuilder()
    sb.add_unsigned_field("row", stored=True)
    sb.add_text_field("body", tokenizer_name="en_stem", index_option="freq")
    writer = tantivy.Index(sb.build(), path=os.path.join(out_dir, "bm25")).writer(threads * 500_000_000, threads)
    schema, row = pa.schema([(c, pa.string()) for c in ("docid", "url", "title", "text")]), 0
    with pa.ipc.new_file(os.path.join(out_dir, "docs.arrow"), schema) as out:
        for f in sorted(glob.glob(os.path.join(parquet_dir, "*.parquet"))):
            for b in pq.ParquetFile(f).iter_batches(10_000, columns=["docid", "url", "text"]):
                docid, url, text = (b.column(c).to_pylist() for c in ("docid", "url", "text"))
                out.write_batch(pa.record_batch([docid, url, [title_of(t) for t in text], text], schema=schema))
                for t in text:
                    writer.add_document(tantivy.Document(row=row, body=t))
                    row += 1
    writer.commit()
    writer.wait_merging_threads()
    return row


class QwenEncoder:
    """Qwen3-Embedding query encoder as BrowseComp-Plus uses it: instruction prefix, last-token pooling, L2 norm."""

    def __init__(self, path, device="cpu"):
        import torch, transformers
        self.torch, self.device = torch, device
        self.tok = transformers.AutoTokenizer.from_pretrained(path, padding_side="left")
        self.model = transformers.AutoModel.from_pretrained(
            path, dtype=torch.float32 if device == "cpu" else torch.float16).to(device).eval()

    def __call__(self, query):
        with self.torch.no_grad():
            batch = self.tok([QUERY_PREFIX + query], truncation=True, max_length=512, return_tensors="pt")
            h = self.model(**batch.to(self.device)).last_hidden_state[:, -1].float()
            return self.torch.nn.functional.normalize(h, dim=-1).cpu().numpy()


class Engine:
    def __init__(self, store_dir, backend, corpus, embeddings=None, encode=None, tokenizer="Qwen/Qwen3-0.6B"):
        import pyarrow as pa, transformers
        self.corpus, self.backend = corpus, backend
        self.tok = transformers.AutoTokenizer.from_pretrained(tokenizer)
        self.docs = pa.ipc.open_file(pa.memory_map(os.path.join(store_dir, "docs.arrow"))).read_all()
        self.row = {d: i for i, d in enumerate(self.docs.column("docid").to_pylist())}
        if backend == "bm25":
            import tantivy
            self.index = tantivy.Index.open(os.path.join(store_dir, "bm25"))
            self.searcher = self.index.searcher()
        else:
            import faiss, numpy as np
            self.encode, self.rows, self.index = encode, [], None
            for f in sorted(glob.glob(embeddings)):  # each pickle is (reps [n, d], docids [n])
                reps, lookup = pickle.load(open(f, "rb"))
                if self.index is None:
                    self.index = faiss.IndexScalarQuantizer(reps.shape[1], faiss.ScalarQuantizer.QT_fp16,
                                                            faiss.METRIC_INNER_PRODUCT)
                for i in range(0, len(reps), 100_000):
                    chunk = np.ascontiguousarray(reps[i:i + 100_000], dtype=np.float32)
                    faiss.normalize_L2(chunk)
                    self.index.add(chunk)
                self.rows += [self.row[str(d)] for d in lookup]
                del reps

    def health(self):
        return {"corpus": self.corpus, "n_docs": len(self.row), "backend": self.backend}

    def _doc(self, row):
        return {c: self.docs.column(c)[row].as_py() for c in ("docid", "title", "url", "text")}

    def snippet(self, text):
        """The first SNIPPET_TOKENS tokens of text, as BrowseComp-Plus's search tool cuts them."""
        # a 16k-char prefix: its first 512 tokens are the full text's unless the cut lands among them (600: a margin)
        tokens = self.tok.encode(text[:PREFIX_CHARS], add_special_tokens=False)
        if len(text) > PREFIX_CHARS and len(tokens) <= 600:
            tokens = self.tok.encode(text, add_special_tokens=False)
        if len(tokens) <= SNIPPET_TOKENS:
            return text
        return self.tok.decode(tokens[:SNIPPET_TOKENS], skip_special_tokens=True)

    def search(self, query, k=K):
        if self.backend == "bm25":
            words = re.findall(r"[^\W_]+", query.lower())  # bare words: no query-language operators from the agent
            if not words:
                return []
            hits = [(s, self.searcher.doc(a)["row"][0])
                    for s, a in self.searcher.search(self.index.parse_query(" ".join(words), ["body"]), k).hits]
        else:
            scores, ids = self.index.search(self.encode(query), k)
            hits = [(s, self.rows[i]) for s, i in zip(scores[0].tolist(), ids[0].tolist()) if i >= 0]
        out = []
        for score, row in hits:
            d = self._doc(row)
            out.append({"docid": d["docid"], "title": d["title"], "url": d["url"],
                        "snippet": self.snippet(d["text"]), "score": float(score)})
        return out

    def open(self, docid, offset=0, length=6000):
        if docid not in self.row:
            return None
        d, offset = self._doc(self.row[docid]), max(0, int(offset))
        return {"docid": docid, "title": d["title"], "url": d["url"], "text": d["text"][offset:offset + int(length)],
                "offset": offset, "total_length": len(d["text"])}


def make_server(engine, host, port):
    class Handler(BaseHTTPRequestHandler):
        def send(self, code, obj):
            body = json.dumps(obj, ensure_ascii=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            self.send(*((200, engine.health()) if self.path == "/health" else (404, {"error": "not found"})))

        def do_POST(self):
            try:
                req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
                if self.path == "/search":
                    return self.send(200, {"results": engine.search(str(req["query"]), int(req.get("k", K)))})
                if self.path == "/open":
                    doc = engine.open(str(req["docid"]), req.get("offset", 0), req.get("length", 6000))
                    return self.send(200, doc) if doc else self.send(404, {"error": f"unknown docid {req['docid']}"})
                self.send(404, {"error": "not found"})
            except (KeyError, ValueError, TypeError, AttributeError) as e:
                self.send(400, {"error": repr(e)})

        def log_message(self, *args):
            pass

    class Server(ThreadingHTTPServer):
        allow_reuse_port = True  # --workers processes share the port; the kernel spreads connections

    return Server((host, port), Handler)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "serve"])
    ap.add_argument("--corpus", choices=sorted(CORPORA), required=True)
    ap.add_argument("--root", default=os.path.expanduser("~/octo-env"), help="holds datasets/ and search-scratch/")
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--backend", choices=["bm25", "dense"], default="bm25")
    ap.add_argument("--dense", default="qwen3-embedding-8b", help="index subdir: qwen3-embedding-{8b,4b,0.6b}")
    ap.add_argument("--encoder", help="Qwen3-Embedding model matching --dense (default: its HF name)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--tokenizer", default="Qwen/Qwen3-0.6B", help="snippet tokenizer (BrowseComp-Plus's) or its copy")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--workers", type=int, default=1, help="serving processes (bm25; each loads its own engine)")
    a = ap.parse_args()
    dataset, dense_glob = CORPORA[a.corpus]
    store = os.path.join(a.root, "search-scratch", a.corpus)
    if a.cmd == "build":
        print(build(os.path.join(a.root, "datasets", dataset, "data"), store, a.threads), "docs ->", store)
    else:
        for _ in range(a.workers - 1):  # one Python process serialises the hits' work: scale across processes
            if os.fork() == 0:
                break
        encode = None
        if a.backend == "dense":
            encode = QwenEncoder(a.encoder or "Qwen/Qwen3-Embedding-" + a.dense.split("-")[-1].upper(), a.device)
        engine = Engine(store, a.backend, a.corpus, os.path.join(a.root, "datasets", dense_glob.format(dense=a.dense)),
                        encode, a.tokenizer)
        print(f"pid {os.getpid()} serving {engine.health()} on {a.host}:{a.port}", flush=True)
        make_server(engine, a.host, a.port).serve_forever()
