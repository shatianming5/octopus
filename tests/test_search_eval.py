"""BrowseComp-Plus grading against a fake OpenAI-compatible judge: the shared judge (official prompt, parser and
non-thinking request), accuracy, the official recall (searched docids) and open recall (opened docids)."""
import json, threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from octopus.envs.search import reward
from octopus.eval import bcp as eval_bcp

BODIES = []


class Judge(BaseHTTPRequestHandler):
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        BODIES.append(body)
        prompt = body["messages"][0]["content"]
        if "[response]: Lyon" in prompt:  # a 200 reply without choices
            return self.reply({"error": "overloaded"})
        assert prompt.startswith("Judge whether the following [response]") and "[correct_answer]: Paris" in prompt
        verdict = ("correct: no\n**extracted_final_answer:** Paris\n**correct:** yes\n**confidence:** 90%"
                   if "[response]: It is Paris" in prompt else "extracted_final_answer: Rome\ncorrect: no\nconfidence: 40")
        self.reply({"choices": [{"message": {"content": verdict}}]})

    def reply(self, obj):
        body = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


def test_evaluate():
    assert eval_bcp.judge is reward.judge  # one judge for training and evaluation
    server = ThreadingHTTPServer(("127.0.0.1", 0), Judge)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    truth = {q: {"query": "Capital of France?", "answer": "Paris", "evidence_docids": ["1", "2", "3", "4"],
                 "gold_docids": ["2"]} for q in ("1", "2", "3", "4")}
    run = [{"query_id": "1", "response": "It is Paris", "retrieved_docids": ["1", "2", "4", "7"],
            "opened_docids": ["2", "4", "9"]},
           {"query_id": 2, "response": "Rome", "retrieved_docids": [], "opened_docids": []},
           {"query_id": "3", "response": "", "retrieved_docids": ["3"], "opened_docids": []},
           {"query_id": "4", "response": "Lyon", "retrieved_docids": [], "opened_docids": []}]
    summary, rows = eval_bcp.evaluate(run, truth, f"http://127.0.0.1:{server.server_address[1]}/v1", "judge")
    server.shutdown()
    assert summary == {"n": 4, "accuracy": 1 / 4, "judge_failures": 1, "recall": 1 / 4, "open_recall": 1 / 8,
                       "open_gold_recall": 1 / 4}
    assert [(r["correct"], r["confidence"]) for r in rows] == [(True, 90.0), (False, 40.0), (False, None), (None, None)]
    assert len(BODIES) == 3  # the empty response is incorrect without a judge call, as the official evaluation
    assert all(b["chat_template_kwargs"] == {"enable_thinking": False} and b["temperature"] == 0.0 for b in BODIES)
    assert not any("Paris" in json.dumps(r) for r in rows)  # per-query output carries no benchmark text


def test_entry_point_writes_no_judge_cache(monkeypatch, tmp_path):
    """python -m octopus.eval.bcp never uses OCTOPUS_JUDGE_CACHE: its keys would hold the decrypted question
    and gold answer on disk."""
    import runpy, sys
    from octopus.envs.search import data
    server = ThreadingHTTPServer(("127.0.0.1", 0), Judge)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    truth = {"1": {"query": "Capital of France?", "answer": "Paris", "evidence_docids": ["1"], "gold_docids": ["1"]}}
    monkeypatch.setattr(data, "browsecomp_plus", lambda dataset_dir: truth)
    (tmp_path / "run.jsonl").write_text(json.dumps({"query_id": "1", "response": "It is Paris",
                                                    "retrieved_docids": ["1"], "opened_docids": []}) + "\n")
    monkeypatch.setenv("OCTOPUS_JUDGE_CACHE", str(tmp_path / "cache.jsonl"))
    monkeypatch.setenv("OCTOPUS_JUDGE_URL", f"http://127.0.0.1:{server.server_address[1]}/v1")  # reward.py's name
    monkeypatch.setattr(sys, "argv", ["eval_bcp", str(tmp_path / "run.jsonl"), "--dataset-dir", "-", "--out",
                                      str(tmp_path / "out.jsonl")])
    monkeypatch.delitem(sys.modules, "octopus.eval.bcp")  # run afresh as __main__
    runpy.run_module("octopus.eval.bcp", run_name="__main__")
    server.shutdown()
    assert json.loads((tmp_path / "out.jsonl").read_text())["correct"] is True
    assert not (tmp_path / "cache.jsonl").exists()


def test_areal_side_entry_points_import_no_harbor():
    """AReaL's python (scripts/alpha/search_eval.sh's AREAL_PY) runs the evaluations and the trainer and has no harbor;
    only the episodes run on harbor's. No module these entry points import at load time may import harbor."""
    import ast
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]

    def file(name):
        path = root / Path(*name.split("."))
        return next((p for p in (path.with_suffix(".py"), path / "__init__.py") if p.exists()), None)

    def imports(name):  # module-level imports (not those inside functions)
        stack, out = list(ast.parse(file(name).read_text()).body), []
        while stack:
            node = stack.pop()
            if isinstance(node, ast.Import):
                out += [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and not node.level:
                out += [node.module] + [f"{node.module}.{a.name}" for a in node.names]
            elif not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                stack += ast.iter_child_nodes(node)
        return out

    for entry in ("octopus.eval.search_dev_eval", "octopus.eval.dev_eval", "octopus.rl.train"):
        seen, stack = {entry}, [entry]
        while stack:
            for dep in imports(stack.pop()):
                assert dep.split(".")[0] != "harbor", f"{entry} imports harbor (via {sorted(seen)})"
                if dep.startswith("octopus") and dep not in seen and file(dep):
                    seen.add(dep)
                    stack.append(dep)
