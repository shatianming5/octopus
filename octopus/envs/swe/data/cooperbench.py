"""CooperBench (arXiv 2601.13295) as 2-task bundles: one feature pair of one PR = one Harbor task.

Source: the HF dataset CooperBench/cooperbench-dataset (a git clone, DS_DIR), the 9 Python repos: 24 PRs ("tasks"),
165 features, 560 pairs. Per PR: combined.patch (the whole PR, every feature, conflicts resolved), runner.sh (the
official test command), feature<i>/{feature.md, feature.patch, tests.patch}. Image akhatua/cooperbench-<repo>:task<id>
(the upstream repo at the PR's base commit in /workspace/repo, editable install in the system Python).

Bundle task dir (Harbor), like build_bundles.py's with n = 2:
  task.toml       healthcheck: /workspace/repo moved to /testbed (a symlink back, so the editable install's .pth still
                  points at the work tree), the system Python linked as /opt/miniconda3/envs/testbed/bin/python (the
                  path every verifier and tool uses), feature statements to /tasks/<k>.md + INDEX.md, `rm -rf .git`
                  (the image's history holds the upstream PR) and one fresh commit
  instruction.md  2 features in /tasks, one repo
  tests/test.sh   INFRA_RESTORE, check.py
  tests/check.py  per task k: every test file of the tree back to the initial commit (the official harness drops test
                  files from the graded patch), k's tests.patch applied, the PR's official test command (runner.sh
                  without its installs); solved = exit 0 and >= 1 passed -> /logs/verifier/tasks.json {k: 0/1},
                  reward.txt = solved fraction
  tests/tasks.json {k: {feature, tests patch path, command}};  tests/tests_<k>.patch feature k's tests;
                  tests/hidden_full/ empty (the RL bundle verifier copies it in; the base tests are in the tree)
  solution/fix_<k>.patch  feature k's gold patch (the RL verifier's check validity reads these);
  solution/combined.patch the PR's combined patch; solve.sh applies it: the "both solved" oracle (the two gold patches
                  together do not always pass both: click 2800 f1+f2 conflict semantically)
Manifest NAME.manifest.jsonl: one row per bundle with repo, pr, features, split (by repo: SPLIT).

usage (node-94 /home/user/harbor-udocker, `source scripts/env.sh`):
  python3 -I cooperbench.py fetch DS_DIR WORK [jobs=4]   the 24 images through MIRRORS (blobs sha256-checked,
                                   Entrypoint runner.sh cleared: it resets and cleans the tree) -> `udocker load`
  PYTHONPATH=<code> harbor-python cooperbench.py build DS_DIR NAME     datasets/NAME + NAME.manifest.jsonl, NAME.manifest.full.jsonl
  PYTHONPATH=<code> harbor-python cooperbench.py accept NAME [conc=12]  per bundle, one container, twice: no patch,
                                   gold A, gold B, combined -> NAME.acceptance.jsonl/.json, NAME-valid/ (+ manifest),
                                   with test.sh wall times
  PYTHONPATH=<code> harbor-python cooperbench.py scan DS_DIR NAME [conc=12]   leak scan, one container per PR -> NAME.leaks.json
  cooperbench_shard.sh SRC CODE NAME CONC REPO...   on another host: accept + scan those repos' bundles, results back
  PYTHONPATH=<code> harbor-python cooperbench.py merge NAME     shard results -> NAME.merged.*, NAME-valid, NAME-train-valid
  PYTHONPATH=<code with the connect guard (#225)> harbor-python cooperbench.py offline NAME [conc=10]   combined patch
                                   graded with the network blocked -> NAME.offline.jsonl, bundles that need the network
  PYTHONPATH=<code> harbor-python cooperbench.py commands NAME  test commands in NAME's tasks.json from COMMANDS
  PYTHONPATH=<code> harbor-python cooperbench.py drop NAME LIST  NAME without LIST's bundles (old version -> NAME-v1)
"""
import base64, concurrent.futures, gzip, hashlib, io, itertools, json, os, random, re, shutil, subprocess, sys, tarfile, \
    threading, time, urllib.error, urllib.parse, urllib.request, zipfile
from pathlib import Path

# repo-level split (coordinator, 2026-10-20)
SPLIT = {"pallets_jinja": "train", "samuelcolvin_dirty_equals": "train", "openai_tiktoken": "train", "pillow": "train",
         "llama_index": "train", "huggingface_datasets": "train", "dottxt_ai_outlines": "val", "pallets_click": "test",
         "dspy": "test"}
# fetch order: the small images, then outlines (a 4.2 GB layer) early so it is not the tail
REPOS = ["pallets_click_task", "pallets_jinja_task", "samuelcolvin_dirty_equals_task", "openai_tiktoken_task", "pillow_task",
         "dottxt_ai_outlines_task", "huggingface_datasets_task", "llama_index_task", "dspy_task"]


CL100K = "https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken"
# files the tests would download, baked into the image as one more layer (grading is offline)
NLTK = "https://cdn.jsdelivr.net/gh/nltk/nltk_data@gh-pages/packages/"  # raw.githubusercontent.com stalls from node-94
LLAMA_STATIC = "workspace/repo/llama-index-core/llama_index/core/_static/"  # gitignored caches llama_index fills on import
EXTRA_FILES = {"akhatua/cooperbench-openai-tiktoken": {  # tiktoken's default cache: /tmp/data-gym-cache/sha1(url)
    "tmp/data-gym-cache/" + hashlib.sha1(CL100K.encode()).hexdigest(): CL100K},
    "akhatua/cooperbench-llama-index": {
    LLAMA_STATIC + "tiktoken_cache/" + hashlib.sha1(CL100K.encode()).hexdigest(): CL100K,
    LLAMA_STATIC + "nltk_cache/corpora/stopwords.zip": NLTK + "corpora/stopwords.zip",
    LLAMA_STATIC + "nltk_cache/tokenizers/punkt_tab.zip": NLTK + "tokenizers/punkt_tab.zip"}}

# the libraries' own switches, set in the image: offline ones, without which every test run waits on a blocked network
# (litellm fetches its model cost map from GitHub, ~11 s per pytest start; huggingface_hub checks cached models), and
# NUM_WORKERS=1: DSPy 8587's tests start a litellm proxy that otherwise forks one worker per CPU, which under PRoot is
# not listening within the tests' 10 s (32 CPUs: 10 s; one worker: ready at once)
EXTRA_ENV = {"akhatua/cooperbench-dspy": ["LITELLM_LOCAL_MODEL_COST_MAP=True", "NUM_WORKERS=1"],
             "akhatua/cooperbench-dottxt-ai-outlines": ["HF_HUB_OFFLINE=1"],
             # its base test loads a llama.cpp model with Llama.from_pretrained, which lists the Hub repo (no offline
             # mode): through the hf-mirror.com proxy instead (reachable from node-134; ~45 s per test run)
             "akhatua/cooperbench-dottxt-ai-outlines:task1706": ["HF_ENDPOINT=https://hf-mirror.com"]}


def image_of(repo_dir: str, task: str) -> str:
    return f"akhatua/cooperbench-{repo_dir.removesuffix('_task').replace('_', '-')}:{task}"


def prs(ds: Path) -> list:
    return [(r, t) for r in REPOS for t in sorted((p.name for p in (ds / r).glob("task*")), key=lambda x: int(x[4:]))]


# ---------------------------------------------------------------- images

# Docker Hub mirrors (2026-10-20/21 tests, /home/user/octopus-sft/coupled/mirrors*.txt): ~0.2-0.6 MB/s per connection
# from node-94, ~2.2 MB/s from node-68; docker.1ms.run redirects to a CDN, docker-0.unsee.tech serves byte ranges itself
MIRRORS = ["docker.1ms.run", "docker-0.unsee.tech"]


def fetch(ds: str, work: str, jobs: int = 4):
    ua = {"User-Agent": "curl/8.5.0"}
    work = Path(work)
    (work / "blobs").mkdir(parents=True, exist_ok=True)
    udocker = [os.environ["N"] + "/bin/udocker"]
    have = subprocess.run([*udocker, "images"], capture_output=True, text=True).stdout
    todo = [image_of(r, t) for r, t in prs(Path(ds)) if image_of(r, t) not in have]
    blob_locks, load_lock, conns, PART = {}, threading.Lock(), threading.Semaphore(48), 32 << 20

    def get_json(repo, ref, accept, tok):
        h = {**ua, "Accept": accept, **({"Authorization": f"Bearer {tok}"} if tok else {})}
        return json.load(urllib.request.urlopen(urllib.request.Request(f"https://{MIRRORS[0]}/v2/{repo}/manifests/{ref}",
                                                                       headers=h), timeout=120))

    def token(repo, mirror=MIRRORS[0]):
        try:
            urllib.request.urlopen(urllib.request.Request(f"https://{mirror}/v2/", headers=ua), timeout=30)
            return ""
        except urllib.error.HTTPError as e:
            wa = e.headers.get("WWW-Authenticate", "")
        kv = {k.strip(): v for k, v in (p.split("=", 1) for p in wa.split(" ", 1)[1].replace('"', "").split(","))}
        q = urllib.parse.urlencode({"service": kv.get("service", ""), "scope": f"repository:{repo}:pull"})
        d = json.load(urllib.request.urlopen(urllib.request.Request(f"{kv['realm']}?{q}", headers=ua), timeout=30))
        return d.get("token") or d.get("access_token", "")

    def part(locate, i, a, b, dest):
        # a layer comes as 32 MB byte ranges over many connections spread over the mirrors; a rerun keeps finished ranges
        with conns:
            for attempt in range(20):
                if dest.exists() and dest.stat().st_size == b - a + 1:
                    return True
                url, head = locate(MIRRORS[(i + attempt) % len(MIRRORS)], attempt > 0)
                if url:  # --max-filesize: a server that ignores the range would send the whole layer
                    subprocess.run(["curl", "-sS", "-f", "--retry", "3", "-m", "1800", "--max-filesize", str(b - a + 1),
                                    "-r", f"{a}-{b}", *head, "-o", str(dest), url])
                else:
                    time.sleep(30)  # 429 or an unreachable mirror
        return False

    def blob(repo, digest, size):
        path = work / "blobs" / digest.split(":")[1]
        parts = work / "parts" / digest.split(":")[1]
        where, where_lock = {}, threading.Lock()

        def locate(mirror, failed):  # tokens and signed CDN URLs expire: fresh ones after a failure or 10 min
            with where_lock:
                url, head, t = where.get(mirror, ("", [], 0.0))
                if not url or time.time() - t > 600 or (failed and time.time() - t > 30):
                    try:
                        tok = token(repo, mirror)
                    except Exception:
                        tok = None
                    head = ["-H", f"Authorization: Bearer {tok}"] if tok else []
                    reg = f"https://{mirror}/v2/{repo}/blobs/{digest}"
                    got = subprocess.run(["curl", "-sS", "-m", "30", "-r", "0-0", "-o", "/dev/null", "-w",
                                          "%{http_code} %{redirect_url}", *head, reg], capture_output=True, text=True).stdout
                    code, _, redirect = got.partition(" ")
                    url, head = (redirect, []) if redirect else (reg if code in ("200", "206") else "", head)
                    where[mirror] = (url, head, time.time())
                return url, head

        with blob_locks.setdefault(digest, threading.Lock()):  # images of one repo share most layers
            for attempt in range(8):
                if path.exists():
                    with open(path, "rb") as f:
                        if hashlib.file_digest(f, "sha256").hexdigest() == digest.split(":")[1]:
                            shutil.rmtree(parts, ignore_errors=True)
                            return path
                    path.unlink()
                if attempt:
                    print(f"{repo} blob {digest[:19]} retry {attempt}", flush=True)
                parts.mkdir(parents=True, exist_ok=True)
                spans = [(i, a, min(a + PART, size) - 1, parts / str(i)) for i, a in enumerate(range(0, size, PART))]
                with concurrent.futures.ThreadPoolExecutor(len(spans)) as pool:
                    if all(pool.map(lambda x: part(locate, *x), spans)):
                        with open(path, "wb") as out:
                            for *_, dest in spans:
                                with open(dest, "rb") as f:
                                    shutil.copyfileobj(f, out, 1 << 20)
        raise RuntimeError(f"blob {digest} failed")

    def extra_layer(repo):
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as tf:
            for dest, url in EXTRA_FILES[repo].items():
                for attempt in range(5):  # resets happen (openaipublic from node-136)
                    try:
                        data = urllib.request.urlopen(urllib.request.Request(url, headers=ua), timeout=600).read()
                        break
                    except OSError:
                        if attempt == 4:
                            raise
                        time.sleep(30)
                info = tarfile.TarInfo(str(Path(dest).parent))
                info.type, info.mode = tarfile.DIRTYPE, 0o777
                tf.addfile(info)
                files = {dest: data}
                if dest.endswith(".zip"):  # unpacked next to the zip, as nltk.download leaves it (find() misses the zip)
                    with zipfile.ZipFile(io.BytesIO(data)) as z:
                        files |= {str(Path(dest).parent / n): z.read(n) for n in z.namelist() if not n.endswith("/")}
                for name, blob_ in files.items():
                    info = tarfile.TarInfo(name)
                    info.size, info.mode = len(blob_), 0o644
                    tf.addfile(info, io.BytesIO(blob_))
        lid = hashlib.sha256(buf.getvalue()).hexdigest()
        (work / "blobs" / lid).write_bytes(buf.getvalue())
        return lid, work / "blobs" / lid

    def one(image):
        repo, tag = image.split(":")
        t0, tok = time.time(), token(repo)
        idx = get_json(repo, tag, "application/vnd.oci.image.index.v1+json,"
                       "application/vnd.docker.distribution.manifest.list.v2+json", tok)
        dg = next(m["digest"] for m in idx["manifests"] if m.get("platform", {}).get("architecture") == "amd64")
        man = get_json(repo, dg, "application/vnd.oci.image.manifest.v1+json,"
                       "application/vnd.docker.distribution.manifest.v2+json", tok)
        cfg = json.loads(blob(repo, man["config"]["digest"], man["config"]["size"]).read_text())
        cfg["config"]["Entrypoint"] = None  # runner.sh: `git reset --hard && git clean -fdx`, then pip install
        cfg["config"]["Env"] = cfg["config"].get("Env", []) + EXTRA_ENV.get(image, EXTRA_ENV.get(repo, []))
        layers = [(l["digest"].split(":")[1], blob(repo, l["digest"], l["size"])) for l in man["layers"]]
        layers += [extra_layer(repo)] if repo in EXTRA_FILES else []
        print(f"DOWNLOADED {image} {sum(l['size'] for l in man['layers']) / 1e6:.0f} MB {time.time() - t0:.0f}s", flush=True)
        with load_lock:  # one `udocker load` at a time into the shared UDOCKER_DIR
            save = work / "save"
            shutil.rmtree(save, ignore_errors=True)
            save.mkdir()
            config = json.dumps(cfg).encode()  # udocker keeps it in its shared layers dir by this name
            config_name = hashlib.sha256(config).hexdigest() + ".json"
            (save / config_name).write_bytes(config)
            parent, ids = None, []
            for lid, path in layers:  # legacy docker-save layout, what udocker 1.3.17 loads
                (save / lid).mkdir()
                (save / lid / "VERSION").write_text("1.0")
                (save / lid / "json").write_text(json.dumps({"id": lid, **({"parent": parent} if parent else {})}))
                with gzip.open(path) as src, open(save / lid / "layer.tar", "wb") as dst:
                    shutil.copyfileobj(src, dst, 1 << 20)
                parent = lid
                ids.append(lid)
            (save / "manifest.json").write_text(json.dumps([{"Config": config_name, "RepoTags": [image],
                                                              "Layers": [f"{i}/layer.tar" for i in ids]}]))
            (save / "repositories").write_text(json.dumps({repo: {tag: ids[-1]}}))
            with tarfile.open(work / "image.tar", "w") as tf:
                for name in sorted(os.listdir(save)):
                    tf.add(save / name, arcname=name)
            shutil.rmtree(save)
            rc = subprocess.run([*udocker, "load", "-i", str(work / "image.tar")]).returncode
            (work / "image.tar").unlink()
        print(f"LOADED {image} rc={rc} {time.time() - t0:.0f}s", flush=True)
        return rc

    def safe(image):  # one failed image does not stop the others; a rerun resumes it
        try:
            return one(image)
        except Exception as exc:
            print(f"FAILED {image}: {exc}", flush=True)
            return 1

    with concurrent.futures.ThreadPoolExecutor(jobs) as pool:
        bad = [i for i, rc in zip(todo, pool.map(safe, todo)) if rc]
    print(f"images fetched: {len(todo) - len(bad)}/{len(todo)}, failed {bad}")



# ---------------------------------------------------------------- tasks

# The PR's official test command: runner.sh after its patches, without the installs (the image has them; grading is
# offline). TESTS_PATCH is the feature's tests.patch (runner.sh's /patches/$TEST_PATCH); the agent's tree is always
# the "feature patch", so runner.sh's `if [[ -n "$FEATURE_PATCH" ]]` branches are taken.
PATCH_TARGETS = r"""
for file in $(grep -oE '^\+\+\+ b/.*' "$TESTS_PATCH" | sed 's|^+++ b/||' | sort -u); do
    if [[ -e "$file" ]]; then timeout 300 python -m pytest "$file" -v || exit $?; else exit 1; fi
done"""
COMMANDS = {
    "pallets_click_task/task2068": "python -m pytest tests/test_termui.py -v",
    "pallets_click_task/task2800": "python -m pytest tests/test_shell_completion.py tests/test_context.py -v",
    "pallets_click_task/task2956": "python -m pytest tests/test_arguments.py tests/test_imports.py tests/test_info_dict.py "
                                   "tests/test_options.py -v",
    "pallets_jinja_task/task1465": "python -m pytest tests/test_async_filters.py tests/test_filters.py -v",
    "pallets_jinja_task/task1559": "python -m pytest tests/test_ext.py -v",
    "pallets_jinja_task/task1621": "python -m pytest tests/test_loader.py -v",
    "samuelcolvin_dirty_equals_task/task43": "python -m pytest tests/test_other.py -v",
    "openai_tiktoken_task/task0": 'if compgen -G "tests/test_feature*.py" > /dev/null; then UNIT_TEST_SPEC="tests/test_feature*.py"; '
                                  'else UNIT_TEST_SPEC="tests/"; fi\npython -m pytest $UNIT_TEST_SPEC -v',
    "pillow_task/task25": 'python -m pytest "Tests/test_image.py" -v',
    "pillow_task/task290": 'python -m pytest "Tests/test_image_quantize.py" -v',
    "pillow_task/task68": 'python -m pytest "Tests/test_imageops.py" -v',
    "dspy_task/task8394": 'timeout 300 python -m pytest "tests/clients/test_cache.py" -v --tb=short' + PATCH_TARGETS,
    "dspy_task/task8563": 'timeout 300 python -m pytest "tests/adapters/test_tool.py" "tests/adapters/test_chat_adapter.py" '
                          '-v --tb=short',
    "dspy_task/task8587": 'timeout 300 python -m pytest "tests/streaming/test_streaming.py" -v --tb=short',
    "dspy_task/task8635": 'timeout 300 python -m pytest "tests/teleprompt/test_utils.py" "tests/propose/test_grounded_proposer.py" '
                          '-v --tb=short\nfor file in tests/propose/test_grounded_proposer1.py tests/propose/test_grounded_proposer2.py '
                          'tests/propose/test_grounded_proposer4.py tests/propose/test_grounded_proposer5.py '
                          'tests/propose/test_grounded_proposer6.py; do\n    if [[ -e "$file" ]]; then python -m pytest "$file" -v; fi\n'
                          'done' + PATCH_TARGETS,
    "llama_index_task/task17070": "python -m pytest llama-index-core/tests/evaluation/test_metrics.py -v",
    "llama_index_task/task17244": 'timeout 300 python -m pytest "llama-index-core/tests/base/llms/test_types.py" -v',
    # the base tests on the mock_pdf_bytes / pdf_url fixtures download a PDF from w3.org: deselected like 7309's (node
    # ids are relative to pytest's rootdir, llama-index-core/: the llama-index-core/... ids deselected nothing)
    "llama_index_task/task18813": 'timeout 300 python -m pytest "llama-index-core/tests/base/llms/test_types.py" -v' + "".join(
        f' --deselect "tests/base/llms/test_types.py::test_document_block_from_{x}"' for x in ("bytes", "b64", "path", "url")),
    "huggingface_datasets_task/task3997": "python -m pytest tests/features/test_features.py -v",
    "huggingface_datasets_task/task6252": 'timeout 300 python -m pytest "tests/features/test_image.py" -v',
    "huggingface_datasets_task/task7309": 'timeout 300 python -m pytest "tests/io/test_parquet.py" -v '
                                          '--deselect "tests/io/test_parquet.py::test_parquet_read_geoparquet"',
    "dottxt_ai_outlines_task/task1371": "timeout 300 python3 -m pytest tests/test_prompts.py -v",
    "dottxt_ai_outlines_task/task1655": "timeout 300 python -m pytest tests/types/test_custom_types.py -v",
    "dottxt_ai_outlines_task/task1706": """export TRANSFORMERS_NO_TF=1 TRANSFORMERS_NO_FLAX=1 PANDAS_NO_USE_PYARROW=1 LLAMA_CPP_FORCE_CPU=1 PYTORCH_ENABLE_MPS_FALLBACK=1
set +e
timeout 300 python -m pytest tests/backends/test_xgrammar.py -v 2>&1 | tee /tmp/pytest_output.log
PYTEST_EXIT=${PIPESTATUS[0]}
set -e
if grep -qE "[1-9][0-9]* passed" /tmp/pytest_output.log && ! grep -qE "FAILED|ERROR" /tmp/pytest_output.log; then
    if [[ $PYTEST_EXIT -eq 139 ]]; then exit 0; else exit $PYTEST_EXIT; fi
else
    exit ${PYTEST_EXIT:-1}
fi""",
}
PACKAGES = {"pallets_click_task": "click", "pallets_jinja_task": "jinja2", "samuelcolvin_dirty_equals_task": "dirty_equals",
            "openai_tiktoken_task": "tiktoken", "pillow_task": "PIL", "dspy_task": "dspy", "llama_index_task": "llama_index.core",
            "huggingface_datasets_task": "datasets", "dottxt_ai_outlines_task": "outlines"}

CHECK = r"""import json, os, re, subprocess, time
from pathlib import Path

# Per task k, CooperBench's grading: the work tree's test files back to the initial commit (the official harness drops
# test files from the graded patch: cooperbench eval/sandbox.py _filter_test_files, same patterns), k's tests.patch
# applied, the PR's official test command (runner.sh without its installs); solved = exit 0 and the first
# "N passed" with N > 0 (sandbox.py _run_tests / _parse_results).
TEST_DIR = re.compile(r"/(tests?|__tests__|spec|testdata)/")
TEST_FILE = re.compile(r"(^|/)(test_[^/]+|[^/]+_test\.[A-Za-z0-9]+|[^/]+\.(test|spec)\.[A-Za-z0-9]+"
                       r"|[^/]*Test[s]?\.(java|kt|cs|scala)|tests?\.py)$")
tasks = json.loads(Path(__file__).with_name("tasks.json").read_text())


def git(*args):
    return subprocess.run(["git", *args], cwd="/testbed", capture_output=True, text=True)


def reset_tests():
    git("reset", "-q", "HEAD")  # take_snap/apply did git add -A; a new test file is staged, not in HEAD
    is_test = lambda p: TEST_DIR.search("/" + p) or TEST_FILE.search(p)
    tracked = [p for p in git("diff", "--name-only", "-z", "HEAD").stdout.split("\0") if p and is_test(p)]
    if tracked:
        git("checkout", "HEAD", "--", *tracked)
    for p in git("ls-files", "--others", "--exclude-standard", "-z").stdout.split("\0"):
        if p and is_test(p):
            os.remove(os.path.join("/testbed", p))


solved, seconds, details = {}, {}, {}
for k, t in tasks.items():
    reset_tests()
    start = time.time()
    applied = git("apply", "--ignore-whitespace", "--ignore-space-change", t["tests"])
    if applied.returncode:
        solved[k], seconds[k], details[k] = 0, 0.0, {"tests_patch": applied.stderr[-1000:]}
        continue
    try:
        run = subprocess.run(["bash", "-c", "set -e\n" + t["command"]], cwd="/testbed", capture_output=True, text=True,
                             env={**os.environ, "TESTS_PATCH": t["tests"]}, timeout=900)
        rc, out = run.returncode, run.stdout + run.stderr
    except subprocess.TimeoutExpired:
        rc, out = None, "timeout 900 s"
    passed = re.search(r"(\d+) passed", out)
    solved[k] = int(rc == 0 and bool(passed) and int(passed.group(1)) > 0)
    seconds[k] = round(time.time() - start, 1)
    details[k] = {"rc": rc, "passed": int(passed.group(1)) if passed else 0, "tail": out[-3000:]}
reset_tests()
res = Path("/logs/verifier")
res.mkdir(parents=True, exist_ok=True)
(res / "tasks.json").write_text(json.dumps(solved))
(res / "seconds.json").write_text(json.dumps(seconds))
(res / "details.json").write_text(json.dumps(details))
(res / "reward.txt").write_text(f"{sum(solved.values()) / len(solved)}\n")
"""

INSTRUCTION = """The repository in /testbed needs {n} new features. Each one is a task described in /tasks/<k>.md \
(k = 1..{n}); /tasks/INDEX.md lists them, one line per task. Implement as many tasks as you can by changing the code in \
/testbed; each task is graded on its own by hidden tests.
"""
SOLVE_SH = "#!/bin/bash\ncd /testbed && git apply --whitespace=nowarn /solution/combined.patch\n"


# llama_index 17244/18813 install the root pyproject's dependencies, which bring a PyPI llama-index-core (not editable)
# into site-packages; out with it, and a .pth puts the tree there instead, as 17070's editable install does
LLAMA_TREE = ('SP=$(python -c "import site;print(site.getsitepackages()[0])") && rm -rf $SP/llama_index/core'
              ' && echo /testbed/llama-index-core > $SP/llama_index_core_testbed.pth')
SETUP_EXTRA = {"llama_index_task/task17244": LLAMA_TREE, "llama_index_task/task18813": LLAMA_TREE}


def setup_command(tasks: list, extra: str = "") -> str:
    """Healthcheck: the repo to /testbed (Harbor's work dir; /workspace/repo stays as a symlink, the editable install
    points there), the system Python where the verifiers look for it, `extra` (SETUP_EXTRA), statements, fresh history.
    uv's cache goes too: it keeps whole copies of installed packages (llama_index 18813: its PyPI llama-index-core, gold
    lines included; site-packages has hard links to it, so imports are unaffected)."""
    from octopus.envs.swe.data.build_bundles import tasks_tar
    steps = ["rm -rf /workspace/repo/.git /root/.cache/uv", "rmdir /testbed", "mv /workspace/repo /testbed", "ln -s /testbed /workspace/repo",
             "mkdir -p /opt/miniconda3/envs/testbed/bin", "ln -sf $(command -v python) /opt/miniconda3/envs/testbed/bin/python",
             *([extra] if extra else []),
             f"mkdir -p /tasks && printf %s {tasks_tar(tasks)} | base64 -d | tar -xz -C /tasks",
             "cd /testbed && git init -q && git add -A",
             'git -c user.name=octopus -c user.email=octopus@localhost commit -qm "Initial commit"',
             "touch .git/octopus_ready"]
    return f"if [ ! -f /testbed/.git/octopus_ready ]; then {' && '.join(steps)}; fi && test -f /testbed/.git/octopus_ready"


def build(ds: str, name: str):
    from octopus.envs.swe.taskset import INFRA_RESTORE, branch_setup, task_config_text, write_task_tree
    from octopus.envs.swe.data.regen_tasks import N
    ds, dest = Path(ds), N / "datasets" / name
    if dest.exists():
        raise SystemExit(f"{dest} exists; pick a new name")
    test_sh = "#!/bin/bash\nset +e\ncd /testbed\n" + INFRA_RESTORE + "/opt/miniconda3/envs/testbed/bin/python /tests/check.py\n"
    sha = subprocess.run(["git", "-C", str(ds), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    rows = []
    for repo, pr in prs(ds):
        src = ds / repo / pr
        feats = sorted((int(p.name[7:]) for p in src.glob("feature*")))
        for pair in itertools.combinations(feats, 2):
            bundle = f"{repo.removesuffix('_task')}.{pr}.f{pair[0]}-f{pair[1]}"
            tasks = [{"statement": (src / f"feature{f}/feature.md").read_text().strip()} for f in pair]
            spec = {"instance_id": bundle, "image": image_of(repo, pr), "verifier_shell": test_sh,
                    "instruction": INSTRUCTION.format(n=2)}
            d = write_task_tree(dest, spec, check_py=CHECK)
            config = task_config_text(spec).replace(f"'{branch_setup(bundle)}'", f"'{setup_command(tasks, SETUP_EXTRA.get(f'{repo}/{pr}', ''))}'")
            config = config.replace("[agent]\ntimeout_sec = 300.0", "[agent]\ntimeout_sec = 3600.0")
            (d / "task.toml").write_text(config.replace("[verifier]\ntimeout_sec = 180.0", "[verifier]\ntimeout_sec = 1800.0"))
            (d / "tests/hidden_full").mkdir()  # the RL bundle verifier copies it in; the base test files are HEAD's
            (d / "solution").mkdir()
            graded = {}
            for k, f in enumerate(pair, 1):
                shutil.copy(src / f"feature{f}/tests.patch", d / f"tests/tests_{k}.patch")
                shutil.copy(src / f"feature{f}/feature.patch", d / f"solution/fix_{k}.patch")
                graded[str(k)] = {"feature": f, "tests": f"/tests/tests_{k}.patch", "command": COMMANDS[f"{repo}/{pr}"]}
            (d / "tests/tasks.json").write_text(json.dumps(graded))
            shutil.copy(src / "combined.patch", d / "solution/combined.patch")
            (d / "solution/solve.sh").write_text(SOLVE_SH)
            (d / "solution/solve.sh").chmod(0o755)
            rows.append({"bundle": bundle, "repo": repo.removesuffix("_task"), "pr": pr, "features": list(pair),
                         "split": SPLIT[repo.removesuffix("_task")],
                         "image": image_of(repo, pr), "n": 2, "dataset_sha": sha})
    for suffix in ("manifest", "manifest.full"):  # cooperbench_shard.sh cuts NAME.manifest.jsonl to its repos
        Path(f"{dest}.{suffix}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(f"{len(rows)} bundles -> {dest}")


# ---------------------------------------------------------------- acceptance

VARIANTS = {"none": "", "A": "/solution/fix_1.patch", "B": "/solution/fix_2.patch", "combined": "/solution/combined.patch"}


def judge(rounds: list) -> list:
    """Why a bundle fails: none (0,0), gold A solves 1, gold B solves 2, combined (1,1), both rounds equal."""
    want = {"none": {"1": 0, "2": 0}, "combined": {"1": 1, "2": 1}}
    why = []
    for v in VARIANTS:
        got = [r.get(v, {}).get("tasks") for r in rounds]
        if got[0] is None or got[0] != got[1]:
            why.append(f"{v}: flaky or no result {got}")
        elif v in want and got[0] != want[v]:
            why.append(f"{v}: {got[0]}")
        elif v in ("A", "B") and not got[0]["12"[v == "B"]]:
            why.append(f"{v}: gold does not solve its task {got[0]}")
    return why


async def accept_bundle(d: Path, row: dict, sem, variants: dict = VARIANTS, rounds: int = 2, env_vars: dict = None) -> dict:
    import tomllib
    from octopus.envs.swe.data.validate_tasks import make_env
    setup = tomllib.loads((d / "task.toml").read_text())["environment"]["healthcheck"]["command"]
    out = {"bundle": row["bundle"], "rounds": [{} for _ in range(rounds)]}
    async with sem:
        env = make_env(row["image"])
        try:
            await env.start(False)
            start = time.time()
            done = await env.exec(setup, timeout_sec=1200)
            out["setup_seconds"] = round(time.time() - start, 1)
            if done.return_code:
                out["error"] = f"setup rc {done.return_code}: {(done.stdout + done.stderr)[-500:]}"
                return out
            await env.upload_dir(d / "tests", "/tests")
            await env.upload_dir(d / "solution", "/solution")
            for run in out["rounds"]:
                for v, patch in variants.items():
                    apply = f" && git apply --whitespace=nowarn {patch}" if patch else ""
                    r = await env.exec("cd /testbed && git reset -q --hard && git clean -fdq && rm -rf /logs/verifier && "
                                       f"mkdir -p /logs/verifier{apply} && s=$(date +%s.%N) && bash /tests/test.sh "
                                       "> /logs/verifier/test-stdout.txt 2>&1; e=$(date +%s.%N); cat /logs/verifier/tasks.json;"
                                       " echo; echo $s $e; cat /logs/verifier/details.json", env=env_vars, timeout_sec=2000)
                    lines = r.stdout.strip().splitlines()
                    try:
                        s, e = map(float, lines[1].split())
                        run[v] = {"tasks": json.loads(lines[0]), "wall": round(e - s, 1),
                                  "passed": {k: x["passed"] for k, x in json.loads(lines[2]).items()},
                                  "tails": {k: x.get("tail", "")[-1500:] for k, x in json.loads(lines[2]).items()
                                            if not json.loads(lines[0]).get(k)}}
                    except (IndexError, ValueError):
                        run[v] = {"tasks": None, "out": (r.stdout + r.stderr)[-500:]}
        except Exception as exc:
            out["error"] = f"{type(exc).__name__}: {exc}"[:500]
        finally:
            await env.stop(True)
    return out


async def accept(name: str, conc: int = 12):
    """Every bundle of datasets/NAME -> NAME.acceptance.jsonl (resumable), NAME.acceptance.json, NAME-valid/."""
    import asyncio
    from octopus.envs.swe.data.regen_tasks import N
    src, valid = N / "datasets" / name, N / "datasets" / f"{name}-valid"
    log = Path(f"{src}.acceptance.jsonl")
    done = {r["bundle"]: r for r in map(json.loads, log.open())} if log.exists() else {}
    rows = [r for r in map(json.loads, open(f"{src}.manifest.jsonl")) if r["bundle"] not in done]
    sem = asyncio.Semaphore(conc)
    for fut in asyncio.as_completed([accept_bundle(src / r["bundle"], r, sem) for r in rows]):
        res = await fut
        res["why"] = [res["error"]] if res.get("error") else judge(res["rounds"])
        res["accepted"] = not res["why"]
        done[res["bundle"]] = res
        with log.open("a") as f:
            f.write(json.dumps(res) + "\n")
        print(json.dumps({k: res[k] for k in ("bundle", "accepted", "why")}), flush=True)
    valid.mkdir(exist_ok=True)
    keep = [r for r in map(json.loads, open(f"{src}.manifest.jsonl")) if done[r["bundle"]]["accepted"]]
    for r in keep:
        if not (valid / r["bundle"]).exists():
            shutil.copytree(src / r["bundle"], valid / r["bundle"])
    Path(f"{valid}.manifest.jsonl").write_text("".join(json.dumps(r) + "\n" for r in keep))
    Path(f"{src}.acceptance.json").write_text(json.dumps(done, indent=1))
    print(f"{len(keep)}/{len(done)} bundles accepted -> {valid}")


async def offline(name: str, conc: int = 10):
    """Which bundles of datasets/NAME (its manifest) need the network to grade. Each bundle's combined patch is graded
    with OCTOPUS_NO_EGRESS=1 on test.sh (the udocker connect guard refuses connections off the host, as for agent tool
    calls), in a seeded shuffled order (a PR's bundles together load the host and time out). A bundle not solving both
    tasks is graded again with the network and without it: it needs the network when it solves both with it and not
    without; a failure both ways or one that does not repeat is load or tooling. -> NAME.offline.jsonl (every pass),
    the need-network list printed (cooperbench_needs_network.txt for drop)."""
    import asyncio
    from octopus.envs.swe.data.regen_tasks import N
    src = N / "datasets" / name
    rows = [json.loads(line) for line in open(f"{src}.manifest.jsonl")]
    order = random.Random(0).sample(rows, len(rows))
    sem, off = asyncio.Semaphore(conc), {"OCTOPUS_NO_EGRESS": "1"}
    grade = lambda r, env: accept_bundle(src / r["bundle"], r, sem, {"combined": VARIANTS["combined"]}, 1, env)
    solved = lambda o: not o.get("error") and o["rounds"][0]["combined"].get("tasks") == {"1": 1, "2": 1}
    first = dict(zip((r["bundle"] for r in order), await asyncio.gather(*(grade(r, off) for r in order))))
    again = [r for r in order if not solved(first[r["bundle"]])]
    redo = await asyncio.gather(*(grade(r, env) for r in again for env in (None, off)))
    recheck = {r["bundle"]: {"network": redo[2 * i], "no_egress": redo[2 * i + 1]} for i, r in enumerate(again)}
    need = [r["bundle"] for r in rows if r["bundle"] in recheck
            and solved(recheck[r["bundle"]]["network"]) and not solved(recheck[r["bundle"]]["no_egress"])]
    Path(f"{src}.offline.jsonl").write_text("".join(json.dumps({"bundle": r["bundle"], "no_egress": first[r["bundle"]],
                                                                **({"recheck": recheck[r["bundle"]]} if r["bundle"] in recheck else {})}) + "\n"
                                                    for r in rows))
    print(json.dumps({"bundles": len(rows), "rechecked": [r["bundle"] for r in again], "need_network": need}))


def commands(name: str):
    """Rewrite every bundle's test command in datasets/NAME/*/tests/tasks.json from COMMANDS (after a COMMANDS fix)."""
    from octopus.envs.swe.data.regen_tasks import N
    changed = 0
    for r in map(json.loads, open(N / "datasets" / f"{name}.manifest.jsonl")):
        f = N / "datasets" / name / r["bundle"] / "tests/tasks.json"
        tasks = json.loads(f.read_text())
        cmd = COMMANDS[f"{r['repo']}_task/{r['pr']}"]
        if any(t["command"] != cmd for t in tasks.values()):
            f.write_text(json.dumps({k: {**t, "command": cmd} for k, t in tasks.items()}))
            changed += 1
    print(f"{name}: {changed} bundles' commands rewritten")


def drop(name: str, listing: str):
    """datasets/NAME without the bundles listed in LISTING (one id per line, # comments): the old version is renamed
    NAME-v1 (then -v2, ...), NAME and NAME.manifest.jsonl are rewritten from it."""
    from octopus.envs.swe.data.regen_tasks import N
    D = N / "datasets"
    out = set(line.split("#")[0].strip() for line in open(listing)) - {""}
    rows = [json.loads(line) for line in open(D / f"{name}.manifest.jsonl")]
    if out - {r["bundle"] for r in rows}:  # before any rename: a typo leaves NAME as it was
        raise SystemExit(f"not in {name}: {sorted(out - {r['bundle'] for r in rows})}")
    old = next(f"{name}-v{i}" for i in range(1, 100) if not (D / f"{name}-v{i}").exists())
    (D / name).rename(D / old)
    (D / f"{name}.manifest.jsonl").rename(D / f"{old}.manifest.jsonl")
    keep = [r for r in rows if r["bundle"] not in out]
    (D / name).mkdir()
    for r in keep:
        shutil.copytree(D / old / r["bundle"], D / name / r["bundle"])
    (D / f"{name}.manifest.jsonl").write_text("".join(json.dumps(r) + "\n" for r in keep))
    print(f"{name}: {len(rows)} -> {len(keep)} bundles; previous version {old}")


async def scan_pr(ds: Path, d: Path, row: dict, sem) -> dict:
    """Leak scan of one PR's image (fresh container): the repo is the upstream base commit with no local change, then
    after the healthcheck one commit and no other ref or .git, the package imports from /testbed with no
    site-packages copy, no patch file, and no added line of any feature's gold/tests patch or combined.patch (>= 40
    chars, absent from the base tree) in a site-packages copy of the package (any line; counted before the healthcheck
    too) or elsewhere outside /testbed (a file with 5 or more distinct such lines: single generic lines such as
    `with tempfile.TemporaryDirectory() as tmpdir:` turn up in pip and setuptools). Patch files are listed."""
    import tomllib
    from octopus.envs.swe.data.validate_tasks import make_env
    src = ds / f"{row['repo']}_task" / row["pr"]
    base = re.search(r"git checkout ([0-9a-f]{40})", (src / "Dockerfile").read_text()).group(1)
    lines = {l[1:].strip() for p in [src / "combined.patch", *src.glob("feature*/*.patch")]
             for l in p.read_text(errors="replace").splitlines() if l.startswith("+") and not l.startswith("+++")}
    pats = "\n".join(sorted(l for l in lines if len(l) >= 40)) + "\n"
    pkg = PACKAGES[f"{row['repo']}_task"]
    setup = tomllib.loads((d / "task.toml").read_text())["environment"]["healthcheck"]["command"]
    out = {"pr": f"{row['repo']}/{row['pr']}", "base": base}
    async with sem:
        env = make_env(row["image"])
        try:
            await env.start(False)
            sh = lambda c, t=1800: env.exec(c, timeout_sec=t)
            r = await sh("cd /workspace/repo && git rev-parse HEAD && git status --porcelain | head -20")
            out["head"], *out["dirty"] = r.stdout.strip().splitlines() or [""]
            # the PR's added lines that are not in the base tree; any of them in a copy of the package anywhere outside
            # the tree (a `{pkg}/__init__.py` under site-packages, caches, ...; line by line, no threshold), before and
            # after the healthcheck
            copies = (f"find / -xdev \\( -path /proc -o -path /sys -o -path /testbed -o -path /workspace/repo \\) -prune -o "
                      f"-type f -path '*/{pkg.replace('.', '/')}/__init__.py' -print 2>/dev/null")
            site = f"$({copies} | xargs -r -n1 dirname)"
            count = f"grep -rhoF -f /logs/new {site} /dev/null 2>/dev/null | sort -u | wc -l"
            r = await sh(f"mkdir -p /logs && printf %s {base64.b64encode(pats.encode()).decode()} | base64 -d > /logs/pats && "
                         "cd /workspace/repo && grep -rhoF -f /logs/pats . 2>/dev/null | sort -u > /logs/inbase; "
                         f"grep -vxF -f /logs/inbase /logs/pats > /logs/new; wc -l < /logs/new; {count}")
            out["gold_lines"], out["site_gold_lines_image"] = (r.stdout.split() + ["?", "?"])[:2]
            out["setup_rc"] = (await sh(setup)).return_code
            out["site_gold_lines"] = (await sh(count)).stdout.strip()
            r = await sh("cd /testbed && git rev-list --all | wc -l && git for-each-ref | wc -l && "
                         "find / -xdev -name .git -not -path /testbed/.git 2>/dev/null | grep -v ^/proc | head")
            out["commits"], out["refs"], *out["other_git"] = r.stdout.split()
            r = await sh(f"cd / && python -c 'import {pkg}, os; print(os.path.realpath({pkg}.__file__))'")
            out["import_path"] = r.stdout.strip() or r.stderr[-300:]
            out["site_copies"] = (await sh(f"{copies} | head")).stdout.split()
            r = await sh("find / -xdev \\( -path /proc -o -path /sys -o -path /tests -o -path /solution \\) -prune -o -type f "
                         "\\( -name '*.patch' -o -name '*.diff' -o -name feature.md \\) -print 2>/dev/null | head")
            out["patch_files"] = r.stdout.split()
            # anywhere else: files with 5 or more distinct such lines
            r = await sh("grep -rIoF -f /logs/new $(ls -d /* | grep -vxE '/(proc|sys|dev|testbed|tests|solution|logs|tasks)')"
                         " 2>/dev/null | sort -u | cut -d: -f1 | uniq -c | awk '$1 >= 5' | sort -rn | head -20")
            out["gold_hits"] = [x.strip() for x in r.stdout.strip().splitlines()]
        except Exception as exc:
            out["error"] = f"{type(exc).__name__}: {exc}"[:500]
        finally:
            await env.stop(True)
    out["leak"] = bool(out.get("error") or out.get("head") != base or out.get("dirty") or out.get("setup_rc")
                       or out.get("commits") != "1" or out.get("refs") != "1" or out.get("other_git")
                       or not str(out.get("import_path", "")).startswith("/testbed/")
                       or out.get("site_gold_lines") != "0" or out.get("gold_hits"))  # site_copies: listed (scipy/datasets is not datasets)
    return out


async def scan(ds: str, name: str, conc: int = 12):
    """scan_pr for every PR of datasets/NAME (its first bundle's healthcheck) -> NAME.leaks.json."""
    import asyncio
    from octopus.envs.swe.data.regen_tasks import N
    src = N / "datasets" / name
    first = {}
    for r in map(json.loads, open(f"{src}.manifest.jsonl")):
        first.setdefault((r["repo"], r["pr"]), r)
    sem = asyncio.Semaphore(conc)
    res = await asyncio.gather(*(scan_pr(Path(ds), src / r["bundle"], r, sem) for r in first.values()))
    Path(f"{src}.leaks.json").write_text(json.dumps(res, indent=1))
    for r in res:
        print(json.dumps(r), flush=True)
    print(f"leaks: {[r['pr'] for r in res if r['leak']]}")


def merge(name: str):
    """Shard results (cooperbench_shard.sh: NAME.acceptance.<host>.jsonl, NAME.leaks.<host>.json; inputs are not changed)
    into NAME.merged.acceptance.jsonl / .json and NAME.merged.leaks.json (one row per bundle / PR; a judged row beats a
    tool-error row; two judged rows that disagree are a conflict, reported and left out), then NAME-valid/ and the Train
    part NAME-train-valid/ (+ manifests). Bundle list and order: NAME.manifest.full.jsonl (written by build)."""
    import glob
    from octopus.envs.swe.data.regen_tasks import N
    D = N / "datasets"
    acc, conflicts = {}, set()
    for f in sorted(glob.glob(str(D / f"{name}.acceptance.*.jsonl"))):
        for line in open(f):
            r = json.loads(line)
            old = acc.get(r["bundle"])
            if old and not old.get("error") and not r.get("error") and (old["accepted"], old["why"]) != (r["accepted"], r["why"]):
                conflicts.add(r["bundle"])
            if not old or old.get("error") and not r.get("error"):
                acc[r["bundle"]] = r
    for b in conflicts:
        del acc[b]
    rows = [json.loads(line) for line in open(D / f"{name}.manifest.full.jsonl")]
    (D / f"{name}.merged.acceptance.jsonl").write_text("".join(json.dumps(acc[r["bundle"]]) + "\n" for r in rows if r["bundle"] in acc))
    (D / f"{name}.merged.acceptance.json").write_text(json.dumps(acc, indent=1))
    leaks = {}
    for f in sorted(glob.glob(str(D / f"{name}.leaks.*.json"))):
        for r in json.load(open(f)):
            leaks[r["pr"]] = r
    (D / f"{name}.merged.leaks.json").write_text(json.dumps(list(leaks.values()), indent=1))
    for out, keep in ((f"{name}-valid", lambda r: True), (f"{name}-train-valid", lambda r: r["split"] == "train")):
        ok = [r for r in rows if r["bundle"] in acc and acc[r["bundle"]]["accepted"] and keep(r)]
        (D / out).mkdir(exist_ok=True)
        for r in ok:
            if not (D / out / r["bundle"]).exists():
                shutil.copytree(D / name / r["bundle"], D / out / r["bundle"])
        (D / f"{out}.manifest.jsonl").write_text("".join(json.dumps(r) + "\n" for r in ok))
        print(out, len(ok))
    print("judged", len(acc), "of", len(rows), "; conflicts", sorted(conflicts), "; leak scan PRs", len(leaks))

if __name__ == "__main__":
    cmd, *args = sys.argv[1:]
    if cmd == "fetch":
        fetch(args[0], args[1], *map(int, args[2:3]))
    elif cmd == "build":
        build(*args)
    elif cmd == "accept":
        import asyncio
        asyncio.run(accept(args[0], *map(int, args[1:2])))
    elif cmd == "offline":
        import asyncio
        asyncio.run(offline(args[0], *map(int, args[1:2])))
    elif cmd == "commands":
        commands(args[0])
    elif cmd == "drop":
        drop(*args)
    elif cmd == "merge":
        merge(args[0])
    elif cmd == "scan":
        import asyncio
        asyncio.run(scan(args[0], args[1], *map(int, args[2:3])))
