"""Select Harbor tasks from grok-scale rows. No lighthouse scoring."""

from __future__ import annotations

import json
import os
from pathlib import Path

# The Harbor environment (`harbor run -e`) of every run started from this code: udocker goldens, or on hosts without
# room for them octopus.envs.swe.singularity:SingularityEnvironment
HARBOR_ENV = os.environ.get("OCTOPUS_HARBOR_ENV", "octopus.envs.swe.udocker:UdockerEnvironment")

HOLDOUT_IDS = frozenset({
    "amueller__word_cloud.ec24191c.lm_rewrite__jhxlqo3m",
    "prettytable__prettytable.ca90b055.lm_rewrite__guayblhb",
})

def branch_setup(instance_id: str) -> str:
    """Put /testbed on the instance's SWE-smith branch (Bug Patch, then Remove F2P Tests) and replace
    its git history with one commit of that tree.

    The image is shared by every instance of a repo and sits on a clean main, so without the
    checkout the agent edits bug-free code. The history goes because it holds the answer: main is the
    fix, the branch tip hides the tests. Runs as the Harbor healthcheck, i.e. once, before the agent.
    """
    ref = f"origin/{instance_id}"
    return (
        "cd /testbed && if [ ! -f .git/octopus_ready ]; then"
        f" git checkout -q --force {ref} && rm -rf .git && git init -q && git add -A"
        ' && git -c user.name=octopus -c user.email=octopus@localhost commit -qm "Initial commit"'
        " && touch .git/octopus_ready; fi && test -f .git/octopus_ready"
    )


# test-infrastructure edits of the agent are undone before grading: no fix needs them, and a patch that edits a
# conftest fixture or pytest config can make the hidden tests pass without fixing anything (seen once in RL)
INFRA_RESTORE = r"""git diff --name-only HEAD | grep -E '(^|/)(conftest\.py|pytest\.ini|tox\.ini|setup\.cfg|pyproject\.toml)$' | while read -r f; do git checkout -q HEAD -- "$f" 2>/dev/null || { git rm -q -f --cached "$f" 2>/dev/null; rm -f "$f"; }; done
git ls-files --others --exclude-standard | grep -E '(^|/)conftest\.py$' | xargs -r rm -f
"""

HIDDEN_RESTORE = INFRA_RESTORE + """if [ -d /tests/hidden_full ]; then cp -a /tests/hidden_full/. /testbed/; fi
if [ -s /tests/hidden_tests.patch ] && ! git apply --3way --whitespace=nowarn /tests/hidden_tests.patch; then
  mkdir -p /logs/verifier; echo 0 > /logs/verifier/reward.txt; exit 0
fi
"""


def verifier_shell(instance_id: str = "") -> str:
    """Put back the tests the branch hid, from files stored with the task (the agent never sees
    /tests), and grade the files that held them. Hidden files the bug never touched are restored
    wholesale from tests/hidden_full (overwriting whatever the agent wrote there, as SWE-smith does);
    only files that also carry the bug get the removed hunks back (3-way), so the agent's fix in them
    stays. A patch that cannot be restored scores 0."""
    return VERIFIER_SHELL.replace("cd /testbed\n", "cd /testbed\n" + HIDDEN_RESTORE, 1).replace(
        "/tests/check.py\n", "/tests/check.py $(cat /tests/hidden_files.txt)\n", 1,
    )


def plan_verifier_shell(instance_id: str = "") -> str:
    """Hidden tests on the final tree, then each branch patch alone on its fork-point snapshot
    (/tmp/octopus-plan/plan.tsv, written by the agent after the episode's last tool call), then the no-fork control
    snapshot if present. Writes /logs/verifier/reward.txt, branches.json, and restores the final tree."""
    return r"""#!/bin/bash
set +e
cd /testbed
mkdir -p /logs/verifier /tmp/octopus-plan

take_snap() {
  head=$(git rev-parse HEAD)
  git add -A
  stash=$(git stash create)
  echo "$head $stash"
}
# index == tree after both: git apply --3way refuses a file whose tree differs from the index
restore() {
  set -- $1
  git reset -q --hard "$1" && git clean -fdq
  if [ -n "${2:-}" ]; then git stash apply -q "$2" && git add -A; fi
}
grade() {
  dest=$1
""" + INFRA_RESTORE + r"""
  if [ -d /tests/hidden_full ]; then cp -a /tests/hidden_full/. /testbed/; fi
  if [ -s /tests/hidden_tests.patch ] && ! git apply --3way --whitespace=nowarn /tests/hidden_tests.patch; then
    echo 0 > "$dest"; return
  fi
  /opt/miniconda3/envs/testbed/bin/python /tests/check.py $(cat /tests/hidden_files.txt)
  if [ $? -eq 0 ]; then echo 1 > "$dest"; else echo 0 > "$dest"; fi
}

final=$(take_snap)
grade /logs/verifier/reward.txt
: > /tmp/octopus-plan/grades.txt
if [ -f /tmp/octopus-plan/plan.tsv ]; then
  while IFS="$(printf '\t')" read -r -u 3 key point patch; do  # fd 3: tests that read stdin must not eat rows
    [ -n "$key" ] || continue
    restore "$point"
    if [ -n "${patch:-}" ] && [ -f "$patch" ]; then
      if ! git apply --3way --whitespace=nowarn "$patch"; then
        echo "$key 0" >> /tmp/octopus-plan/grades.txt
        continue
      fi
    fi
    grade /tmp/octopus-plan/g.txt
    echo "$key $(cat /tmp/octopus-plan/g.txt)" >> /tmp/octopus-plan/grades.txt
  done 3< /tmp/octopus-plan/plan.tsv
fi
if [ -f /tmp/octopus-plan/control.snap ]; then
  restore "$(cat /tmp/octopus-plan/control.snap)"
  grade /logs/verifier/control.txt
fi
restore "$final"
python3 - <<'PY'
import json
from pathlib import Path
branches = []
p = Path("/tmp/octopus-plan/grades.txt")
if p.exists():
    for line in p.read_text().splitlines():
        key, _, bit = line.partition(" ")
        branches.append({"key": key, "passed": bit.strip() == "1"})
ctrl = Path("/logs/verifier/control.txt")
out = {"branches": branches}
if ctrl.exists():
    out["control"] = {"passed": ctrl.read_text().strip() == "1"}
Path("/logs/verifier/branches.json").write_text(json.dumps(out))
PY
"""


def bundle_verifier_shell() -> str:
    """OCTOPUS_BUNDLE plan episodes: first each branch patch alone on the bundle's initial tree (plan.tsv rows: key,
    initial snapshot, task id, patch; "-" for none, as read collapses empty tab fields), graded by the bundle's check.py with only the named task's ids (every task when
    the row names none or a task the bundle lacks), then the no-fork control's tree (control.snap) on every task into
    control_tasks.json, then the final tree on every task, so check.py's last writes (tasks.json, seconds.json,
    reward.txt = solved fraction) are the final tree's. branches.json: per branch key, task (None = every task), tasks
    {k: 0/1}, passed (its task solved; with no task, any task); control {tasks} when graded. checks.json (from
    checks.tsv: key, kind fork|own, initial snapshot, task id, base64 check): "<key>:<kind>" -> exit codes on the
    initial tree (bug) and with /tests/fixes/fix_<task>.patch (fix): the plan reward's check validity."""
    return r"""#!/bin/bash
set +e
cd /testbed
mkdir -p /logs/verifier /tmp/octopus-plan/check
PY=/opt/miniconda3/envs/testbed/bin/python
take_snap() {
  head=$(git rev-parse HEAD)
  git add -A
  stash=$(git stash create)
  echo "$head $stash"
}
restore() {
  set -- $1
  git reset -q --hard "$1" && git clean -fdq
  if [ -n "${2:-}" ]; then git stash apply -q "$2" && git add -A; fi
}
grade() {  # $1: check.py (its tasks.json sits next to it)
""" + INFRA_RESTORE + r"""
  cp -a /tests/hidden_full/. /testbed/
  rm -f /logs/verifier/tasks.json
  $PY "$1"
}
final=$(take_snap)
: > /tmp/octopus-plan/grades.txt
if [ -f /tmp/octopus-plan/plan.tsv ]; then
  while IFS="$(printf '\t')" read -r -u 3 key point task patch; do  # fd 3: tests that read stdin must not eat rows
    [ -n "$key" ] || continue
    restore "$point"
    if [ "$patch" = - ] || ! git apply --3way --whitespace=nowarn "$patch"; then
      printf '%s\t%s\t{}\n' "$key" "$task" >> /tmp/octopus-plan/grades.txt  # nothing to grade: the initial tree fails
      continue
    fi
    python3 - "$task" <<'PY'
import json, sys
tasks = json.load(open("/tests/tasks.json"))
k = sys.argv[1]
json.dump({k: tasks[k]} if k in tasks else tasks, open("/tmp/octopus-plan/check/tasks.json", "w"))
PY
    cp /tests/check.py /tmp/octopus-plan/check/check.py
    grade /tmp/octopus-plan/check/check.py
    printf '%s\t%s\t%s\n' "$key" "$task" "$(cat /logs/verifier/tasks.json 2>/dev/null || echo '{}')" >> /tmp/octopus-plan/grades.txt
  done 3< /tmp/octopus-plan/plan.tsv
fi
if [ -f /tmp/octopus-plan/control.snap ]; then
  restore "$(cat /tmp/octopus-plan/control.snap)"
  grade /tests/check.py
  [ -f /logs/verifier/tasks.json ] && cp /logs/verifier/tasks.json /logs/verifier/control_tasks.json
fi
if [ -f /tmp/octopus-plan/checks.tsv ]; then  # each fork's acceptance check on the initial tree and with its fix
  S=/tmp/octopus-plan/checks; T=/testbed; rm -rf $S; mkdir -p $S; i=0
  while IFS="$(printf '\t')" read -r -u 3 key kind point task b64; do
    [ -n "$key" ] || continue
    i=$((i + 1))
    [ $i = 1 ] && restore "$point"
    cp -a /testbed $S/$i.bug && cp -a /testbed $S/$i.fix
    git -C $S/$i.fix apply --whitespace=nowarn "/tests/fixes/fix_$task.patch" > /dev/null 2>&1; ap=$?
    check=$(printf %s "$b64" | base64 -d)
    for side in bug fix; do
      (cd $S/$i.$side && timeout 60 bash -lc "${check//$T/$S/$i.$side}" > /dev/null 2>&1 < /dev/null
       echo $? > $S/$i.$side.rc) &
    done
    printf '%s\t%s\t%s\t%s\t%s\n' "$i" "$key" "$kind" "$task" "$ap" >> $S/rows
  done 3< /tmp/octopus-plan/checks.tsv
  wait
  python3 - "$S" <<'PY'
import json, sys
from pathlib import Path
S = Path(sys.argv[1])
out = {}
for line in (S / "rows").read_text().splitlines() if (S / "rows").exists() else []:
    i, key, kind, task, ap = line.split("\t")
    rc = {side: (S / f"{i}.{side}.rc").read_text().strip() for side in ("bug", "fix")}
    out[f"{key}:{kind}"] = {"task": task, "fix_applied": ap == "0", **rc}
Path("/logs/verifier/checks.json").write_text(json.dumps(out))
PY
  rm -rf $S
fi
restore "$final"
grade /tests/check.py
python3 - <<'PY'
import json
from pathlib import Path
known = json.load(open("/tests/tasks.json"))
branches = []
for line in Path("/tmp/octopus-plan/grades.txt").read_text().splitlines():
    key, task, solved = line.split("\t", 2)
    solved = json.loads(solved or "{}")
    task = task if task in known else None
    passed = bool(solved.get(task)) if task else any(solved.values())
    branches.append({"key": key, "task": task, "tasks": solved, "passed": passed})
out = {"branches": branches}
ctrl = Path("/logs/verifier/control_tasks.json")
if ctrl.exists():
    out["control"] = {"tasks": json.loads(ctrl.read_text())}
Path("/logs/verifier/branches.json").write_text(json.dumps(out))
PY
"""


VERIFIER_SHELL = """#!/bin/bash
set +e
cd /testbed
/opt/miniconda3/envs/testbed/bin/python /tests/check.py
rc=$?
mkdir -p /logs/verifier
if [ "$rc" -eq 0 ]; then
  echo 1 > /logs/verifier/reward.txt
else
  echo 0 > /logs/verifier/reward.txt
fi
"""

DEFAULT_CHECK = """import importlib.util
import subprocess
import sys
from pathlib import Path

# Grade the test files the instance branch hid (argv); they hold its fail-to-pass tests. Tests that
# fail on the clean reference too (sandbox limits, network) are listed in known_failing.txt.
hidden = sys.argv[1:]
if not hidden:
    sys.exit(1)
known = Path(__file__).with_name("known_failing.txt")
deselect = [a for node in (known.read_text().split() if known.exists() else []) for a in ("--deselect", node)]
cov = ["--no-cov"] if importlib.util.find_spec("pytest_cov") else []
try:
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "--tb=line", "-p", "no:cacheprovider", *cov, *deselect, *hidden],
        cwd="/testbed",
        timeout=100,
    )
except subprocess.TimeoutExpired:
    sys.exit(1)
sys.exit(completed.returncode)
"""


def instance_org(instance_id: str) -> str:
    return instance_id.split("__", 1)[0]


def is_excluded(instance_id: str, excluded_ids: set[str]) -> bool:
    if instance_id in excluded_ids or instance_id in HOLDOUT_IDS:
        return True
    return instance_org(instance_id) == "bottlepy"


def select_task_specs(rows, local_images, excluded_ids):
    """Return Harbor specs for rows whose image is local and not excluded.

    ``instruction`` is the row issue text. The verifier shell is not copied
    into that instruction.
    """
    images = set(local_images)
    blocked = set(excluded_ids)
    specs = []
    for row in rows:
        task = row["solver_task"]
        instance_id = task["instance_id"]
        image = task["image"]
        if is_excluded(instance_id, blocked) or image not in images:
            continue
        issue = task["issue"]
        specs.append({
            "instance_id": instance_id,
            "image": image,
            "repo": task.get("repo") or "",
            "instruction": issue,
            "verifier_shell": verifier_shell(instance_id),
        })
    return specs


def task_config_text(spec: dict) -> str:
    """task.toml naming the existing image. No proxy URL and no API key."""
    name = spec["instance_id"].replace(".", "-")
    return (
        'schema_version = "1.4"\n'
        "[task]\n"
        f'name = "octopus/{name}"\n'
        'authors = [{ name = "octopus" }]\n'
        "[environment]\n"
        f'docker_image = "{spec["image"]}"\n'
        "gpus = 0\n"
        "[environment.healthcheck]\n"
        f"command = '{branch_setup(spec['instance_id'])}'\n"
        "timeout_sec = 300.0\n"
        "[agent]\n"
        "timeout_sec = 300.0\n"
        "[verifier]\n"
        "timeout_sec = 180.0\n"
    )


def render_task_files(spec: dict, *, check_py: str = DEFAULT_CHECK, fix_patch: str | None = None) -> dict[str, str]:
    files = {
        "instruction.md": spec["instruction"],
        "task.toml": task_config_text(spec),
        "tests/test.sh": spec["verifier_shell"],
        "tests/check.py": check_py,
    }
    if fix_patch is not None:
        files["solution/fix.patch"] = fix_patch
    return files


def write_task_tree(root: Path, spec: dict, **kwargs) -> Path:
    dest = root / spec["instance_id"]
    dest.mkdir(parents=True, exist_ok=True)
    env_dir = dest / "environment"
    env_dir.mkdir(exist_ok=True)
    (env_dir / ".gitkeep").write_text("")
    for rel, text in render_task_files(spec, **kwargs).items():
        path = dest / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        if rel.endswith(".sh"):
            path.chmod(0o755)
    return dest


def load_rows(directory: Path) -> list:
    rows = []
    for path in sorted(directory.glob("row*.json")):
        rows.append(json.loads(path.read_text()))
    return rows
