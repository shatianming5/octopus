"""Rebuild Harbor task dirs to the SWE-smith branch semantics of octopus.envs.swe.taskset.

For each task it reads the instance branch from the image's golden repo on the host and writes:
  task.toml                 healthcheck = branch_setup (checkout the branch, drop git history)
  tests/test.sh             verifier_shell (restore hidden tests from the files below, grade them)
  tests/check.py            DEFAULT_CHECK
  tests/hidden_files.txt    files "Remove F2P Tests" touched (graded)
  tests/hidden_full/<path>  their full pre-removal content, for those the Bug Patch did not touch:
                            restored wholesale, whatever the agent wrote there (as SWE-smith does)
  tests/hidden_tests.patch  git diff <branch> <branch>~1 limited to hidden files the Bug Patch also
                            touched (tests inside the buggy module): only the removed hunks go back
  solution/fix.patch        the reference fix on the branch tree (Bug Patch reverted); solve.sh applies it
Oracle tasks (a hand-written check.py and instruction for one behavior, made for the old clean-main
setup) grade something other than the instance bug and are left out too. A task whose reference fix does not apply to the branch tree (e.g. "Remove F2P Tests" deleted the very
module the bug is in) cannot be solved as posed; it is left out and listed in <new dataset>.skipped.json.
usage: regen_tasks.py <source dataset> <new dataset>   (names under datasets/)"""
import json
import hashlib, re, shutil, subprocess, sys, tempfile, tomllib
from pathlib import Path

from octopus.envs.swe.taskset import DEFAULT_CHECK, branch_setup, verifier_shell

N = Path("/home/user/harbor-udocker")


def git(repo, *args, **kw) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True, **kw).stdout


def golden_repo(image: str) -> Path:
    key = hashlib.sha1(image.encode()).hexdigest()[:16]
    cid = (N / "golden" / ".golden" / f"{key}.cid").read_text().split()[0]
    return N / "golden" / cid / "ROOT" / "testbed"


def reference_fix(repo: Path, name: str) -> str:
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(["git", "clone", "-q", "--shared", "--no-checkout", str(repo), tmp], check=True)
        git(tmp, "fetch", "-q", str(repo), f"+refs/remotes/origin/{name}:refs/heads/task")
        git(tmp, "checkout", "-q", "task")
        revert = git(tmp, "diff", "--binary", "task~1", "task~2")
        subprocess.run(["git", "-C", tmp, "apply", "--3way"], input=revert, check=True, capture_output=True, text=True)
        return git(tmp, "diff", "--binary", "HEAD")


def main(src: Path, dest: Path):
    if dest.exists():
        raise SystemExit(f"{dest} exists; pick a new name")
    dest.mkdir()
    count = 0
    skipped = {}
    for task in sorted(src.iterdir()):
        if not (task / "task.toml").is_file():
            continue
        check = (task / "tests" / "check.py").read_text()
        if '"--maxfail=1"' not in check and check != DEFAULT_CHECK:  # the old emitted check, or the current one
            skipped[task.name] = "oracle task: its check.py/instruction grade a different behavior than the instance bug"
            continue
        config = (task / "task.toml").read_text()
        repo = golden_repo(tomllib.loads(config)["environment"]["docker_image"])
        ref = f"origin/{task.name}"
        try:
            fix = reference_fix(repo, task.name)
        except subprocess.CalledProcessError as exc:
            skipped[task.name] = f"reference fix does not apply to the branch tree: {(exc.stderr or '')[-200:]}"
            continue
        out = dest / task.name
        shutil.copytree(task, out)
        config = re.sub(r"\[environment\.healthcheck\]\n(?:[a-z_]+ = .*\n)*", "", config)
        config = config.replace(
            "gpus = 0\n",
            f"gpus = 0\n[environment.healthcheck]\ncommand = '{branch_setup(task.name)}'\ntimeout_sec = 300.0\n", 1,
        )
        config = re.sub(r"(\[verifier\]\ntimeout_sec = )[0-9.]+", r"\g<1>180.0", config)
        tomllib.loads(config)
        (out / "task.toml").write_text(config)
        tests = out / "tests"
        (tests / "test.sh").write_text(verifier_shell())
        (tests / "check.py").write_text(DEFAULT_CHECK)
        hidden = git(repo, "diff", "--name-only", f"{ref}~1", ref).split()
        bug_files = set(git(repo, "diff", "--name-only", f"{ref}~2", f"{ref}~1").split())
        (tests / "hidden_files.txt").write_text("".join(f"{h}\n" for h in hidden))
        for stale in ("hidden_new.txt", "hidden_tests.patch"):
            (tests / stale).unlink(missing_ok=True)
        shutil.rmtree(tests / "hidden_full", ignore_errors=True)
        for path in hidden:
            if path in bug_files:
                continue
            target = tests / "hidden_full" / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(subprocess.run(
                ["git", "-C", str(repo), "show", f"{ref}~1:{path}"], check=True, capture_output=True,
            ).stdout)
        shared = [h for h in hidden if h in bug_files]
        (tests / "hidden_tests.patch").write_text(git(repo, "diff", "--binary", ref, f"{ref}~1", "--", *shared) if shared else "")
        solution = out / "solution"
        shutil.rmtree(solution, ignore_errors=True)
        solution.mkdir()
        (solution / "fix.patch").write_text(fix)
        (solution / "solve.sh").write_text("#!/bin/bash\ncd /testbed && git apply --whitespace=nowarn /solution/fix.patch\n")
        (solution / "solve.sh").chmod(0o755)
        count += 1
    Path(f"{dest}.skipped.json").write_text(json.dumps(skipped, indent=1))
    print(f"{count} tasks -> {dest} ({len(skipped)} skipped, see {dest}.skipped.json)")


if __name__ == "__main__":
    main(N / "datasets" / sys.argv[1], N / "datasets" / sys.argv[2])
