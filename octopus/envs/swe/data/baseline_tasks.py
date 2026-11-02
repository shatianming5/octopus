"""On clean main (the reference fix) run each task's hidden test files and record the tests that
fail anyway (sandbox limits, network) into tests/known_failing.txt, which check.py deselects.
usage: baseline_tasks.py <dataset> [concurrency=16]"""
import asyncio, json, re, sys, tomllib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from validate_tasks import N, make_env  # noqa: E402

PYTEST = "/opt/miniconda3/envs/testbed/bin/python -m pytest -q -rfE --tb=no -p no:cacheprovider"


async def baseline(task: Path, sem):
    async with sem:
        config = tomllib.loads((task / "task.toml").read_text())
        ref = f"origin/{task.name}"
        env = make_env(config["environment"]["docker_image"])
        try:
            await env.start(False)
            hidden = (await env.exec(f"cd /testbed && git diff --name-only {ref}~1 {ref}")).stdout.split()
            cov = (await env.exec("/opt/miniconda3/envs/testbed/bin/python -c 'import pytest_cov'")).return_code == 0
            run = await env.exec(f"cd /testbed && timeout 150 {PYTEST} {'--no-cov' if cov else ''} {' '.join(hidden)} 2>&1", timeout_sec=180)
        finally:
            await env.stop(True)
        failing = sorted(set(re.findall(r"^(?:FAILED|ERROR) (\S+)", run.stdout, re.M)))
        (task / "tests" / "known_failing.txt").write_text("".join(f"{n}\n" for n in failing))
        print(json.dumps({"task": task.name, "hidden": hidden, "rc": run.return_code, "known_failing": failing}), flush=True)


async def main(data: Path, conc: int):
    sem = asyncio.Semaphore(conc)
    await asyncio.gather(*(baseline(t, sem) for t in sorted(data.iterdir()) if (t / "task.toml").is_file()))


if __name__ == "__main__":
    asyncio.run(main(N / "datasets" / sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 16))
