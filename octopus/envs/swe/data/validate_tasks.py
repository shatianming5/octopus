"""Check every task of a dataset without a model: after the branch setup the verifier must give 0,
and with the branch's Bug Patch reverted (a fresh container each) it must give 1. Only such tasks
measure a fix.
usage: validate_tasks.py <dataset> <out.jsonl> [concurrency=16]"""
import asyncio, importlib, json, logging, sys, tomllib
from pathlib import Path
from types import SimpleNamespace

from octopus.envs.swe.taskset import HARBOR_ENV

N = Path("/home/user/harbor-udocker")


def make_env(image: str):
    """An environment of the OCTOPUS_HARBOR_ENV class for IMAGE, outside a Harbor trial."""
    module, name = HARBOR_ENV.split(":")
    env = object.__new__(getattr(importlib.import_module(module), name))
    env.task_env_config = SimpleNamespace(docker_image=image)
    env.logger = logging.getLogger("validate")
    env.session_id = "validate"
    env._init_state()
    return env


async def verify(task: Path, config: dict, revert_bug: bool) -> dict:
    """One fresh container, as Harbor runs it: branch setup, optional reference fix, then test.sh once."""
    ref = f"origin/{task.name}"
    row = {}
    env = make_env(config["environment"]["docker_image"])
    try:
        await env.start(False)
        row["setup_rc"] = (await env.exec(config["environment"]["healthcheck"]["command"])).return_code
        if revert_bug:  # the reference fix: undo the branch's Bug Patch commit
            row["revert_rc"] = (await env.exec(f"cd /testbed && git diff {ref}~1 {ref}~2 | git apply")).return_code
        await env.upload_dir(task / "tests", "/tests")
        run = "bash /tests/test.sh >/logs/verifier/out.txt 2>&1; cat /logs/verifier/reward.txt; tail -3 /logs/verifier/out.txt"
        out = (await env.exec(run, timeout_sec=240)).stdout.strip().splitlines()
        row["reward"], row["tail"] = (out[0] if out else ""), " | ".join(out[1:])[-300:]
    except Exception as exc:
        row["error"] = f"{type(exc).__name__}: {exc}"[:300]
    finally:
        await env.stop(True)
    return row


async def check(task: Path, sem: asyncio.Semaphore, sink):
    async with sem:
        config = tomllib.loads((task / "task.toml").read_text())
        unfixed = await verify(task, config, revert_bug=False)
        fixed = await verify(task, config, revert_bug=True)
        row = {"task": task.name, "unfixed": unfixed.get("reward"), "fixed": fixed.get("reward"),
               "setup_rc": unfixed.get("setup_rc"), "revert_rc": fixed.get("revert_rc"),
               "unfixed_tail": unfixed.get("tail"), "fixed_tail": fixed.get("tail")}
        if unfixed.get("error") or fixed.get("error"):
            row["error"] = unfixed.get("error") or fixed.get("error")
        row["valid"] = row["unfixed"] == "0" and row["fixed"] == "1"
        sink.write(json.dumps(row) + "\n"); sink.flush()
        print(json.dumps({k: row[k] for k in ("task", "unfixed", "fixed", "valid")}), flush=True)


async def main(data: Path, out: Path, conc: int):
    done = {json.loads(l)["task"] for l in out.open()} if out.exists() else set()
    tasks = [t for t in sorted(data.iterdir()) if (t / "task.toml").is_file() and t.name not in done]
    sem = asyncio.Semaphore(conc)
    with out.open("a") as sink:
        await asyncio.gather(*(check(t, sem, sink) for t in tasks))


if __name__ == "__main__":
    asyncio.run(main(N / "datasets" / sys.argv[1], Path(sys.argv[2]), int(sys.argv[3]) if len(sys.argv) > 3 else 16))
