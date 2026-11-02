"""Rollout-only load (no trainer, no optimizer step): training's AReaL proxy in this process (octopus/rl/proxy.py,
the repo's grpo.yaml with tokenizer_path MODEL) on one SGLang server with the policy LoRA, and bundle plan
episodes (Harbor + udocker, HarborEpisodeAgent as RL runs them, its run flags from the environment), PARALLEL at a time, until N episodes ran or
DURATION seconds passed (then the running ones are cancelled: Harbor removes their containers). Slice 6 of
docs/cross-branch-read.md puts octopus/tools/mount_load.py on top of it. With SWITCH > 0, every SWITCH seconds the server
loads ADAPTER again as the next version (rq1-v1, rq1-v2, ...: AReaL's names after a weight update), new requests use it
(the engine's version, and OUT_JSON.lora for mount_load), and versions older than the last KEEP are unloaded.

usage: python -m octopus.tools.rollout_load SGLANG_ADDR PROXY_PORT TASKS_TXT N PARALLEL DURATION OUT_JSON MODEL
       [SWITCH ADAPTER KEEP]
"""
import asyncio
import itertools
import json
import sys
import time

from areal.experimental.openai.proxy import proxy_rollout_server as srv
from areal.experimental.openai.proxy.server import StartSessionRequest

from octopus.rl import proxy
from octopus.rl.config import CONFIG, load, set_derived_settings, workflow_kwargs
from octopus.envs.swe.rl import HarborEpisodeAgent

addr, port, tasks_txt, n, parallel, duration, out, model = sys.argv[1:9]
switch, adapter, keep = (float(sys.argv[9]), sys.argv[10], int(sys.argv[11])) if len(sys.argv) > 9 else (0, None, 0)
config = load(CONFIG, [f"tokenizer_path={model}"])  # the proxy and the episodes as grpo.yaml runs them
set_derived_settings(config)


async def switch_versions(eng):
    """A weight update's adapter switch: the same weights under the next version's name."""
    import aiohttp

    async with aiohttp.ClientSession() as http:
        for v in itertools.count(1):
            await asyncio.sleep(switch)
            async with http.post(f"http://{addr}/load_lora_adapter",
                                 json={"lora_name": f"rq1-v{v}", "lora_path": adapter}) as r:
                assert r.status == 200, await r.text()
            eng.set_version(v)
            open(f"{out}.lora", "w").write(f"rq1-v{v}")
            print(f"{time.strftime('%H:%M:%S')} adapter rq1-v{v}", flush=True)
            if v - keep >= 0:
                async with http.post(f"http://{addr}/unload_lora_adapter", json={"lora_name": f"rq1-v{v - keep}"}) as r:
                    assert r.status == 200, await r.text()


async def main():
    async with proxy.serve(config, [addr], "rq1-v0", int(port)) as url:
        await run(srv._engine, url)


async def run(eng, url: str):
    tasks = [x.strip() for x in open(tasks_txt) if x.strip()]
    agent = HarborEpisodeAgent(**{**workflow_kwargs(config), "jobs_dir": f"{out}.jobs", "parallel": int(parallel)})

    async def one(task_dir, k):
        s = srv.start_session(StartSessionRequest(task_id=f"{task_dir.rsplit('/', 1)[-1]}-{k}"))
        t = time.time()
        try:
            rec = await agent.run_episode({"task_dir": task_dir, "kind": "plan", "plan_control": True}, url, s.api_key, s.session_id)
        except Exception as e:
            rec = {"error": f"{type(e).__name__}: {e}"[:300]}
        return {"task_dir": task_dir, "sample": k, "session": s.session_id, "wall_s": round(time.time() - t),
                **{key: rec.get(key) for key in ("error", "trial", "ended_with", "y")}}

    open(f"{out}.lora", "w").write("rq1-v0")
    switching = asyncio.create_task(switch_versions(eng)) if switch else None
    runs = [asyncio.create_task(one(t, k)) for k, t in zip(range(int(n)), itertools.cycle(tasks))]
    done, pending = await asyncio.wait(runs, timeout=float(duration))
    for r in [*pending, *([switching] if switching else [])]:
        r.cancel()
    await asyncio.gather(*pending, return_exceptions=True)
    json.dump({"episodes": [r.result() for r in done], "cancelled": len(pending),
               "interactions": sum(len(sd.completions) for sd in srv._session_cache.values())},
              open(out, "w"), indent=1, default=str)


asyncio.run(main())
