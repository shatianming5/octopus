"""Build every golden for a Harbor dataset up front. Shares flock + records with running trials."""
import logging, sys, threading, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from octopus.envs.swe.udocker import UdockerEnvironment

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
from collections import Counter
counts = Counter(
    line.split("=", 1)[1].strip().strip('"')
    for toml in Path(sys.argv[1]).glob("*/task.toml")
    for line in toml.read_text().splitlines()
    if line.strip().startswith("docker_image")
)
images = [image for image, _ in counts.most_common()]  # unblock the most trials first
print(f"{len(images)} images", flush=True)

def build(image):
    env = object.__new__(UdockerEnvironment)
    env.task_env_config = SimpleNamespace(docker_image=image)
    env.logger = logging.getLogger("prebuild")
    env._stopping = threading.Event(); env._state_lock = threading.Lock(); env._procs = set()
    t = time.time()
    try:
        cid = env._golden(image)
        print(f"ok {time.time()-t:6.0f}s {cid} {image}", flush=True)
    except Exception as exc:
        print(f"FAIL {image}: {exc}", flush=True)

with ThreadPoolExecutor(int(sys.argv[2]) if len(sys.argv) > 2 else 4) as pool:
    list(pool.map(build, images))
print("prebuild done", flush=True)
