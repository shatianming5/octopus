"""Copy a just-saved LoRA weight-update dir to the remote rollout hosts before their SGLang servers are told to load it.

AReaL's rollout workers load `<fileroot>/.../weight_update_v<N>` by path on every server once the trainer's rank 0
publishes the update key; a server on another machine needs the files at the same path there first. ChunkedHeadFSDPPPOActor
pushes right after the save and before that key, so a failed push fails the weight update instead of leaving a remote
server on an old version.
"""

import os
import subprocess
import time

# ssh destinations user@host:port of the remote rollout hosts, as reachable from the trainer (rl_run.sh REMOTE)
DESTS = os.environ.get("OCTOPUS_ADAPTER_PUSH", "").split()


def push_adapter(path: str, dests: list[str], attempts: int = 4) -> None:
    """tar-stream `path` to each destination at the same path, replacing the weight_update_v* dirs pushed before it
    (the servers keep loaded adapters in memory). ssh's own failure (exit 255: e.g. a remote sshd whose unauthenticated
    slots are full drops the handshake) is tried again, `attempts` times in all; the remote side is idempotent. Raises
    CalledProcessError or TimeoutExpired on any other failure."""
    parent, name = os.path.split(path.rstrip("/"))
    for dest in dests:
        host, port = dest.rsplit(":", 1)
        remote = f"mkdir -p {parent} && cd {parent} && rm -rf weight_update_v* && tar -xf -"
        for attempt in range(attempts):
            tar = subprocess.Popen(["tar", "-C", parent, "-cf", "-", name], stdout=subprocess.PIPE)
            try:
                ran = subprocess.run(["ssh", "-p", port, "-o", "BatchMode=yes", "-o", "ConnectTimeout=20", host, remote],
                                     stdin=tar.stdout, timeout=90)
            finally:
                tar.stdout.close()
            tar.wait()
            if ran.returncode == 255 and attempt + 1 < attempts:
                time.sleep(2 ** attempt)
                continue
            if ran.returncode:
                raise subprocess.CalledProcessError(ran.returncode, ran.args)
            if tar.returncode:
                raise subprocess.CalledProcessError(tar.returncode, tar.args)
            break
