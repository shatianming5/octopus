"""octopus/envs/swe/udocker_nokill.c under a real PRoot (rootfs /, as udocker P1 runs a trial), through UdockerEnvironment.exec:
kill, pkill, killall, os.kill, killpg, pidfd_send_signal reach the command's own processes and nothing else, not the
host's and not a concurrent command's of the same trial (fork branches' tool calls). Kills that would reach beyond a
dedicated `nokill-probe` binary use SIGURG (ignored by default), so a shim that fails to load harms nothing. And a
command's exec timeout, stop() or `kill -9 0` takes its setsid'd and `timeout`-wrapped processes along."""
import asyncio, os, shutil, subprocess, tempfile, threading, time
from pathlib import Path

import pytest

pytest.importorskip("harbor")

from octopus.envs.swe import udocker as ue  # noqa: E402

UDOCKER_BIN = Path(os.environ.get("UDOCKER_DIR") or Path.home() / ".udocker") / "bin"
PROOT = next((p for p in (UDOCKER_BIN / "proot-x86_64-4_8_0", UDOCKER_BIN / "proot-x86_64") if p.exists()), None)  # udocker's pick
PROOT = PROOT or shutil.which("proot")
pytestmark = pytest.mark.skipif(not PROOT or not shutil.which("gcc"), reason="needs proot and gcc")


@pytest.fixture
def probe(tmp_path):
    path = tmp_path / "nokill-probe"  # a sleep under its own name: pkill/killall match only these
    shutil.copy(shutil.which("sleep"), path)
    return path


@pytest.fixture
def env(tmp_path):
    """Commands run as a trial's do, each under its own PRoot, with the shim preloaded."""
    env = object.__new__(ue.UdockerEnvironment)
    env._ipc, env._procs = tmp_path, set()
    env._proot_env = {**os.environ, "LD_PRELOAD": str(ue._nokill_so())}
    env._stopping, env._state_lock = threading.Event(), threading.Lock()
    env._cid, env._started = "", True
    env._command_argv = lambda cmd: [str(PROOT), "--kill-on-exit", "-r", "/", "/bin/bash", "-c",
                                     cmd.replace("/logs/udocker_ipc", str(tmp_path))]
    return env


def alive(probe, arg: str) -> bool:
    deadline = time.time() + 3
    while subprocess.run(["pgrep", "-f", f"{probe} {arg}"], capture_output=True).returncode == 0:
        if time.time() > deadline:
            return True
        time.sleep(0.1)
    return False


def test_shim_needs_no_libc_but_errno():
    out = subprocess.run(["nm", "-D", "--undefined-only", str(ue._nokill_so())], capture_output=True, text=True).stdout
    assert out.split() == ["U", "__errno_location"]


def test_kills_reach_only_the_commands_own_processes(env, tmp_path, probe):
    host = subprocess.Popen([probe, "300"])
    h, g = host.pid, os.getpgrp()
    # 303: a group whose leader is gone, as `os.killpg(p.pid)` meets after p exited with children left
    script = """
python3 -c 'import os; os.kill({h}, 9)' 2>&1 | tail -1
python3 -c 'import os; os.kill({o}, 9)' 2>&1 | tail -1
python3 -c 'import os, signal; signal.pidfd_send_signal(os.pidfd_open({o}), 9)' 2>&1 | tail -1
python3 -c 'import ctypes, os; c = ctypes.CDLL(None, use_errno=True); print("sigqueue", c.sigqueue({h}, 9, 0), ctypes.get_errno(), "pidfd_send_signal", c.pidfd_send_signal(os.pidfd_open({h}), 9, None, 0), ctypes.get_errno(), "sigqueueinfo", c.syscall(129, {h}, 9, None), ctypes.get_errno(), c.syscall(297, {h}, {h}, 9, None), ctypes.get_errno())'
kill -9 {h} 2>/dev/null; echo kill=$?
kill -9 {o} 2>/dev/null; echo sibling=$?
kill -URG -1 2>/dev/null; echo all=$?
kill -URG -{g} 2>/dev/null; echo group=$?
kill -0 {h}; echo sig0=$?
kill -9 999999999 2>&1 | grep -o 'No such process'
python3 -c 'import os, subprocess, time; p = subprocess.Popen(["true"]); time.sleep(0.5); os.kill(p.pid, 9); print("zombie ok")'
{probe} 301 & kill %1; wait $!; echo job=$?
{probe} 301 & pkill -9 -x nokill-probe; wait $!; echo pkill=$?
{probe} 301 & pkill -9 -f nokill-probe; wait $!; echo pkillf=$?
{probe} 301 & pkill -q 1 -9 -x nokill-probe; wait $!; echo pkillq=$?
if command -v killall >/dev/null; then {probe} 301 & killall -q -9 nokill-probe; wait $!; echo killall=$?; else echo killall=137; fi
python3 -c 'import os, signal; os.killpg(os.getpgrp(), signal.SIGURG); print("own group ok")'
python3 -c 'import os, subprocess, time; p = subprocess.Popen(["sh", "-c", "$0 303 & exit", "{probe}"], start_new_session=True); p.wait(); os.killpg(p.pid, 9); time.sleep(0.5); print("leaderless", subprocess.run(["pgrep", "-f", "nokill-prob[e] 303"]).returncode)'
python3 -c 'import subprocess; p = subprocess.Popen(["sleep", "300"]); p.terminate(); print("terminate", p.wait())'
timeout 1 sleep 5; echo timeout=$?
"""

    async def main():
        sibling = asyncio.create_task(env.exec(f"{probe} 302", cwd=str(tmp_path)))  # a concurrent command
        await asyncio.sleep(1.5)
        o = int(subprocess.check_output(["pgrep", "-f", f"{probe} 302"]))
        out = (await env.exec(script.format(h=h, o=o, g=g, probe=probe), cwd=str(tmp_path))).stdout
        survived = host.poll() is None and alive(probe, "302")
        await env.stop(delete=False)
        return out, survived, (await sibling).return_code
    try:
        out, survived, sibling_rc = asyncio.run(main())
    finally:
        host.kill()
    assert out.split("\n")[:-1] == [
        "PermissionError: [Errno 1] Operation not permitted", "PermissionError: [Errno 1] Operation not permitted",
        "PermissionError: [Errno 1] Operation not permitted",
        "sigqueue -1 1 pidfd_send_signal -1 1 sigqueueinfo -1 1 -1 1",
        "kill=1", "sibling=1", "all=1", "group=1", "sig0=0",
        "No such process", "zombie ok", "job=143", "pkill=137", "pkillf=137", "pkillq=137", "killall=137", "own group ok",
        "leaderless 1", "terminate -15", "timeout=124",
    ]
    assert survived and sibling_rc == 137
    assert not alive(probe, "302")


def test_timeout_and_stop_kill_setsid_and_timeout_wrapped_processes(env, tmp_path, probe):
    spawn = "setsid {p} {a} & timeout 300 {p} {b} & sleep 30"
    r = asyncio.run(env.exec(spawn.format(p=probe, a=306, b=307), cwd=str(tmp_path), timeout_sec=2))
    assert r.return_code == 124
    assert not alive(probe, "306") and not alive(probe, "307")

    async def stopped():
        task = asyncio.create_task(env.exec(spawn.format(p=probe, a=308, b=309), cwd=str(tmp_path)))
        await asyncio.sleep(1.5)
        assert alive(probe, "308") and alive(probe, "309")
        await env.stop(delete=False)
        return await task
    assert asyncio.run(stopped()).return_code == 137
    assert not alive(probe, "308") and not alive(probe, "309")


def test_kill_0_ends_the_command_without_orphans(env, tmp_path, probe):
    t = time.time()
    r = asyncio.run(env.exec(f"setsid {probe} 310 & {probe} 311 & sleep 0.5; kill -9 0", cwd=str(tmp_path),
                             timeout_sec=20))
    assert r.return_code == 137 and time.time() - t < 10
    assert not alive(probe, "310") and not alive(probe, "311")


def test_agent_tool_calls_connect_only_on_this_host(env, tmp_path):
    """OCTOPUS_NO_EGRESS=1 (the agent's commands): connects off the host, or to a proxy port on loopback, fail with EACCES,
    even with the variable unset inside; loopback, IPv4-mapped loopback and unix sockets work. Without it (healthcheck, verifier) nothing changes."""
    sock = Path(tempfile.mkdtemp(dir="/tmp")) / "s"  # sun_path has 108 bytes; tmp_path can be longer
    off = "python3 -c 'import socket; socket.create_connection((\"192.0.2.1\", 80), 1)' 2>&1 | tail -1"  # TEST-NET-1
    tool = f"""
python3 -m http.server 18765 --bind 127.0.0.1 >/dev/null 2>&1 & sleep 1
curl -s -o /dev/null -w 'loopback %{{http_code}}\\n' http://127.0.0.1:18765/
python3 -c 'import socket; socket.create_connection(("::ffff:127.0.0.1", 18765)); print("mapped ok")'
python3 -c 'import socket; s = socket.socket(socket.AF_UNIX); s.bind("{sock}"); s.listen(); socket.socket(socket.AF_UNIX).connect("{sock}"); print("unix ok")'
python3 -c 'import ctypes, socket; a = bytes([10, 0, 0, 80, 0, 0, 0, 0, 0x20, 1, 0xd, 0xb8] + [0] * 12); print("in6 24 bytes", ctypes.CDLL(None, use_errno=True).connect(socket.socket(socket.AF_INET6).fileno(), a, 24), ctypes.get_errno())'
kill %1
python3 -c 'import socket; print("proxy ports", sorted({{socket.socket(f).connect_ex((h, p)) for h, f in (("127.0.0.1", 2), ("127.0.0.2", 2), ("0.0.0.0", 2), ("::1", 10), ("::ffff:127.0.0.1", 10), ("::", 10)) for p in (10808, 10809, 8118, 7890, 7891, 1080)}}))'
curl -sSv https://pypi.org 2>&1 | grep -o 'Permission denied' | sort -u
python3 -c "import urllib.request; urllib.request.urlopen('https://pypi.org')" 2>&1 | grep -o 'Permission denied' | sort -u
GIT_CURL_VERBOSE=1 git ls-remote https://github.com/pallets/click 2>&1 | grep -o 'Permission denied' | sort -u
pip download --no-deps --retries 1 -d {tmp_path} six 2>&1 | grep -o 'Permission denied' | sort -u
env -u OCTOPUS_NO_EGRESS {off}
"""
    out = asyncio.run(env.exec(tool, cwd=str(tmp_path), env={"OCTOPUS_NO_EGRESS": "1"})).stdout
    shutil.rmtree(sock.parent)
    assert out.split("\n")[:-1] == ["loopback 200", "mapped ok", "unix ok", "in6 24 bytes -1 13", "proxy ports [13]",
                                    *["Permission denied"] * 4,
                                    "PermissionError: [Errno 13] Permission denied"]
    assert "Permission denied" not in asyncio.run(env.exec(off, cwd=str(tmp_path))).stdout
