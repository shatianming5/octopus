"""Harbor environment backed by udocker. No Docker daemon, no compose."""
from __future__ import annotations

import asyncio
import fcntl
import functools
import hashlib
import json
import logging
import os
import re
import shlex
import shutil
import subprocess
import tempfile
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from harbor.environments.base import BaseEnvironment, ExecResult


# exec threads poll until the command ends; keep them out of asyncio's small default pool.
_POOL = ThreadPoolExecutor(
    max_workers=int(os.environ.get("UDOCKER_THREADS", "128")),
    thread_name_prefix="udocker",
)
# stop()'s own: execs can hold every _POOL thread (no timeout, or cancelled by the caller but still running).
_STOPS = ThreadPoolExecutor(max_workers=32, thread_name_prefix="udocker-stop")


# Copying a 3.3G rootfs is disk-bound; unbounded parallel creates pushed every start past 600s.
_CREATE_SLOTS = threading.BoundedSemaphore(int(os.environ.get("UDOCKER_CREATE_SLOTS", "4")))
_CID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")


# udocker has no pid namespace of its own and runs as us: an agent's `pkill -9 python` once killed the host's model
# servers. Every trial preloads udocker_nokill.c (its /etc/ld.so.preload), so kill & co. reach only the calling
# command's processes (each command has its own PRoot), and agent tool calls (exec env OCTOPUS_NO_EGRESS=1) cannot
# connect off the host.
_NOKILL = "/usr/lib/octopus-nokill.so"

# Commands inherit the read end; this process never closes the write end, so a command's read sees EOF exactly when
# Harbor is gone (a hard kill never reaches stop()): it then kills the command's shell, the wrapper (PRoot's first child)
# exits, and PRoot kills the rest (--kill-on-exit).
_ALIVE_R, _ALIVE_W = os.pipe()

# `udocker run` with a marker command, run by udocker's own interpreter: prints the proot argv and environment udocker
# would exec, and execs nothing. Not from the cwd: `python -c` puts it first on sys.path, and listing a big one (/tmp, a
# jobs dir) took ~40 s.
_CAPTURE = r"""
import sys
del sys.path[0]
import json, os, runpy, subprocess
def capture(cmd, **kw):
    sys.stdout.write("PROOT" + json.dumps({"argv": cmd, "env": dict(os.environ)}) + "\n"); sys.stdout.flush(); os._exit(0)
subprocess.call = capture
sys.argv = [sys.argv[1], "-q", "run", sys.argv[2], "/bin/bash", "-lc", "__CMD__"]
runpy.run_path(sys.argv[0], run_name="__main__")
"""


async def _off_loop(fn, *args, pool=None):
    return await asyncio.get_running_loop().run_in_executor(pool or _POOL, fn, *args)


# Path to a fuse-overlayfs binary: trial containers become an overlay (a small writable upper dir) over the golden's
# read-only ROOT instead of a full copy. A copy of a 3-4 GB, ~60k-file golden took 45 s median (p90 168 s) with 48
# episodes copying at once; mounting takes ~0 s, and commands inside run ~10% slower through FUSE. Empty: copy.
_OVERLAY = os.environ.get("UDOCKER_OVERLAY", "").strip()


def _unmount(cid: str) -> None:
    """Unmount a trial's overlay ROOT (no-op if it is not mounted) and remove its upper/work dirs."""
    root = _containers_dir() / cid / "ROOT"
    if os.path.ismount(root):
        subprocess.run(["fusermount3", "-u", "-z", str(root)], capture_output=True, timeout=60)
    shutil.rmtree(_containers_dir() / f"{cid}.ovl", ignore_errors=True)


@functools.cache
def _nokill_so() -> Path:
    """udocker_nokill.c built once per process with the host's gcc into our ~/.cache (not /tmp: whoever can swap the file
    runs code in every trial as us; a rename: concurrent first builds do not clash); it needs no libc but errno, so it
    fits every image."""
    out = Path.home() / ".cache" / "octopus-nokill"
    out.mkdir(mode=0o700, parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=out)
    os.close(fd)
    src = Path(__file__).with_name("udocker_nokill.c")
    built = subprocess.run(["gcc", "-shared", "-fPIC", "-O2", "-nostdlib", "-fno-stack-protector", "-o", tmp, str(src)],
                           capture_output=True, text=True)
    if built.returncode:
        os.unlink(tmp)
        raise RuntimeError(f"building {src} failed: {built.stderr[-2000:]}")
    os.replace(tmp, out / "nokill.so")
    return out / "nokill.so"


def _containers_dir() -> Path:
    explicit = os.environ.get("UDOCKER_CONTAINERS", "").strip()
    if explicit:
        return Path(explicit)
    return Path(os.environ.get("UDOCKER_DIR") or Path.home() / ".udocker") / "containers"


def _golden_dir() -> Path:
    """Where pristine per-image containers live. May differ from the trial containers dir
    (e.g. goldens on disk, trial copies on /dev/shm)."""
    explicit = os.environ.get("UDOCKER_GOLDEN_DIR", "").strip()
    return Path(explicit) if explicit else _containers_dir()


def _parse_cid(stdout: bytes | None) -> str:
    found = _CID.findall((stdout or b"").decode(errors="replace"))
    return found[-1] if found else ""


def _kill_group(proc: subprocess.Popen) -> None:
    """SIGKILL proc's children, then its process group. A command's PRoot kills every tracee (setsid'd ones, those in
    their own group) when its first child dies (--kill-on-exit), not when it is SIGKILLed itself: it gets 5 s for that."""
    try:
        children = Path(f"/proc/{proc.pid}/task/{proc.pid}/children").read_text().split()
    except OSError:
        children = []
    for pid in children:
        try:
            os.kill(int(pid), 9)
        except OSError:
            pass
    for _ in range(100 if children else 0):
        try:
            if Path(f"/proc/{proc.pid}/stat").read_text().rsplit(")", 1)[1].split()[0] == "Z":
                break
        except OSError:
            break
        time.sleep(0.05)
    try:
        os.killpg(proc.pid, 9)
    except OSError:
        proc.kill()


class StartCancelled(RuntimeError):
    pass


class UdockerEnvironment(BaseEnvironment):
    default_user = None

    @staticmethod
    def type() -> str:
        return "udocker"

    @classmethod
    def preflight(cls) -> None:
        if not shutil.which("udocker"):
            raise SystemExit("udocker is not on PATH")

    @classmethod
    def resource_capabilities(cls):
        return None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._init_state()

    def _init_state(self) -> None:
        self._cid = ""
        self._root: Path | None = None
        self._ipc: Path | None = None
        # stop() can arrive while _start_sync still runs in a thread (Harbor start timeout).
        self._stopping = threading.Event()
        self._state_lock = threading.Lock()
        self._procs: set[subprocess.Popen] = set()  # tracked commands running now: stop() kills them
        self._proot: list[str] = []  # _capture_proot
        self._proot_env: dict[str, str] = {}
        self._started = False

    def _udocker(self) -> str:
        return shutil.which("udocker") or "udocker"

    def _run(
        self,
        args: list[str],
        timeout: int = 600,
        input_bytes: bytes | None = None,
        track: bool = True,
        env: dict[str, str] | None = None,
    ) -> subprocess.CompletedProcess:
        """Run one udocker command. Tracked commands are killed (with their children) by stop()."""
        return self._run_argv([self._udocker(), *args], timeout, input_bytes, track, env)

    def _run_argv(
        self,
        argv: list[str],
        timeout: int = 600,
        input_bytes: bytes | None = None,
        track: bool = True,
        env: dict[str, str] | None = None,
        pass_fds: tuple[int, ...] = (),
    ) -> subprocess.CompletedProcess:
        proc = subprocess.Popen(
            argv,
            stdin=subprocess.PIPE if input_bytes is not None else subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
            env=env,  # None: ours
            pass_fds=pass_fds,
        )
        if track:
            with self._state_lock:
                self._procs.add(proc)
                cancelled = self._stopping.is_set()
            if cancelled:
                _kill_group(proc)
        try:
            out, err = proc.communicate(input_bytes, timeout=timeout)
        except subprocess.TimeoutExpired:
            _kill_group(proc)
            proc.communicate(timeout=10)  # bounded: a process that outlived PRoot may still hold our pipes
            raise
        finally:
            if track:
                with self._state_lock:
                    self._procs.discard(proc)
        return subprocess.CompletedProcess(proc.args, proc.returncode, out, err)

    def _image(self) -> str:
        image = getattr(self.task_env_config, "docker_image", None)
        if not image:
            raise RuntimeError("task has no docker_image; udocker env needs a prebuilt image")
        return image

    def _container_root(self) -> Path:
        if self._root and self._root.is_dir():
            return self._root
        if not self._cid:
            raise RuntimeError("udocker container was never created")
        direct = _containers_dir() / self._cid / "ROOT"
        if direct.is_dir():
            self._root = direct
            return direct
        proc = self._run(["inspect", "-p", self._cid], timeout=30, track=False)
        path = (proc.stdout or b"").decode().strip()
        if not path:
            raise RuntimeError(f"udocker inspect failed for {self._cid}: {proc.stderr.decode()[-300:]}")
        root = Path(path)
        if root.name != "ROOT":
            root = root / "ROOT"
        self._root = root
        return root

    def _ipc_dir(self) -> Path:
        if self._ipc and self._ipc.is_dir():
            return self._ipc
        ipc = self._container_root() / "logs" / "udocker_ipc"
        ipc.mkdir(parents=True, exist_ok=True)
        try:
            os.chmod(ipc, 0o777)
        except OSError:
            pass
        self._ipc = ipc
        return ipc

    def _publish_ipc_links(self, ipc: Path) -> None:
        """Host-visible index of live trial IPC dirs (UDOCKER_IPC_INDEX). Nothing is written into the
        task's environment dir: that is the dataset, shared by every trial of the task."""
        raw = os.environ.get("UDOCKER_IPC_INDEX", "").strip()
        if not raw:
            return
        try:
            index = Path(raw)
            index.mkdir(parents=True, exist_ok=True)
            dest = index / f"{self.session_id}__{self._cid}"[:180]
            if dest.is_symlink() or dest.exists():
                dest.unlink()
            dest.symlink_to(ipc)
            (ipc / "HOST_PATH.txt").write_text(f"{ipc}\nindex={dest}\ncid={self._cid}\n")
        except OSError:
            pass

    def _validate_definition(self):
        if not getattr(self.task_env_config, "docker_image", None):
            raise FileNotFoundError("udocker environment requires task docker_image")

    async def start(self, force_build: bool) -> None:
        # Every udocker call blocks for seconds; off the loop so other trials keep running.
        await _off_loop(self._start_sync)

    def _ensure_image(self, image: str) -> None:
        listed = self._run(["images"], timeout=60, track=False)
        have = (listed.stdout or b"").decode()
        if image in have or image.split(":")[0] in have:
            return
        pulled = self._run(["pull", image], 3600, track=False)
        if pulled.returncode != 0:
            raise RuntimeError(
                "udocker pull failed for "
                f"{image}: {(pulled.stderr or b'').decode()[-400:]}"
            )

    def _golden(self, image: str) -> str:
        """One pristine, never-run container per image. Trials clone it instead of re-extracting layers.

        Shared by every trial of the image, so it is built untracked: a cancelled trial does not kill it.
        """
        root = _golden_dir()
        at_golden = {**os.environ, "UDOCKER_CONTAINERS": str(root)}
        meta = root / ".golden"
        meta.mkdir(parents=True, exist_ok=True)
        key = hashlib.sha1(image.encode()).hexdigest()[:16]
        record = meta / f"{key}.cid"
        # Lock on the (fast) trials dir: opening a file on a busy HDD can wait on its journal for a minute.
        locks = _containers_dir() / ".golden-locks"
        locks.mkdir(parents=True, exist_ok=True)
        with open(locks / f"{key}.lock", "a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if record.is_file():
                lines = record.read_text().split()
                cid = lines[0] if lines else ""
                if cid and (root / cid / "ROOT").is_dir():
                    return cid
            self._ensure_image(image)
            with _CREATE_SLOTS:
                created = self._run(["create", image], timeout=1800, track=False, env=at_golden)
            cid = _parse_cid(created.stdout)
            if created.returncode != 0 or not cid:
                if cid:
                    self._run(["rm", "-f", cid], timeout=300, track=False, env=at_golden)
                raise RuntimeError(f"udocker create failed for {image}: {(created.stderr or b'').decode()[-400:]}")
            record.write_text(f"{cid}\n{image}\n")
            self.logger.info("udocker golden %s image=%s", cid, image)
            return cid

    def _check_cancel(self) -> None:
        if self._stopping.is_set():
            self._discard()
            raise StartCancelled("udocker start cancelled by stop()")

    def _discard(self) -> None:
        with self._state_lock:
            cid, self._cid = self._cid, ""
        self._root = None
        self._ipc = None
        if cid:
            _unmount(cid)  # first: removing through a mounted overlay would walk the whole golden
            self._run(["rm", "-f", cid], timeout=300, track=False)
            # A copy killed halfway is not a valid udocker container; rm may leave it.
            shutil.rmtree(_containers_dir() / cid, ignore_errors=True)

    def _clone(self, golden: str) -> subprocess.CompletedProcess:
        """Copy the golden container into the trials dir under a fresh id (what `udocker clone` does)."""
        src = _golden_dir() / golden
        cid = str(uuid.uuid4())
        dest = _containers_dir() / cid
        with self._state_lock:
            self._cid = cid
        dest.mkdir(parents=True)
        if _OVERLAY:  # small metadata files copied, ROOT mounted as an overlay over the golden's
            ovl = _containers_dir() / f"{cid}.ovl"
            pipe = (
                f"cp -a {' '.join(shlex.quote(str(p)) for p in src.iterdir() if p.name != 'ROOT')} {shlex.quote(str(dest))}/ "
                f"&& mkdir -p {shlex.quote(str(dest / 'ROOT'))} {shlex.quote(str(ovl / 'upper'))} {shlex.quote(str(ovl / 'work'))} "
                f"&& {shlex.quote(_OVERLAY)} -o lowerdir={src / 'ROOT'},upperdir={ovl / 'upper'},workdir={ovl / 'work'} "
                f"{shlex.quote(str(dest / 'ROOT'))} < /dev/null > /dev/null 2>&1"  # it stays as a daemon: not on our pipes
            )
            copied = self._run_argv(["/bin/bash", "-o", "pipefail", "-c", pipe], timeout=120)
            return subprocess.CompletedProcess(copied.args, copied.returncode, f"{cid}\n".encode(), copied.stderr)
        pipe = (
            f"tar -C {shlex.quote(str(src))} -c --one-file-system -S --xattrs -f - . "
            f"| tar -C {shlex.quote(str(dest))} -x -f -"
        )
        copied = self._run_argv(["/bin/bash", "-o", "pipefail", "-c", pipe], timeout=900)
        return subprocess.CompletedProcess(copied.args, copied.returncode, f"{cid}\n".encode(), copied.stderr)

    def _start_sync(self) -> None:
        image = self._image()
        golden = self._golden(image)
        with _CREATE_SLOTS:
            self._check_cancel()
            cloned = self._clone(golden)
        cid = self._cid
        if cloned.returncode != 0 or not cid:
            self._check_cancel()
            self._discard()
            raise RuntimeError(f"udocker clone of {golden} failed: {(cloned.stderr or b'').decode()[-400:]}")
        self._check_cancel()
        self._run(["setup", "--execmode=P1", cid], timeout=120)
        root = self._container_root()
        for rel in (
            "tmp", "testbed", "logs/verifier", "logs/agent",
            "logs/artifacts", "logs/udocker_ipc", "tests",
        ):
            path = root / rel
            path.mkdir(parents=True, exist_ok=True)
            try:
                os.chmod(path, 0o777)
            except OSError:
                pass
        shutil.copyfile(_nokill_so(), root / _NOKILL.lstrip("/"))
        (root / "etc/ld.so.preload").write_text(_NOKILL + "\n")
        self.logger.info("udocker started %s image=%s root=%s", cid, image, root)
        ipc = self._ipc_dir()
        self._publish_ipc_links(ipc)
        self._check_cancel()
        self._capture_proot()
        self._check_cancel()
        self._started = True

    def _capture_proot(self) -> None:
        """What `udocker run` execs, captured once per trial: its proot argv, and the environment a login shell leaves.
        A command then starts in ~10 ms as `bash -c` with that environment, as under the old long-lived `bash -l`
        worker; `udocker run ... bash -lc` per command cost ~0.35 s (~0.3 s of it the login scripts under PRoot)."""
        script = self._udocker()
        with open(script) as f:
            python = shlex.split(f.readline()[2:])  # its shebang interpreter
        got = self._run_argv([*python, "-c", _CAPTURE, script, self._cid], timeout=300)
        line = next((x for x in (got.stdout or b"").decode(errors="replace").splitlines() if x.startswith("PROOT")), "")
        cap = json.loads(line[5:]) if line else {}
        if not cap or cap["argv"][-3:] != ["/bin/bash", "-lc", "__CMD__"]:
            raise RuntimeError(f"udocker run capture failed for {self._cid}: {(got.stderr or b'').decode()[-400:]}")
        proot = cap["argv"][:-3]
        if "--kill-on-exit" not in proot:  # timeout, stop() and a dead Harbor rely on it to end a command's tracees
            raise RuntimeError(f"udocker's proot argv has no --kill-on-exit: {proot}")
        login = self._run_argv([*proot, "/bin/bash", "-lc", "env -0"], timeout=120, env=cap["env"])
        if login.returncode != 0:
            raise RuntimeError(f"login shell in {self._cid} failed: {(login.stderr or b'').decode()[-400:]}")
        pairs = (x.split("=", 1) for x in login.stdout.decode(errors="replace").split("\0") if "=" in x)
        self._proot, self._proot_env = proot, {**cap["env"], **dict(pairs)}

    def _command_argv(self, command: str) -> list[str]:
        # Keep udocker's default host /dev /proc /sys binds: without them multiprocessing has no
        # /dev/shm and pty tests have no /dev/pts, so those tests fail on clean code too.
        return [*self._proot, "/bin/bash", "-c", command]

    async def stop(self, delete: bool):
        with self._state_lock:
            self._stopping.set()
            procs = list(self._procs)
        await asyncio.gather(*(_off_loop(_kill_group, proc, pool=_STOPS) for proc in procs if proc.poll() is None))
        # A start that never finished is garbage whatever delete says; the start thread removes
        # anything it creates after this point (_check_cancel).
        if self._cid and (delete or not self._started):
            await _off_loop(self._discard, pool=_STOPS)

    async def exec(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
        user: str | int | None = None,
    ) -> ExecResult:
        # One process per command: concurrent callers (fork branches) run side by side.
        work = cwd or "/testbed"
        # No timeout_sec: the caller bounds the call itself (Harbor wraps the verifier in the task's verifier
        # timeout and the healthcheck in its own); a fixed 180 s here cut every verifier longer than that.
        timeout = int(timeout_sec) if timeout_sec else None
        script = f"#!/bin/bash\ncd {shlex.quote(work)} || exit 1\n{command}\n"
        return await _off_loop(self._exec_sync, script, timeout, env or {})

    def _exec_sync(self, script: str, timeout: int | None, env: dict[str, str]) -> ExecResult:
        name = f"c-{uuid.uuid4().hex}.sh"  # a file, not argv: a command can be longer than one argument may be
        path = self._ipc_dir() / name
        path.write_text(script)
        fd = _ALIVE_R  # the command runs with it closed; a watcher blocks on it (no polling) and is killed at the end
        wrapper = (f"exec 9<&{fd} {fd}<&-; /bin/bash /logs/udocker_ipc/{name} 9<&- & c=$!; "
                   "{ read -u 9 _; kill -9 $c; } & w=$!; wait $c; rc=$?; kill $w 2>/dev/null; exit $rc")
        try:
            # exactly udocker's environment: it leaves out ours (API keys, ssh), which the agent must not see. The
            # caller's env goes to PRoot's own environment, where the command cannot change it (OCTOPUS_NO_EGRESS).
            # Singularity captures none: ours, as before
            done = self._run_argv(self._command_argv(wrapper), timeout, env={**(self._proot_env or os.environ), **env},
                                  pass_fds=(fd,))
        except subprocess.TimeoutExpired:
            return ExecResult(stdout="", stderr=f"udocker exec timed out after {timeout}s", return_code=124)
        finally:
            path.unlink(missing_ok=True)
        if self._stopping.is_set():  # the caller gave up (e.g. its timeout) and stopped the environment
            return ExecResult(stdout="", stderr="udocker environment stopped", return_code=137)
        return ExecResult(stdout=done.stdout.decode(errors="replace"), stderr=done.stderr.decode(errors="replace"),
                          return_code=done.returncode)

    async def upload_file(self, source_path: Path | str, target_path: str):
        await _off_loop(self._upload_file, source_path, target_path)

    def _upload_file(self, source_path: Path | str, target_path: str):
        src = Path(source_path)
        dest = self._container_root() / target_path.lstrip("/")
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)

    async def upload_dir(self, source_dir: Path | str, target_dir: str):
        await _off_loop(self._upload_dir, source_dir, target_dir)

    def _upload_dir(self, source_dir: Path | str, target_dir: str):
        src = Path(source_dir)
        dest = self._container_root() / target_dir.lstrip("/")
        dest.mkdir(parents=True, exist_ok=True)
        shutil.copytree(src, dest, dirs_exist_ok=True)

    async def download_file(self, source_path: str, target_path: Path | str):
        if not self._cid:
            return
        await _off_loop(self._download_file, source_path, target_path)

    def _download_file(self, source_path: str, target_path: Path | str):
        src = self._container_root() / source_path.lstrip("/")
        dest = Path(target_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)

    async def download_dir(self, source_dir: str, target_dir: Path | str):
        if not self._cid:
            return
        await _off_loop(self._download_dir, source_dir, target_dir)

    def _download_dir(self, source_dir: str, target_dir: Path | str):
        src = self._container_root() / source_dir.lstrip("/")
        dest = Path(target_dir)
        dest.mkdir(parents=True, exist_ok=True)
        if not src.is_dir():
            raise FileNotFoundError(f"udocker missing {src}")
        shutil.copytree(src, dest, dirs_exist_ok=True)
