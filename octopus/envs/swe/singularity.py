"""Harbor environment on Singularity, for hosts where udocker goldens do not fit (NVwulf: a golden is ~82k files and
3.2 GB on Lustre under a 10M-file quota; compute nodes have no local disk).

One read-only SIF per image in SINGULARITY_SIF_DIR, with the image's /root next to it as <name>.root.tar
(scripts/nvwulf/pull_sifs.sh). Each trial gets a RAM dir under SINGULARITY_TRIALS_DIR (default /dev/shm) holding a
writable copy of the image's /testbed and /root plus empty /tmp, /logs, /tests, /solution and /tasks, each bound at its
path; the rest of the image stays read-only. Commands run as in UdockerEnvironment (one `bash -l` process per
command, the script under /logs/udocker_ipc, the same exec timeout rules) inside `singularity exec --containall`: its
own pid and ipc namespaces and /dev, so an agent's pkill reaches only its own command.
Select it in Harbor with `-e octopus.envs.swe.singularity:SingularityEnvironment` (OCTOPUS_HARBOR_ENV).

Differences from udocker: commands run as the invoking user, not a fake root, and the image outside the bound
directories cannot be written (e.g. no `pip install` into the conda env), unless SINGULARITY_TRIAL_OVERLAY=1 (a host setting
like SINGULARITY_SIF_DIR; not SINGULARITY_OVERLAY, which Apptainer reads itself): then every
trial is one Apptainer instance (`instance start` at start, `instance stop` at discard) with a writable overlay dir
(`--overlay`: Apptainer's unprivileged fuse-overlayfs, one mount per trial), and each command an `exec
instance://` into it, so writes anywhere persist from one command to the next and concurrent commands (fork branches)
share one view, as in a udocker container (coupled-a's healthcheck removes vendored copies of the task's library under
/opt/miniconda3). Not one `exec --overlay` per command: two fuse-overlayfs mounts of one upper dir lose each other's
writes. /testbed and /tmp are then the image's own under that overlay, not binds (a test renaming a file from
/testbed to /tmp fails across two binds), and the host reaches the container's files through the overlay's upper dir
except while the instance runs: then files outside the bound dirs go through a staging dir under /logs and a cp
inside the instance (fuse-overlayfs does not see the host write into its upper dir); the instance's pid namespace
holds every command of the trial.
"""
from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import uuid
from pathlib import Path

from octopus.envs.swe.udocker import _CREATE_SLOTS, UdockerEnvironment

# writable per-trial directories, bound at /<name>
BINDS = ("testbed", "root", "tmp", "logs", "tests", "solution", "tasks")
OVERLAY = os.environ.get("SINGULARITY_TRIAL_OVERLAY") == "1"  # the rest of the image writable through a per-trial overlay


def sif_name(image: str) -> str:
    """swebench/swesmith.x86_64.<repo>:transport-<x> and jyangballin/swesmith.x86_64.<repo>:latest -> <one SIF name>."""
    return image.rsplit("/", 1)[-1].split(":", 1)[0] + ".sif"


class SingularityEnvironment(UdockerEnvironment):
    @staticmethod
    def type() -> str:
        return "singularity"

    @classmethod
    def preflight(cls) -> None:
        if not shutil.which("singularity"):
            raise SystemExit("singularity is not on PATH")

    def _sif(self) -> Path:
        sif = Path(os.environ["SINGULARITY_SIF_DIR"]) / sif_name(self._image())
        if not sif.is_file():
            raise RuntimeError(f"no SIF {sif} for {self._image()} (scripts/nvwulf/pull_sifs.sh)")
        return sif

    def _start_sync(self) -> None:
        sif = self._sif()
        root = Path(os.environ.get("SINGULARITY_TRIALS_DIR", "/dev/shm")) / f"octopus-{uuid.uuid4()}"
        with self._state_lock:
            self._cid, self._root = str(root), root
        fs = root / "overlay" / "upper" if OVERLAY else root  # the container's / as the host sees it (_container_root)
        for name in BINDS[1:]:  # testbed comes from the copy below (from the image itself under the overlay)
            (fs / name).mkdir(parents=True)
        for rel in ("logs/verifier", "logs/agent", "logs/artifacts", "logs/udocker_ipc"):
            (fs / rel).mkdir(parents=True)
        self._check_cancel()
        with _CREATE_SLOTS:
            self._check_cancel()
            copied = self._run_argv([
                "/bin/bash", "-c",
                'tar -xf "$1" -C "$2/root" && { [ -n "$4" ] || singularity exec --containall --bind "$2:/mnt/octopus-trial" "$3" '
                'cp -a /testbed /mnt/octopus-trial/; }', "_", str(sif.with_suffix(".root.tar")), str(fs), str(sif),
                "1" if OVERLAY else "",
            ], timeout=900)
        if copied.returncode != 0:
            self._check_cancel()
            self._discard()
            raise RuntimeError(f"copying /root and /testbed of {sif} failed: {(copied.stderr or b'').decode()[-400:]}")
        if OVERLAY:
            self._check_cancel()
            started = self._run_argv(["singularity", "instance", "start", *self._options(), str(sif), root.name], timeout=300)
            if started.returncode != 0:
                self._check_cancel()
                self._discard()
                raise RuntimeError(f"singularity instance start of {sif} failed: {(started.stderr or b'').decode()[-400:]}")
        self.logger.info("singularity started %s image=%s", root, sif)
        self._check_cancel()
        self._started = True

    def _container_root(self) -> Path:
        if not self._root:
            raise RuntimeError("singularity trial was never created")
        return self._root / "overlay" / "upper" if OVERLAY else self._root

    def _options(self) -> list[str]:
        root = self._container_root()
        binds = [f"--bind={root / name}:/{name}" for name in BINDS if name != "root"]
        if OVERLAY:  # /testbed and /tmp from the overlay, one mount: a rename between them works (EXDEV across binds)
            binds = [b for b in binds if not b.endswith((":/testbed", ":/tmp"))]
            binds += ["--no-mount=tmp", f"--overlay={root.parent}"]
        return ["--containall", f"--home={root / 'root'}:/root", *binds]

    def _command_argv(self, command: str) -> list[str]:
        image = f"instance://{self._container_root().parents[1].name}" if OVERLAY else str(self._sif())
        return ["singularity", "exec", "--pwd=/", *([] if OVERLAY else self._options()), image, "/bin/bash", "-lc", command]

    def _staged(self, path: str, inside: str, copy_in: bool, host_copy) -> None:
        """Overlay mode, a container path outside the bound dirs: host_copy(stage dir) fills or reads a staging dir
        under /logs/udocker_ipc, `inside` (the cp in the instance, $S = that dir) moves it to or from path."""
        stage = self._ipc_dir() / f"stage-{uuid.uuid4().hex}"
        stage.mkdir()
        try:
            if copy_in:
                host_copy(stage)
            done = self._run_argv(self._command_argv(f"S=/logs/udocker_ipc/{stage.name}; " + inside), 600)
            if done.returncode != 0:
                raise FileNotFoundError(f"singularity {path}: {(done.stderr or b'').decode()[-300:]}")
            if not copy_in:
                host_copy(stage)
        finally:
            shutil.rmtree(stage, ignore_errors=True)

    def _bound(self, path: str) -> bool:
        return not OVERLAY or path.lstrip("/").split("/", 1)[0] in ("root", "logs", "tests", "solution", "tasks")

    def _upload_file(self, source_path, target_path: str):
        if self._bound(target_path):
            return super()._upload_file(source_path, target_path)
        q = shlex.quote(target_path)
        self._staged(target_path, f'mkdir -p "$(dirname {q})" && cp -a "$S/f" {q}', True,
                     lambda st: shutil.copy2(source_path, st / "f"))

    def _upload_dir(self, source_dir, target_dir: str):
        if self._bound(target_dir):
            return super()._upload_dir(source_dir, target_dir)
        q = shlex.quote(target_dir)
        self._staged(target_dir, f'mkdir -p {q} && cp -a "$S/d/." {q}/', True,
                     lambda st: shutil.copytree(source_dir, st / "d"))

    def _download_file(self, source_path: str, target_path):
        if self._bound(source_path):
            return super()._download_file(source_path, target_path)
        Path(target_path).parent.mkdir(parents=True, exist_ok=True)
        self._staged(source_path, f'cp -a {shlex.quote(source_path)} "$S/f"', False,
                     lambda st: shutil.copy2(st / "f", target_path))

    def _download_dir(self, source_dir: str, target_dir):
        if self._bound(source_dir):
            return super()._download_dir(source_dir, target_dir)
        self._staged(source_dir, f'[ -d {shlex.quote(source_dir)} ] && cp -a {shlex.quote(source_dir)}/. "$S/"', False,
                     lambda st: shutil.copytree(st, target_dir, dirs_exist_ok=True))

    def _discard(self) -> None:
        with self._state_lock:
            root, self._cid = self._root, ""
        self._root = None
        self._ipc = None
        if root and OVERLAY:  # no instance (yet): stop fails, nothing to undo
            try:
                subprocess.run(["singularity", "instance", "stop", root.name], capture_output=True, timeout=120)
            except subprocess.TimeoutExpired:
                pass
        if root:
            shutil.rmtree(root, ignore_errors=True)
