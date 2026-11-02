"""SingularityEnvironment with SINGULARITY_TRIAL_OVERLAY=1 on a host that has Apptainer and the SIFs (Alpha compute node,
SINGULARITY_SIF_DIR set, SINGULARITY_TEST_IMAGE one of its images): two concurrent commands of one trial (as fork
branches run) each make a directory tree under /tmp and append to their own half of files of /testbed, while a third
command renames a file from /testbed to /tmp; afterwards a fresh command sees every write of both, the rename, and a
file the host uploaded to /tmp. Two fuse-overlayfs mounts over one upper dir (an `exec --overlay` per command) fail it."""
import asyncio
import os
import shutil

import pytest

pytest.importorskip("harbor")
if not (shutil.which("singularity") and os.environ.get("SINGULARITY_TEST_IMAGE")):
    pytest.skip("needs singularity, SINGULARITY_SIF_DIR and SINGULARITY_TEST_IMAGE", allow_module_level=True)

from octopus.envs.swe import singularity as se  # noqa: E402
from octopus.envs.swe.data import validate_tasks  # noqa: E402

N = 200


def test_concurrent_commands_share_one_overlay(monkeypatch, tmp_path):
    monkeypatch.setattr(se, "OVERLAY", True)
    monkeypatch.setattr(validate_tasks, "HARBOR_ENV", "octopus.envs.swe.singularity:SingularityEnvironment")
    env = validate_tasks.make_env(os.environ["SINGULARITY_TEST_IMAGE"])

    async def run():
        await env.start(False)
        try:
            files = (await env.exec("git ls-files | grep -v ' ' | head -n %d" % (2 * N))).stdout.split()  # the image's
            assert len(files) >= 2
            half = len(files) // 2
            def writer(tag, part):
                return ("set -e; mkdir -p /tmp/%s/a/b; for i in $(seq %d); do echo %s-$i > /tmp/%s/a/b/f$i; done; "
                        "for f in %s; do printf '\\n%%s\\n' %s >> \"$f\"; done" % (tag, N, tag, tag, " ".join(part), tag))
            renamer = "set -e; echo moved > /testbed/zz_move; mkdir -p /tmp/r; python3 -c " \
                      "'import os; os.rename(\"/testbed/zz_move\", \"/tmp/r/zz_move\")'"
            done = await asyncio.gather(env.exec(writer("x", files[:half])), env.exec(writer("y", files[half:])),
                                        env.exec(renamer))
            assert [d.return_code for d in done] == [0, 0, 0], [d.stderr[-300:] for d in done]
            (tmp_path / "up").mkdir()
            (tmp_path / "up" / "f").write_text("host")
            await env.upload_dir(tmp_path / "up", "/tmp/up")
            check = ("for t in x y; do for i in $(seq %d); do [ \"$(cat /tmp/$t/a/b/f$i)\" = $t-$i ] || echo /tmp/$t/a/b/f$i; done; done; "
                     "for f in %s; do [ \"$(tail -n 1 \"$f\")\" = x ] || echo $f; done; "
                     "for f in %s; do [ \"$(tail -n 1 \"$f\")\" = y ] || echo $f; done; "
                     "[ \"$(cat /tmp/r/zz_move)\" = moved ] || echo zz_move; [ \"$(cat /tmp/up/f)\" = host ] || echo up"
                     % (N, " ".join(files[:half]), " ".join(files[half:])))
            assert (await env.exec(check)).stdout.split() == []
            await env.download_dir("/tmp/x", tmp_path / "down")
            await env.download_file("/tmp/r/zz_move", tmp_path / "moved")
            assert (tmp_path / "down" / "a" / "b" / "f1").read_text() == "x-1\n"
            assert (tmp_path / "moved").read_text() == "moved\n"
        finally:
            await env.stop(True)

    asyncio.run(run())
