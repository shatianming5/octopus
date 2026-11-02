import subprocess

import pytest

from octopus.rl.adapter_push import push_adapter


def test_push_replaces_old_version_and_failure_raises(tmp_path, monkeypatch):
    # fake ssh: runs the remote command locally with <tmp>/local mapped to <tmp>/remote; port 1 is an unreachable host
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "ssh").write_text('#!/bin/bash\n[ "$2" = 1 ] && exit 255\nexec bash -c "$(sed s#/local/#/remote/#g <<< "${@: -1}")"\n')
    (bin_dir / "ssh").chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}:/usr/bin:/bin")
    monkeypatch.setattr("octopus.rl.adapter_push.time.sleep", lambda s: None)
    src = tmp_path / "local" / "default" / "weight_update_v7"
    src.mkdir(parents=True)
    (src / "adapter_model.safetensors").write_bytes(b"w7")
    remote = tmp_path / "remote" / "default"
    (remote / "weight_update_v6").mkdir(parents=True)
    (remote / "epoch0epochstep4globalstep4").mkdir()

    push_adapter(str(src), ["u@h:22"])
    assert (remote / "weight_update_v7" / "adapter_model.safetensors").read_bytes() == b"w7"
    assert sorted(p.name for p in remote.iterdir()) == ["epoch0epochstep4globalstep4", "weight_update_v7"]

    with pytest.raises(subprocess.CalledProcessError):
        push_adapter(str(src), ["u@h:1"])


def test_push_tries_again_when_ssh_itself_fails(tmp_path, monkeypatch):
    """A remote sshd with its unauthenticated slots full drops the handshake (ssh exits 255): the push is tried again;
    a failure of the remote command itself is not."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    calls = tmp_path / "calls"
    (bin_dir / "ssh").write_text(f'#!/bin/bash\necho x >> {calls}\n[ "$(wc -l < {calls})" = 1 ] && exit 255\n'
                                 '[ "$2" = 2 ] && exit 3\nexec bash -c "$(sed s#/local/#/remote/#g <<< "${@: -1}")"\n')
    (bin_dir / "ssh").chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}:/usr/bin:/bin")
    monkeypatch.setattr("octopus.rl.adapter_push.time.sleep", lambda s: None)
    src = tmp_path / "local" / "default" / "weight_update_v8"
    src.mkdir(parents=True)
    (src / "adapter_model.safetensors").write_bytes(b"w8")
    push_adapter(str(src), ["u@h:22"])
    assert (tmp_path / "remote" / "default" / "weight_update_v8" / "adapter_model.safetensors").read_bytes() == b"w8"
    assert calls.read_text().count("x") == 2
    with pytest.raises(subprocess.CalledProcessError) as e:
        push_adapter(str(src), ["u@h:2"])
    assert e.value.returncode == 3 and calls.read_text().count("x") == 3  # not retried
