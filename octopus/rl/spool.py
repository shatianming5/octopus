"""Rollout spool (OCTOPUS_ROLLOUT_SPOOL=dir, off when unset): finished GRPO groups survive a trainer restart.

AReaL's recover checkpoint saves the dataloader position after each step, but the rollout controller draws prompts
ahead of training (rollout_controller.prepare_batch -> BatchTaskDispatcher.active_submit_and_wait), so the groups in
flight or finished-but-unconsumed at a checkpoint are past the position it saves: a resume skips their prompts and
waits a whole group duration for its first batch (bundle-e9: 27-37 min per resume, against ~6 min per step otherwise).
The prompts a process draws after its last checkpoint, by contrast, are drawn again by the next resume from it.

rollout worker   save: each finished group with reward variance -> ready/<head version>~<task>~<id>.pt (torch.save of
                 the trajectory AReaL trains on: tensors with versions, logprobs, rewards, rollout_group)
controller       recover (startup, at the resumed version): ready/, taken/ and trained/ of the previous process ->
                 resume/, one group per task (the newest head version at most the resumed one, then the newest file);
                 the task's other groups are deleted, each with a log line
rollout worker   take: a new group call returns its own task's resume/ group, else the oldest one, whose head version
                 is within max_head_offpolicyness of the current version (moved to taken/) instead of running its own
                 prompt; with none left, a task whose group a call already took is rejected (logged) instead of run
                 again
controller       consume (prepare_batch): the groups a train step takes -> trained/
controller       commit (after a recover checkpoint is written, octopus/rl/train.py): trained/ is deleted

A group is gone only once a recover checkpoint covers the step that trained on it. A resume restarts after the last
checkpoint, so the groups of the steps after it (the step a crash interrupted; with recover.freq_steps > 1 also the
finished steps since the checkpoint, whose updates the resume discards) go back to resume/, and the groups of
checkpointed steps do not. take reuses those whose head version is at most the resumed version; a group sampled by a
policy newer than that (one the resume discarded) is dropped. A kill between the checkpoint write and commit's deletes
retrains that step's groups once.

Reusing a group is training on it as if the crash had not happened: the decoupled PPO loss weighs each token by its
stored behavior logprob ("logprobs", from the rollout) against a proximal logprob of the current policy (grpo.yaml
prox_logp_method reuse_train_logp: the training forward's own logprob), so a group of head version h trained at
version v is the same off-policy sample whether v - h came from waiting in the queue or from a restart, and the
head-version filter keeps v - h within the run's bound. The prompt the substituted call drew is skipped for this
epoch, as the spooled group's own prompt would have been without the spool.

One group per task: since a resume redraws the prompts drawn after its checkpoint, a spooled group's prompt may come
back and, before this rule, ran again beside its spooled group; each restart from one checkpoint added another group of
the task (search-v9-rest 10-29: restarts at version 41 left up to 4 groups of one qid, 3 of search-5271 in one step).
recover keeps one group per task and deletes the others (the log names each: file, head version, mtime). A redrawn
prompt's call takes its own group; a call whose task's group another call already took (taken/ or trained/) returns no
trajectory, so AReaL rejects it and draws another prompt rather than train a second group of the task. A group trained
and deleted by commit no longer marks its task, so a later redraw in the same process runs it again, in another step.
A mark kept until the next recover would close that gap but would also reject the task's first draw of the next
epoch: a substituted group may be one whose prompt the resume skipped (drawn ahead of the checkpoint), and that prompt
comes back only in the next epoch, which a rollout worker cannot tell from a redraw.
"""

from __future__ import annotations

import glob
import os
import time
import uuid

import torch

DIR = os.environ.get("OCTOPUS_ROLLOUT_SPOOL", "")
KEY = "octopus_spool"  # trajectory key: its spool file, moved to trained/ by consume


def head_version(traj: dict) -> int:
    """Oldest policy version among the trained tokens (AReaL's head offpolicyness is measured from it)."""
    versions = traj["versions"][traj["loss_mask"].bool()]
    return int(versions.min()) if versions.numel() else -1


def save(traj: dict, name: str) -> str:
    os.makedirs(f"{DIR}/ready", exist_ok=True)
    path = f"{DIR}/ready/{head_version(traj):06d}~{name}~{uuid.uuid4().hex[:8]}.pt"
    torch.save({k: v for k, v in traj.items() if k != KEY}, path + ".tmp")
    os.replace(path + ".tmp", path)  # a reader never sees a partial file
    return path


def _head(path: str) -> int:
    return int(os.path.basename(path).split("~", 1)[0])


def _task(path: str) -> str:
    return os.path.basename(path).split("~", 1)[1].rsplit("~", 1)[0]


def recover(version: int) -> int:
    """Controller startup at the resumed policy version: the previous process's groups no recover checkpoint covers
    become this one's resume/ groups, one per task: the newest head version at most `version` (one take can use), then
    the newest file; a task with no such group keeps its newest, which take drops."""
    os.makedirs(f"{DIR}/resume", exist_ok=True)
    for path in glob.glob(f"{DIR}/ready/*.pt") + glob.glob(f"{DIR}/taken/*.pt") + glob.glob(f"{DIR}/trained/*.pt"):
        os.replace(path, f"{DIR}/resume/{os.path.basename(path)}")
    kept = {}
    for mtime, path in sorted(((os.path.getmtime(p), p) for p in glob.glob(f"{DIR}/resume/*.pt")),
                              key=lambda x: (_head(x[1]) <= version, _head(x[1]), x[0])):
        if old := kept.get(_task(path)):
            os.remove(old[1])
            print(f"[octopus] spool: deleted {os.path.basename(old[1])} (head {_head(old[1])}, mtime "
                  f"{time.strftime('%m-%d %H:%M:%S', time.localtime(old[0]))}), kept {os.path.basename(path)}",
                  flush=True)
        kept[_task(path)] = mtime, path
    return len(kept)


def take(version: int, max_offpolicy: int, task: str) -> dict | None:
    """This task's usable resume/ group, else the oldest, claimed for this call (rename is atomic across rollout
    workers); None: run the prompt; {}: none left and this task's group is already out, so reject the call."""
    os.makedirs(f"{DIR}/taken", exist_ok=True)
    for path in sorted(glob.glob(f"{DIR}/resume/*.pt"), key=lambda p: (_task(p) != task, p)):
        name = os.path.basename(path)
        if not 0 <= version - _head(path) <= max_offpolicy:
            try:
                os.remove(path)  # too stale (only gets staler), or newer than the policy: a discarded policy's sample
            except FileNotFoundError:
                pass
            continue
        try:
            os.replace(path, f"{DIR}/taken/{name}")
        except FileNotFoundError:  # another rollout worker claimed it
            continue
        traj = torch.load(f"{DIR}/taken/{name}", weights_only=False)
        traj[KEY] = f"{DIR}/taken/{name}"
        print(f"[octopus] spool: reused {name} at version {version}", flush=True)
        return traj
    # checked after resume/: a group renamed out of resume/ meanwhile is in taken/ (or trained/) by now
    if out := [p for p in glob.glob(f"{DIR}/taken/*.pt") + glob.glob(f"{DIR}/trained/*.pt") if _task(p) == task]:
        print(f"[octopus] spool: rejected a call of {task}: its group {os.path.basename(out[0])} is already out",
              flush=True)
        return {}
    return None


def consume(batch: list[dict]) -> None:
    """A train step took these groups: kept in trained/ until a recover checkpoint covers the step (commit)."""
    for d in batch:
        path = d.pop(KEY, None)
        if path:
            os.makedirs(f"{DIR}/trained", exist_ok=True)
            try:
                os.replace(path, f"{DIR}/trained/{os.path.basename(path)}")
            except FileNotFoundError:
                pass


def commit() -> None:
    """A recover checkpoint was written: every trained/ group was taken by a step at or before it."""
    for path in glob.glob(f"{DIR}/trained/*.pt"):
        os.remove(path)
