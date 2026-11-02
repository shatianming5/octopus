"""Finished groups survive a trainer restart (octopus/rl/spool.py); live EPISODES_PER_SERVER, MB_TOKENS and ACT_GPU_GB."""

import asyncio
import os
from types import SimpleNamespace

import numpy as np
import torch

import areal.infra.controller.train_controller as train_controller
import areal.infra.workflow_executor as workflow_executor
from areal.api.cli_args import MicroBatchSpec
from areal.utils.data import concat_batch

import octopus.envs.swe.rl as swe_rl
import octopus.rl.train as train
import octopus.rl.workflow as workflow
from octopus.rl import actor, dp_balance, spool
from octopus.rl.actor import apply_live_mb_tokens

rng = np.random.default_rng(0)


def _interaction(n, rollout_index, reward, version):
    t = {"input_ids": torch.tensor(rng.integers(1000, 2000, n))[None], "attention_mask": torch.ones(1, n, dtype=torch.long),
         "loss_mask": torch.tensor([[0] * 5 + [1] * (n - 5)]), "logprobs": -torch.rand(1, n),
         "versions": torch.tensor([[-1] * 5 + [version] * (n - 5)]), "rewards": torch.tensor([reward]),
         "octopus_adv": torch.full((1, n), reward)}
    return SimpleNamespace(to_tensor_dict=lambda: t, rollout_index=rollout_index, rollout_reward=None, octopus_plan=1.0)


def _merged(rewards, version):
    """A plan group as GroupedRolloutWorkflow.arun_episode returns it: rollout i's interactions, each episode 2 rows."""
    return {f"{i}-{j}": _interaction(int(rng.integers(20, 60)), i, r, version + j)
            for i, r in enumerate(rewards) for j in range(2)}


def _run_group_call(monkeypatch, merged, version, ofp=2, task="t"):
    """One AReaL group call through our wrapper, then the executor's concat (only when the wrapper returned interactions)."""
    async def episode(self, engine, data):
        return merged

    monkeypatch.setattr(workflow, "_group_arun_episode", episode)
    engine = SimpleNamespace(get_version=lambda: version, config=SimpleNamespace(max_head_offpolicyness=ofp))
    out = asyncio.run(workflow._group_arun_with_rewards(None, engine, {"task_dir": f"/x/{task}", "kind": "plan-free"}))
    if all(isinstance(v, SimpleNamespace) for v in out.values()):
        out = workflow_executor.concat_tensor_interactions(out)
    rows = out["input_ids"].shape[0]
    out["rollout_group"].validate_rows(rows)  # what _execute_workflow checks on any returned trajectory
    return out


def _recover_save(monkeypatch, global_step, written):
    """ExternalSGLangTrainer's recover save after a step; written=False: a step recover.freq_steps skips."""
    def save(self, epoch, epoch_step, global_step):
        if written:  # RecoverHandler.dump records the step it wrote
            self.recover_handler.last_step_info = SimpleNamespace(global_step=global_step)

    monkeypatch.setattr(train.PPOTrainer, "_save_recover_checkpoint", save)
    trainer = object.__new__(train.ExternalSGLangTrainer)
    trainer.recover_handler = SimpleNamespace(last_step_info=SimpleNamespace(global_step=-1))
    trainer._save_recover_checkpoint(epoch=0, epoch_step=global_step, global_step=global_step)


def test_finished_groups_are_reused_once_after_a_restart(monkeypatch, tmp_path):
    monkeypatch.setattr(spool, "DIR", str(tmp_path / "spool"))
    monkeypatch.setattr(train_controller, "_item_weight", train_controller._item_weight)
    monkeypatch.setattr(train_controller, "balanced_greedy_partition", train_controller.balanced_greedy_partition)
    # first trainer process (version 10): four groups finish, step 0 takes one and checkpoints, step 1 takes another
    # and dies before its checkpoint (search-v9-rest 10-27: OOM in the train step, its groups were lost)
    trained = _run_group_call(monkeypatch, _merged([1.0, 0.0, 0.0, 1.0], 9), 10, task="a")
    flat = _run_group_call(monkeypatch, _merged([0.0, 0.0, 0.0, 0.0], 9), 10, task="flat")  # no GRPO signal
    old = _run_group_call(monkeypatch, _merged([1.0, -1.0, 0.0, 0.0], 7), 10, task="old")  # head 7: stale at 10
    kept = _run_group_call(monkeypatch, _merged([0.5, 0.0, 0.0, 1.0], 8), 10, task="kept")
    assert spool.KEY not in flat and all(spool.KEY in t for t in (trained, old, kept))
    snapshot = {k: v.clone() for k, v in kept.items() if isinstance(v, torch.Tensor)}
    actor = SimpleNamespace(prepare_batch=lambda *a, **k: [trained],
                            compute_advantages=lambda batch, **k: [{"advantages": None} for _ in batch])
    dp_balance.patch_controller(actor)
    items = actor.prepare_batch()
    assert len(items) == 4 and not any(spool.KEY in d for d in items)  # the key never reaches a train worker
    _recover_save(monkeypatch, 0, written=True)  # step 0 is checkpointed: a restart must not train on "trained" again
    actor.prepare_batch = lambda *a, **k: [dict(kept)]
    dp_balance.patch_controller(actor)
    actor.prepare_batch()  # step 1 takes "kept"; the process dies before its checkpoint
    assert len(list((tmp_path / "spool/ready").glob("*.pt"))) == 1

    # second process, resumed after step 0: "old" (head 7) is past max_head_offpolicyness 2, "kept" (head 8) is not
    assert spool.recover(10) == 2
    reused = _run_group_call(monkeypatch, _merged([1.0, 1.0, 0.0, 0.0], 10), 10, task="new")
    assert all(torch.equal(reused[k], v) for k, v in snapshot.items())
    assert reused[dp_balance.EPISODE_TOKENS] == kept[dp_balance.EPISODE_TOKENS]
    assert reused["rollout_group"] == kept["rollout_group"] and workflow.has_variance(reused)
    assert spool.head_version(reused) == 8
    fresh = _run_group_call(monkeypatch, _merged([1.0, 1.0, 0.0, 0.0], 10), 10, task="new")  # nothing left: runs
    assert spool.head_version(fresh) == 10
    assert not list((tmp_path / "spool/resume").glob("*.pt"))

    # the reused group trains like any other: through the controller and onto a rank
    actor.prepare_batch = lambda *a, **k: [reused]
    dp_balance.patch_controller(actor)
    items = actor.prepare_batch()
    splits, _ = train_controller._dispatch_tensors(items, dp_size=2)
    for split in splits:
        concat_batch(dp_balance.take_rows([dict(d) for d in split]))
    _recover_save(monkeypatch, 1, written=False)  # recover.freq_steps skips step 1: no checkpoint covers it yet
    assert len(list((tmp_path / "spool/trained").glob("*.pt"))) == 1
    _recover_save(monkeypatch, 2, written=True)
    assert not list((tmp_path / "spool/taken").glob("*.pt")) + list((tmp_path / "spool/trained").glob("*.pt"))
    assert spool.recover(11) == 1  # a third process: only "fresh", the group the second one never trained on
    assert spool.take(11, 2, "new")["rollout_group"] == fresh["rollout_group"]


def test_a_spooled_group_is_claimed_by_one_worker_and_staleness_is_two_sided(monkeypatch, tmp_path):
    monkeypatch.setattr(spool, "DIR", str(tmp_path))
    traj = workflow_executor.concat_tensor_interactions(_merged([1.0, 0.0], 5))
    spool.save(traj, "a")
    spool.recover(4)
    assert spool.take(4, 2, "a") is None  # a group newer than the policy: another trial's leftover, dropped
    spool.save(traj, "b")
    spool.recover(7)
    assert spool.take(7, 2, "c") is not None and spool.take(7, 2, "c") is None


def test_one_group_per_task(monkeypatch, tmp_path, capsys):
    """Restarts from one checkpoint redraw the same prompts (search-v9-rest 10-29: 3 groups of search-5271 in a step)."""
    monkeypatch.setattr(spool, "DIR", str(tmp_path))
    group = lambda head: workflow_executor.concat_tensor_interactions(_merged([1.0, 0.0], head))
    paths = [spool.save(group(6), "x"), spool.save(group(5), "x"), spool.save(group(5), "y"),
             spool.save(group(5), "y"), spool.save(group(5), "z"), spool.save(group(8), "x"),
             spool.save(group(9), "v")]
    for i, path in enumerate(paths):
        os.utime(path, (i, i))
    # x: the newest head version the resumed policy 7 can use (8 is newer than it); y: the newest file; v: none usable
    assert spool.recover(7) == 4
    assert sorted(p.name for p in (tmp_path / "resume").iterdir()) == sorted(
        os.path.basename(p) for p in (paths[0], paths[3], paths[4], paths[6]))
    assert capsys.readouterr().out.count("[octopus] spool: deleted") == 3  # the others are gone, each logged
    assert spool.take(7, 2, "y")[spool.KEY].endswith(os.path.basename(paths[3]))  # its own task's group first
    assert spool.take(7, 2, "w")[spool.KEY].endswith(os.path.basename(paths[4]))  # else the oldest: z's
    spool.consume([{spool.KEY: spool.take(7, 2, "z")[spool.KEY]}])  # z's call gets x's group; a step trains on it
    engine = SimpleNamespace(get_version=lambda: 7, config=SimpleNamespace(max_head_offpolicyness=2))
    for task in "xyz":  # each one's group is out: its call is rejected rather than run
        assert spool.take(7, 2, task) == {}
        assert asyncio.run(workflow._group_arun_with_rewards(None, engine, {"task_dir": f"/x/{task}"})) is None
    assert "rejected a call of z" in capsys.readouterr().out
    assert spool.take(7, 2, "w") is None  # w's own prompt runs (v's group is newer than the policy: dropped)


def test_live_episodes_per_server_resizes_within_the_launch_value(monkeypatch, tmp_path):
    live, task = tmp_path / "live.env", tmp_path / "task"  # the task outside jobs_dir (a copy into itself recurses)
    task.mkdir()
    monkeypatch.setenv("OCTOPUS_LIVE_ENV", str(live))
    monkeypatch.setattr(swe_rl.HarborEpisodeAgent, "_slots", None)
    monkeypatch.setattr(swe_rl, "trial_outcome", lambda *a: {"y": 0.5})
    gates, active = [], [0]

    async def harbor_run(command, **kw):
        active[0] += 1
        gates.append(gate := asyncio.Event())
        await gate.wait()
        active[0] -= 1
        return 0

    monkeypatch.setattr(swe_rl, "_run_group", harbor_run)

    async def scenario():
        agent = swe_rl.HarborEpisodeAgent(jobs_dir=str(tmp_path / "jobs"), parallel=4, temperature=1.0,
                                                max_new_tokens=8192, chat_template_kwargs={})
        row = {"task_dir": str(task), "kind": "join", "plan_control": False}
        start = lambda: asyncio.create_task(agent.run_episode(row, "http://h/v1", "k"))
        live.write_text("EPISODES_PER_SERVER=2\n")
        episodes = [start() for _ in range(5)]
        await asyncio.sleep(0.1)
        assert active[0] == 2
        live.write_text("EPISODES_PER_SERVER=2\nEPISODES_PER_SERVER = 9  # the last one wins, capped at launch's 4\n")
        episodes.append(start())  # the next episode start reads it
        await asyncio.sleep(0.1)
        assert active[0] == 4
        live.write_text("EPISODES_PER_SERVER=1\n")
        episodes.append(start())
        for gate in gates[:3]:
            gate.set()
        await asyncio.sleep(0.1)
        assert active[0] == 1  # 3 of 4 finished; the slot cap is now 1, so no waiting episode starts
        live.unlink()  # no file: the current size stays
        while not all(e.done() for e in episodes):
            for gate in gates:
                gate.set()
            await asyncio.sleep(0.05)
        assert len(gates) == 7

    asyncio.run(scenario())


def test_live_mb_tokens(monkeypatch, tmp_path):
    spec = MicroBatchSpec(max_tokens_per_mb=262144)
    monkeypatch.setenv("OCTOPUS_LIVE_ENV", str(tmp_path / "live.env"))
    monkeypatch.setattr(actor, "ACT_GPU_GB", 0.0)
    apply_live_mb_tokens(spec)
    assert spec.max_tokens_per_mb == 262144
    (tmp_path / "live.env").write_text("# tuning\nMB_TOKENS=196608\nACT_GPU_GB=12\n")
    apply_live_mb_tokens(spec)
    assert spec.max_tokens_per_mb == 196608
    assert actor.ACT_GPU_GB == 12.0


def test_the_filter_is_the_trained_signal():
    """has_variance keeps a group by its per-token advantages, read credit included, not by its rewards."""
    group = {"rewards": torch.tensor([1.0, 1.0]), "octopus_adv": torch.zeros(2, 4)}
    assert not workflow.has_variance(group)
    group["octopus_adv"][1, 2] = 0.5  # one read call's credit in a group whose rewards are all equal
    assert workflow.has_variance(group)
