"""RL with AReaL: GRPO over kind=plan free-fork trees (docs/rl-plan-episodes.md), or search episodes
(rl.search_questions, octopus/envs/search/rl.py).

Rollout uses SGLang servers we launch ourselves (scripts/rl_env/rl_run.sh serve, --enable-lora with the zero
adapter as <lora_name>-v0) instead of AReaL-launched ones; each episode is a Harbor trial run by
octopus.envs.swe.rl.HarborEpisodeAgent. Groups whose rewards are all equal carry no signal
and are dropped.

usage: python -m octopus.rl.train --config octopus/rl/grpo.yaml scheduler.type=local \
         rl.servers=127.0.0.1:30200,... [key=value ...]
         rl.plan_tasks=/path/to/plan-tasks  # the kind=plan task dirs (file or directory)
"""

from __future__ import annotations

import importlib
import os
import sys
from copy import deepcopy
from pathlib import Path

from datasets import Dataset

from areal import PPOTrainer
from areal.api.cli_args import load_expr_config
from areal.api.io_struct import LocalInfServerInfo
from areal.engine.sglang_remote import RemoteSGLangEngine
from areal.utils.stats_logger import StatsLogger

from octopus.rl.actor import ChunkedHeadFSDPPPOActor
from octopus.rl.config import (RQ1RLConfig, check_mount_layers, plan_row, save_settings,
                                  set_derived_settings, workflow_kwargs)
from octopus.rl import spool
from octopus.rl.dp_balance import patch_controller


def _plan_task_dirs(spec: str) -> list[str]:
    if not spec:
        return []
    path = Path(spec)
    if path.is_dir():
        return [str(d) for d in sorted(path.iterdir()) if d.is_dir()]
    if path.is_file():
        return [line for line in path.read_text().split() if line]
    return []


def plan_dataset(config: RQ1RLConfig) -> Dataset:
    """One kind=plan row per task dir of rl.plan_tasks, less rl.exclude's."""
    cfg = config.rl
    skip = set(Path(cfg.exclude).read_text().split()) if cfg.exclude else set()
    return Dataset.from_list([plan_row(config, d) for d in _plan_task_dirs(cfg.plan_tasks) if d not in skip])


class ExternalSGLangTrainer(PPOTrainer):
    """PPOTrainer whose training rollout connects to already-running SGLang servers (rl.servers)."""

    def _init_rollout(self, rollout_config, is_eval=False, lora_path=None):
        if is_eval:
            return super()._init_rollout(rollout_config, is_eval=True, lora_path=lora_path)
        rollout_config = deepcopy(rollout_config)  # its scheduling_spec is interpolated from the actor's
        for spec in rollout_config.scheduling_spec:
            spec.gpu = 0  # workers only drive the servers
        infos = [LocalInfServerInfo(host=a.rsplit(":", 1)[0], port=int(a.rsplit(":", 1)[1]), process=None)
                 for a in self.config.rl.servers.split(",")]
        controller = RemoteSGLangEngine.as_controller(rollout_config, self.scheduler)
        controller.initialize(role="rollout", server_args={}, server_infos=infos)
        return controller

    def _create_train_engine(self, actor_config, alloc):
        """The FSDP actor with the chunked LM head (octopus/rl/actor.py); plan episodes dispatched to the
        data-parallel ranks one by one, balanced by tokens (octopus/rl/dp_balance.py)."""
        actor = ChunkedHeadFSDPPPOActor.as_controller(actor_config, self.scheduler)
        actor.create_process_group(parallel_strategy=alloc.parallel)
        patch_controller(actor)
        return actor

    def _save_recover_checkpoint(self, epoch, epoch_step, global_step):
        """Spooled groups are deleted once a recover checkpoint covers the steps that trained on them
        (octopus/rl/spool.py); the handler skips steps between recover.freq_steps and records only the ones it wrote."""
        super()._save_recover_checkpoint(epoch, epoch_step, global_step)
        if spool.DIR and self.recover_handler.last_step_info.global_step == global_step:
            spool.commit()


def keep_weight_updates() -> None:
    """OCTOPUS_KEEP_WEIGHT_UPDATES=1: AReaL's RolloutController keeps each weight_update_v<N> dir instead of removing it
    once the servers loaded it. SGLang keeps a loaded adapter's CPU tensors mapped on its files; on a server of another
    NFS client (rl_run.sh EXTRA_SERVERS) the trainer host's removal turns those pages into SIGBUS (Alpha, job 144687:
    every server died at its first swap after a load). Its only shutil use is that rmtree."""
    import types
    from areal.infra.controller import rollout_controller
    # ponytail: the dirs stay (one adapter per step, ~0.2 GB each); prune old versions if a run's disk use matters
    rollout_controller.shutil = types.SimpleNamespace(rmtree=lambda *a, **k: None)


def fork_ready_timeout(seconds: float = 300) -> None:
    """AReaL's LocalScheduler gives a forked worker (one proxy-rollout per rollout server, started by that rollout
    worker's guard) 60 s to answer /health, then kills it. The worker first imports areal (torch, transformers) from
    /projects NFS: 40-60 s on an Cluster Beta GB200 node, ~60 s for 7 at once, so every fork of Beta job 217588 timed
    out (10-26; an Alpha H200 node forks in ~30 s). The wait polls every 0.5 s: a longer cap costs a healthy start
    nothing, but a worker that dies at startup now shows only after the full cap (the wait sees /health, not the
    process)."""
    from functools import partial
    import inspect
    from areal.infra.scheduler.local import LocalScheduler
    assert "timeout" in inspect.signature(LocalScheduler._wait_for_fork_ready).parameters  # AReaL renamed it: fail now
    LocalScheduler._wait_for_fork_ready = staticmethod(partial(LocalScheduler._wait_for_fork_ready, timeout=seconds))


def main(argv):
    fork_ready_timeout()
    if os.environ.get("OCTOPUS_KEEP_WEIGHT_UPDATES") == "1":
        keep_weight_updates()
    config, _ = load_expr_config(argv, RQ1RLConfig)  # also saved: config.yaml in the run's log dir
    set_derived_settings(config)  # before any worker starts: rollout and train workers inherit it
    check_mount_layers(config.rl.servers.split(","))  # and a layer setting the servers do not share
    # the run's episode settings, once, beside config.yaml: dev_eval runs this run's episodes with them
    save_settings(StatsLogger.get_log_path(config.stats_logger))
    # no valid_dataset: dev evaluation runs octopus/eval/dev_eval.py on saved adapters
    if config.rl.search_questions:  # search questions: the workflow's module builds them
        # (workflow=octopus.envs.search.rl.SearchEpisodeAgent: octopus.envs.search.rl.search_dataset)
        dataset = importlib.import_module(config.workflow.rpartition(".")[0]).search_dataset(config)
    else:
        dataset = plan_dataset(config)
    with ExternalSGLangTrainer(config, train_dataset=dataset) as trainer:
        trainer.weight_update_meta.lora_keep_versions = config.rl.lora_keep_versions  # AReaL default: offpolicyness + 2
        if spool.DIR:  # before train() starts any rollout: the last process's groups no recover checkpoint covers
            info = trainer.recover_info  # AReaL resumes at version last global_step + 1 (PPOTrainer.__init__)
            version = info.last_step_info.global_step + 1 if info else 0
            print(f"[octopus] spool {spool.DIR}: {spool.recover(version)} groups to reuse", flush=True)
        trainer.train(workflow=config.workflow, workflow_kwargs=workflow_kwargs(config),
                      dynamic_filter_fn="octopus.rl.workflow.has_variance" if config.rl.filter_zero_variance else None)


if __name__ == "__main__":
    main(sys.argv[1:])
