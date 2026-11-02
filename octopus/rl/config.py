"""Config of octopus.rl.train, in its own module so AReaL workers can import it by path; and what an evaluation
takes from a run: its config and its episodes' environment (octopus/eval/dev_eval.py)."""

import json
import os
import re
import sys
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from areal.api.cli_args import PPOConfig, to_structured_cfg
from omegaconf import OmegaConf
from omegaconf.errors import ConfigKeyError

from octopus.rl.run_config import CONFIG, request_kwargs


@dataclass
class RLPoolConfig:
    servers: str = ""  # host:port,... of the SGLang servers, one per rollout worker
    jobs_dir: str = "/home/user/harbor-udocker/rl/jobs"
    exclude: str = ""  # file of task dirs to leave out of rl.plan_tasks
    # AReaL unloads adapter version N-keep when it loads N; SGLang's unload waits (holding its LoRA lock) for every
    # request still on that version, so a long episode of an old version stalled update_weights past 600 s. Keep 10.
    lora_keep_versions: int = 10
    # per rollout worker (one SGLang server each): SWE, concurrent Harbor episodes (swe/rl.py slots); search, concurrent
    # groups of n_samples episodes each (rollout.max_concurrent_rollouts = this x servers, rl_run.sh; no episode slots)
    episodes_per_server: int = 8
    filter_zero_variance: bool = True  # drop groups whose advantages are all 0 (off only for pilots)
    init_lora: str = ""  # saved adapter dir the LoRA starts from (a continued run; ADAPTER0 must be the same); "" = zero
    plan_tasks: str = ""  # file of task dirs, or a directory of them: the kind=plan rows
    plan_control: bool = False  # one no-fork continuation from the first fork: an evaluation's (never trained)
    # a branch thread's advantage: "outcome" = its episode's A_main; "gate" = the environment's rule: code,
    # max(A_main, 0) for a committed, merged, non-empty, passing branch, else min(A_main, 0); search, A_main for a
    # branch that closed with a non-empty done answer after a search/open of its subtree, else min(A_main, 0).
    # group.tree_advantage; not an episode setting
    branch_credit: str = "outcome"
    search_questions: str = ""  # JSONL of {qid, question, answer}: a search run (octopus/envs/search/rl.py) instead of the pool
    search_split: str = ""  # file of the qids of search_questions to train on ("" = all)


@dataclass
class RQ1RLConfig(PPOConfig):
    workflow: str = "octopus.envs.swe.rl.HarborEpisodeAgent"
    rl: RLPoolConfig = field(default_factory=RLPoolConfig)


def load(path: str, overrides=()) -> RQ1RLConfig:
    """A run config as train.py's load_expr_config builds it, from a yaml (grpo.yaml, or the config.yaml AReaL saved
    in a run's log dir: yaml.dump's, with python tuples) and key=value overrides; not saved anywhere. A key this code
    does not have fails: the run predates it (e.g. rl.plan_lambda / plan_alpha / plan_cost_unit, gone with PR #213)."""
    with open(path) as f:
        cfg = OmegaConf.merge(OmegaConf.create(yaml.load(f, yaml.FullLoader)), OmegaConf.from_dotlist(list(overrides)))
    try:
        return OmegaConf.to_object(to_structured_cfg(cfg, RQ1RLConfig))
    except ConfigKeyError as e:
        raise SystemExit(f"{path}: {e}\nThe run predates this code (rl.plan_lambda, plan_alpha and plan_cost_unit "
                         "left with PR #213): evaluate it with its own commit") from e


def workflow_kwargs(config: RQ1RLConfig) -> dict:
    """The arguments training builds config.workflow (the episode agent) with."""
    return {"jobs_dir": config.rl.jobs_dir, "parallel": config.rl.episodes_per_server, **request_kwargs(config)}


def row_settings(config: RQ1RLConfig) -> dict:
    """The settings training puts on every dataset row (plan_row, search.rl.search_row)."""
    return {"tokenizer_path": config.tokenizer_path, "plan_control": config.rl.plan_control,
            "branch_credit": config.rl.branch_credit}


def plan_row(config: RQ1RLConfig, task_dir: str) -> dict:
    """The kind=plan dataset row of one task dir (train.plan_dataset)."""
    return {"task_dir": task_dir, "kind": "plan", **row_settings(config)}


def set_derived_settings(config: RQ1RLConfig) -> None:
    """What the rollout workers and their episodes derive from the config (train.py; dev_eval as well)."""
    from octopus.runtime.core import read_card, tool_format
    from octopus.runtime.settings import read_tool

    tool_format(config.rollout.agent.tool_call_parser)  # an unknown format fails here, before any episode
    os.environ["OCTOPUS_TOOL_FORMAT"] = config.rollout.agent.tool_call_parser
    os.environ["SGLANG_FORWARD_UNKNOWN_TOOLS"] = "1"  # the proxy's tool-call parser keeps read_thread calls
    os.environ["OCTOPUS_TOKENIZER"] = config.tokenizer_path  # read_thread's card (core.read_card)
    if read_tool():  # the card renders here, before any episode: an unreadable path or a template that cannot render
        read_card()  # tools fails now


def check_mount_layers(servers: list[str]) -> None:
    """Every server must report this process's OCTOPUS_MOUNT_LAYERS (octopus/serving/kv_mount.py stats; a server without
    the mount patch reports none, as for all layers): training's prefix tree trains the token DAG the servers rolled
    out, and an evaluation runs the run's."""
    from octopus.serving.kv_mount import mount_layers

    want = None if (layers := mount_layers()) is None else sorted(layers)
    for addr in servers:
        with urllib.request.urlopen(f"http://{addr}/get_server_info", timeout=60) as r:
            got = [s.get("octopus_mount_layers") for s in json.load(r)["internal_states"]]
        if any(g != want for g in got):
            raise SystemExit(f"SGLang {addr} mounts in layers {got} (OCTOPUS_MOUNT_LAYERS), this run in {want}: "
                             "start both with the same value (None: every attention layer)")


EPISODE_SETTINGS = "episode_settings.json"  # train.py writes it into the run's log dir, beside AReaL's config.yaml
# Where things are on a host (paths, addresses, resources), and the search judge, whom an evaluation sets (the
# official one): not what an episode does, so neither in EPISODE_SETTINGS nor compared with it.
HOST_ENV = frozenset({
    "OCTOPUS_CODE", "OCTOPUS_HARBOR", "OCTOPUS_HARBOR_ENV", "OCTOPUS_LIVE_ENV", "OCTOPUS_ROLLOUT_SPOOL",
    "OCTOPUS_ADAPTER_PUSH", "OCTOPUS_SGLANG_PY", "OCTOPUS_ACT_GPU_GB", "OCTOPUS_ACT_GPU_TARGET_GB", "OCTOPUS_BLOCK_CHECKPOINT",
    "OCTOPUS_CHUNK_MAJOR", "OCTOPUS_CHUNK_TOKENS", "OCTOPUS_KAPPA",
    "OCTOPUS_COMPILE",
    "OCTOPUS_INIT_LORA", "OCTOPUS_EPISODE_MEM", "OCTOPUS_ROUTE_SERVERS", "OCTOPUS_ROUTE_CAP", "OCTOPUS_ROUTE_KV",
    "OCTOPUS_ROUTE_WAIT", "OCTOPUS_ROUTE_TREE_TOKENS", "OCTOPUS_CURSOR_BYTES",
    "OCTOPUS_ROUTE_GAP",  # gone since #351 (OCTOPUS_ROUTE_TREE_TOKENS); a host that still exports it is no arm
    "OCTOPUS_CURSOR_AUDIT",
    "OCTOPUS_SEARCH_URL", "OCTOPUS_SEARCH_PYTHON", "OCTOPUS_JUDGE_URL", "OCTOPUS_JUDGE_MODEL", "OCTOPUS_JUDGE_KEY",
    "OCTOPUS_JUDGE_CACHE", "OCTOPUS_JUDGE_TIMEOUT", "OCTOPUS_JUDGE_TEMPERATURE", "OCTOPUS_TOKENIZER", "OCTOPUS_HTTP_WORKERS",
    "OCTOPUS_KEEP_WEIGHT_UPDATES"})


def episode_key(k: str) -> bool:
    """An episode setting's name: OCTOPUS_*, not HOST_ENV, and no credential (*_KEY, *_TOKEN, *_SECRET, *_PASSWORD:
    never written to a snapshot, copied or printed)."""
    return k.startswith("OCTOPUS_") and k not in HOST_ENV and not re.search(
        r"_(KEY|TOKEN|SECRET|PASSWORD)$", k)


def episode_settings() -> dict[str, str]:
    """This process's episode settings (episode_key); refuses an environment that still exports a pre-rename GROK_*
    name (a host env file not yet renamed; GROK_MODELS_BASE_URL is the data tools' Grok gateway, kept)."""
    if old := sorted(k for k in os.environ if k.startswith("GROK_") and k != "GROK_MODELS_BASE_URL"):
        raise SystemExit(f"this environment exports pre-rename names {old}: see docs/rename-map-step3.md")
    return {k: v for k, v in os.environ.items() if episode_key(k)}


def run_episode_settings(run: dict[str, str], changes: list[str]) -> None:
    """Make this process's episode settings a run's (its EPISODE_SETTINGS) with KEY=VALUE changes (an evaluation's arm):
    a change may add a key the run lacks, and KEY= drops one. A setting of this process's environment that the result
    lacks or holds with another value fails: it would be a second, silent source. A run recorded before the rename
    (GROK_* keys) is refused: it is evaluated from its own commit."""
    if old := sorted(k for k in run if k.startswith("GROK_")):
        raise SystemExit(f"{EPISODE_SETTINGS} holds pre-rename keys {old}: see docs/rename-map-step3.md")
    changes = dict(c.split("=", 1) for c in changes)
    if bad := [k for k in changes if not episode_key(k)]:
        raise SystemExit(f"--env {bad}: not an episode setting (a host's path or address, or a credential)")
    want = {k: v for k, v in {**run, **changes}.items() if v != "" and episode_key(k)}  # not an old file's host path
    if stray := {k: v for k, v in episode_settings().items() if want.get(k) != v}:
        raise SystemExit(f"episode settings here that the run does not have: {stray} (unset them, or vary one on "
                         "purpose with --env KEY=VALUE)")
    os.environ.update(want)


def save_settings(log_dir) -> None:
    """EPISODE_SETTINGS in log_dir: this process's episode settings (train.py's run; `rl_run.sh settings`)."""
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    Path(log_dir, EPISODE_SETTINGS).write_text(json.dumps(episode_settings(), indent=1, sort_keys=True) + "\n")


def add_run_args(ap) -> None:
    """The dev_eval arguments that name the run whose episodes it runs (eval_config) and the adapter."""
    ap.add_argument("--run", help="the training run's log dir: AReaL's config.yaml and train.py's " + EPISODE_SETTINGS)
    ap.add_argument("--config", default=str(CONFIG), help="without --run: the run config (default the repo's); the "
                    "episodes get this environment's settings, as a training started here would")
    ap.add_argument("--tokenizer", help="this host's copy of the policy tokenizer dir (default the config's)")
    ap.add_argument("--env", action="append", default=[], metavar="KEY=VALUE",
                    help="an episode setting changed on purpose (an eval arm); repeatable")
    ap.add_argument("--lora", help="the adapter the servers serve, named as training names its versions: "
                    "<gconfig.lora_name>-v<version> (e.g. rq1-v0, the run's initial adapter); none with "
                    "rollout.use_lora=false (a base model, no adapter)")
    ap.add_argument("overrides", nargs="*", help="config overrides key=value, as train.py takes them")


def eval_config(args) -> RQ1RLConfig:
    """The config of the run args name (add_run_args), this process's episode settings made the run's."""
    run = json.loads(Path(args.run, EPISODE_SETTINGS).read_text()) if args.run else episode_settings()
    run_episode_settings(run, args.env)
    config = load(Path(args.run, "config.yaml") if args.run else args.config, args.overrides)
    if args.tokenizer:
        config.tokenizer_path = config.rollout.tokenizer_path = args.tokenizer
    set_derived_settings(config)
    return config


def live_env() -> dict[str, str]:
    """KEY=VALUE lines of $OCTOPUS_LIVE_ENV (rl_run.sh: $STATE/live.env), the last assignment of a key winning; {} while
    the file does not exist. Read at each new episode (EPISODES_PER_SERVER, octopus/envs/swe/rl.py) and each
    train-worker step (MB_TOKENS, octopus/rl/actor.py): tuning without a restart."""
    path = os.environ.get("OCTOPUS_LIVE_ENV", "")
    if not path or not os.path.exists(path):
        return {}
    lines = (line.split("#", 1)[0].strip() for line in open(path))
    return {k.strip(): v.strip() for k, v in (line.split("=", 1) for line in lines if "=" in line)}


if __name__ == "__main__":  # `rl_run.sh settings OUT_DIR [key=value ...]`: a run dir for dev_eval --run, no training
    from areal.api.cli_args import save_config

    config = load(CONFIG, sys.argv[2:])
    set_derived_settings(config)
    save_config(config, sys.argv[1])
    save_settings(sys.argv[1])
