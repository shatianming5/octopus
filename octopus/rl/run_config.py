"""The run config's settings episodes share with the trainer: grpo.yaml, and each episode request's settings from
a run config (train.py's, or the one dev_eval runs a run's episodes under: octopus/rl/config.py)."""

import json
from pathlib import Path

CONFIG = Path(__file__).with_name("grpo.yaml")


def request_kwargs(config) -> dict:
    """HarborEpisodeAgent's per-request settings from a run config: the temperature the trainer recomputes logprobs at
    (actor.temperature = gconfig.temperature; a request without one would get the AReaL proxy's 1.0), the reply cap
    and the chat-template switches."""
    return {"temperature": config.gconfig.temperature, "max_new_tokens": config.gconfig.max_new_tokens,
            "chat_template_kwargs": dict(config.rollout.agent.chat_template_kwargs)}


def episode_env(model: str, temperature: float, max_new_tokens: int, chat_template_kwargs: dict,
                top_p: float | None = None) -> dict:
    """An episode's model-request env (the endpoint and key are the caller's): request_kwargs' settings, no retry of a
    failed request (it fails the episode). top_p: an evaluation's (search_dev_eval); training sends none (the proxy's
    1.0, which the trainer's logprobs assume)."""
    sampling = {"temperature": temperature, **({} if top_p is None else {"top_p": top_p})}
    return {"OCTOPUS_MODEL": model, "OCTOPUS_CHAT_TEMPLATE_KWARGS": json.dumps(chat_template_kwargs),
            "OCTOPUS_MAX_TOKENS": str(max_new_tokens), "OCTOPUS_SAMPLING_JSON": json.dumps(sampling),
            "OCTOPUS_HTTP_TIMEOUT": "900", "OCTOPUS_HTTP_ATTEMPTS": "1"}
