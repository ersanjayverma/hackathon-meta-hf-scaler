from __future__ import annotations

import importlib
import inspect
from pathlib import Path
from typing import Any

import yaml

from .models import Action, Observation, Reward
from .tasks import get_email_tasks

REQUIRED_METADATA_KEYS = {
    "name",
    "version",
    "description",
    "observation_schema",
    "action_schema",
    "reward_definition",
    "task_definitions",
    "entry_point",
}


def _load_metadata(path: str | Path) -> dict[str, Any]:
    base = Path(path)
    metadata_path = base if base.name == "openenv.yaml" else base / "openenv.yaml"
    return yaml.safe_load(metadata_path.read_text(encoding="utf-8"))


def _load_environment(entry_point: str) -> Any:
    module_name, class_name = entry_point.split(":")
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def validate_environment(path: str | Path) -> list[str]:
    issues: list[str] = []
    metadata = _load_metadata(path)

    missing = sorted(REQUIRED_METADATA_KEYS.difference(metadata))
    if missing:
        issues.append(f"metadata missing keys: {', '.join(missing)}")
        return issues

    env_cls = _load_environment(metadata["entry_point"])
    reset_signature = inspect.signature(env_cls.reset)
    if len(reset_signature.parameters) != 1:
        issues.append("reset() must not require runtime parameters")

    if not hasattr(env_cls, "step") or not hasattr(env_cls, "state"):
        issues.append("environment is missing required API methods")
        return issues

    task = get_email_tasks()[0]
    env_one = env_cls(task=task, seed=task.seed)
    env_two = env_cls(task=task, seed=task.seed)

    obs_one = env_one.reset()
    obs_two = env_two.reset()
    if not isinstance(obs_one, Observation):
        issues.append("reset() did not return an Observation model")
    if not isinstance(obs_two, Observation):
        issues.append("reset() did not return an Observation model")
    if obs_one.model_dump(mode="json") != obs_two.model_dump(mode="json"):
        issues.append("reset() is not deterministic for the same seed")

    validation_action = env_one.default_validation_action(obs_one)
    step_one = env_one.step(validation_action)
    step_two = env_two.step(validation_action)
    if len(step_one) != 4 or len(step_two) != 4:
        issues.append("step() must return (Observation, Reward, done, info)")
    else:
        next_obs_one, reward_one, done_one, info_one = step_one
        next_obs_two, reward_two, done_two, info_two = step_two
        if not isinstance(next_obs_one, Observation) or not isinstance(next_obs_two, Observation):
            issues.append("step() did not return Observation instances")
        if not isinstance(reward_one, Reward) or not isinstance(reward_two, Reward):
            issues.append("step() did not return Reward instances")
        if next_obs_one.model_dump(mode="json") != next_obs_two.model_dump(mode="json"):
            issues.append("step() is not deterministic for the same seed and action")
        if reward_one.model_dump(mode="json") != reward_two.model_dump(mode="json"):
            issues.append("reward output is not deterministic")
        comparable_info_one = {key: value for key, value in info_one.items() if key != "step_latency"}
        comparable_info_two = {key: value for key, value in info_two.items() if key != "step_latency"}
        if done_one != done_two or comparable_info_one != comparable_info_two:
            issues.append("step() info or done flags are inconsistent")

    observation_schema = metadata["observation_schema"]
    action_schema = metadata["action_schema"]
    if observation_schema.get("title") != Observation.__name__:
        issues.append("openenv.yaml observation_schema does not describe Observation")
    if action_schema.get("title") != Action.__name__:
        issues.append("openenv.yaml action_schema does not describe Action")

    return issues
