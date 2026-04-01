from __future__ import annotations

import logging
import os
from typing import Callable

from openai import OpenAI

from agents.heuristic_agent import HeuristicAgent
from baseline.run_baseline import choose_action
from environments.email_triage_env import EmailTriageEnv
from openenv.config import BENCHMARK_METADATA
from openenv.models import Action, Observation
from openenv.tasks import get_benchmark_graders, get_benchmark_task_names, get_benchmark_tasks


SUCCESS_SCORE_THRESHOLD = 0.8


def _resolve_backend() -> str:
    explicit_backend = os.environ.get("OPENENV_BASELINE_BACKEND")
    if explicit_backend:
        return explicit_backend.strip().lower()
    if (
        os.environ.get("API_BASE_URL")
        and os.environ.get("MODEL_NAME")
        and os.environ.get("HF_TOKEN")
    ):
        return "openai"
    return "heuristic"


def _canonical_tasks_by_name():
    return {task.name: task for task in get_benchmark_tasks()}


def _select_task_name() -> str:
    requested_task = os.environ.get("OPENENV_TASK")
    if requested_task and requested_task in _canonical_tasks_by_name():
        return requested_task
    return get_benchmark_task_names()[0]


def _resolve_model_name(backend: str) -> str:
    if backend == "openai":
        return os.environ.get("MODEL_NAME") or BENCHMARK_METADATA.default_model
    return "heuristic-v1"


def _format_action(action: Action) -> str:
    if action.action_type == "wait":
        return "wait"
    if action.action_type == "ignore":
        return f"ignore('{action.email_id}')"
    if action.action_type == "classify":
        return f"classify('{action.email_id}','{action.category}')"
    if action.action_type == "respond":
        return (
            f"respond('{action.email_id}','{action.response_template}','{action.priority}')"
        )
    return f"escalate('{action.email_id}','{action.priority}')"


def _log_start(task: str, env_name: str, model: str) -> None:
    print(f"[START] task={task} env={env_name} model={model}", flush=True)


def _log_step(step: int, action: Action, reward: float, done: bool, error: str | None) -> None:
    print(
        f"[STEP] step={step} action={_format_action(action)} reward={reward:.2f} "
        f"done={str(done).lower()} error={error or 'null'}",
        flush=True,
    )


def _log_end(success: bool, steps: int, rewards: list[float]) -> None:
    rewards_blob = ",".join(f"{reward:.2f}" for reward in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} rewards={rewards_blob}",
        flush=True,
    )


def _build_action_selector(backend: str) -> Callable[[Observation], Action]:
    if backend == "openai":
        client = OpenAI(
            base_url=os.environ.get("API_BASE_URL"),
            api_key=os.environ.get("HF_TOKEN"),
        )
        model_name = _resolve_model_name(backend)
        return lambda observation: choose_action(client, observation, model_name)

    agent = HeuristicAgent()
    return agent.act


def main() -> None:
    backend = _resolve_backend()
    task_name = _select_task_name()
    task = _canonical_tasks_by_name()[task_name]
    grader = get_benchmark_graders()[task.name]
    action_selector = _build_action_selector(backend)
    model_name = _resolve_model_name(backend)
    env = EmailTriageEnv(task=task, seed=task.seed)
    env_logger = getattr(getattr(env, "logger", None), "_logger", None)
    if env_logger is not None:
        env_logger.setLevel(logging.CRITICAL + 1)

    rewards: list[float] = []
    steps_taken = 0
    success = False

    _log_start(task=task.name, env_name=BENCHMARK_METADATA.benchmark_name, model=model_name)

    try:
        observation = env.reset()
        done = False

        while not done:
            action = action_selector(observation)
            observation, reward, done, info = env.step(action)
            reward_value = float(reward.total)
            rewards.append(reward_value)
            steps_taken += 1
            error = info.get("last_action_error")
            _log_step(steps_taken, action, reward_value, done, error if isinstance(error, str) else None)

        score = grader(env.trajectory)
        success = score >= SUCCESS_SCORE_THRESHOLD
    finally:
        env.close()
        _log_end(success=success, steps=steps_taken, rewards=rewards)


if __name__ == "__main__":
    main()
