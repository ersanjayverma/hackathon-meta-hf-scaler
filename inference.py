from __future__ import annotations

import logging
import os

from openai import OpenAI

from agents.heuristic_agent import HeuristicAgent
from baseline.run_baseline import choose_action
from environments.email_triage_env import EmailTriageEnv
from openenv.config import BENCHMARK_METADATA
from openenv.models import Action, Observation
from openenv.tasks import get_benchmark_task_names, get_benchmark_tasks


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


def _select_tasks():
    requested_task = os.environ.get("OPENENV_TASK")
    tasks_by_name = _canonical_tasks_by_name()
    if requested_task and requested_task in tasks_by_name:
        return [tasks_by_name[requested_task]]
    return [tasks_by_name[name] for name in get_benchmark_task_names()]


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
        return f"respond('{action.email_id}','{action.response_template}','{action.priority}')"
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


def _priority_rank(priority_hint: str) -> int:
    return {"critical": 4, "high": 3, "medium": 2, "low": 1}.get(priority_hint, 0)


def _infer_category(agent: HeuristicAgent, email) -> str:
    return agent._profile(email).category


def _visible_unprocessed_emails(observation: Observation, processed_email_ids: set[str]) -> list:
    visible = {}
    for email in observation.inbox:
        if email.email_id not in processed_email_ids:
            visible[email.email_id] = email
    return sorted(
        visible.values(),
        key=lambda email: (-_priority_rank(email.priority_hint), -email.age, email.email_id),
    )


def _build_openai_classifier(model_name: str):
    client = OpenAI(
        base_url=os.environ.get("API_BASE_URL"),
        api_key=os.environ.get("HF_TOKEN"),
    )
    return lambda observation: choose_action(client, observation, model_name)


def _next_classification_action(
    observation: Observation,
    processed_email_ids: set[str],
    backend: str,
    heuristic_agent: HeuristicAgent,
    llm_classifier,
) -> Action | None:
    candidates = _visible_unprocessed_emails(observation, processed_email_ids)
    if not candidates:
        return None

    next_email = candidates[0]
    fallback_action = Action(
        action_type="classify",
        email_id=next_email.email_id,
        category=_infer_category(heuristic_agent, next_email),
    )

    if backend != "openai":
        return fallback_action

    suggested_action = llm_classifier(observation)
    if (
        suggested_action.action_type == "classify"
        and suggested_action.email_id is not None
        and suggested_action.email_id not in processed_email_ids
        and any(email.email_id == suggested_action.email_id for email in candidates)
    ):
        return suggested_action
    return fallback_action


def _task_email_ids(task) -> set[str]:
    return {str(email["email_id"]) for email in task.initial_state.get("emails", [])}


def _run_task(task, backend: str, model_name: str, heuristic_agent: HeuristicAgent, llm_classifier) -> None:
    env = EmailTriageEnv(task=task, seed=task.seed)
    env_logger = getattr(getattr(env, "logger", None), "_logger", None)
    if env_logger is not None:
        env_logger.setLevel(logging.CRITICAL + 1)

    all_email_ids = _task_email_ids(task)
    processed_email_ids: set[str] = set()
    rewards: list[float] = []
    steps_taken = 0
    success = False

    _log_start(task=task.name, env_name=BENCHMARK_METADATA.benchmark_name, model=model_name)

    try:
        observation = env.reset()

        if not all_email_ids:
            success = True
            return

        while processed_email_ids != all_email_ids:
            action = _next_classification_action(
                observation=observation,
                processed_email_ids=processed_email_ids,
                backend=backend,
                heuristic_agent=heuristic_agent,
                llm_classifier=llm_classifier,
            )

            if action is None:
                unseen_remaining = all_email_ids - processed_email_ids
                visible_ids = {email.email_id for email in observation.inbox}
                if unseen_remaining and not (unseen_remaining & visible_ids):
                    action = Action(action_type="wait")
                else:
                    break

            if action.email_id is not None and action.email_id in processed_email_ids:
                break

            observation, reward, done, info = env.step(action)
            reward_value = float(reward.total)
            rewards.append(reward_value)
            steps_taken += 1
            if action.action_type == "classify" and action.email_id is not None:
                processed_email_ids.add(action.email_id)
            error = info.get("last_action_error")
            _log_step(steps_taken, action, reward_value, done, error if isinstance(error, str) else None)

            if done:
                break

        success = processed_email_ids == all_email_ids
    finally:
        env.close()
        _log_end(success=success, steps=steps_taken, rewards=rewards)


def main() -> None:
    backend = _resolve_backend()
    model_name = _resolve_model_name(backend)
    heuristic_agent = HeuristicAgent()
    llm_classifier = _build_openai_classifier(model_name) if backend == "openai" else None

    for task in _select_tasks():
        _run_task(
            task=task,
            backend=backend,
            model_name=model_name,
            heuristic_agent=heuristic_agent,
            llm_classifier=llm_classifier,
        )


if __name__ == "__main__":
    main()
