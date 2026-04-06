from __future__ import annotations

import logging
from typing import Any, Callable

from openai import OpenAI

from baseline.run_baseline import choose_action_with_diagnostics
from environments.email_triage_env import EmailTriageEnv
from openenv.config import BENCHMARK_METADATA
from openenv.models import Action, Observation
from openenv.runtime_config import (
    API_BASE_URL,
    BENCHMARK,
    MAX_STEPS,
    MAX_TOKENS,
    MODEL_NAME,
    OPENAI_BASE_URL,
    SUCCESS_SCORE_THRESHOLD,
    TASK_NAME,
    TEMPERATURE,
    runtime_api_base_url,
    runtime_api_key,
    runtime_benchmark_name,
    runtime_hf_token,
    runtime_max_steps,
    runtime_max_tokens,
    runtime_model_name,
    runtime_openai_api_key,
    runtime_success_score_threshold,
    runtime_temperature,
    runtime_task_name,
)
from openenv.tasks import Task, get_benchmark_graders, get_benchmark_task_names, get_benchmark_tasks

Classifier = Callable[[Observation], tuple[Action, str | None]]


def _is_hf_model(model: str) -> bool:
    """HF models have org/name format (contain '/'), OpenAI models don't."""
    return "/" in model


def _validate_runtime_config() -> None:
    if not runtime_api_base_url(API_BASE_URL):
        raise ValueError("API_BASE_URL must be configured")
    if not runtime_model_name(MODEL_NAME):
        raise ValueError("MODEL_NAME must be configured")
    if runtime_max_steps(MAX_STEPS) <= 0:
        raise ValueError("MAX_STEPS must be positive")
    if runtime_temperature(TEMPERATURE) < 0.0:
        raise ValueError("TEMPERATURE must be non-negative")
    if runtime_max_tokens(MAX_TOKENS) <= 0:
        raise ValueError("MAX_TOKENS must be positive")
    if not 0.0 <= runtime_success_score_threshold(SUCCESS_SCORE_THRESHOLD) <= 1.0:
        raise ValueError("SUCCESS_SCORE_THRESHOLD must be between 0.0 and 1.0")
    if not (runtime_hf_token() or runtime_openai_api_key()):
        raise ValueError("HF_TOKEN or OPENAI_API_KEY must be set for LLM inference")


def _canonical_tasks_by_name() -> dict[str, Task]:
    return {task.name: task for task in get_benchmark_tasks()}


def _select_tasks() -> list[Task]:
    requested_task = runtime_task_name(TASK_NAME)
    tasks_by_name = _canonical_tasks_by_name()
    if requested_task and requested_task in tasks_by_name:
        return [tasks_by_name[requested_task]]
    return [tasks_by_name[name] for name in get_benchmark_task_names()]


def _resolve_model_name() -> str:
    return runtime_model_name(MODEL_NAME) or BENCHMARK_METADATA.default_model


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


def _log_end(success: bool, steps: int, score: float, rewards: list[float]) -> None:
    rewards_blob = ",".join(f"{reward:.2f}" for reward in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_blob}",
        flush=True,
    )


def _resolve_benchmark_name() -> str:
    configured_benchmark = runtime_benchmark_name(BENCHMARK)
    if configured_benchmark == BENCHMARK:
        return BENCHMARK_METADATA.benchmark_name
    return configured_benchmark


def _completion_ratio(processed_email_ids: set[str], all_email_ids: set[str]) -> float:
    return len(processed_email_ids & all_email_ids) / max(len(all_email_ids), 1)


def _score_episode(
    processed_email_ids: set[str],
    all_email_ids: set[str],
) -> float:
    return _completion_ratio(processed_email_ids, all_email_ids)


def _build_openai_classifier(model_name: str) -> Classifier:
    # Route to the right API based on model type
    if _is_hf_model(model_name):
        # HF model -> HF router + HF_TOKEN
        base_url = runtime_api_base_url(API_BASE_URL)
        api_key = runtime_hf_token() or runtime_api_key()
    else:
        # OpenAI model (gpt-*) -> OpenAI API + OPENAI_API_KEY
        base_url = runtime_api_base_url(OPENAI_BASE_URL)
        api_key = runtime_openai_api_key() or runtime_api_key()
    client = OpenAI(
        base_url=base_url,
        api_key=api_key,
    )

    def classify(observation: Observation) -> tuple[Action, str | None]:
        return choose_action_with_diagnostics(client, observation, model_name)

    return classify


def _filter_observation_for_llm(observation: Observation, env_state: dict[str, Any] | None) -> Observation:
    """Return observation with inbox filtered to only unclassified emails."""
    if env_state is None:
        return observation
    classified_ids = set(env_state.get("classifications", {}).keys())
    unclassified_inbox = [e for e in observation.inbox if e.email_id not in classified_ids]
    if not unclassified_inbox:
        return observation  # all classified — caller will skip LLM anyway
    return observation.model_copy(update={"inbox": unclassified_inbox})


def _is_redundant_action(action: Action, env_state: dict[str, Any] | None) -> bool:
    """Check if the action repeats something already done on this email."""
    if env_state is None or action.email_id is None:
        return False
    eid = action.email_id
    if action.action_type == "classify" and eid in env_state.get("classifications", {}):
        return True
    if action.action_type == "respond" and eid in env_state.get("responses", {}):
        return True
    if action.action_type == "escalate" and eid in env_state.get("escalations", {}):
        return True
    if action.action_type == "ignore" and eid in set(env_state.get("ignored", [])):
        return True
    return False


def _handled_email_ids(env_state: dict[str, Any] | None) -> set[str]:
    """Return IDs of emails already classified/responded/escalated/ignored — no need to send to LLM again."""
    if env_state is None:
        return set()
    handled: set[str] = set()
    handled.update(env_state.get("classifications", {}).keys())
    handled.update(env_state.get("responses", {}).keys())
    handled.update(env_state.get("escalations", {}).keys())
    handled.update(env_state.get("ignored", []))
    return handled


def _next_action(
    observation: Observation,
    llm_classifier: Classifier,
    env_state: dict[str, Any] | None = None,
) -> tuple[Action, str | None]:
    wait_action = Action(action_type="wait")
    visible_email_ids = {email.email_id for email in observation.inbox}

    if not visible_email_ids:
        return wait_action, None

    # Skip LLM entirely if all visible emails are already classified
    classified_ids = set((env_state or {}).get("classifications", {}).keys())
    unclassified = visible_email_ids - classified_ids
    if not unclassified:
        return wait_action, None

    # Only show unclassified emails to the LLM so it can't re-pick classified ones
    filtered_obs = _filter_observation_for_llm(observation, env_state)
    suggested_action, model_error = llm_classifier(filtered_obs)

    # If LLM suggests a redundant action, wait instead of repeating
    if _is_redundant_action(suggested_action, env_state):
        return wait_action, model_error

    if suggested_action.action_type == "wait":
        return wait_action, model_error
    if suggested_action.email_id is None or suggested_action.email_id not in visible_email_ids:
        return wait_action, model_error
    return suggested_action, model_error


def _task_email_ids(task: Task) -> set[str]:
    return {str(email["email_id"]) for email in task.initial_state.get("emails", [])}


def _read_progress(env: Any, task: Task, observation: Observation) -> tuple[set[str], set[str]]:
    task_email_ids = _task_email_ids(task)
    state_fn = getattr(env, "state", None)
    if callable(state_fn):
        state = state_fn()
        processed_email_ids = set(state.get("processed_email_ids", []))
        state_all_email_ids = processed_email_ids | set(state.get("remaining_email_ids", []))
        return processed_email_ids, state_all_email_ids or task_email_ids

    visible_email_ids = {email.email_id for email in observation.inbox}
    return task_email_ids - visible_email_ids, task_email_ids


def _run_task(
    task: Task,
    model_name: str,
    llm_classifier: Classifier,
) -> None:
    env = EmailTriageEnv(task=task, seed=task.seed)
    env_logger = getattr(getattr(env, "logger", None), "_logger", None)
    if env_logger is not None:
        env_logger.setLevel(logging.CRITICAL + 1)

    # Only track initial task emails for scoring — ignore spawned follow-ups
    initial_email_ids = _task_email_ids(task)
    processed_email_ids: set[str] = set()
    rewards: list[float] = []
    steps_taken = 0
    score = 0.0
    success = False
    step_budget = max(int(getattr(task, "max_steps", 0) or 0), runtime_max_steps(MAX_STEPS))

    _log_start(task=task.name, env_name=_resolve_benchmark_name(), model=model_name)

    try:
        observation = env.reset()
        processed_email_ids, _ = _read_progress(env, task, observation)

        if not initial_email_ids:
            success = True
            return

        done = False
        while steps_taken < step_budget and not done:
            # Check if all initial emails are completed — no need to continue
            if initial_email_ids.issubset(processed_email_ids):
                break

            # Use env state to ensure every email gets properly classified
            env_state = env.state() if callable(getattr(env, "state", None)) else None
            action, _ = _next_action(
                observation=observation,
                llm_classifier=llm_classifier,
                env_state=env_state,
            )

            observation, reward, done, info = env.step(action)
            reward_value = float(reward.total)
            rewards.append(reward_value)
            steps_taken += 1
            env_error = info.get("last_action_error")
            _log_step(
                steps_taken,
                action,
                reward_value,
                done,
                env_error if isinstance(env_error, str) else None,
            )
            processed_email_ids, _ = _read_progress(env, task, observation)
            if done:
                break

        # Use the official grader for the canonical benchmark score
        graders = get_benchmark_graders()
        if task.name in graders:
            score = graders[task.name](env.trajectory)
        else:
            score = _score_episode(processed_email_ids, initial_email_ids)
        success = score >= runtime_success_score_threshold(SUCCESS_SCORE_THRESHOLD)
    finally:
        env.close()
        _log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


def main() -> None:
    _validate_runtime_config()
    model_name = _resolve_model_name()
    llm_classifier = _build_openai_classifier(model_name)

    for task in _select_tasks():
        _run_task(
            task=task,
            model_name=model_name,
            llm_classifier=llm_classifier,
        )


if __name__ == "__main__":
    main()
