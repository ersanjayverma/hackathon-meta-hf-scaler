from __future__ import annotations

import json
import logging
import sys
from typing import Any, Callable

from pydantic import ValidationError

from environments.email_triage_env import EmailTriageEnv
from openenv.config import BENCHMARK_METADATA, EMAIL_TRIAGE_CONFIG
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
    runtime_baseline_backend,
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

logger = logging.getLogger(__name__)
if not logger.handlers:
    _stderr_handler = logging.StreamHandler(sys.stderr)
    _stderr_handler.setLevel(logging.WARNING)
    logger.addHandler(_stderr_handler)
    logger.propagate = False

Classifier = Callable[[Observation], tuple[Action, str | None]]


def _is_hf_model(model: str) -> bool:
    """HF models have org/name format (contain '/'), OpenAI models don't."""
    return "/" in model


# ── LLM action parsing & normalization ──────────────────────────────────────

VALID_ACTION_TYPES = {"classify", "respond", "escalate", "ignore", "wait"}
VALID_PRIORITIES = {"low", "medium", "high", "critical"}
SAFE_DEFAULT_PRIORITY = "medium"

SYSTEM_PROMPT = """
You are an email triage agent operating in a time-sensitive environment.

Your objective is to maximize total reward by:
1. Preventing SLA breaches
2. Reducing system stress
3. Correctly classifying emails
4. Taking timely and effective actions

STRICT RULES (DO NOT VIOLATE):
1. NEVER choose "wait" if there is any actionable email.
2. ALWAYS prioritize emails closest to SLA breach.
3. DO NOT repeat actions on the same email unless new information is received.
4. EVERY email must be either classified, responded to, escalated, or ignored (only if clearly spam).
5. If an email is urgent -> respond or escalate immediately.
6. If unsure -> classify first, then act.
7. DO NOT stall. Inaction is heavily penalized.

PRIORITIZATION STRATEGY:
Order emails by:
1. Imminent SLA deadline
2. Urgency (urgent > escalation > normal > spam)
3. Whether already partially handled
Always pick the highest priority email.

ACTION GUIDELINES:
- classify: Use when category is unknown.
- respond: Use when email requires acknowledgment or info.
- escalate: Use when issue is critical or cannot be handled directly.
- ignore: ONLY for clear spam.
- wait: ONLY if backlog == 0

ANTI-FAILURE GUARDS:
- If stress > 0 -> prioritize clearing backlog immediately
- If SLA breach risk exists -> override all other logic
- If same action repeated twice -> choose a different action

Allowed enum values:
- action_type: classify, respond, escalate, ignore, wait
- category: spam, urgent, normal, escalation
- response_template: acknowledge, resolve, request_info, escalate_notice, none
- priority: low, medium, high, critical

Field rules:
- wait: all other fields must be null or omitted
- ignore: email_id is required; category, response_template, and priority must be null or omitted
- classify: email_id and category are required; response_template and priority must be null or omitted
- respond: email_id, response_template, and priority are required; category must be null or omitted
- escalate: email_id and priority are required; category and response_template must be null or omitted

Never invent enum values.
Never output free text inside enum fields.
Think step-by-step internally, but ONLY output a single JSON object.
""".strip()


def _build_safe_default_payload(observation: Observation) -> dict[str, Any]:
    if observation.inbox:
        return {"action_type": "ignore", "email_id": observation.inbox[0].email_id}
    return {"action_type": "wait"}


def _compact_error_message(message: str, *, limit: int = 240) -> str:
    compact = " ".join(message.split())
    return compact if len(compact) <= limit else f"{compact[: limit - 3]}..."


def extract_json_object(raw_output: str) -> dict[str, Any] | None:
    text = raw_output.strip()
    candidates: list[str] = []
    if text:
        candidates.append(text)
        if text.startswith("```"):
            lines = text.splitlines()
            if len(lines) >= 3:
                candidates.append("\n".join(lines[1:-1]).strip())
    start = text.find("{")
    if start != -1:
        depth = 0
        in_string = False
        escaped = False
        for index in range(start, len(text)):
            char = text[index]
            if escaped:
                escaped = False
                continue
            if char == "\\" and in_string:
                escaped = True
                continue
            if char == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    candidates.append(text[start : index + 1])
                    break
    for candidate in candidates:
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    return None


def normalize_category(value: Any) -> str:
    if not isinstance(value, str):
        return "normal"
    normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "spam": "spam", "junk": "spam", "newsletter": "spam",
        "urgent": "urgent", "urgent_support": "urgent", "high_priority": "urgent",
        "normal": "normal", "routine": "normal", "general": "normal",
        "escalation": "escalation", "escalate": "escalation",
    }
    if normalized in aliases:
        return aliases[normalized]
    if "spam" in normalized or "newsletter" in normalized:
        return "spam"
    if "urgent" in normalized or "critical" in normalized:
        return "urgent"
    if "escalat" in normalized:
        return "escalation"
    return "normal"


def normalize_response_template(value: Any) -> str:
    if not isinstance(value, str):
        return "none"
    normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "acknowledge": "acknowledge", "ack": "acknowledge",
        "resolve": "resolve", "resolution": "resolve",
        "request_info": "request_info", "request_more_info": "request_info", "ask_for_info": "request_info",
        "escalate_notice": "escalate_notice", "escalation_notice": "escalate_notice",
        "none": "none", "no_response": "none",
    }
    if normalized in aliases:
        return aliases[normalized]
    if "request" in normalized and "info" in normalized:
        return "request_info"
    if "escalat" in normalized and "notice" in normalized:
        return "escalate_notice"
    if "ack" in normalized:
        return "acknowledge"
    if "resolv" in normalized:
        return "resolve"
    return "none"


def normalize_priority(value: Any) -> str:
    if isinstance(value, int):
        return {1: "low", 2: "medium", 3: "medium", 4: "high", 5: "critical"}.get(value, SAFE_DEFAULT_PRIORITY)
    if not isinstance(value, str):
        return SAFE_DEFAULT_PRIORITY
    normalized = value.strip().lower()
    if normalized in VALID_PRIORITIES:
        return normalized
    if normalized.isdigit():
        return normalize_priority(int(normalized))
    if "critical" in normalized or "p0" in normalized or "sev0" in normalized or "sev1" in normalized:
        return "critical"
    if "high" in normalized or "urgent" in normalized:
        return "high"
    if "medium" in normalized or "normal" in normalized:
        return "medium"
    if "low" in normalized:
        return "low"
    return SAFE_DEFAULT_PRIORITY


def normalize_decision_payload(payload: dict[str, Any] | None, observation: Observation) -> dict[str, Any]:
    safe_default = _build_safe_default_payload(observation)
    visible_ids = {email.email_id for email in observation.inbox}
    source = payload or {}

    raw_action_type = source.get("action_type")
    action_type = raw_action_type if isinstance(raw_action_type, str) else safe_default["action_type"]
    action_type = action_type.strip().lower() if isinstance(action_type, str) else safe_default["action_type"]
    if action_type not in VALID_ACTION_TYPES:
        action_type = safe_default["action_type"]

    raw_email_id = source.get("email_id")
    email_id = raw_email_id if isinstance(raw_email_id, str) and raw_email_id in visible_ids else safe_default.get("email_id")

    if action_type == "wait":
        return {"action_type": "wait"}
    if email_id is None:
        return safe_default
    if action_type == "ignore":
        return {"action_type": "ignore", "email_id": email_id}
    if action_type == "classify":
        return {
            "action_type": "classify",
            "email_id": email_id,
            "category": normalize_category(source.get("category")),
        }
    if action_type == "escalate":
        return {
            "action_type": "escalate",
            "email_id": email_id,
            "priority": normalize_priority(source.get("priority")),
        }
    response_template = normalize_response_template(source.get("response_template"))
    if response_template == "none":
        return safe_default
    return {
        "action_type": "respond",
        "email_id": email_id,
        "response_template": response_template,
        "priority": normalize_priority(source.get("priority")),
    }


def _action_from_payload(payload: dict[str, Any], observation: Observation) -> tuple[Action, str | None]:
    try:
        return Action(**payload), None
    except ValidationError as exc:
        fallback_payload = _build_safe_default_payload(observation)
        logger.error("action_validation_failed payload=%s fallback=%s", payload, fallback_payload)
        return Action(**fallback_payload), f"model_action_validation_failed: {_compact_error_message(str(exc))}"


def _call_llm(client: OpenAI, observation: Observation, model: str) -> tuple[Action, str | None]:
    safe_default = _build_safe_default_payload(observation)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": json.dumps({
                "step": getattr(observation, "step_index", None),
                "inbox": [
                    {"email_id": e.email_id, "subject": getattr(e, "subject", ""), "body": getattr(e, "body", "")}
                    for e in observation.inbox
                ],
            }),
        },
    ]
    try:
        token_limit = runtime_max_tokens(MAX_TOKENS)
        token_param = {"max_tokens": token_limit} if _is_hf_model(model) else {"max_completion_tokens": token_limit}
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.0,
            response_format={"type": "json_object"},
            **token_param,
        )
        raw_output = response.choices[0].message.content or ""
        parsed_payload = extract_json_object(raw_output)
    except Exception as exc:
        logger.error("llm_request_failed error=%s fallback=%s", exc, safe_default)
        return Action(**safe_default), f"model_request_failed: {_compact_error_message(str(exc))}"

    if parsed_payload is None:
        logger.error("llm_response_parse_failed raw_output=%s fallback=%s", raw_output, safe_default)
        parse_detail = raw_output if raw_output else "empty response"
        return Action(**safe_default), f"model_response_parse_failed: {_compact_error_message(parse_detail)}"

    normalized_payload = normalize_decision_payload(parsed_payload, observation)
    return _action_from_payload(normalized_payload, observation)

# ── Runtime config & task selection ─────────────────────────────────────────


def _validate_runtime_config() -> None:
    backend = runtime_baseline_backend()
    if backend == "heuristic":
        return
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


def _compute_score(rewards: list[float]) -> float:
    """Score = clamp(sum(rewards) / (steps_taken * max_reward_per_step), 0, 1).

    steps_taken = len(rewards).
    max_reward_per_step = 1.0 (from config).
    Deterministic: same rewards = same score.
    """
    if not rewards:
        return 0.0
    steps_taken = len(rewards)
    max_possible = float(steps_taken) * EMAIL_TRIAGE_CONFIG.max_reward_per_step
    raw = sum(rewards) / max_possible
    return max(0.0, min(1.0, raw))


def _score_episode(
    processed_email_ids: set[str],
    all_email_ids: set[str],
) -> float:
    return _completion_ratio(processed_email_ids, all_email_ids)


def _build_openai_classifier(model_name: str) -> Classifier:
    from openai import OpenAI

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
        return _call_llm(client, observation, model_name)

    return classify


def _filter_observation_for_llm(observation: Observation, env_state: dict[str, Any] | None) -> Observation:
    """Return observation with inbox filtered to only unhandled emails."""
    if env_state is None:
        return observation
    handled = _handled_email_ids(env_state)
    unhandled_inbox = [e for e in observation.inbox if e.email_id not in handled]
    if hasattr(observation, "model_copy"):
        return observation.model_copy(update={"inbox": unhandled_inbox})
    observation.inbox = unhandled_inbox
    return observation


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
    # Only show unclassified emails to the LLM to avoid redundant classifications
    filtered_obs = _filter_observation_for_llm(observation, env_state)
    return llm_classifier(filtered_obs)


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
    step_budget = task.max_steps

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

            # If all visible emails are already handled, stop — nothing left to do
            handled = _handled_email_ids(env_state)
            visible_ids = {e.email_id for e in observation.inbox}
            if visible_ids and visible_ids.issubset(handled):
                break

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

        # Score = clamp(sum(rewards) / (steps_taken * max_reward_per_step), 0, 1)
        score = _compute_score(rewards)
        success = score >= 0.6
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
