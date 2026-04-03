from __future__ import annotations

import json
import logging
from pathlib import Path
from datetime import datetime, timezone
from typing import Any

from openai import OpenAI
from pydantic import ValidationError

from agents.heuristic_agent import HeuristicAgent
from environments.email_triage_env import EmailTriageEnv
from openenv.config import BENCHMARK_METADATA
from openenv.models import Action, Observation
from openenv.runtime_config import (
    API_BASE_URL,
    ENV_OPENAI_API_KEY,
    ENV_OPENENV_BASELINE_BACKEND,
    MAX_TOKENS,
    MODEL_NAME,
    TEMPERATURE,
    runtime_api_base_url,
    runtime_api_key,
    runtime_baseline_backend,
    runtime_has_openai_config,
    runtime_max_tokens,
    runtime_model_name,
    runtime_temperature,
)
from openenv.tasks import (
    get_benchmark_graders,
    get_benchmark_task_names,
    get_benchmark_tasks,
    get_email_tasks,
    get_graders,
)

VALID_ACTION_TYPES = {"classify", "respond", "escalate", "ignore", "wait"}
VALID_PRIORITIES = {"low", "medium", "high", "critical"}
SAFE_DEFAULT_PRIORITY = "medium"
BASELINE_SYSTEM_PROMPT = """
You are an email triage agent for the OpenEnv benchmark.

Return exactly one JSON object and nothing else.
Do not include markdown, code fences, commentary, or explanations.

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
""".strip()

logger = logging.getLogger(__name__)
RUNTIME_STATS = {"api_failures": 0, "fallback_actions": 0}
CANONICAL_BASELINE_BACKEND = "heuristic"
OPTIONAL_BASELINE_BACKENDS = {"heuristic", "openai"}


def build_safe_default_payload(observation: Observation) -> dict[str, Any]:
    if observation.inbox:
        return {"action_type": "ignore", "email_id": observation.inbox[0].email_id}
    return {"action_type": "wait"}


def _compact_error_message(message: str, *, limit: int = 240) -> str:
    compact_message = " ".join(message.split())
    if len(compact_message) <= limit:
        return compact_message
    return f"{compact_message[: limit - 3]}..."


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
        "spam": "spam",
        "junk": "spam",
        "newsletter": "spam",
        "urgent": "urgent",
        "urgent_support": "urgent",
        "high_priority": "urgent",
        "normal": "normal",
        "routine": "normal",
        "general": "normal",
        "escalation": "escalation",
        "escalate": "escalation",
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
        "acknowledge": "acknowledge",
        "ack": "acknowledge",
        "resolve": "resolve",
        "resolution": "resolve",
        "request_info": "request_info",
        "request_more_info": "request_info",
        "ask_for_info": "request_info",
        "escalate_notice": "escalate_notice",
        "escalation_notice": "escalate_notice",
        "none": "none",
        "no_response": "none",
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
    safe_default = build_safe_default_payload(observation)
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


def action_from_payload(payload: dict[str, Any], observation: Observation) -> tuple[Action, str | None]:
    try:
        return Action(**payload), None
    except ValidationError as exc:
        fallback_payload = build_safe_default_payload(observation)
        RUNTIME_STATS["fallback_actions"] += 1
        logger.error("action_validation_failed payload=%s fallback=%s", payload, fallback_payload)
        return Action(**fallback_payload), f"model_action_validation_failed: {_compact_error_message(str(exc))}"


def choose_action_with_diagnostics(client: OpenAI, observation: Observation, model: str) -> tuple[Action, str | None]:
    safe_default = build_safe_default_payload(observation)
    raw_output = ""
    parsed_payload: dict[str, Any] | None = None
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.0,  # keep deterministic
            max_tokens=runtime_max_tokens(MAX_TOKENS),
            response_format={"type": "json_object"}  
        )
        raw_output = raw_output = response.choices[0].message.content or ""
        parsed_payload = extract_json_object(raw_output)
    except Exception as exc:
        RUNTIME_STATS["api_failures"] += 1
        RUNTIME_STATS["fallback_actions"] += 1
        logger.error("llm_request_failed error=%s fallback=%s", exc, safe_default)
        return Action(**safe_default), f"model_request_failed: {_compact_error_message(str(exc))}"

    if parsed_payload is None:
        RUNTIME_STATS["fallback_actions"] += 1
        logger.error("llm_response_parse_failed raw_output=%s fallback=%s", raw_output, safe_default)
        parse_detail = raw_output if raw_output else "empty response"
        return Action(**safe_default), f"model_response_parse_failed: {_compact_error_message(parse_detail)}"

    normalized_payload = normalize_decision_payload(parsed_payload, observation)
    logger.info("llm_raw_output=%s", raw_output)
    logger.info("llm_parsed_json=%s", parsed_payload)
    logger.info("llm_normalized_decision=%s", normalized_payload)
    return action_from_payload(normalized_payload, observation)


def choose_action(client: OpenAI, observation: Observation, model: str) -> Action:
    action, _ = choose_action_with_diagnostics(client, observation, model)
    return action


def verify_openai_api(client: OpenAI, model: str) -> None:
    try:
        response = client.responses.create(
            model=model,
            input="ping",
            temperature=runtime_temperature(TEMPERATURE),
            max_output_tokens=runtime_max_tokens(MAX_TOKENS),
        )
        logger.info("openai_api_healthcheck_ok output=%s", response.output_text)
    except Exception as exc:
        logger.error("openai_api_healthcheck_failed error=%s", exc)
        raise RuntimeError(f"OpenAI Responses API health check failed for model {model}") from exc


def _reset_runtime_stats() -> None:
    RUNTIME_STATS["api_failures"] = 0
    RUNTIME_STATS["fallback_actions"] = 0


def _resolve_backend(backend: str | None) -> str:
    requested = backend or runtime_baseline_backend()
    if requested is None:
        return "openai" if runtime_has_openai_config(api_base_url_default=API_BASE_URL, model_name_default=MODEL_NAME) else CANONICAL_BASELINE_BACKEND
    requested = requested.strip().lower()
    if requested not in OPTIONAL_BASELINE_BACKENDS:
        return CANONICAL_BASELINE_BACKEND
    return requested


def run_baseline(
    *,
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    output_path: Path | None = None,
    backend: str | None = None,
    include_supplemental: bool = False,
) -> dict[str, object]:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    _reset_runtime_stats()

    resolved_backend = _resolve_backend(backend)
    requested_model_name = model or runtime_model_name(BENCHMARK_METADATA.default_model) or MODEL_NAME
    model_name = requested_model_name if resolved_backend == "openai" else "heuristic-v1"
    resolved_api_key = api_key if api_key is not None else runtime_api_key()
    resolved_base_url = base_url if base_url is not None else runtime_api_base_url(API_BASE_URL)
    client: OpenAI | None = None
    if resolved_backend == "openai":
        if not resolved_api_key:
            raise RuntimeError(f"{ENV_OPENENV_BASELINE_BACKEND}=openai requires HF_TOKEN or {ENV_OPENAI_API_KEY}")
        client = OpenAI(api_key=resolved_api_key, base_url=resolved_base_url)
        verify_openai_api(client, requested_model_name)
    else:
        logger.info("using_canonical_heuristic_baseline")

    tasks = get_email_tasks(include_supplemental=True) if include_supplemental else get_benchmark_tasks()
    graders = get_graders() if include_supplemental else get_benchmark_graders()
    results: dict[str, object] = {
        "model": model_name,
        "base_url": resolved_base_url,
        "backend": resolved_backend,
        "tasks": [],
        "average_score": 0.0,
        "api_failures": 0,
        "fallback_actions": 0,
        "benchmark": {
            "task_set": "canonical_benchmark" if not include_supplemental else "full_task_catalog",
            "canonical": not include_supplemental,
            "benchmark_task_names": list(get_benchmark_task_names()),
            "evaluated_task_names": [task.name for task in tasks],
        },
        "metadata": {
            **BENCHMARK_METADATA.to_dict(),
            "model_name": model_name,
            "requested_model_name": requested_model_name,
            "base_url": resolved_base_url,
            "backend": resolved_backend,
            "canonical_baseline_backend": CANONICAL_BASELINE_BACKEND,
            "benchmark_mode": "deterministic_canonical" if resolved_backend == "heuristic" and not include_supplemental else "custom",
            "task_count": len(tasks),
            "include_supplemental": include_supplemental,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        },
    }
    scores: list[float] = []

    for task in tasks:
        env = EmailTriageEnv(task=task, seed=task.seed)
        observation = env.reset()
        done = False
        total_reward = 0.0
        action_log: list[dict[str, object]] = []
        heuristic_agent = HeuristicAgent()

        while not done:
            if client is not None:
                action = choose_action(client, observation, model=requested_model_name)
            else:
                action = heuristic_agent.act(observation)
            observation, reward, done, info = env.step(action)
            total_reward += reward.total
            action_log.append(
                {
                    "step": observation.step_index,
                    "action": action.model_dump(mode="json"),
                    "reward": reward.model_dump(mode="json"),
                    "info": info,
                }
            )
            print(
                f"task={task.name} step={observation.step_index} action={action.action_type} "
                f"reward={reward.total:.2f}"
            )

        score = graders[task.name](env.trajectory)
        if not 0.0 <= score <= 1.0:
            raise RuntimeError(f"grader for {task.name} returned out-of-range score: {score}")
        scores.append(score)
        task_result = {
            "task": task.name,
            "difficulty": task.difficulty,
            "seed": task.seed,
            "score": score,
            "total_reward": total_reward,
            "actions": action_log,
        }
        results["tasks"].append(task_result)
        print(f"task={task.name} final_score={score:.3f} total_reward={total_reward:.2f}")
        env.close()

    results["average_score"] = sum(scores) / len(scores)
    results["api_failures"] = RUNTIME_STATS["api_failures"]
    results["fallback_actions"] = RUNTIME_STATS["fallback_actions"]
    resolved_output_path = output_path or Path("baseline/results/baseline_results.json")
    resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_output_path.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
    print(f"average_score={results['average_score']:.3f}")
    print(f"results_file={resolved_output_path}")
    return results


def run() -> None:
    run_baseline()


if __name__ == "__main__":
    run()
