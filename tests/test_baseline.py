from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

from environments.email_triage_env import EmailTriageEnv
from openenv.tasks import get_benchmark_task_names, get_benchmark_tasks, get_email_tasks

MODULE_PATH = Path(__file__).resolve().parents[1] / "baseline" / "run_baseline.py"
MODULE_SPEC = importlib.util.spec_from_file_location("baseline_run_baseline", MODULE_PATH)
assert MODULE_SPEC is not None and MODULE_SPEC.loader is not None
baseline_run_baseline = importlib.util.module_from_spec(MODULE_SPEC)
MODULE_SPEC.loader.exec_module(baseline_run_baseline)

choose_action = baseline_run_baseline.choose_action
extract_json_object = baseline_run_baseline.extract_json_object
normalize_decision_payload = baseline_run_baseline.normalize_decision_payload
verify_openai_api = baseline_run_baseline.verify_openai_api
RUNTIME_STATS = baseline_run_baseline.RUNTIME_STATS


class FakeResponse:
    def __init__(self, output_text: str) -> None:
        self.output_text = output_text


class FakeResponses:
    def __init__(self, output_text: str | Exception) -> None:
        self.output_text = output_text
        self.calls: list[dict[str, Any]] = []

    def create(self, **_: Any) -> FakeResponse:
        self.calls.append(_)
        if isinstance(self.output_text, Exception):
            raise self.output_text
        return FakeResponse(self.output_text)


class FakeClient:
    def __init__(self, output_text: str | Exception) -> None:
        self.responses = FakeResponses(output_text)


def reset_runtime_stats() -> None:
    RUNTIME_STATS["api_failures"] = 0
    RUNTIME_STATS["fallback_actions"] = 0


def test_extract_json_object_from_markdown_fence() -> None:
    payload = extract_json_object(
        '```json\n{"action_type":"classify","email_id":"e-001","category":"urgent"}\n```'
    )
    assert payload == {"action_type": "classify", "email_id": "e-001", "category": "urgent"}


def test_normalize_decision_payload_maps_invalid_values() -> None:
    task = get_benchmark_tasks()[0]
    env_observation = EmailTriageEnv(task=task, seed=task.seed).reset()
    normalized = normalize_decision_payload(
        {
            "action_type": "classify",
            "email_id": env_observation.inbox[0].email_id,
            "category": "urgent_support",
        },
        env_observation,
    )
    assert normalized["category"] == "urgent"


def test_choose_action_falls_back_for_non_json_output() -> None:
    reset_runtime_stats()
    task = get_benchmark_tasks()[0]

    client = FakeClient("I think this is urgent and needs help immediately.")
    observation = EmailTriageEnv(task=task, seed=task.seed).reset()
    action = choose_action(client, observation, model="test")
    assert action.action_type == "ignore"
    assert action.email_id == observation.inbox[0].email_id
    assert "text" not in client.responses.calls[0]
    assert client.responses.calls[0]["input"][0]["role"] == "system"
    assert client.responses.calls[0]["input"][1]["role"] == "user"
    assert RUNTIME_STATS["fallback_actions"] == 1
    assert RUNTIME_STATS["api_failures"] == 0


def test_choose_action_normalizes_invalid_template_and_priority() -> None:
    reset_runtime_stats()
    task = get_benchmark_tasks()[1]

    observation = EmailTriageEnv(task=task, seed=task.seed).reset()
    action = choose_action(
        FakeClient(
            '{"action_type":"respond","email_id":"%s","response_template":"Please reassure them and ask questions","priority":5}'
            % observation.inbox[0].email_id
        ),
        observation,
        model="test",
    )
    assert action.action_type == "ignore"
    assert action.email_id == observation.inbox[0].email_id
    assert RUNTIME_STATS["fallback_actions"] == 0


def test_verify_openai_api_uses_ping() -> None:
    client = FakeClient("pong")
    verify_openai_api(client, model="test")
    assert client.responses.calls[0]["input"] == "ping"


def test_choose_action_counts_api_failures() -> None:
    reset_runtime_stats()
    task = get_benchmark_tasks()[0]
    observation = EmailTriageEnv(task=task, seed=task.seed).reset()
    action = choose_action(FakeClient(RuntimeError("boom")), observation, model="test")
    assert action.action_type == "ignore"
    assert RUNTIME_STATS["api_failures"] == 1
    assert RUNTIME_STATS["fallback_actions"] == 1


def test_run_baseline_writes_benchmark_metadata(tmp_path: Path, monkeypatch) -> None:
    benchmark_tasks = get_benchmark_tasks()
    monkeypatch.setattr(baseline_run_baseline, "get_benchmark_tasks", lambda: benchmark_tasks[:1])
    monkeypatch.setattr(
        baseline_run_baseline,
        "get_benchmark_graders",
        lambda: {benchmark_tasks[0].name: lambda _: 1.0},
    )
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("OPENENV_BASELINE_BACKEND", "heuristic")
    result = baseline_run_baseline.run_baseline(
        model="test-model-v1",
        api_key=None,
        base_url=None,
        output_path=tmp_path / "baseline_results.json",
    )
    metadata = result["metadata"]
    assert metadata["output_schema_version"] == "baseline_results/v2"
    assert metadata["model_name"] == "heuristic-v1"
    assert metadata["requested_model_name"] == "test-model-v1"
    assert metadata["task_count"] == 1
    assert result["benchmark"]["canonical"] is True
    assert result["benchmark"]["task_set"] == "canonical_benchmark"


def test_run_baseline_is_deterministic_for_canonical_tasks(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("OPENENV_BASELINE_BACKEND", "heuristic")
    first = baseline_run_baseline.run_baseline(output_path=tmp_path / "first.json")
    second = baseline_run_baseline.run_baseline(output_path=tmp_path / "second.json")

    assert first["backend"] == "heuristic"
    assert first["average_score"] == 1.0
    assert second["average_score"] == 1.0
    assert first["benchmark"]["benchmark_task_names"] == list(get_benchmark_task_names())
    assert [task["task"] for task in first["tasks"]] == list(get_benchmark_task_names())
    assert {task["task"]: task["score"] for task in first["tasks"]} == {
        "task_easy_classification": 1.0,
        "task_medium_prioritization": 1.0,
        "task_hard_thread_reasoning": 1.0,
    }
    assert [
        (task["task"], task["difficulty"], task["seed"], task["score"])
        for task in first["tasks"]
    ] == [
        (task["task"], task["difficulty"], task["seed"], task["score"])
        for task in second["tasks"]
    ]
