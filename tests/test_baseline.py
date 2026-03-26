from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

from environments.email_triage_env import EmailTriageEnv
from openenv.tasks import get_email_tasks

MODULE_PATH = Path(__file__).resolve().parents[1] / "baseline" / "run_baseline.py"
MODULE_SPEC = importlib.util.spec_from_file_location("baseline_run_baseline", MODULE_PATH)
assert MODULE_SPEC is not None and MODULE_SPEC.loader is not None
baseline_run_baseline = importlib.util.module_from_spec(MODULE_SPEC)
MODULE_SPEC.loader.exec_module(baseline_run_baseline)

choose_action = baseline_run_baseline.choose_action
extract_json_object = baseline_run_baseline.extract_json_object
normalize_decision_payload = baseline_run_baseline.normalize_decision_payload


class FakeResponse:
    def __init__(self, output_text: str) -> None:
        self.output_text = output_text


class FakeResponses:
    def __init__(self, output_text: str | Exception) -> None:
        self.output_text = output_text

    def create(self, **_: Any) -> FakeResponse:
        if isinstance(self.output_text, Exception):
            raise self.output_text
        return FakeResponse(self.output_text)


class FakeClient:
    def __init__(self, output_text: str | Exception) -> None:
        self.responses = FakeResponses(output_text)


def test_extract_json_object_from_markdown_fence() -> None:
    payload = extract_json_object(
        '```json\n{"action_type":"classify","email_id":"e-001","category":"urgent"}\n```'
    )
    assert payload == {"action_type": "classify", "email_id": "e-001", "category": "urgent"}


def test_normalize_decision_payload_maps_invalid_values() -> None:
    task = get_email_tasks()[0]
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
    task = get_email_tasks()[0]

    observation = EmailTriageEnv(task=task, seed=task.seed).reset()
    action = choose_action(FakeClient("I think this is urgent and needs help immediately."), observation, model="test")
    assert action.action_type == "ignore"
    assert action.email_id == observation.inbox[0].email_id


def test_choose_action_normalizes_invalid_template_and_priority() -> None:
    task = get_email_tasks()[1]

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
