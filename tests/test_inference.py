from __future__ import annotations

import importlib.util
import io
import sys
from contextlib import redirect_stdout
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "inference.py"
MODULE_SPEC = importlib.util.spec_from_file_location("openenv_inference", MODULE_PATH)
assert MODULE_SPEC is not None and MODULE_SPEC.loader is not None
inference_module = importlib.util.module_from_spec(MODULE_SPEC)
sys.modules[MODULE_SPEC.name] = inference_module
MODULE_SPEC.loader.exec_module(inference_module)

Action = inference_module.Action


class FakeReward:
    def __init__(self, total: float) -> None:
        self.total = total


class FakeObservation:
    def __init__(self, inbox, step_index: int = 0) -> None:
        self.inbox = inbox
        self.step_index = step_index


class FakeEmail:
    def __init__(self, email_id: str) -> None:
        self.email_id = email_id
        self.subject = "routine request"
        self.body = "please review the request"
        self.sender = "ops@customer.example"


class FakeTask:
    def __init__(self, name: str, email_ids: list[str], seed: int = 101) -> None:
        self.name = name
        self.seed = seed
        self.max_steps = len(email_ids)
        self.difficulty = "easy"
        self.description = "test"
        self.initial_state = {"emails": [{"email_id": eid} for eid in email_ids]}


class FakeEnv:
    def __init__(self, task, seed: int | None = None) -> None:
        self.task = task
        self.seed = seed
        self._step_index = 0
        self._emails = [FakeEmail(eid) for eid in [e["email_id"] for e in task.initial_state["emails"]]]
        self._processed: set[str] = set()
        self.trajectory = []

    def reset(self):
        self._step_index = 0
        return FakeObservation(list(self._emails), self._step_index)

    def step(self, action):
        self._step_index += 1
        if action.email_id:
            self._processed.add(action.email_id)
            self._emails = [e for e in self._emails if e.email_id != action.email_id]
        obs = FakeObservation(list(self._emails), self._step_index)
        return obs, FakeReward(1.0), len(self._emails) == 0, {}

    def state(self):
        return {
            "processed_email_ids": list(self._processed),
            "remaining_email_ids": [e.email_id for e in self._emails],
            "classifications": {eid: "normal" for eid in self._processed},
            "responses": {},
            "escalations": {},
            "ignored": [],
        }

    def close(self) -> None:
        pass


def _fake_classifier_factory():
    """Returns a classifier that classifies the first inbox email as 'normal'."""
    def classifier(observation):
        if observation.inbox:
            return Action(action_type="classify", email_id=observation.inbox[0].email_id, category="normal"), None
        return Action(action_type="wait"), None
    return classifier


def test_select_tasks_defaults_to_all_canonical_tasks(monkeypatch) -> None:
    tasks = [
        FakeTask("task_easy_classification", ["e-001"]),
        FakeTask("task_medium_prioritization", ["e-101"]),
        FakeTask("task_hard_thread_reasoning", ["e-201"]),
    ]
    monkeypatch.setattr(inference_module, "get_benchmark_tasks", lambda: tasks)
    monkeypatch.setattr(
        inference_module,
        "get_benchmark_task_names",
        lambda: tuple(t.name for t in tasks),
    )
    monkeypatch.delenv("OPENENV_TASK", raising=False)
    assert [t.name for t in inference_module._select_tasks()] == [t.name for t in tasks]


def test_score_episode_full_completion() -> None:
    score = inference_module._score_episode(
        processed_email_ids={"e-001", "e-002"},
        all_email_ids={"e-001", "e-002"},
    )
    assert score == 1.0


def test_score_episode_partial_completion() -> None:
    score = inference_module._score_episode(
        processed_email_ids={"e-001"},
        all_email_ids={"e-001", "e-002"},
    )
    assert score == 0.5


def test_extract_json_object_from_markdown_fence() -> None:
    payload = inference_module.extract_json_object(
        '```json\n{"action_type":"classify","email_id":"e-001","category":"urgent"}\n```'
    )
    assert payload == {"action_type": "classify", "email_id": "e-001", "category": "urgent"}


def test_normalize_category_maps_aliases() -> None:
    assert inference_module.normalize_category("urgent_support") == "urgent"
    assert inference_module.normalize_category("junk") == "spam"
    assert inference_module.normalize_category("routine") == "normal"
    assert inference_module.normalize_category("escalate") == "escalation"


def test_normalize_priority_from_int() -> None:
    assert inference_module.normalize_priority(5) == "critical"
    assert inference_module.normalize_priority(1) == "low"
    assert inference_module.normalize_priority(2) == "medium"


def test_run_task_completes_all_emails(monkeypatch) -> None:
    task = FakeTask("task_easy_classification", ["e-001", "e-002"])
    monkeypatch.setattr(inference_module, "EmailTriageEnv", FakeEnv)
    monkeypatch.setattr(inference_module, "get_benchmark_graders", lambda: {})

    output = io.StringIO()
    with redirect_stdout(output):
        inference_module._run_task(
            task=task,
            model_name="test-model",
            llm_classifier=_fake_classifier_factory(),
        )

    lines = output.getvalue().strip().splitlines()
    assert any("[START]" in line for line in lines)
    assert any("[END]" in line for line in lines)
    assert any("score=1.000" in line for line in lines)
