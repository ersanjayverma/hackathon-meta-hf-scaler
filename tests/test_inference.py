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


class FakeReward:
    def __init__(self, total: float) -> None:
        self.total = total


class FakeObservation:
    def __init__(self, inbox) -> None:
        self.inbox = inbox


class FakeEmail:
    def __init__(self, email_id: str, priority_hint: str = "medium", age: int = 0) -> None:
        self.email_id = email_id
        self.priority_hint = priority_hint
        self.age = age
        self.subject = "routine request"
        self.body = "please review the request"
        self.sender = "ops@customer.example"


class FakeTask:
    def __init__(self, name: str, email_ids: list[str], seed: int = 101) -> None:
        self.name = name
        self.seed = seed
        self.initial_state = {"emails": [{"email_id": email_id} for email_id in email_ids]}


class FakeEnv:
    def __init__(self, task, seed: int | None = None) -> None:
        self.task = task
        self.seed = seed
        self._step_index = 0
        self._emails = [FakeEmail(email_id) for email_id in [item["email_id"] for item in task.initial_state["emails"]]]

    def reset(self):
        self._step_index = 0
        return FakeObservation(list(self._emails))

    def step(self, action):
        self._step_index += 1
        remaining = [email for email in self._emails if email.email_id != action.email_id]
        self._emails = remaining
        return FakeObservation(list(self._emails)), FakeReward(0.5), False, {}

    def close(self) -> None:
        return None


def test_inference_main_runs_all_canonical_tasks(monkeypatch) -> None:
    tasks = [
        FakeTask("task_easy_classification", ["e-001", "e-002"]),
        FakeTask("task_medium_prioritization", ["e-101"]),
        FakeTask("task_hard_thread_reasoning", ["e-201"]),
    ]

    monkeypatch.setattr(inference_module, "EmailTriageEnv", FakeEnv)
    monkeypatch.setattr(inference_module, "get_benchmark_tasks", lambda: tasks)
    monkeypatch.setattr(
        inference_module,
        "get_benchmark_task_names",
        lambda: tuple(task.name for task in tasks),
    )
    monkeypatch.delenv("OPENENV_TASK", raising=False)

    output = io.StringIO()
    with redirect_stdout(output):
        inference_module.main()

    lines = output.getvalue().strip().splitlines()
    assert lines == [
        "[START] task=task_easy_classification env=email_triage_benchmark model=heuristic-v1",
        "[STEP] step=1 action=classify('e-001','normal') reward=0.50 done=false error=null",
        "[STEP] step=2 action=classify('e-002','normal') reward=0.50 done=false error=null",
        "[END] success=true steps=2 rewards=0.50,0.50",
        "[START] task=task_medium_prioritization env=email_triage_benchmark model=heuristic-v1",
        "[STEP] step=1 action=classify('e-101','normal') reward=0.50 done=false error=null",
        "[END] success=true steps=1 rewards=0.50",
        "[START] task=task_hard_thread_reasoning env=email_triage_benchmark model=heuristic-v1",
        "[STEP] step=1 action=classify('e-201','normal') reward=0.50 done=false error=null",
        "[END] success=true steps=1 rewards=0.50",
    ]


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
        lambda: tuple(task.name for task in tasks),
    )
    monkeypatch.delenv("OPENENV_TASK", raising=False)
    assert [task.name for task in inference_module._select_tasks()] == [task.name for task in tasks]


def test_resolve_backend_prefers_explicit_env(monkeypatch) -> None:
    monkeypatch.setenv("OPENENV_BASELINE_BACKEND", "heuristic")
    monkeypatch.setenv("API_BASE_URL", "https://example.invalid/v1")
    monkeypatch.setenv("MODEL_NAME", "test-model")
    monkeypatch.setenv("HF_TOKEN", "token")
    assert inference_module._resolve_backend() == "heuristic"


def test_resolve_backend_uses_external_env_when_complete(monkeypatch) -> None:
    monkeypatch.delenv("OPENENV_BASELINE_BACKEND", raising=False)
    monkeypatch.setenv("API_BASE_URL", "https://example.invalid/v1")
    monkeypatch.setenv("MODEL_NAME", "test-model")
    monkeypatch.setenv("HF_TOKEN", "token")
    assert inference_module._resolve_backend() == "openai"


def test_resolve_backend_falls_back_to_internal_when_env_incomplete(monkeypatch) -> None:
    monkeypatch.delenv("OPENENV_BASELINE_BACKEND", raising=False)
    monkeypatch.setenv("API_BASE_URL", "https://example.invalid/v1")
    monkeypatch.setenv("MODEL_NAME", "test-model")
    monkeypatch.delenv("HF_TOKEN", raising=False)
    assert inference_module._resolve_backend() == "heuristic"
