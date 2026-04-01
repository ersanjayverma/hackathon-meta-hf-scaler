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
    inbox: list[object] = []


class FakeEnv:
    def __init__(self, task, seed: int | None = None) -> None:
        self.task = task
        self.seed = seed
        self.trajectory = ["trajectory"]
        self._step_index = 0

    def reset(self):
        self._step_index = 0
        return FakeObservation()

    def step(self, action):
        self._step_index += 1
        done = self._step_index >= 2
        return FakeObservation(), FakeReward(0.5), done, {}

    def close(self) -> None:
        return None


class FakeTask:
    def __init__(self, name: str, seed: int = 101) -> None:
        self.name = name
        self.seed = seed


def test_inference_main_emits_required_line_protocol(monkeypatch) -> None:
    fake_task = FakeTask("task_easy_classification")
    actions = [
        inference_module.Action(action_type="classify", email_id="e-001", category="spam"),
        inference_module.Action(action_type="ignore", email_id="e-001"),
    ]

    monkeypatch.setattr(inference_module, "EmailTriageEnv", FakeEnv)
    monkeypatch.setattr(inference_module, "get_benchmark_tasks", lambda: [fake_task])
    monkeypatch.setattr(inference_module, "get_benchmark_task_names", lambda: (fake_task.name,))
    monkeypatch.setattr(inference_module, "get_benchmark_graders", lambda: {fake_task.name: lambda _: 1.0})
    monkeypatch.setattr(inference_module, "_build_action_selector", lambda _: lambda __: actions.pop(0))
    monkeypatch.delenv("OPENENV_TASK", raising=False)

    output = io.StringIO()
    with redirect_stdout(output):
        inference_module.main()

    lines = output.getvalue().strip().splitlines()
    assert lines == [
        "[START] task=task_easy_classification env=email_triage_benchmark model=heuristic-v1",
        "[STEP] step=1 action=classify('e-001','spam') reward=0.50 done=false error=null",
        "[STEP] step=2 action=ignore('e-001') reward=0.50 done=true error=null",
        "[END] success=true steps=2 rewards=0.50,0.50",
    ]


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
