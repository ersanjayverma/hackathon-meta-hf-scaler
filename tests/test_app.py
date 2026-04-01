from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

from fastapi.testclient import TestClient

from agents.heuristic_agent import HeuristicAgent
from environments.email_triage_env import EmailTriageEnv
from openenv.tasks import get_benchmark_task_names, get_benchmark_tasks, get_email_tasks, get_graders

MODULE_PATH = Path(__file__).resolve().parents[1] / "app.py"
sys.path.insert(0, str(MODULE_PATH.parent))
import server.app as server_app
MODULE_SPEC = importlib.util.spec_from_file_location("openenv_app", MODULE_PATH)
assert MODULE_SPEC is not None and MODULE_SPEC.loader is not None
app_module = importlib.util.module_from_spec(MODULE_SPEC)
sys.modules[MODULE_SPEC.name] = app_module
MODULE_SPEC.loader.exec_module(app_module)

client = TestClient(app_module.app)


def test_root_and_health() -> None:
    assert client.get("/").json() == {"status": "ok"}
    assert client.get("/health").json() == {"status": "ok"}


def test_tasks_endpoint_exposes_schema() -> None:
    response = client.get("/tasks")
    assert response.status_code == 200
    payload = response.json()
    assert [task["name"] for task in payload["tasks"]] == list(get_benchmark_task_names())
    assert len(payload["supplemental_tasks"]) >= 1
    assert payload["action_schema"]["title"] == "Action"
    assert payload["observation_schema"]["title"] == "Observation"
    assert payload["reward_schema"]["title"] == "Reward"


def test_reset_step_and_state_round_trip() -> None:
    reset_response = client.post("/reset", json={"task_name": "task_easy_classification"})
    assert reset_response.status_code == 200
    observation = reset_response.json()
    assert observation["task_name"] == "task_easy_classification"

    step_response = client.post(
        "/step",
        json={"action_type": "classify", "email_id": "e-001", "category": "spam"},
    )
    assert step_response.status_code == 200
    step_payload = step_response.json()
    assert "observation" in step_payload
    assert "reward" in step_payload
    assert isinstance(step_payload["done"], bool)

    state_response = client.get("/state")
    assert state_response.status_code == 200
    state_payload = state_response.json()
    assert state_payload["task"]["name"] == "task_easy_classification"


def test_reset_accepts_empty_post_body() -> None:
    response = client.post("/reset")
    assert response.status_code == 200
    payload = response.json()
    assert payload["task_name"] in set(get_benchmark_task_names())
    assert payload["step_index"] == 0


def test_reset_accepts_null_json_body() -> None:
    response = client.post("/reset", content="null", headers={"Content-Type": "application/json"})
    assert response.status_code == 200
    assert response.json()["step_index"] == 0


def test_reset_accepts_whitespace_json_body() -> None:
    response = client.post("/reset", content="   ", headers={"Content-Type": "application/json"})
    assert response.status_code == 200
    assert response.json()["step_index"] == 0


def test_reset_accepts_array_json_body() -> None:
    response = client.post("/reset", content="[]", headers={"Content-Type": "application/json"})
    assert response.status_code == 200
    assert response.json()["step_index"] == 0


def test_reset_accepts_non_json_text_body() -> None:
    response = client.post("/reset", content="task_name=task_easy_classification")
    assert response.status_code == 200
    assert response.json()["step_index"] == 0


def test_reset_ignores_unknown_json_fields() -> None:
    response = client.post(
        "/reset",
        json={
            "repo_url": "https://huggingface.co/spaces/blackhatbadshah/scaler-hackathon-meta",
            "submission_id": 11,
            "submission_validation_id": 61,
        },
    )
    assert response.status_code == 200
    assert response.json()["step_index"] == 0


def test_grader_endpoint_returns_bounded_score() -> None:
    task = get_benchmark_tasks()[0]
    env = EmailTriageEnv(task=task, seed=task.seed)
    observation = env.reset()
    env.step(HeuristicAgent().act(observation))
    response = client.post(
        "/grader",
        json={
            "task_name": task.name,
            "trajectory": [step.model_dump(mode="json") for step in env.trajectory],
        },
    )
    assert response.status_code == 200
    score = response.json()["score"]
    assert 0.0 <= score <= 1.0
    env.close()


def test_baseline_endpoint_returns_metadata(monkeypatch) -> None:
    monkeypatch.setattr(
        server_app,
        "run_baseline",
        lambda model=None, backend=None, include_supplemental=False: {
            "model": "heuristic-v1",
            "backend": "heuristic",
            "api_failures": 0,
            "fallback_actions": 0,
            "average_score": 1.0,
            "tasks": [{"task": "task_easy_classification", "score": 1.0}],
            "benchmark": {
                "canonical": True,
                "task_set": "canonical_benchmark",
                "benchmark_task_names": list(get_benchmark_task_names()),
            },
        },
    )
    response = client.post("/baseline", json={})
    assert response.status_code == 200
    payload = response.json()
    assert payload["average_score"] == 1.0
    assert payload["metadata"]["backend"] == "heuristic"
    assert payload["benchmark"]["canonical"] is True


def test_graders_are_deterministic_and_bounded() -> None:
    tasks = get_benchmark_tasks()
    graders = get_graders()
    assert [task.name for task in tasks] == list(get_benchmark_task_names())
    for task in tasks:
        env_one = EmailTriageEnv(task=task, seed=task.seed)
        env_two = EmailTriageEnv(task=task, seed=task.seed)
        agent = HeuristicAgent()

        obs_one = env_one.reset()
        obs_two = env_two.reset()
        done_one = False
        done_two = False
        while not done_one and not done_two:
            action_one = agent.act(obs_one)
            action_two = agent.act(obs_two)
            obs_one, _, done_one, _ = env_one.step(action_one)
            obs_two, _, done_two, _ = env_two.step(action_two)

        score_one = float(graders[task.name](env_one.trajectory))
        score_two = float(graders[task.name](env_two.trajectory))
        assert score_one == score_two
        assert 0.0 <= score_one <= 1.0
        env_one.close()
        env_two.close()
