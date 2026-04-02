from __future__ import annotations

from environments.email_triage_env import EmailTriageEnv
from openenv.grader import build_task_grader, grade_processed_ids, processed_ids_from_trajectory
from openenv.models import Action
from openenv.tasks import get_benchmark_tasks


def test_grade_processed_ids_is_simple_ratio() -> None:
    assert grade_processed_ids(["a"], ["a", "b"]) == 0.5
    assert grade_processed_ids(["a", "b"], ["a", "b"]) == 1.0
    assert grade_processed_ids([], []) == 1.0


def test_processed_ids_from_trajectory_uses_last_observation() -> None:
    task = get_benchmark_tasks()[0]
    env = EmailTriageEnv(task=task, seed=task.seed)
    observation = env.reset()
    env.step(Action(action_type="classify", email_id=observation.inbox[0].email_id, category="spam"))
    assert processed_ids_from_trajectory(env.trajectory) == list(env.trajectory[-1].observation.completed_email_ids)
    env.close()


def test_build_task_grader_scores_partial_completion() -> None:
    task = get_benchmark_tasks()[0]
    grader = build_task_grader(email["email_id"] for email in task.initial_state["emails"])
    env = EmailTriageEnv(task=task, seed=task.seed)
    observation = env.reset()
    env.step(Action(action_type="classify", email_id=observation.inbox[0].email_id, category="spam"))
    score = grader(env.trajectory)
    assert 0.0 <= score <= 1.0
    assert score == 0.25
    env.close()
