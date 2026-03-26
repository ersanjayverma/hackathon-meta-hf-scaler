from __future__ import annotations

from pathlib import Path

from agents.heuristic_agent import HeuristicAgent
from environments.email_triage_env import EmailTriageEnv
from openenv.models import Action, Observation, Reward
from openenv.replay import ReplayStore
from openenv.tasks import get_email_tasks, get_graders
from openenv.validation import validate_environment


def test_step_determinism() -> None:
    task = get_email_tasks()[0]
    env_a = EmailTriageEnv(task=task, seed=task.seed)
    env_b = EmailTriageEnv(task=task, seed=task.seed)
    obs_a = env_a.reset()
    obs_b = env_b.reset()
    assert obs_a.model_dump(mode="json") == obs_b.model_dump(mode="json")

    action = Action(action_type="classify", email_id="e-001", category="spam")
    step_a = env_a.step(action)
    step_b = env_b.step(action)
    assert step_a[0].model_dump(mode="json") == step_b[0].model_dump(mode="json")
    assert step_a[1].model_dump(mode="json") == step_b[1].model_dump(mode="json")
    assert step_a[2] == step_b[2]


def test_grader_correctness() -> None:
    task = get_email_tasks()[0]
    env = EmailTriageEnv(task=task, seed=task.seed)
    obs = env.reset()
    env.step(Action(action_type="classify", email_id=obs.inbox[0].email_id, category="spam"))
    env.step(Action(action_type="classify", email_id=obs.inbox[1].email_id, category="urgent"))
    score = get_graders()[task.name](env.trajectory)
    assert 0.8 <= score <= 1.0


def test_reward_consistency() -> None:
    task = get_email_tasks()[1]
    env = EmailTriageEnv(task=task, seed=task.seed)
    observation = env.reset()
    action = HeuristicAgent().act(observation)
    _, reward, _, info = env.step(action)
    assert abs(reward.total - sum(info["reward_breakdown"].values())) < 1e-9


def test_api_compliance() -> None:
    task = get_email_tasks()[2]
    env = EmailTriageEnv(task=task, seed=task.seed)
    observation = env.reset()
    assert isinstance(observation, Observation)
    next_observation, reward, done, info = env.step(Action(action_type="wait"))
    assert isinstance(next_observation, Observation)
    assert isinstance(reward, Reward)
    assert isinstance(done, bool)
    assert isinstance(info, dict)
    assert isinstance(env.state(), dict)


def test_replay_round_trip(tmp_path: Path) -> None:
    task = get_email_tasks()[0]
    env = EmailTriageEnv(task=task, seed=task.seed)
    observation = env.reset()
    env.step(Action(action_type="classify", email_id=observation.inbox[0].email_id, category="spam"))
    replay_path = tmp_path / "episode.json"
    ReplayStore.save(env.episode_recorder, replay_path)
    loaded = ReplayStore.load(replay_path)
    assert len(loaded.transitions) == 1


def test_openenv_validate_passes() -> None:
    issues = validate_environment("environments")
    assert issues == []
