from __future__ import annotations

from pathlib import Path

from agents.heuristic_agent import HeuristicAgent
from environments.email_triage_env import EmailTriageEnv
from openenv.models import Action, Observation, Reward
from openenv.replay import ReplayStore
from openenv.tasks import Task, get_email_tasks, get_graders
from openenv.models import EmailSpec
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
    assert 0.35 <= score <= 0.45


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


def test_wrong_action_schedules_delayed_consequences() -> None:
    task = get_email_tasks()[0]
    env = EmailTriageEnv(task=task, seed=task.seed)
    env.reset()

    _, reward, done, info = env.step(Action(action_type="ignore", email_id="e-002"))

    assert not done
    assert reward.total < 0.2
    assert info["scheduled_events"]
    scheduled_types = {event["event_type"] for event in info["scheduled_events"]}
    assert {"followup_email", "penalty", "escalation", "sla_breach"} <= scheduled_types
    assert info["system_state"]["stress"] > 0.0


def test_delayed_followup_and_penalty_trigger_later() -> None:
    task = get_email_tasks()[0]
    env = EmailTriageEnv(task=task, seed=task.seed)
    env.reset()
    env.step(Action(action_type="ignore", email_id="e-002"))

    observation, reward, done, info = env.step(Action(action_type="wait"))
    assert not done
    assert info["triggered_events"] == []
    assert all(email.email_id != "e-002-f0-2" for email in observation.inbox)

    observation, reward, done, info = env.step(Action(action_type="wait"))
    assert not done
    assert any(event["event_type"] == "followup_email" for event in info["triggered_events"])
    assert any(email.email_id == "e-002-f0-2" for email in observation.inbox)

    observation, reward, done, info = env.step(Action(action_type="wait"))
    assert not done
    assert "missed_important" in reward.components
    assert reward.components["missed_important"] < 0.0
    assert info["system_state"]["stress"] >= 4.5


def test_initial_emails_receive_deterministic_sla_deadlines() -> None:
    task = get_email_tasks()[0]
    env_a = EmailTriageEnv(task=task, seed=task.seed)
    env_b = EmailTriageEnv(task=task, seed=task.seed)

    env_a.reset()
    env_b.reset()

    deadlines_a = {spec.email_id: spec.deadline_step for spec in env_a._email_specs.values()}
    deadlines_b = {spec.email_id: spec.deadline_step for spec in env_b._email_specs.values()}
    assert deadlines_a == deadlines_b
    assert all(deadline is not None and deadline >= 3 for deadline in deadlines_a.values())


def test_unresolved_email_triggers_sla_breach_penalty() -> None:
    task = get_email_tasks()[0]
    env = EmailTriageEnv(task=task, seed=task.seed)
    env.reset()
    deadline = env._email_specs["e-002"].deadline_step
    assert deadline is not None

    reward = None
    info = {}
    for _ in range(deadline + 1):
        _, reward, _, info = env.step(Action(action_type="wait"))
        if "sla_breach" in reward.components:
            break

    assert reward is not None
    assert reward.components["sla_breach"] == -10.0
    assert info["system_state"]["sla_breaches"] >= 1.0
    assert info["system_state"]["stress"] >= 15.0


def test_repeated_sla_breaches_trigger_overload_and_cascade() -> None:
    task = get_email_tasks()[1]
    env = EmailTriageEnv(task=task, seed=task.seed)
    env.reset()

    triggered_types: set[str] = set()
    pending_counts: list[int] = []
    for _ in range(task.max_steps):
        observation, reward, done, info = env.step(Action(action_type="wait"))
        triggered_types.update(event["event_type"] for event in info["triggered_events"])
        pending_counts.append(len(observation.inbox))
        if "system_overload" in triggered_types:
            break
        if done:
            break

    assert "sla_breach" in triggered_types
    assert "system_overload" in triggered_types
    assert info["system_state"]["sla_breaches"] >= 3.0
    assert info["system_state"]["stress"] >= 20.0
    assert max(pending_counts) >= 3


def test_episode_stays_alive_after_inbox_is_temporarily_resolved() -> None:
    task = get_email_tasks()[0]
    env = EmailTriageEnv(task=task, seed=task.seed)
    env.reset()

    env.step(Action(action_type="classify", email_id="e-001", category="spam"))
    observation, _, done, _ = env.step(Action(action_type="ignore", email_id="e-001"))
    assert not done
    assert observation.remaining_steps == task.max_steps - 2


def test_episode_can_end_on_stable_resolution() -> None:
    task = Task(
        name="test_stable_resolution",
        description="Single spam mail for stable resolution testing.",
        initial_state={
            "emails": [
                EmailSpec(
                    email_id="s-001",
                    sender="ads@promo.example",
                    subject="Limited time offer",
                    body="Buy now and unsubscribe below.",
                    thread_id="s-thread",
                    arrival_step=0,
                    priority_hint="low",
                    noise_score=0.9,
                    true_category="spam",
                    classification_deadline=1,
                    response_deadline=3,
                    escalation_deadline=3,
                ).model_dump(mode="json")
            ]
        },
        success_criteria="Resolve the only email.",
        max_steps=6,
        difficulty="easy",
        seed=404,
    )
    env = EmailTriageEnv(task=task, seed=task.seed)
    env.reset()
    env.step(Action(action_type="classify", email_id="s-001", category="spam"))
    _, _, done, info = env.step(Action(action_type="ignore", email_id="s-001"))
    assert done
    assert info["termination_reason"] == "stable_resolution"


def test_observation_uses_coarsened_noise_signal() -> None:
    task = get_email_tasks()[0]
    env = EmailTriageEnv(task=task, seed=task.seed)
    observation = env.reset()
    easy_by_id = {email.email_id: email for email in observation.inbox}
    assert easy_by_id["e-001"].noise_score == 0.9
    assert easy_by_id["e-002"].noise_score == 0.2


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
