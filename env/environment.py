from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from environments.email_triage_env import EmailTriageEnv
from env.reward import compute_reward, score_trajectory
from env.state import AgentAction, EnvironmentState, EnvironmentStepResult
from openenv.models import Action, Observation
from openenv.tasks import Task, get_email_tasks, get_graders


@dataclass(slots=True)
class OpenEnvGymLikeEnv:
    task_name: str | None = None
    seed: int | None = None
    deterministic: bool = True
    _task_lookup: dict[str, Task] = field(default_factory=lambda: {task.name: task for task in get_email_tasks()})
    _env: EmailTriageEnv | None = None

    def __post_init__(self) -> None:
        if self.task_name is None:
            self.task_name = get_email_tasks()[0].name

    @property
    def current_task(self) -> Task:
        assert self.task_name is not None
        return self._task_lookup[self.task_name]

    @property
    def action_space(self) -> dict[str, Any]:
        return {
            "action_type": ["classify", "respond", "escalate", "ignore", "wait"],
            "category": ["spam", "urgent", "normal", "escalation"],
            "response_template": ["acknowledge", "resolve", "request_info", "escalate_notice", "none"],
            "priority": ["low", "medium", "high", "critical"],
        }

    @property
    def observation_space(self) -> dict[str, Any]:
        return Observation.model_json_schema()

    def reset(self, *, seed: int | None = None, task_name: str | None = None) -> EnvironmentState:
        if task_name is not None:
            self.task_name = task_name
        if seed is not None:
            self.seed = seed
        task = self.current_task
        effective_seed = self.seed if self.seed is not None else task.seed
        self._env = EmailTriageEnv(task=task, seed=effective_seed)
        observation = self._env.reset()
        return self._state_from_observation(observation)

    def step(self, action: Action | AgentAction | dict[str, Any]) -> EnvironmentStepResult:
        env = self._require_env()
        parsed_action = self._coerce_action(action)
        next_observation, reward, done, info = env.step(parsed_action)
        grader_score = score_trajectory(env.trajectory, get_graders().get(env.task.name)) if done else None
        graded_reward = compute_reward(reward, done, grader_score)
        enriched_info = dict(info)
        if grader_score is not None:
            enriched_info["grader_score"] = grader_score
        return EnvironmentStepResult(
            next_state=self._state_from_observation(next_observation),
            reward=graded_reward.reward,
            done=done,
            info=enriched_info,
        )

    def close(self) -> None:
        if self._env is not None:
            self._env.close()
            self._env = None

    def _require_env(self) -> EmailTriageEnv:
        if self._env is None:
            raise RuntimeError("environment has not been reset")
        return self._env

    def _state_from_observation(self, observation: Observation) -> EnvironmentState:
        env = self._require_env()
        return EnvironmentState(
            observation=observation,
            history_length=len(env.trajectory),
            metadata={
                "task_name": env.task.name,
                "max_steps": env.task.max_steps,
                "deterministic": self.deterministic,
                "seed": env._seed,
            },
        )

    @staticmethod
    def _coerce_action(action: Action | AgentAction | dict[str, Any]) -> Action:
        if isinstance(action, Action):
            return action
        if isinstance(action, AgentAction):
            return action.action
        if isinstance(action, dict):
            return Action(**action)
        raise TypeError(f"unsupported action type: {type(action)!r}")
