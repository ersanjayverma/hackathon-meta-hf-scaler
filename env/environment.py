from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from environments.email_triage_env import EmailTriageEnv
from env.codec import DiscreteActionCodec, ObservationEncoder, TextTrajectoryFormatter
from env.reward import compute_reward, score_trajectory
from env.spaces import BoxSpace, DiscreteSpace
from env.state import AgentAction, EnvironmentState, EnvironmentStepResult
from openenv.models import Action, Observation
from openenv.tasks import Task, get_email_tasks, get_graders


@dataclass(slots=True)
class OpenEnvGymLikeEnv:
    task_name: str | None = None
    seed: int | None = None
    deterministic: bool = True
    max_inbox_size: int = 12
    _task_lookup: dict[str, Task] = field(default_factory=lambda: {task.name: task for task in get_email_tasks()})
    _codec: DiscreteActionCodec = field(init=False)
    _encoder: ObservationEncoder = field(init=False)
    _formatter: TextTrajectoryFormatter = field(init=False)
    _env: EmailTriageEnv | None = None

    def __post_init__(self) -> None:
        if self.task_name is None:
            self.task_name = get_email_tasks()[0].name
        self._codec = DiscreteActionCodec(max_inbox_size=self.max_inbox_size)
        self._encoder = ObservationEncoder(max_inbox_size=self.max_inbox_size)
        self._formatter = TextTrajectoryFormatter(max_emails=self.max_inbox_size)

    @property
    def current_task(self) -> Task:
        assert self.task_name is not None
        return self._task_lookup[self.task_name]

    @property
    def action_space(self) -> dict[str, Any]:
        return {
            "discrete": DiscreteSpace(self._codec.action_count),
            "semantic": {
                "action_type": ["classify", "respond", "escalate", "ignore", "wait"],
                "category": ["spam", "urgent", "normal", "escalation"],
                "response_template": ["acknowledge", "resolve", "request_info", "escalate_notice", "none"],
                "priority": ["low", "medium", "high", "critical"],
            },
        }

    @property
    def observation_space(self) -> dict[str, Any]:
        return {
            "structured": Observation.model_json_schema(),
            "vector": BoxSpace(low=0.0, high=1.0, shape=(self._encoder.feature_count,)),
        }

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

    def step(self, action: Action | AgentAction | dict[str, Any] | int) -> EnvironmentStepResult:
        env = self._require_env()
        current_observation = env._observation()
        parsed_action = self._coerce_action(action, current_observation)
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
        action_mask = self._codec.encode_mask(observation)
        vector = self._encoder.encode(observation)
        return EnvironmentState(
            observation=observation,
            history_length=len(env.trajectory),
            vector_observation=vector.tolist(),
            action_mask=action_mask.tolist(),
            text_observation=self._formatter.render(observation),
            metadata={
                "task_name": env.task.name,
                "max_steps": env.task.max_steps,
                "deterministic": self.deterministic,
                "seed": env._seed,
                "action_count": self._codec.action_count,
                "feature_count": self._encoder.feature_count,
            },
        )

    def _coerce_action(self, action: Action | AgentAction | dict[str, Any] | int, observation: Observation) -> Action:
        if isinstance(action, Action):
            return action
        if isinstance(action, AgentAction):
            return action.action
        if isinstance(action, dict):
            return Action(**action)
        if isinstance(action, int):
            return self._codec.decode(action, observation)
        raise TypeError(f"unsupported action type: {type(action)!r}")

    def reset_gym(self, *, seed: int | None = None, task_name: str | None = None) -> tuple[Any, dict[str, Any]]:
        state = self.reset(seed=seed, task_name=task_name)
        return state.vector, {"action_mask": state.action_mask, "text_observation": state.text_observation}

    def step_gym(self, action: int) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        result = self.step(action)
        terminated = result.done
        truncated = bool(result.info.get("termination_reason") == "max_steps")
        return (
            result.next_state.vector,
            float(result.reward.total),
            terminated,
            truncated,
            {
                **result.info,
                "action_mask": result.next_state.action_mask,
                "text_observation": result.next_state.text_observation,
            },
        )
