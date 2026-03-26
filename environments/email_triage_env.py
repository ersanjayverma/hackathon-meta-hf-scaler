from __future__ import annotations

from typing import Any, Optional

import numpy as np

from openenv.base_env import BaseEnv
from openenv.engine import EnvironmentEngine
from openenv.logger import StructuredLogger
from openenv.models import Action, ActionTrace, EmailSpec, Observation, Reward, StepRecord
from openenv.replay import EpisodeRecorder
from openenv.tasks import Task


class EmailTriageEnv(BaseEnv[Observation, Action, Reward]):
    def __init__(self, task: Task, seed: Optional[int] = None) -> None:
        self.task = task
        self.logger = StructuredLogger("openenv.email_triage")
        super().__init__(engine=EnvironmentEngine(self.logger), seed=seed if seed is not None else task.seed)
        self._closed = False
        self._episode_recorder = EpisodeRecorder(
            environment_name=self.__class__.__name__,
            seed=self._seed,
            config={"task": self.task.model_dump(mode="json")},
        )
        self._initial_email_specs = [EmailSpec(**item) for item in self.task.initial_state["emails"]]
        self._email_specs: dict[str, EmailSpec] = {}
        self._visible_ids: list[str] = []
        self._completed_ids: list[str] = []
        self._step_index = 0
        self._action_history: list[ActionTrace] = []
        self._trajectory: list[StepRecord] = []
        self._classifications: dict[str, str] = {}
        self._responses: dict[str, str] = {}
        self._escalations: dict[str, int] = {}
        self._ignored: set[str] = set()
        self._last_action_key: Optional[tuple[str, str]] = None

    @property
    def trajectory(self) -> list[StepRecord]:
        return list(self._trajectory)

    @property
    def episode_recorder(self) -> EpisodeRecorder:
        return self._episode_recorder

    def reset(self) -> Observation:
        with self._lock:
            self._closed = False
            self._rng = np.random.default_rng(self._seed)
            self._email_specs = {spec.email_id: spec for spec in self._initial_email_specs}
            self._visible_ids = []
            self._completed_ids = []
            self._step_index = 0
            self._action_history = []
            self._trajectory = []
            self._classifications = {}
            self._responses = {}
            self._escalations = {}
            self._ignored = set()
            self._last_action_key = None
            self._episode_recorder = EpisodeRecorder(
                environment_name=self.__class__.__name__,
                seed=self._seed,
                config={"task": self.task.model_dump(mode="json")},
            )
            self._ingest_arrivals()
            return self._observation()

    def default_validation_action(self, observation: Observation) -> Action:
        return Action(action_type="wait")

    def step(self, action: Action) -> tuple[Observation, Reward, bool, dict[str, Any]]:
        with self._lock:
            if self._closed:
                raise RuntimeError("environment is closed")
            observation_before = self._observation()

            def _advance() -> tuple[Observation, Reward, bool, dict[str, Any]]:
                reward_components: dict[str, float] = {}
                info: dict[str, Any] = {"task_name": self.task.name}
                loop_key = (action.action_type, action.email_id or "none")
                if loop_key == self._last_action_key:
                    reward_components["loop_penalty"] = -0.1
                    info["loop_detected"] = True
                self._last_action_key = loop_key

                if action.action_type == "wait":
                    reward_components["wait_penalty"] = self._wait_penalty()
                else:
                    info.update(self._apply_action(action, reward_components))

                reward_components.update(self._deadline_penalties())
                total = sum(reward_components.values())
                reward = Reward(
                    total=total,
                    components=reward_components,
                    reason="dense trajectory-aware reward",
                )

                self._action_history.append(
                    ActionTrace(
                        step_index=self._step_index,
                        email_id=action.email_id,
                        action_type=action.action_type,
                        summary=self._summarize_action(action),
                    )
                )
                self._step_index += 1
                self._ingest_arrivals()
                done = self._is_done()
                observation_after = self._observation()
                info.update(
                    {
                        "reward_breakdown": reward.components,
                        "pending_emails": len(observation_after.inbox),
                        "backlog_size": len(observation_after.inbox),
                        "action_trace": self._action_history[-1].model_dump(mode="json"),
                        "seed": self._seed,
                    }
                )
                step_record = StepRecord(
                    step_index=self._step_index - 1,
                    action=action,
                    reward=reward,
                    observation=observation_after,
                    done=done,
                    info=info,
                )
                self._trajectory.append(step_record)
                self._episode_recorder.record(
                    state=observation_before.model_dump(mode="json"),
                    action=action.model_dump(mode="json"),
                    reward=reward.total,
                    done=done,
                    info=info,
                    next_state=observation_after.model_dump(mode="json"),
                )
                return observation_after, reward, done, info

            return self._engine.timed_step(_advance)

    def state(self) -> dict:
        with self._lock:
            return {
                "task": self.task.model_dump(mode="json"),
                "seed": self._seed,
                "step_index": self._step_index,
                "visible_ids": list(self._visible_ids),
                "completed_ids": list(self._completed_ids),
                "classifications": dict(self._classifications),
                "responses": dict(self._responses),
                "escalations": dict(self._escalations),
                "ignored": sorted(self._ignored),
                "closed": self._closed,
                "trajectory": [step.model_dump(mode="json") for step in self._trajectory],
                "rng_state": self._rng.bit_generator.state,
            }

    def snapshot(self) -> dict:
        return self.state()

    def restore(self, snapshot: dict) -> None:
        with self._lock:
            self.task = Task(**snapshot["task"])
            self._initial_email_specs = [EmailSpec(**item) for item in self.task.initial_state["emails"]]
            self._email_specs = {spec.email_id: spec for spec in self._initial_email_specs}
            self._seed = int(snapshot["seed"])
            self._rng = np.random.default_rng(self._seed)
            self._rng.bit_generator.state = snapshot["rng_state"]
            self._step_index = int(snapshot["step_index"])
            self._visible_ids = list(snapshot["visible_ids"])
            self._completed_ids = list(snapshot["completed_ids"])
            self._classifications = dict(snapshot["classifications"])
            self._responses = dict(snapshot["responses"])
            self._escalations = {str(key): int(value) for key, value in snapshot["escalations"].items()}
            self._ignored = set(snapshot["ignored"])
            self._closed = bool(snapshot["closed"])
            self._trajectory = [StepRecord(**item) for item in snapshot["trajectory"]]
            self._action_history = [
                ActionTrace(
                    step_index=step.step_index,
                    email_id=step.action.email_id,
                    action_type=step.action.action_type,
                    summary=step.info.get("action_trace", {}).get("summary", step.action.action_type),
                )
                for step in self._trajectory
            ]

    def render(self, mode: str = "human") -> None:
        observation = self._observation()
        if mode == "human":
            print(
                f"task={observation.task_name} step={observation.step_index} "
                f"remaining={observation.remaining_steps} inbox={len(observation.inbox)}"
            )
        elif mode == "json":
            print(observation.model_dump_json(indent=2))
        else:
            raise ValueError(f"unsupported render mode: {mode}")

    def close(self) -> None:
        with self._lock:
            self._closed = True

    def _observation(self) -> Observation:
        inbox = [
            self._email_specs[email_id].to_view(self._step_index, seen=True)
            for email_id in self._visible_ids
            if email_id not in self._completed_ids
        ]
        return Observation(
            task_name=self.task.name,
            step_index=self._step_index,
            max_steps=self.task.max_steps,
            remaining_steps=max(self.task.max_steps - self._step_index, 0),
            seed=self._seed,
            inbox=inbox,
            completed_email_ids=list(self._completed_ids),
            action_history=list(self._action_history),
        )

    def _ingest_arrivals(self) -> None:
        for spec in self._initial_email_specs:
            if spec.arrival_step <= self._step_index and spec.email_id not in self._visible_ids:
                self._visible_ids.append(spec.email_id)

    def _apply_action(self, action: Action, reward_components: dict[str, float]) -> dict[str, Any]:
        info: dict[str, Any] = {}
        if action.email_id not in self._visible_ids or action.email_id in self._completed_ids:
            reward_components["invalid_target"] = -0.3
            info["invalid_target"] = True
            return info

        spec = self._email_specs[action.email_id]
        if action.action_type == "classify":
            if self._classifications.get(spec.email_id) is not None:
                reward_components["redundant_action"] = -0.1
            self._classifications[spec.email_id] = action.category or ""
            reward_components["classification"] = 0.3 if action.category == spec.true_category else -0.35
        elif action.action_type == "respond":
            if not spec.requires_response:
                reward_components["unnecessary_response"] = -0.2
            self._responses[spec.email_id] = action.response_template or "none"
            reward_components["response_correctness"] = (
                0.35 if action.response_template == spec.response_template else -0.25
            )
            if self._step_index <= spec.response_deadline:
                reward_components["timeliness"] = reward_components.get("timeliness", 0.0) + 0.15
        elif action.action_type == "escalate":
            self._escalations[spec.email_id] = self._step_index
            should_escalate = spec.requires_escalation or (
                spec.escalation_trigger_step is not None and self._step_index >= spec.escalation_trigger_step
            )
            reward_components["escalation"] = 0.4 if should_escalate else -0.3
            if should_escalate and self._step_index <= spec.escalation_deadline:
                reward_components["timeliness"] = reward_components.get("timeliness", 0.0) + 0.2
        elif action.action_type == "ignore":
            self._ignored.add(spec.email_id)
            reward_components["ignore"] = 0.15 if spec.true_category == "spam" else -0.25

        info["resolved_email_id"] = spec.email_id if self._is_terminal_email(spec.email_id) else None
        if self._is_terminal_email(spec.email_id):
            self._completed_ids.append(spec.email_id)
            reward_components["completion"] = reward_components.get("completion", 0.0) + 0.1
        return info

    def _wait_penalty(self) -> float:
        urgent_open = [
            email_id
            for email_id in self._visible_ids
            if email_id not in self._completed_ids
            and self._email_specs[email_id].priority_hint in {"high", "critical"}
        ]
        return -0.05 if urgent_open else 0.0

    def _deadline_penalties(self) -> dict[str, float]:
        penalties: dict[str, float] = {}
        for email_id in self._visible_ids:
            if email_id in self._completed_ids:
                continue
            spec = self._email_specs[email_id]
            if self._step_index > spec.classification_deadline and email_id not in self._classifications:
                penalties["missed_classification"] = penalties.get("missed_classification", 0.0) - 0.2
            if spec.requires_response and self._step_index > spec.response_deadline and email_id not in self._responses:
                penalties["missed_response"] = penalties.get("missed_response", 0.0) - 0.35
            should_escalate = spec.requires_escalation or (
                spec.escalation_trigger_step is not None and self._step_index >= spec.escalation_trigger_step
            )
            if should_escalate and self._step_index > spec.escalation_deadline and email_id not in self._escalations:
                penalties["missed_escalation"] = penalties.get("missed_escalation", 0.0) - 0.4
        return penalties

    def _is_terminal_email(self, email_id: str) -> bool:
        spec = self._email_specs[email_id]
        if email_id in self._ignored:
            return True
        classified = email_id in self._classifications
        responded = (not spec.requires_response) or email_id in self._responses
        should_escalate = spec.requires_escalation or (
            spec.escalation_trigger_step is not None and self._step_index >= spec.escalation_trigger_step
        )
        escalated = (not should_escalate) or email_id in self._escalations
        return classified and responded and escalated

    def _is_done(self) -> bool:
        no_future_arrivals = all(spec.arrival_step <= self._step_index for spec in self._initial_email_specs)
        all_terminal = all(self._is_terminal_email(email_id) for email_id in self._visible_ids)
        return self._step_index >= self.task.max_steps or (no_future_arrivals and all_terminal)

    def _summarize_action(self, action: Action) -> str:
        if action.action_type == "wait":
            return "waited for more context"
        parts = [action.action_type, action.email_id or ""]
        if action.category:
            parts.append(action.category)
        if action.response_template:
            parts.append(action.response_template)
        return ":".join(parts)
