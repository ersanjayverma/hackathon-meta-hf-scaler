from __future__ import annotations

import random
from typing import Any, Optional

import numpy as np

from openenv.base_env import BaseEnv
from openenv.engine import EnvironmentEngine, ScheduledEvent
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
        self._system_state: dict[str, float] = {"stress": 0.0, "sla_breaches": 0.0}
        self._event_counter = 0
        self._spawned_email_ids: set[str] = set()

    @property
    def trajectory(self) -> list[StepRecord]:
        return list(self._trajectory)

    @property
    def episode_recorder(self) -> EpisodeRecorder:
        return self._episode_recorder

    def reset(self) -> Observation:
        with self._lock:
            self._closed = False
            random.seed(self._seed)
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
            self._system_state = {"stress": 0.0, "sla_breaches": 0.0}
            self._event_counter = 0
            self._spawned_email_ids = set()
            self._engine.event_queue.restore([])
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

            def _advance() -> tuple[Observation, Reward, bool, dict[str, Any]]:
                reward_components: dict[str, float] = {}
                info: dict[str, Any] = {"task_name": self.task.name}
                triggered_events = self._process_due_events(reward_components)
                observation_before = self._observation()
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
                reward_components["stress_penalty"] = -(self._system_state["stress"] * 0.02)
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
                scheduled_events = self._engine.event_queue.snapshot()
                info.update(
                    {
                        "reward_breakdown": reward.components,
                        "pending_emails": len(observation_after.inbox),
                        "backlog_size": len(observation_after.inbox),
                        "action_trace": self._action_history[-1].model_dump(mode="json"),
                        "seed": self._seed,
                        "scheduled_events": scheduled_events,
                        "triggered_events": triggered_events,
                        "system_state": dict(self._system_state),
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
                "system_state": dict(self._system_state),
                "event_queue": self._engine.event_queue.snapshot(),
                "email_specs": [spec.model_dump(mode="json") for spec in self._email_specs.values()],
                "spawned_email_ids": sorted(self._spawned_email_ids),
                "event_counter": self._event_counter,
            }

    def snapshot(self) -> dict:
        return self.state()

    def restore(self, snapshot: dict) -> None:
        with self._lock:
            self.task = Task(**snapshot["task"])
            self._initial_email_specs = [EmailSpec(**item) for item in self.task.initial_state["emails"]]
            self._email_specs = {
                spec.email_id: spec for spec in [EmailSpec(**item) for item in snapshot.get("email_specs", snapshot["task"]["initial_state"]["emails"])]
            }
            self._seed = int(snapshot["seed"])
            random.seed(self._seed)
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
            self._system_state = dict(snapshot.get("system_state", {"stress": 0.0, "sla_breaches": 0.0}))
            self._engine.event_queue.restore(snapshot.get("event_queue", []))
            self._spawned_email_ids = set(snapshot.get("spawned_email_ids", []))
            self._event_counter = int(snapshot.get("event_counter", 0))
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
            self._observed_view(self._email_specs[email_id])
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
        for spec in sorted(self._email_specs.values(), key=lambda item: (item.arrival_step, item.email_id)):
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
            reward_components["classification"] = 0.35 if action.category == spec.true_category else -0.05
            if action.category != spec.true_category:
                self._register_mistake(spec, "misclassified")
        elif action.action_type == "respond":
            if not spec.requires_response:
                reward_components["unnecessary_response"] = -0.05
            self._responses[spec.email_id] = action.response_template or "none"
            reward_components["response_correctness"] = (
                0.45 if action.response_template == spec.response_template else -0.05
            )
            if self._step_index <= spec.response_deadline:
                reward_components["timeliness"] = reward_components.get("timeliness", 0.0) + 0.2
            if action.response_template != spec.response_template:
                self._register_mistake(spec, "wrong_response")
        elif action.action_type == "escalate":
            self._escalations[spec.email_id] = self._step_index
            should_escalate = spec.requires_escalation or (
                spec.escalation_trigger_step is not None and self._step_index >= spec.escalation_trigger_step
            )
            reward_components["escalation"] = 0.5 if should_escalate else -0.05
            if should_escalate and self._step_index <= spec.escalation_deadline:
                reward_components["timeliness"] = reward_components.get("timeliness", 0.0) + 0.25
            if not should_escalate:
                self._register_mistake(spec, "premature_escalation")
        elif action.action_type == "ignore":
            self._ignored.add(spec.email_id)
            reward_components["ignore"] = 0.1 if spec.true_category == "spam" else -0.05
            if spec.true_category != "spam":
                self._register_mistake(spec, "ignored_important")

        info["resolved_email_id"] = spec.email_id if self._is_terminal_email(spec.email_id) else None
        if self._is_terminal_email(spec.email_id):
            self._completed_ids.append(spec.email_id)
            reward_components["completion"] = reward_components.get("completion", 0.0) + 0.15
        return info

    def _wait_penalty(self) -> float:
        urgent_open = [
            email_id
            for email_id in self._visible_ids
            if email_id not in self._completed_ids
            and self._email_specs[email_id].priority_hint in {"high", "critical"}
        ]
        return -0.08 if urgent_open else 0.0

    def _deadline_penalties(self) -> dict[str, float]:
        penalties: dict[str, float] = {}
        for email_id in self._visible_ids:
            if email_id in self._completed_ids:
                continue
            spec = self._email_specs[email_id]
            if self._step_index > spec.classification_deadline and email_id not in self._classifications:
                penalties["missed_classification"] = penalties.get("missed_classification", 0.0) - 0.1
            if spec.requires_response and self._step_index > spec.response_deadline and email_id not in self._responses:
                penalties["missed_response"] = penalties.get("missed_response", 0.0) - 0.15
            should_escalate = spec.requires_escalation or (
                spec.escalation_trigger_step is not None and self._step_index >= spec.escalation_trigger_step
            )
            if should_escalate and self._step_index > spec.escalation_deadline and email_id not in self._escalations:
                penalties["missed_escalation"] = penalties.get("missed_escalation", 0.0) - 0.2
                self._system_state["sla_breaches"] += 1
                self._system_state["stress"] += 2.0
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
        critical_failure = self._system_state["stress"] >= 40.0 or self._system_state["sla_breaches"] >= 6.0
        return self._step_index >= self.task.max_steps or critical_failure

    def _summarize_action(self, action: Action) -> str:
        if action.action_type == "wait":
            return "waited for more context"
        parts = [action.action_type, action.email_id or ""]
        if action.category:
            parts.append(action.category)
        if action.response_template:
            parts.append(action.response_template)
        return ":".join(parts)

    def _observed_view(self, spec: EmailSpec) -> Any:
        hint = self._observed_priority(spec)
        return spec.to_view(self._step_index, seen=True).model_copy(update={"priority_hint": hint})

    def _observed_priority(self, spec: EmailSpec) -> str:
        subject = spec.subject.lower()
        sender = spec.sender.lower()
        if any(token in subject for token in ("outage", "failing", "global", "urgent")):
            return "high"
        if any(token in sender for token in ("ceo@", "vip@", "noc@")):
            return "high"
        if spec.noise_score > 0.8:
            return "low"
        if any(token in subject for token in ("migration", "review", "status")):
            return "medium"
        return "medium"

    def _process_due_events(self, reward_components: dict[str, float]) -> list[dict[str, Any]]:
        def _handle(event: ScheduledEvent) -> None:
            if event.event_type == "penalty":
                amount = float(event.payload["amount"])
                reward_components[event.payload["reason"]] = reward_components.get(event.payload["reason"], 0.0) + amount
                self._system_state["stress"] += float(event.payload.get("stress", 0.0))
                self._system_state["sla_breaches"] += float(event.payload.get("sla_breach", 0.0))
            elif event.event_type == "escalation":
                reward_components["delayed_escalation"] = reward_components.get("delayed_escalation", 0.0) - 0.6
                self._system_state["stress"] += 4.0
                self._system_state["sla_breaches"] += 1.0
            elif event.event_type == "followup_email":
                self._spawn_followup_email(event.payload)

        return self._engine.process_due_events(self._step_index, _handle)

    def _register_mistake(self, spec: EmailSpec, reason: str) -> None:
        self._system_state["stress"] += 1.5
        if reason == "ignored_important":
            self._schedule_followup_email(
                spec,
                delay=2,
                subject_prefix="Follow-up:",
                body_suffix="Still waiting on a response.",
            )
            self._schedule_penalty(spec, delay=3, reason="missed_important", amount=-0.8, stress=3.0)
            self._schedule_escalation(spec, delay=5)
        elif reason == "misclassified":
            self._schedule_followup_email(
                spec,
                delay=2,
                subject_prefix="Correction:",
                body_suffix="Please re-check the urgency of this issue.",
            )
            self._schedule_penalty(spec, delay=4, reason="misclassification_fallout", amount=-0.7, stress=2.0)
        elif reason == "wrong_response":
            self._schedule_followup_email(
                spec,
                delay=2,
                subject_prefix="Customer reply:",
                body_suffix="That did not address the problem.",
            )
            self._schedule_penalty(spec, delay=3, reason="bad_response_fallout", amount=-0.6, stress=2.0)
        elif reason == "premature_escalation":
            self._schedule_penalty(spec, delay=2, reason="false_alarm", amount=-0.5, stress=1.0)

    def _schedule_penalty(self, spec: EmailSpec, delay: int, reason: str, amount: float, stress: float = 0.0) -> None:
        self._engine.schedule_in(
            current_tick=self._step_index,
            delay=delay,
            event_type="penalty",
            payload={
                "email_id": spec.email_id,
                "reason": reason,
                "amount": amount,
                "stress": stress,
                "sla_breach": 1.0 if "missed" in reason else 0.0,
            },
        )

    def _schedule_escalation(self, spec: EmailSpec, delay: int) -> None:
        self._engine.schedule_in(
            current_tick=self._step_index,
            delay=delay,
            event_type="escalation",
            payload={"email_id": spec.email_id},
            priority=-1,
        )

    def _schedule_followup_email(self, spec: EmailSpec, delay: int, subject_prefix: str, body_suffix: str) -> None:
        followup_id = f"{spec.email_id}-f{self._step_index}-{delay}"
        if followup_id in self._spawned_email_ids:
            return
        self._spawned_email_ids.add(followup_id)
        payload = {
            "email_id": followup_id,
            "sender": spec.sender,
            "subject": f"{subject_prefix} {spec.subject}",
            "body": f"{spec.body} {body_suffix}",
            "thread_id": spec.thread_id,
            "arrival_step": self._step_index + delay,
            "priority_hint": "high" if spec.priority_hint in {"high", "critical"} else "medium",
            "noise_score": max(spec.noise_score - 0.15, 0.01),
            "true_category": "escalation" if spec.priority_hint in {"high", "critical"} else "urgent",
            "response_template": "acknowledge",
            "requires_response": True,
            "requires_escalation": spec.priority_hint in {"high", "critical"},
            "escalation_trigger_step": self._step_index + delay + 1,
            "classification_deadline": self._step_index + delay + 1,
            "response_deadline": self._step_index + delay + 2,
            "escalation_deadline": self._step_index + delay + 4,
        }
        self._engine.schedule_in(
            current_tick=self._step_index,
            delay=delay,
            event_type="followup_email",
            payload=payload,
        )

    def _spawn_followup_email(self, payload: dict[str, Any]) -> None:
        spec = EmailSpec(**payload)
        self._email_specs[spec.email_id] = spec
        if spec.email_id not in self._visible_ids and spec.arrival_step <= self._step_index:
            self._visible_ids.append(spec.email_id)
