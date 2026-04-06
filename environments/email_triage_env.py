from __future__ import annotations

import random
from typing import Any, Optional

import numpy as np

from openenv.base_env import BaseEnv
from openenv.config import EMAIL_TRIAGE_CONFIG
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
        self._action_repeat_count: int = 0
        self._cumulative_reward: float = 0.0
        self._system_state: dict[str, float] = {"stress": 0.0, "sla_breaches": 0.0}
        self._event_counter = 0
        self._spawned_email_ids: set[str] = set()
        self._overload_triggered = False
        self._overload_level = 0

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
            self._email_specs = {}
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
            self._action_repeat_count = 0
            self._cumulative_reward = 0.0
            self._system_state = {"stress": 0.0, "sla_breaches": 0.0}
            self._event_counter = 0
            self._spawned_email_ids = set()
            self._overload_triggered = False
            self._overload_level = 0
            self._engine.event_queue.restore([])
            self._episode_recorder = EpisodeRecorder(
                environment_name=self.__class__.__name__,
                seed=self._seed,
                config={"task": self.task.model_dump(mode="json")},
            )
            for spec in self._initial_email_specs:
                self._register_email(spec)
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
                    self._action_repeat_count += 1
                    context_multiplier = self._loop_context_multiplier(action)
                    scaled_penalty = EMAIL_TRIAGE_CONFIG.loop_penalty * (
                        1.0 + self._action_repeat_count * EMAIL_TRIAGE_CONFIG.repetition_decay
                    ) * context_multiplier
                    reward_components["loop_penalty"] = max(scaled_penalty, EMAIL_TRIAGE_CONFIG.reward_floor)
                    info["loop_detected"] = True
                    info["loop_repeat_count"] = self._action_repeat_count
                else:
                    self._action_repeat_count = 0
                self._last_action_key = loop_key

                if action.action_type == "wait":
                    reward_components["wait_penalty"] = self._wait_penalty()
                else:
                    info.update(self._apply_action(action, reward_components))

                reward_components.update(self._deadline_penalties())
                reward = self._build_reward(reward_components)

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
                termination_reason = self._termination_reason()
                done = termination_reason is not None
                observation_after = self._observation()
                info.update(self._build_step_info(observation_after, reward, triggered_events, termination_reason))
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
            processed_email_ids = set(self._completed_ids)
            all_email_ids = set(self._email_specs)
            return {
                "processed_email_ids": list(processed_email_ids),
                "remaining_email_ids": sorted(all_email_ids - processed_email_ids),
                "steps_taken": self._step_index,
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
                "overload_triggered": self._overload_triggered,
                "overload_level": self._overload_level,
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
            self._overload_triggered = bool(snapshot.get("overload_triggered", False))
            self._overload_level = int(snapshot.get("overload_level", 0))
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
        inbox = self._build_inbox()
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

    def _register_email(self, spec: EmailSpec) -> None:
        deadline_step = spec.deadline_step
        if deadline_step is None:
            deadline_step = spec.arrival_step + random.randint(3, 7)
            spec = spec.model_copy(update={"deadline_step": deadline_step})
        self._email_specs[spec.email_id] = spec
        self._schedule_sla_breach(spec)

    def _build_inbox(self) -> list[Any]:
        return [
            self._observed_view(self._email_specs[email_id])
            for email_id in self._visible_ids
            if email_id not in self._completed_ids
        ]

    def _build_reward(self, reward_components: dict[str, float]) -> Reward:
        components = dict(reward_components)
        components["stress_penalty"] = -(self._system_state["stress"] * EMAIL_TRIAGE_CONFIG.stress_penalty_scale)
        components["sla_pressure_penalty"] = -(
            self._system_state["sla_breaches"] * EMAIL_TRIAGE_CONFIG.sla_pressure_penalty_scale
        )
        if self._system_state["stress"] > EMAIL_TRIAGE_CONFIG.system_collapse_stress:
            components["system_collapse"] = EMAIL_TRIAGE_CONFIG.system_collapse_penalty
        raw_total = sum(components.values())
        clamped_total = max(min(raw_total, EMAIL_TRIAGE_CONFIG.reward_ceiling), EMAIL_TRIAGE_CONFIG.reward_floor)
        if abs(clamped_total - raw_total) > 1e-9:
            components["reward_clamp"] = clamped_total - raw_total
        self._cumulative_reward += clamped_total
        return Reward(
            total=clamped_total,
            components=components,
            reason="dense trajectory-aware reward",
        )

    def _build_step_info(
        self,
        observation: Observation,
        reward: Reward,
        triggered_events: list[dict[str, Any]],
        termination_reason: str | None,
    ) -> dict[str, Any]:
        info = {
            "reward_breakdown": reward.components,
            "pending_emails": len(observation.inbox),
            "backlog_size": len(observation.inbox),
            "action_trace": self._action_history[-1].model_dump(mode="json"),
            "seed": self._seed,
            "scheduled_events": self._engine.event_queue.snapshot(),
            "triggered_events": triggered_events,
            "system_state": dict(self._system_state),
            "events_triggered": triggered_events,
            "emails_pending": len(observation.inbox),
            "step": self._step_index,
            "sla_breaches": self._system_state["sla_breaches"],
            "stress": self._system_state["stress"],
            "open_actionable_emails": len(self._pending_email_ids()),
        }
        if termination_reason is not None:
            info["termination_reason"] = termination_reason
            info["termination_diagnostics"] = self._termination_diagnostics(termination_reason)
        return info

    def _termination_diagnostics(self, reason: str) -> dict[str, Any]:
        diag: dict[str, Any] = {"rule": reason}
        if reason == "system_collapse":
            diag["stress"] = self._system_state["stress"]
            diag["threshold"] = EMAIL_TRIAGE_CONFIG.system_collapse_stress
        elif reason == "cumulative_failure":
            diag["cumulative_reward"] = self._cumulative_reward
            diag["threshold"] = EMAIL_TRIAGE_CONFIG.cumulative_reward_floor
        elif reason == "failure_collapse":
            window = EMAIL_TRIAGE_CONFIG.failure_collapse_window
            diag["window"] = window
            diag["last_n_rewards"] = [s.reward.total for s in self._trajectory[-window:]]
            diag["condition"] = "all(r < 0.0)"
        elif reason == "stable_resolution":
            diag["pending_emails"] = len(self._pending_email_ids())
            diag["future_arrivals"] = sum(
                1 for s in self._email_specs.values() if s.arrival_step > self._step_index
            )
        elif reason == "max_steps":
            diag["step_index"] = self._step_index
            diag["max_steps"] = self.task.max_steps
        return diag

    def _apply_action(self, action: Action, reward_components: dict[str, float]) -> dict[str, Any]:
        info: dict[str, Any] = {}
        resolved_cleanly = True
        if action.email_id not in self._visible_ids or action.email_id in self._completed_ids:
            reward_components["invalid_target"] = -0.5
            info["invalid_target"] = True
            return info

        spec = self._email_specs[action.email_id]
        urgency_weight = self._urgency_weight(spec)
        sla_bonus = self._sla_proximity_bonus(spec)

        if action.action_type == "classify":
            if self._classifications.get(spec.email_id) is not None:
                reward_components["redundant_action"] = -0.2
            self._classifications[spec.email_id] = action.category or ""
            if action.category == spec.true_category:
                reward_components["classification"] = 1.0
                reward_components["sla_proximity"] = sla_bonus
            elif action.category is not None and self._is_adjacent_category(action.category, spec.true_category):
                reward_components["classification"] = 0.3
                resolved_cleanly = False
                self._register_mistake(spec, "misclassified")
            elif action.category is not None and self._is_harmful_misclass(action.category, spec.true_category):
                reward_components["classification"] = -1.0 * urgency_weight
                resolved_cleanly = False
                self._register_mistake(spec, "misclassified")
            else:
                reward_components["classification"] = -0.5 * urgency_weight
                resolved_cleanly = False
                self._register_mistake(spec, "misclassified")
        elif action.action_type == "respond":
            if not spec.requires_response:
                reward_components["unnecessary_response"] = -0.3
                resolved_cleanly = False
            self._responses[spec.email_id] = action.response_template or "none"
            if action.response_template == spec.response_template:
                reward_components["response_correctness"] = 1.0
                reward_components["sla_proximity"] = sla_bonus
            elif action.response_template in ("acknowledge", "request_info"):
                reward_components["response_correctness"] = 0.3
            else:
                reward_components["response_correctness"] = -0.5 * urgency_weight
                resolved_cleanly = False
                self._register_mistake(spec, "wrong_response")
            if self._step_index <= spec.response_deadline:
                reward_components["timeliness"] = reward_components.get("timeliness", 0.0) + 0.15 * urgency_weight
        elif action.action_type == "escalate":
            self._escalations[spec.email_id] = self._step_index
            should_escalate = spec.requires_escalation or (
                spec.escalation_trigger_step is not None and self._step_index >= spec.escalation_trigger_step
            )
            if should_escalate:
                reward_components["escalation"] = 1.0
                reward_components["sla_proximity"] = sla_bonus
            else:
                reward_components["escalation"] = -1.0 * urgency_weight
                resolved_cleanly = False
                self._register_mistake(spec, "premature_escalation")
            if should_escalate and self._step_index <= spec.escalation_deadline:
                reward_components["timeliness"] = reward_components.get("timeliness", 0.0) + 0.2 * urgency_weight
        elif action.action_type == "ignore":
            self._ignored.add(spec.email_id)
            if spec.true_category == "spam":
                reward_components["ignore"] = 0.3
            elif spec.true_category == "normal":
                reward_components["ignore"] = -0.5 * urgency_weight
                resolved_cleanly = False
                self._register_mistake(spec, "ignored_important")
            else:
                reward_components["ignore"] = -1.0 * urgency_weight
                resolved_cleanly = False
                self._register_mistake(spec, "ignored_important")

        info["resolved_email_id"] = spec.email_id if self._is_terminal_email(spec.email_id) else None
        if self._is_terminal_email(spec.email_id):
            self._completed_ids.append(spec.email_id)
            if resolved_cleanly:
                reward_components["completion"] = reward_components.get("completion", 0.0) + 0.15
        return info

    def _urgency_weight(self, spec: EmailSpec) -> float:
        """Priority-based multiplier: urgent/critical errors hurt more."""
        weights = {"critical": 1.5, "high": 1.3, "medium": 1.0, "low": 0.8}
        return weights.get(spec.priority_hint, 1.0)

    def _sla_proximity_bonus(self, spec: EmailSpec) -> float:
        """Bonus for acting on an email close to its SLA deadline.

        Returns 0.0–0.2: higher when closer to deadline (time pressure reward).
        Returns 0.0 if already past deadline or no deadline.
        """
        deadline = spec.classification_deadline
        if self._step_index > deadline:
            return 0.0
        remaining = deadline - self._step_index
        if remaining <= 0:
            return 0.2
        if remaining == 1:
            return 0.15
        if remaining == 2:
            return 0.05
        return 0.0

    def _is_adjacent_category(self, predicted: str, actual: str) -> bool:
        adjacent = {
            ("normal", "urgent"), ("urgent", "normal"),
            ("urgent", "escalation"), ("escalation", "urgent"),
            ("normal", "escalation"), ("escalation", "normal"),
        }
        return (predicted, actual) in adjacent

    def _is_harmful_misclass(self, predicted: str, actual: str) -> bool:
        return (predicted == "spam" and actual in ("urgent", "escalation")) or \
               (predicted == "normal" and actual == "escalation")

    def _loop_context_multiplier(self, action: Action) -> float:
        """Repeated actions on urgent emails incur harsher penalty."""
        if action.email_id is None or action.email_id not in self._email_specs:
            return 1.0
        spec = self._email_specs[action.email_id]
        base = self._urgency_weight(spec)
        # Repeating wait or ignore on a high-priority email is especially bad
        if action.action_type in ("wait", "ignore") and spec.priority_hint in ("high", "critical"):
            return base * 1.5
        return base

    def _wait_penalty(self) -> float:
        """Context-aware wait penalty: scales with urgency and SLA proximity."""
        penalty = 0.0
        for email_id in self._visible_ids:
            if email_id in self._completed_ids:
                continue
            spec = self._email_specs[email_id]
            sla_remaining = spec.classification_deadline - self._step_index
            if spec.priority_hint in ("high", "critical"):
                # Base urgent penalty, amplified by SLA proximity
                base = EMAIL_TRIAGE_CONFIG.urgent_wait_penalty
                if sla_remaining <= 0:
                    penalty += base * 2.0  # Already breached — urgent
                elif sla_remaining <= 1:
                    penalty += base * 1.5  # About to breach
                else:
                    penalty += base
            elif sla_remaining <= 0:
                # Non-urgent but past deadline
                penalty += EMAIL_TRIAGE_CONFIG.urgent_wait_penalty * 0.5
        return max(penalty, EMAIL_TRIAGE_CONFIG.reward_floor)

    def _deadline_penalties(self) -> dict[str, float]:
        penalties: dict[str, float] = {}
        for email_id in self._visible_ids:
            if email_id in self._completed_ids:
                continue
            spec = self._email_specs[email_id]
            if self._step_index > spec.classification_deadline and email_id not in self._classifications:
                penalties["missed_classification"] = (
                    penalties.get("missed_classification", 0.0) + EMAIL_TRIAGE_CONFIG.missed_classification_penalty
                )
            if spec.requires_response and self._step_index > spec.response_deadline and email_id not in self._responses:
                penalties["missed_response"] = (
                    penalties.get("missed_response", 0.0) + EMAIL_TRIAGE_CONFIG.missed_response_penalty
                )
            should_escalate = spec.requires_escalation or (
                spec.escalation_trigger_step is not None and self._step_index >= spec.escalation_trigger_step
            )
            if should_escalate and self._step_index > spec.escalation_deadline and email_id not in self._escalations:
                penalties["missed_escalation"] = (
                    penalties.get("missed_escalation", 0.0) + EMAIL_TRIAGE_CONFIG.missed_escalation_penalty
                )
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

    def _termination_reason(self) -> str | None:
        # RULE 1: system stress exceeds hard ceiling → immediate halt
        if self._system_state["stress"] > EMAIL_TRIAGE_CONFIG.system_collapse_stress:
            return "system_collapse"

        # RULE 2: cumulative reward below floor → agent is net-destructive
        if self._cumulative_reward < EMAIL_TRIAGE_CONFIG.cumulative_reward_floor:
            return "cumulative_failure"

        # RULE 3: N consecutive negative rewards → agent is in death spiral
        # Deterministic: exactly failure_collapse_window (3) consecutive r < 0 → done
        window = EMAIL_TRIAGE_CONFIG.failure_collapse_window
        if len(self._trajectory) >= window:
            last_n = self._trajectory[-window:]
            if all(s.reward.total < 0.0 for s in last_n):
                return "failure_collapse"

        # RULE 4: all emails resolved + no future arrivals + no pending events → done
        if EMAIL_TRIAGE_CONFIG.stable_resolution_ends_episode and self._is_stably_resolved():
            return "stable_resolution"

        # RULE 5: hard budget
        if self._step_index >= self.task.max_steps:
            return "max_steps"

        return None

    def _is_stably_resolved(self) -> bool:
        if self._step_index == 0:
            return False
        if self._pending_email_ids():
            return False
        if any(spec.arrival_step > self._step_index for spec in self._email_specs.values()):
            return False
        return not self._has_meaningful_future_events()

    def _pending_email_ids(self) -> list[str]:
        return [email_id for email_id in self._visible_ids if email_id not in self._completed_ids]

    def _has_meaningful_future_events(self) -> bool:
        unresolved = set(self._pending_email_ids())
        for event in self._engine.event_queue.snapshot():
            if event["event_type"] == "system_overload":
                return True
            email_id = event["payload"].get("email_id")
            if email_id is None:
                continue
            if email_id in unresolved:
                return True
        return False

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
        observed_noise = self._observed_noise(spec)
        return spec.to_view(self._step_index, seen=True).model_copy(
            update={"priority_hint": hint, "noise_score": observed_noise}
        )

    def _observed_priority(self, spec: EmailSpec) -> str:
        subject = spec.subject.lower()
        sender = spec.sender.lower()
        if any(token in subject for token in ("checkout", "all regions", "global", "production outage")):
            return "critical"
        if any(token in subject for token in ("outage", "failing", "timeout", "urgent")):
            return "high"
        if any(token in sender for token in ("ceo@", "vip@", "noc@")):
            return "high"
        if spec.noise_score > 0.85:
            return "low"
        if any(token in subject for token in ("migration", "review", "status")):
            return "medium"
        return "medium"

    def _observed_noise(self, spec: EmailSpec) -> float:
        text = f"{spec.subject} {spec.body}".lower()
        if any(token in text for token in ("newsletter", "webinar", "unsubscribe", "offer", "sponsorship")):
            return 0.9
        if any(token in text for token in ("outage", "timeout", "checkout", "payroll", "migration")):
            return 0.2
        return 0.5

    def _process_due_events(self, reward_components: dict[str, float]) -> list[dict[str, Any]]:
        def _handle(event: ScheduledEvent) -> None:
            if event.event_type == "penalty":
                amount = float(event.payload["amount"])
                reward_components[event.payload["reason"]] = reward_components.get(event.payload["reason"], 0.0) + amount
                self._system_state["stress"] += float(event.payload.get("stress", 0.0))
                self._system_state["sla_breaches"] += float(event.payload.get("sla_breach", 0.0))
            elif event.event_type == "escalation":
                reward_components["delayed_escalation"] = reward_components.get("delayed_escalation", 0.0) - 0.2
                self._system_state["stress"] += 1.5
                self._system_state["sla_breaches"] += 0.5
            elif event.event_type == "followup_email":
                self._spawn_followup_email(event.payload)
            elif event.event_type == "sla_breach":
                email_id = str(event.payload["email_id"])
                if email_id not in self._completed_ids and email_id in self._email_specs:
                    reward_components["sla_breach"] = reward_components.get("sla_breach", 0.0) - 0.5
                    self._system_state["sla_breaches"] += 1.0
                    self._system_state["stress"] += 3.0
                    self._maybe_schedule_system_overload()
            elif event.event_type == "system_overload":
                reward_components["system_overload"] = reward_components.get("system_overload", 0.0) - 0.3
                self._system_state["stress"] += 5.0
                self._overload_level += 1
                self._spawn_overload_emails()

        return self._engine.process_due_events(self._step_index, _handle)

    def _register_mistake(self, spec: EmailSpec, reason: str) -> None:
        self._system_state["stress"] += 0.5
        if reason == "ignored_important":
            self._schedule_followup_email(
                spec,
                delay=2,
                subject_prefix="Follow-up:",
                body_suffix="Still waiting on a response.",
            )
            self._schedule_penalty(spec, delay=3, reason="missed_important", amount=-0.3, stress=1.0)
            self._schedule_escalation(spec, delay=5)
        elif reason == "misclassified":
            self._schedule_followup_email(
                spec,
                delay=2,
                subject_prefix="Correction:",
                body_suffix="Please re-check the urgency of this issue.",
            )
            self._schedule_penalty(spec, delay=4, reason="misclassification_fallout", amount=-0.25, stress=0.8)
        elif reason == "wrong_response":
            self._schedule_followup_email(
                spec,
                delay=2,
                subject_prefix="Customer reply:",
                body_suffix="That did not address the problem.",
            )
            self._schedule_penalty(spec, delay=3, reason="bad_response_fallout", amount=-0.2, stress=0.8)
        elif reason == "premature_escalation":
            self._schedule_penalty(spec, delay=2, reason="false_alarm", amount=-0.15, stress=0.5)

    def _schedule_penalty(self, spec: EmailSpec, delay: int, reason: str, amount: float, stress: float = 0.0) -> None:
        self._schedule_relative_event(
            delay=delay,
            event_type="penalty",
            payload={
                "email_id": spec.email_id,
                "reason": reason,
                "amount": amount,
                "stress": stress,
                "sla_breach": 0.0,
            },
        )

    def _schedule_sla_breach(self, spec: EmailSpec) -> None:
        if spec.deadline_step is None:
            return
        self._schedule_absolute_event(
            tick=spec.deadline_step,
            event_type="sla_breach",
            payload={"email_id": spec.email_id},
        )

    def _schedule_escalation(self, spec: EmailSpec, delay: int) -> None:
        self._schedule_relative_event(
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
            "noise_score": min(max(spec.noise_score - 0.15 + (self._overload_level * 0.05), 0.01), 1.0),
            "true_category": "escalation" if spec.priority_hint in {"high", "critical"} else "urgent",
            "response_template": "acknowledge",
            "requires_response": True,
            "requires_escalation": spec.priority_hint in {"high", "critical"},
            "escalation_trigger_step": self._step_index + delay + 1,
            "classification_deadline": self._step_index + delay + 1,
            "response_deadline": self._step_index + delay + 2,
            "escalation_deadline": self._step_index + delay + 4,
            "deadline_step": self._step_index + delay + random.randint(3, 7),
        }
        self._schedule_relative_event(
            delay=delay,
            event_type="followup_email",
            payload=payload,
        )

    def _spawn_followup_email(self, payload: dict[str, Any]) -> None:
        spec = EmailSpec(**payload)
        self._register_email(spec)
        if spec.email_id not in self._visible_ids and spec.arrival_step <= self._step_index:
            self._visible_ids.append(spec.email_id)

    def _maybe_schedule_system_overload(self) -> None:
        if self._system_state["sla_breaches"] >= 3.0 and not self._overload_triggered:
            self._overload_triggered = True
            self._schedule_relative_event(
                delay=1,
                event_type="system_overload",
                payload={},
                priority=-2,
            )

    def _spawn_overload_emails(self) -> None:
        base_step = self._step_index + 1
        for offset in range(2):
            email_id = f"overload-{self._step_index}-{offset}"
            if email_id in self._spawned_email_ids:
                continue
            self._spawned_email_ids.add(email_id)
            noise = min(0.35 + (self._overload_level * 0.12) + (offset * 0.08), 0.98)
            spec = EmailSpec(
                email_id=email_id,
                sender="ops-overload@system.example",
                subject=f"Queue pressure alert {self._overload_level}-{offset}",
                body="Backlog has increased and customers are reporting degraded service.",
                thread_id=f"overload-thread-{self._overload_level}",
                arrival_step=base_step + offset,
                priority_hint="high",
                noise_score=noise,
                true_category="urgent" if offset == 0 else "escalation",
                response_template="acknowledge",
                requires_response=True,
                requires_escalation=offset == 1,
                escalation_trigger_step=base_step + offset + 1,
                classification_deadline=base_step + offset + 1,
                response_deadline=base_step + offset + 2,
                escalation_deadline=base_step + offset + 3,
                deadline_step=base_step + offset + random.randint(3, 7),
            )
            self._register_email(spec)

    def _schedule_relative_event(
        self,
        *,
        delay: int,
        event_type: str,
        payload: dict[str, Any],
        priority: int = 0,
    ) -> None:
        self._schedule_absolute_event(
            tick=self._step_index + delay,
            event_type=event_type,
            payload=payload,
            priority=priority,
        )

    def _schedule_absolute_event(
        self,
        *,
        tick: int,
        event_type: str,
        payload: dict[str, Any],
        priority: int = 0,
    ) -> None:
        self._event_counter += 1
        self._engine.event_queue.schedule(
            ScheduledEvent(
                tick=tick,
                priority=priority,
                event_type=event_type,
                payload=payload,
            )
        )
