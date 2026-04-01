from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from openenv.models import Action, ActionTrace, Observation

PRIORITY_TO_FLOAT = {"low": 0.25, "medium": 0.5, "high": 0.75, "critical": 1.0}
CATEGORY_ORDER = ["spam", "urgent", "normal", "escalation"]
RESPONSE_ORDER = ["acknowledge", "resolve", "request_info", "escalate_notice"]
ESCALATE_PRIORITIES = ["medium", "high", "critical"]
SLOT_ACTION_WIDTH = 12
GLOBAL_FEATURE_LABELS = (
    "step_ratio",
    "remaining_ratio",
    "inbox_ratio",
    "completed_ratio",
    "history_ratio",
    "has_inbox",
    "max_priority",
    "mean_noise",
    "stress_ratio",
    "sla_breach_ratio",
    "actionable_ratio",
    "pending_event_ratio",
)
PER_EMAIL_FEATURE_LABELS = (
    "age_ratio",
    "noise_score",
    "observed_priority",
    "seen_flag",
    "incident_signal",
    "promo_signal",
    "ops_sender_signal",
    "thread_signal",
    "classified_flag",
    "responded_flag",
    "escalated_flag",
    "ignored_flag",
)
GLOBAL_FEATURE_COUNT = len(GLOBAL_FEATURE_LABELS)
PER_EMAIL_FEATURE_COUNT = len(PER_EMAIL_FEATURE_LABELS)


@dataclass(slots=True)
class EmailActionStatus:
    classified: bool = False
    responded: bool = False
    escalated: bool = False
    ignored: bool = False

    def progressed(self) -> bool:
        return self.classified or self.responded or self.escalated or self.ignored


@dataclass(slots=True)
class EncoderContext:
    stress: float = 0.0
    sla_breaches: float = 0.0
    open_actionable_emails: int = 0
    pending_event_count: int = 0
    progress_by_email: dict[str, EmailActionStatus] = field(default_factory=dict)


def action_progress_by_email(action_history: list[ActionTrace]) -> dict[str, EmailActionStatus]:
    progress: dict[str, EmailActionStatus] = {}
    for trace in action_history:
        if trace.email_id is None:
            continue
        status = progress.setdefault(trace.email_id, EmailActionStatus())
        if trace.action_type == "classify":
            status.classified = True
        elif trace.action_type == "respond":
            status.responded = True
        elif trace.action_type == "escalate":
            status.escalated = True
        elif trace.action_type == "ignore":
            status.ignored = True
    return progress


@dataclass(slots=True)
class DiscreteActionCodec:
    max_inbox_size: int = 12
    wait_action: int = 0

    @property
    def action_count(self) -> int:
        return 1 + (self.max_inbox_size * SLOT_ACTION_WIDTH)

    def encode_mask(self, observation: Observation, context: EncoderContext | None = None) -> np.ndarray:
        context = self._normalized_context(observation, context)
        mask = np.zeros(self.action_count, dtype=np.int8)
        mask[self.wait_action] = 1
        visible_slots = min(len(observation.inbox), self.max_inbox_size)
        for slot in range(visible_slots):
            start = 1 + (slot * SLOT_ACTION_WIDTH)
            email = observation.inbox[slot]
            status = context.progress_by_email.get(email.email_id, EmailActionStatus())
            if not status.ignored and not (status.responded and status.escalated):
                mask[start] = 1
            if not status.classified and not status.ignored:
                mask[start + 1 : start + 5] = 1
            if not status.responded and not status.ignored:
                mask[start + 5 : start + 9] = 1
            if not status.escalated and not status.ignored:
                mask[start + 9 + self._preferred_escalation_offset(email)] = 1
        return mask

    def decode(self, action_id: int, observation: Observation) -> Action:
        if action_id <= 0:
            return Action(action_type="wait")
        slot = (action_id - 1) // SLOT_ACTION_WIDTH
        offset = (action_id - 1) % SLOT_ACTION_WIDTH
        if slot >= len(observation.inbox) or slot >= self.max_inbox_size:
            return Action(action_type="wait")

        email = observation.inbox[slot]
        if offset == 0:
            return Action(action_type="ignore", email_id=email.email_id)
        if 1 <= offset <= 4:
            return Action(
                action_type="classify",
                email_id=email.email_id,
                category=CATEGORY_ORDER[offset - 1],
            )
        if 5 <= offset <= 8:
            return Action(
                action_type="respond",
                email_id=email.email_id,
                response_template=RESPONSE_ORDER[offset - 5],
                priority=email.priority_hint,
            )
        return Action(
            action_type="escalate",
            email_id=email.email_id,
            priority=ESCALATE_PRIORITIES[offset - 9],
        )

    def _preferred_escalation_offset(self, email: object) -> int:
        priority_hint = getattr(email, "priority_hint", "medium")
        if priority_hint == "critical":
            return 2
        if priority_hint == "high":
            return 1
        return 0

    def _normalized_context(
        self, observation: Observation, context: EncoderContext | None
    ) -> EncoderContext:
        if context is None:
            return EncoderContext(progress_by_email=action_progress_by_email(observation.action_history))
        if context.progress_by_email:
            return context
        return EncoderContext(
            stress=context.stress,
            sla_breaches=context.sla_breaches,
            open_actionable_emails=context.open_actionable_emails,
            pending_event_count=context.pending_event_count,
            progress_by_email=action_progress_by_email(observation.action_history),
        )


@dataclass(slots=True)
class ObservationEncoder:
    max_inbox_size: int = 12
    max_age: float = 20.0
    max_stress: float = 100.0
    max_sla_breaches: float = 8.0
    max_pending_events: float = 16.0
    incident_keywords: tuple[str, ...] = ("outage", "timeout", "failing", "checkout", "payroll")
    promo_keywords: tuple[str, ...] = ("newsletter", "unsubscribe", "offer", "webinar", "sponsorship")
    ops_senders: tuple[str, ...] = ("ceo@", "vip@", "noc@", "support@", "security@", "ops@")

    @property
    def feature_count(self) -> int:
        return GLOBAL_FEATURE_COUNT + (self.max_inbox_size * PER_EMAIL_FEATURE_COUNT)

    @property
    def global_feature_labels(self) -> tuple[str, ...]:
        return GLOBAL_FEATURE_LABELS

    @property
    def per_email_feature_labels(self) -> tuple[str, ...]:
        return PER_EMAIL_FEATURE_LABELS

    def encode(self, observation: Observation, context: EncoderContext | None = None) -> np.ndarray:
        context = self._normalized_context(observation, context)
        global_features = np.array(
            [
                min(observation.step_index / max(observation.max_steps, 1), 1.0),
                min(observation.remaining_steps / max(observation.max_steps, 1), 1.0),
                min(len(observation.inbox) / max(self.max_inbox_size, 1), 1.0),
                min(len(observation.completed_email_ids) / max(observation.max_steps, 1), 1.0),
                min(len(observation.action_history) / max(observation.max_steps, 1), 1.0),
                1.0 if observation.inbox else 0.0,
                max((PRIORITY_TO_FLOAT[email.priority_hint] for email in observation.inbox), default=0.0),
                float(sum(email.noise_score for email in observation.inbox) / max(len(observation.inbox), 1)),
                min(context.stress / max(self.max_stress, 1.0), 1.0),
                min(context.sla_breaches / max(self.max_sla_breaches, 1.0), 1.0),
                min(context.open_actionable_emails / max(self.max_inbox_size, 1), 1.0),
                min(context.pending_event_count / max(self.max_pending_events, 1.0), 1.0),
            ],
            dtype=np.float32,
        )

        encoded_emails: list[float] = []
        for email in observation.inbox[: self.max_inbox_size]:
            text = f"{email.subject} {email.body}".lower()
            sender = email.sender.lower()
            status = context.progress_by_email.get(email.email_id, EmailActionStatus())
            encoded_emails.extend(
                [
                    min(email.age / self.max_age, 1.0),
                    float(email.noise_score),
                    PRIORITY_TO_FLOAT[email.priority_hint],
                    1.0 if email.seen else 0.0,
                    1.0 if any(token in text for token in self.incident_keywords) else 0.0,
                    1.0 if any(token in text for token in self.promo_keywords) else 0.0,
                    1.0 if any(token in sender for token in self.ops_senders) else 0.0,
                    1.0 if "thread" in email.thread_id or "-" in email.thread_id else 0.0,
                    1.0 if status.classified else 0.0,
                    1.0 if status.responded else 0.0,
                    1.0 if status.escalated else 0.0,
                    1.0 if status.ignored else 0.0,
                ]
            )

        missing_slots = self.max_inbox_size - min(len(observation.inbox), self.max_inbox_size)
        if missing_slots > 0:
            encoded_emails.extend([0.0] * (missing_slots * PER_EMAIL_FEATURE_COUNT))

        return np.concatenate([global_features, np.asarray(encoded_emails, dtype=np.float32)])

    def _normalized_context(
        self, observation: Observation, context: EncoderContext | None
    ) -> EncoderContext:
        if context is None:
            return EncoderContext(
                open_actionable_emails=len(observation.inbox),
                progress_by_email=action_progress_by_email(observation.action_history),
            )
        if context.progress_by_email:
            return context
        return EncoderContext(
            stress=context.stress,
            sla_breaches=context.sla_breaches,
            open_actionable_emails=context.open_actionable_emails or len(observation.inbox),
            pending_event_count=context.pending_event_count,
            progress_by_email=action_progress_by_email(observation.action_history),
        )


@dataclass(slots=True)
class TextTrajectoryFormatter:
    max_emails: int = 12

    def render(self, observation: Observation, context: EncoderContext | None = None) -> str:
        context = context or EncoderContext(
            open_actionable_emails=len(observation.inbox),
            progress_by_email=action_progress_by_email(observation.action_history),
        )
        lines = [
            f"task={observation.task_name}",
            f"step={observation.step_index}/{observation.max_steps}",
            f"remaining={observation.remaining_steps}",
            f"inbox={len(observation.inbox)}",
            f"stress={context.stress:.1f}",
            f"sla_breaches={context.sla_breaches:.1f}",
            f"actionable={context.open_actionable_emails}",
            f"pending_events={context.pending_event_count}",
        ]
        for index, email in enumerate(observation.inbox[: self.max_emails]):
            status = context.progress_by_email.get(email.email_id, EmailActionStatus())
            lines.append(
                f"[{index}] sender={email.sender} priority={email.priority_hint} age={email.age} "
                f"progress={self._progress_summary(status)} subject={email.subject}"
            )
        return "\n".join(lines)

    def _progress_summary(self, status: EmailActionStatus) -> str:
        flags: list[str] = []
        if status.classified:
            flags.append("classified")
        if status.responded:
            flags.append("responded")
        if status.escalated:
            flags.append("escalated")
        if status.ignored:
            flags.append("ignored")
        return ",".join(flags) if flags else "none"
