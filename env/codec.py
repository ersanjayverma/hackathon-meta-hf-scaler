from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from openenv.models import Action, Observation

PRIORITY_TO_FLOAT = {"low": 0.25, "medium": 0.5, "high": 0.75, "critical": 1.0}
CATEGORY_ORDER = ["spam", "urgent", "normal", "escalation"]
RESPONSE_ORDER = ["acknowledge", "resolve", "request_info", "escalate_notice"]
ESCALATE_PRIORITIES = ["medium", "high", "critical"]
SLOT_ACTION_WIDTH = 12
GLOBAL_FEATURE_COUNT = 8
PER_EMAIL_FEATURE_COUNT = 8


@dataclass(slots=True)
class DiscreteActionCodec:
    max_inbox_size: int = 12
    wait_action: int = 0

    @property
    def action_count(self) -> int:
        return 1 + (self.max_inbox_size * SLOT_ACTION_WIDTH)

    def encode_mask(self, observation: Observation) -> np.ndarray:
        mask = np.zeros(self.action_count, dtype=np.int8)
        mask[self.wait_action] = 1
        visible_slots = min(len(observation.inbox), self.max_inbox_size)
        for slot in range(visible_slots):
            start = 1 + (slot * SLOT_ACTION_WIDTH)
            mask[start : start + SLOT_ACTION_WIDTH] = 1
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


@dataclass(slots=True)
class ObservationEncoder:
    max_inbox_size: int = 12
    max_age: float = 20.0
    incident_keywords: tuple[str, ...] = ("outage", "timeout", "failing", "checkout", "payroll")
    promo_keywords: tuple[str, ...] = ("newsletter", "unsubscribe", "offer", "webinar", "sponsorship")
    ops_senders: tuple[str, ...] = ("ceo@", "vip@", "noc@", "support@", "security@", "ops@")

    @property
    def feature_count(self) -> int:
        return GLOBAL_FEATURE_COUNT + (self.max_inbox_size * PER_EMAIL_FEATURE_COUNT)

    def encode(self, observation: Observation) -> np.ndarray:
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
            ],
            dtype=np.float32,
        )

        encoded_emails: list[float] = []
        for email in observation.inbox[: self.max_inbox_size]:
            text = f"{email.subject} {email.body}".lower()
            sender = email.sender.lower()
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
                ]
            )

        missing_slots = self.max_inbox_size - min(len(observation.inbox), self.max_inbox_size)
        if missing_slots > 0:
            encoded_emails.extend([0.0] * (missing_slots * PER_EMAIL_FEATURE_COUNT))

        return np.concatenate([global_features, np.asarray(encoded_emails, dtype=np.float32)])


@dataclass(slots=True)
class TextTrajectoryFormatter:
    max_emails: int = 12

    def render(self, observation: Observation) -> str:
        lines = [
            f"task={observation.task_name}",
            f"step={observation.step_index}/{observation.max_steps}",
            f"remaining={observation.remaining_steps}",
            f"inbox={len(observation.inbox)}",
        ]
        for index, email in enumerate(observation.inbox[: self.max_emails]):
            lines.append(
                f"[{index}] sender={email.sender} priority={email.priority_hint} age={email.age} subject={email.subject}"
            )
        return "\n".join(lines)
