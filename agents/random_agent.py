from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from openenv.models import Action, Observation


@dataclass(slots=True)
class RandomAgent:
    seed: int | None = None

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed)

    def act(self, observation: Observation) -> Action:
        if not observation.inbox or self._rng.random() < 0.2:
            return Action(action_type="wait")
        email = observation.inbox[int(self._rng.integers(0, len(observation.inbox)))]
        action_type = self._rng.choice(["classify", "respond", "escalate", "ignore"])
        if action_type == "classify":
            return Action(
                action_type="classify",
                email_id=email.email_id,
                category=str(self._rng.choice(["spam", "urgent", "normal", "escalation"])),
            )
        if action_type == "respond":
            return Action(
                action_type="respond",
                email_id=email.email_id,
                response_template=str(
                    self._rng.choice(["acknowledge", "resolve", "request_info", "escalate_notice"])
                ),
                priority=email.priority_hint,
            )
        if action_type == "escalate":
            return Action(
                action_type="escalate",
                email_id=email.email_id,
                priority="critical",
            )
        return Action(action_type="ignore", email_id=email.email_id)
