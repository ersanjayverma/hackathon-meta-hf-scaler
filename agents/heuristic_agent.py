from __future__ import annotations

from dataclasses import dataclass

from openenv.models import Action, Observation


@dataclass(slots=True)
class HeuristicAgent:
    def act(self, observation: Observation) -> Action:
        if not observation.inbox:
            return Action(action_type="wait")

        inbox = sorted(
            observation.inbox,
            key=lambda email: (
                {"critical": 0, "high": 1, "medium": 2, "low": 3}[email.priority_hint],
                -email.age,
            ),
        )
        email = inbox[0]
        text = f"{email.subject} {email.body}".lower()

        if any(token in text for token in ("unsubscribe", "offer", "webinar", "newsletter")):
            if "spam" not in {trace.summary.split(":")[-1] for trace in observation.action_history if trace.email_id == email.email_id}:
                return Action(action_type="classify", email_id=email.email_id, category="spam")
            return Action(action_type="ignore", email_id=email.email_id)

        if any(token in text for token in ("outage", "failing", "timeout", "checkout")):
            if "classify" not in {trace.action_type for trace in observation.action_history if trace.email_id == email.email_id}:
                category = "escalation" if "global" in text or "checkout" in text else "urgent"
                return Action(action_type="classify", email_id=email.email_id, category=category)
            if "global" in text or "checkout" in text:
                if "escalate" not in {trace.action_type for trace in observation.action_history if trace.email_id == email.email_id}:
                    return Action(action_type="escalate", email_id=email.email_id, priority="critical")
                return Action(
                    action_type="respond",
                    email_id=email.email_id,
                    response_template="escalate_notice",
                    priority="critical",
                )
            return Action(
                action_type="respond",
                email_id=email.email_id,
                response_template="acknowledge",
                priority=email.priority_hint,
            )

        if "migration" in text or "review" in text:
            if "classify" not in {trace.action_type for trace in observation.action_history if trace.email_id == email.email_id}:
                return Action(action_type="classify", email_id=email.email_id, category="normal")
            return Action(
                action_type="respond",
                email_id=email.email_id,
                response_template="request_info",
                priority=email.priority_hint,
            )

        return Action(action_type="wait")
