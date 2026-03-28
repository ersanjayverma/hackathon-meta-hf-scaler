from __future__ import annotations

from dataclasses import dataclass

import torch

from openenv.models import Action, Observation


@dataclass(slots=True)
class HeuristicAgent:
    def act(self, observation: Observation) -> Action:
        if not observation.inbox:
            return Action(action_type="wait")

        priority_values = {"critical": 4.0, "high": 3.0, "medium": 2.0, "low": 1.0}
        inbox = list(observation.inbox)
        priority_scores = torch.tensor([priority_values[email.priority_hint] for email in inbox], dtype=torch.float32)
        age_scores = torch.tensor([float(email.age) for email in inbox], dtype=torch.float32)
        noise_scores = torch.tensor([float(email.noise_score) for email in inbox], dtype=torch.float32)

        # Use a small torch-based ranking pass to prioritize urgent, older, low-noise emails.
        ranking_scores = priority_scores * 3.0 + age_scores * 0.25 - noise_scores
        email = inbox[int(torch.argmax(ranking_scores).item())]
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
