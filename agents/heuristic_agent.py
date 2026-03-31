from __future__ import annotations

from dataclasses import dataclass

from openenv.models import Action, Observation


@dataclass(slots=True)
class HeuristicAgent:
    spam_keywords: tuple[str, ...] = (
        "unsubscribe",
        "webinar",
        "newsletter",
        "offer",
        "discount",
        "promotion",
        "sponsorship",
        "subscribe",
    )
    incident_keywords: tuple[str, ...] = (
        "outage",
        "timeout",
        "failing",
        "failure",
        "incident",
        "blocked",
        "critical",
        "urgent",
    )
    escalation_keywords: tuple[str, ...] = (
        "checkout",
        "global",
        "security",
        "revenue",
        "all regions",
        "escalat",
        "sev1",
        "sev0",
    )
    routine_keywords: tuple[str, ...] = (
        "migration",
        "review",
        "status",
        "checklist",
        "update",
        "share",
        "confirm",
        "please send",
    )
    sender_urgency_keywords: tuple[str, ...] = ("ceo@", "vip@", "noc@", "security@", "support@")
    sender_spam_keywords: tuple[str, ...] = ("promo@", "news@", "events@", "marketing@vendor")

    def act(self, observation: Observation) -> Action:
        if not observation.inbox:
            return Action(action_type="wait")

        ranked_emails = sorted(
            observation.inbox,
            key=lambda email: (-self._email_score(observation, email.email_id), email.age, email.email_id),
        )

        for email in ranked_emails:
            action = self._next_action_for_email(observation, email.email_id)
            if action is not None:
                return action

        fallback_email = ranked_emails[0]
        if not self._was_taken(observation, fallback_email.email_id, "classify"):
            return Action(action_type="classify", email_id=fallback_email.email_id, category=self._default_category(fallback_email))
        if not self._was_taken(observation, fallback_email.email_id, "respond"):
            return Action(
                action_type="respond",
                email_id=fallback_email.email_id,
                response_template=self._default_response(fallback_email),
                priority=self._default_priority(fallback_email),
            )
        return Action(action_type="wait")

    def _next_action_for_email(self, observation: Observation, email_id: str) -> Action | None:
        email = next(item for item in observation.inbox if item.email_id == email_id)
        profile = self._profile(email)

        if profile["spam"]:
            if not self._was_taken(observation, email_id, "classify"):
                return Action(action_type="classify", email_id=email_id, category="spam")
            if not self._was_taken(observation, email_id, "ignore"):
                return Action(action_type="ignore", email_id=email_id)
            return None

        if profile["escalation"]:
            if not self._was_taken(observation, email_id, "classify"):
                return Action(action_type="classify", email_id=email_id, category="escalation")
            if not self._was_taken(observation, email_id, "escalate"):
                return Action(action_type="escalate", email_id=email_id, priority="critical")
            if not self._was_taken(observation, email_id, "respond"):
                return Action(
                    action_type="respond",
                    email_id=email_id,
                    response_template="escalate_notice",
                    priority="critical",
                )
            return None

        if profile["incident"]:
            if not self._was_taken(observation, email_id, "classify"):
                return Action(action_type="classify", email_id=email_id, category="urgent")
            if not self._was_taken(observation, email_id, "respond"):
                return Action(
                    action_type="respond",
                    email_id=email_id,
                    response_template="acknowledge",
                    priority=self._default_priority(email),
                )
            return None

        if profile["routine"]:
            if not self._was_taken(observation, email_id, "classify"):
                return Action(action_type="classify", email_id=email_id, category="normal")
            if not self._was_taken(observation, email_id, "respond"):
                return Action(
                    action_type="respond",
                    email_id=email_id,
                    response_template="request_info",
                    priority=self._default_priority(email),
                )
            return None

        if not self._was_taken(observation, email_id, "classify"):
            return Action(action_type="classify", email_id=email_id, category=self._default_category(email))
        if not self._was_taken(observation, email_id, "respond"):
            return Action(
                action_type="respond",
                email_id=email_id,
                response_template=self._default_response(email),
                priority=self._default_priority(email),
            )
        return None

    def _email_score(self, observation: Observation, email_id: str) -> float:
        email = next(item for item in observation.inbox if item.email_id == email_id)
        profile = self._profile(email)

        score = 0.0
        if profile["escalation"]:
            score += 8.0
        elif profile["incident"]:
            score += 6.0
        elif profile["routine"]:
            score += 3.0
        elif profile["spam"]:
            score += 1.0

        score += self._priority_weight(email.priority_hint)
        score += min(float(email.age) * 0.2, 1.0)
        score -= float(email.noise_score) * 0.5

        actions_taken = self._actions_for_email(observation, email_id)
        if "classify" not in actions_taken:
            score += 0.8
        if profile["spam"] and "ignore" not in actions_taken:
            score += 0.4
        if not profile["spam"] and "respond" not in actions_taken:
            score += 0.6
        if profile["escalation"] and "escalate" not in actions_taken:
            score += 1.0
        return score

    def _profile(self, email) -> dict[str, bool]:
        text = f"{email.subject} {email.body}".lower()
        sender = email.sender.lower()
        spam = self._contains_any(text, self.spam_keywords) or self._contains_any(sender, self.sender_spam_keywords)
        incident = self._contains_any(text, self.incident_keywords) or self._contains_any(sender, self.sender_urgency_keywords)
        escalation = self._contains_any(text, self.escalation_keywords)
        routine = self._contains_any(text, self.routine_keywords)
        if spam and (incident or escalation):
            spam = False
        if escalation:
            incident = True
        return {
            "spam": spam,
            "incident": incident,
            "escalation": escalation,
            "routine": routine and not incident,
        }

    def _actions_for_email(self, observation: Observation, email_id: str) -> set[str]:
        return {trace.action_type for trace in observation.action_history if trace.email_id == email_id}

    def _was_taken(self, observation: Observation, email_id: str, action_type: str) -> bool:
        return action_type in self._actions_for_email(observation, email_id)

    @staticmethod
    def _priority_weight(priority_hint: str) -> float:
        return {
            "critical": 2.5,
            "high": 1.75,
            "medium": 0.9,
            "low": 0.2,
        }.get(priority_hint, 0.0)

    @staticmethod
    def _contains_any(text: str, keywords: tuple[str, ...]) -> bool:
        return any(keyword in text for keyword in keywords)

    def _default_category(self, email) -> str:
        profile = self._profile(email)
        if profile["spam"]:
            return "spam"
        if profile["escalation"]:
            return "escalation"
        if profile["incident"]:
            return "urgent"
        return "normal"

    def _default_response(self, email) -> str:
        profile = self._profile(email)
        if profile["escalation"]:
            return "escalate_notice"
        if profile["incident"]:
            return "acknowledge"
        return "request_info"

    @staticmethod
    def _default_priority(email) -> str:
        if email.priority_hint in {"critical", "high"}:
            return email.priority_hint
        return "medium"
