from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from openenv.models import Action, Observation


@dataclass(slots=True)
class EmailHistory:
    classified: bool = False
    responded: bool = False
    escalated: bool = False
    ignored: bool = False


@dataclass(slots=True)
class EmailProfile:
    category: str
    response_template: str | None
    priority: str
    requires_escalation: bool
    spam: bool
    urgent: bool
    escalation: bool
    routine: bool
    followup: bool
    importance: float


@dataclass(slots=True)
class HeuristicAgent:
    spam_keywords: tuple[str, ...] = (
        "unsubscribe",
        "newsletter",
        "offer",
        "discount",
        "promotion",
        "sponsorship",
        "buy three get one",
        "webinar invite",
        "bundle",
    )
    incident_keywords: tuple[str, ...] = (
        "outage",
        "timeout",
        "failing",
        "failure",
        "incident",
        "blocked",
        "degraded service",
        "payroll",
        "login outage",
        "queue pressure",
    )
    escalation_keywords: tuple[str, ...] = (
        "escalation request",
        "escalate",
        "security",
        "revenue is at risk",
        "revenue at risk",
        "spreading globally",
        "queue pressure alert",
        "customer checkouts are failing",
        "checkout timeout complaints",
        "global and customer checkouts are failing",
    )
    routine_keywords: tuple[str, ...] = (
        "migration",
        "review",
        "checklist",
        "status update",
        "announcement text",
        "approve",
        "dependencies",
        "confirm",
        "plan before tonight",
    )
    followup_keywords: tuple[str, ...] = (
        "follow-up:",
        "correction:",
        "customer reply:",
        "still waiting",
        "did not address",
        "re-check the urgency",
    )
    sender_urgency_keywords: tuple[str, ...] = ("ceo@", "vip@", "noc@", "security@", "support@", "ops-overload@")
    sender_spam_keywords: tuple[str, ...] = ("promo@", "news@", "events@", "marketing@vendor")
    sender_internal_keywords: tuple[str, ...] = ("@internal.example", "pm@", "ops@", "support@", "customer.example")

    def act(self, observation: Observation) -> Action:
        if not observation.inbox:
            return Action(action_type="wait")

        history = self._history_by_email(observation)
        ranked_emails = sorted(
            observation.inbox,
            key=lambda email: (
                -self._email_score(email, self._profile(email), history.get(email.email_id, EmailHistory())),
                email.age,
                email.email_id,
            ),
        )

        for email in ranked_emails:
            profile = self._profile(email)
            email_history = history.get(email.email_id, EmailHistory())
            action = self._next_action(email, profile, email_history)
            if action is not None:
                return action

        fallback_email = ranked_emails[0]
        fallback_profile = self._profile(fallback_email)
        fallback_history = history.get(fallback_email.email_id, EmailHistory())
        return self._safe_progress_action(fallback_email, fallback_profile, fallback_history)

    def _next_action(self, email, profile: EmailProfile, history: EmailHistory) -> Action | None:
        if profile.spam:
            if not history.classified:
                return Action(action_type="classify", email_id=email.email_id, category="spam")
            if not history.ignored:
                return Action(action_type="ignore", email_id=email.email_id)
            return None

        if not history.classified:
            return Action(action_type="classify", email_id=email.email_id, category=profile.category)

        if profile.requires_escalation and not history.escalated:
            return Action(action_type="escalate", email_id=email.email_id, priority=profile.priority)

        if profile.response_template is not None and not history.responded:
            return Action(
                action_type="respond",
                email_id=email.email_id,
                response_template=profile.response_template,
                priority=profile.priority,
            )

        if profile.urgent and not history.responded:
            return Action(
                action_type="respond",
                email_id=email.email_id,
                response_template="acknowledge",
                priority=profile.priority,
            )

        return None

    def _safe_progress_action(self, email, profile: EmailProfile, history: EmailHistory) -> Action:
        if not history.classified:
            return Action(action_type="classify", email_id=email.email_id, category=profile.category)
        if profile.requires_escalation and not history.escalated:
            return Action(action_type="escalate", email_id=email.email_id, priority=profile.priority)
        if profile.response_template is not None and not history.responded:
            return Action(
                action_type="respond",
                email_id=email.email_id,
                response_template=profile.response_template,
                priority=profile.priority,
            )
        if profile.spam and not history.ignored:
            return Action(action_type="ignore", email_id=email.email_id)
        return Action(action_type="wait")

    def _email_score(self, email, profile: EmailProfile, history: EmailHistory) -> float:
        score = profile.importance
        score += min(float(email.age) * 0.2, 1.5)
        score += self._priority_weight(email.priority_hint)

        if not history.classified:
            score += 1.6
        if profile.response_template is not None and not history.responded:
            score += 1.4
        if profile.requires_escalation and not history.escalated:
            score += 1.8
        if profile.spam and not history.ignored:
            score += 0.5
        if history.ignored:
            score -= 4.0

        return score

    def _profile(self, email) -> EmailProfile:
        subject = email.subject.lower()
        body = email.body.lower()
        text = f"{subject} {body}"
        sender = email.sender.lower()

        is_followup = self._contains_any(text, self.followup_keywords)
        has_incident_signal = self._contains_any(text, self.incident_keywords)
        has_escalation_signal = self._contains_any(text, self.escalation_keywords)
        has_routine_signal = self._contains_any(text, self.routine_keywords)
        has_spam_signal = self._contains_any(text, self.spam_keywords)
        sender_is_urgent = self._contains_any(sender, self.sender_urgency_keywords)
        sender_is_spam = self._contains_any(sender, self.sender_spam_keywords)
        sender_is_internal = self._contains_any(sender, self.sender_internal_keywords)

        internal_review = sender_is_internal and has_routine_signal
        spam = (has_spam_signal or sender_is_spam) and not internal_review
        escalation = has_escalation_signal or (
            is_followup and has_incident_signal and (sender_is_urgent or email.priority_hint in {"high", "critical"})
        )
        urgent = has_incident_signal or is_followup or sender_is_urgent
        routine = has_routine_signal and not urgent

        if spam and (urgent or escalation or routine):
            spam = False

        if escalation:
            category = "escalation"
            requires_escalation = True
            response_template = "acknowledge" if is_followup else "escalate_notice"
            importance = 9.0
        elif urgent:
            category = "urgent"
            requires_escalation = False
            response_template = "acknowledge"
            importance = 7.0
        elif spam:
            category = "spam"
            requires_escalation = False
            response_template = None
            importance = 1.5
        else:
            category = "normal"
            requires_escalation = False
            response_template = "request_info"
            importance = 4.0 if routine else 3.0

        return EmailProfile(
            category=category,
            response_template=response_template,
            priority=self._default_priority(email, escalation=escalation, urgent=urgent),
            requires_escalation=requires_escalation,
            spam=spam,
            urgent=urgent,
            escalation=escalation,
            routine=routine,
            followup=is_followup,
            importance=importance,
        )

    def _history_by_email(self, observation: Observation) -> dict[str, EmailHistory]:
        history: dict[str, EmailHistory] = {}
        for trace in getattr(observation, "action_history", []):
            if trace.email_id is None:
                continue
            state = history.setdefault(trace.email_id, EmailHistory())
            if trace.action_type == "classify":
                state.classified = True
            elif trace.action_type == "respond":
                state.responded = True
            elif trace.action_type == "escalate":
                state.escalated = True
            elif trace.action_type == "ignore":
                state.ignored = True
        return history

    @staticmethod
    def _contains_any(text: str, keywords: Iterable[str]) -> bool:
        return any(keyword in text for keyword in keywords)

    @staticmethod
    def _priority_weight(priority_hint: str) -> float:
        return {
            "critical": 2.0,
            "high": 1.5,
            "medium": 0.75,
            "low": 0.15,
        }.get(priority_hint, 0.0)

    @staticmethod
    def _default_priority(email, *, escalation: bool, urgent: bool) -> str:
        if escalation:
            return "critical" if email.priority_hint == "critical" else "high"
        if urgent and email.priority_hint in {"high", "critical"}:
            return email.priority_hint
        return "medium"
