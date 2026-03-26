from __future__ import annotations

from typing import Callable

from pydantic import Field

from .models import Action, EmailSpec, StepRecord, VersionedModel


class Task(VersionedModel):
    name: str
    description: str
    initial_state: dict
    success_criteria: str
    max_steps: int = Field(gt=0)
    difficulty: str
    seed: int


def _classification_task() -> Task:
    emails = [
        EmailSpec(
            email_id="e-001",
            sender="ads@promo.example",
            subject="Limited time offer",
            body="Act now and buy three get one free. Unsubscribe link below.",
            thread_id="t-001",
            arrival_step=0,
            priority_hint="low",
            noise_score=0.92,
            true_category="spam",
            classification_deadline=1,
            response_deadline=3,
            escalation_deadline=3,
        ),
        EmailSpec(
            email_id="e-002",
            sender="ceo@customer.example",
            subject="Production outage impacting payroll",
            body="Payroll exports are failing for all regions. Need immediate help.",
            thread_id="t-002",
            arrival_step=0,
            priority_hint="critical",
            noise_score=0.05,
            true_category="urgent",
            response_template="acknowledge",
            requires_response=True,
            classification_deadline=1,
            response_deadline=2,
            escalation_deadline=4,
        ),
    ]
    return Task(
        name="task_easy_classification",
        description="Classify obvious spam and urgent operational mail.",
        initial_state={"emails": [email.model_dump(mode="json") for email in emails]},
        success_criteria="Correctly classify both visible emails within one step each.",
        max_steps=4,
        difficulty="easy",
        seed=101,
    )


def _prioritization_task() -> Task:
    emails = [
        EmailSpec(
            email_id="e-101",
            sender="support@vip.example",
            subject="Enterprise login outage",
            body="Our entire finance team is blocked from login. Please confirm ownership.",
            thread_id="t-101",
            arrival_step=0,
            priority_hint="critical",
            noise_score=0.02,
            true_category="urgent",
            response_template="acknowledge",
            requires_response=True,
            classification_deadline=1,
            response_deadline=1,
            escalation_deadline=3,
        ),
        EmailSpec(
            email_id="e-102",
            sender="news@vendor.example",
            subject="Quarterly product newsletter",
            body="Highlights from this quarter and upcoming webinars.",
            thread_id="t-102",
            arrival_step=0,
            priority_hint="low",
            noise_score=0.87,
            true_category="spam",
            classification_deadline=2,
            response_deadline=3,
            escalation_deadline=3,
        ),
        EmailSpec(
            email_id="e-103",
            sender="ops@customer.example",
            subject="Need status update on migration",
            body="Please send an update on the database migration scheduled for tonight.",
            thread_id="t-103",
            arrival_step=1,
            priority_hint="medium",
            noise_score=0.14,
            true_category="normal",
            response_template="request_info",
            requires_response=True,
            classification_deadline=2,
            response_deadline=3,
            escalation_deadline=4,
        ),
    ]
    return Task(
        name="task_medium_prioritization",
        description="Prioritize urgent mail, ignore obvious spam, and send the right response.",
        initial_state={"emails": [email.model_dump(mode="json") for email in emails]},
        success_criteria="Handle the urgent email first and choose correct responses with no redundant actions.",
        max_steps=6,
        difficulty="medium",
        seed=202,
    )


def _thread_reasoning_task() -> Task:
    emails = [
        EmailSpec(
            email_id="e-201",
            sender="noc@customer.example",
            subject="Intermittent API timeout",
            body="Seeing intermittent timeout spikes in eu-west. Need triage.",
            thread_id="thread-outage",
            arrival_step=0,
            priority_hint="high",
            noise_score=0.11,
            true_category="urgent",
            response_template="acknowledge",
            requires_response=True,
            requires_escalation=False,
            classification_deadline=1,
            response_deadline=1,
            escalation_deadline=4,
        ),
        EmailSpec(
            email_id="e-202",
            sender="noc@customer.example",
            subject="Update: issue spreading globally",
            body="Timeout spikes are now global and customer checkouts are failing.",
            thread_id="thread-outage",
            arrival_step=2,
            priority_hint="critical",
            noise_score=0.04,
            true_category="escalation",
            response_template="escalate_notice",
            requires_response=True,
            requires_escalation=True,
            escalation_trigger_step=2,
            classification_deadline=3,
            response_deadline=3,
            escalation_deadline=3,
        ),
        EmailSpec(
            email_id="e-203",
            sender="marketing@internal.example",
            subject="Approve webinar copy",
            body="Need a quick review of tomorrow's webinar announcement text.",
            thread_id="t-203",
            arrival_step=1,
            priority_hint="low",
            noise_score=0.22,
            true_category="normal",
            response_template="request_info",
            requires_response=True,
            classification_deadline=3,
            response_deadline=5,
            escalation_deadline=5,
        ),
    ]
    return Task(
        name="task_hard_thread_reasoning",
        description="Track a multi-step outage thread and escalate only when the follow-up evidence arrives.",
        initial_state={"emails": [email.model_dump(mode="json") for email in emails]},
        success_criteria="Acknowledge the first outage note, avoid premature escalation, then escalate immediately after the follow-up arrives.",
        max_steps=7,
        difficulty="hard",
        seed=303,
    )


def get_email_tasks() -> list[Task]:
    return [_classification_task(), _prioritization_task(), _thread_reasoning_task()]


def _task_one_grade(trajectory: list[StepRecord]) -> float:
    if not trajectory:
        return 0.0
    correct = 0.0
    penalty = 0.0
    seen_ids: set[str] = set()
    for step in trajectory:
        action = step.action
        if action.email_id:
            seen_ids.add(action.email_id)
        if action.action_type == "classify":
            if action.email_id == "e-001" and action.category == "spam":
                correct += 0.5
            elif action.email_id == "e-002" and action.category == "urgent":
                correct += 0.5
            else:
                penalty += 0.25
        elif action.action_type != "wait":
            penalty += 0.1
    if seen_ids != {"e-001", "e-002"}:
        penalty += 0.2
    return max(0.0, min(1.0, correct - penalty))


def _task_two_grade(trajectory: list[StepRecord]) -> float:
    score = 0.0
    first_non_wait = next((step for step in trajectory if step.action.action_type != "wait"), None)
    if first_non_wait and first_non_wait.action.email_id == "e-101":
        score += 0.25
    if any(
        step.action.action_type == "classify"
        and step.action.email_id == "e-101"
        and step.action.category == "urgent"
        for step in trajectory
    ):
        score += 0.2
    if any(
        step.action.action_type == "respond"
        and step.action.email_id == "e-101"
        and step.action.response_template == "acknowledge"
        for step in trajectory
    ):
        score += 0.25
    if any(
        step.action.action_type == "ignore" and step.action.email_id == "e-102"
        for step in trajectory
    ):
        score += 0.15
    if any(
        step.action.action_type == "respond"
        and step.action.email_id == "e-103"
        and step.action.response_template == "request_info"
        for step in trajectory
    ):
        score += 0.15
    loops = sum(1 for step in trajectory if step.info.get("loop_detected"))
    score -= min(loops * 0.05, 0.2)
    return max(0.0, min(1.0, score))


def _task_three_grade(trajectory: list[StepRecord]) -> float:
    score = 0.0
    escalated_before_follow_up = any(
        step.action.action_type == "escalate"
        and step.action.email_id == "e-201"
        and step.step_index < 2
        for step in trajectory
    )
    if escalated_before_follow_up:
        score -= 0.2
    if any(
        step.action.action_type == "respond"
        and step.action.email_id == "e-201"
        and step.action.response_template == "acknowledge"
        for step in trajectory
    ):
        score += 0.25
    if any(
        step.action.action_type == "classify"
        and step.action.email_id == "e-202"
        and step.action.category == "escalation"
        for step in trajectory
    ):
        score += 0.2
    if any(
        step.action.action_type == "escalate"
        and step.action.email_id == "e-202"
        and step.step_index <= 3
        for step in trajectory
    ):
        score += 0.35
    if any(
        step.action.action_type == "respond"
        and step.action.email_id == "e-202"
        and step.action.response_template == "escalate_notice"
        for step in trajectory
    ):
        score += 0.2
    loops = sum(1 for step in trajectory if step.info.get("loop_detected"))
    score -= min(loops * 0.05, 0.2)
    return max(0.0, min(1.0, score))


def get_graders() -> dict[str, Callable[[list[StepRecord]], float]]:
    return {
        "task_easy_classification": _task_one_grade,
        "task_medium_prioritization": _task_two_grade,
        "task_hard_thread_reasoning": _task_three_grade,
    }
