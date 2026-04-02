from __future__ import annotations

from typing import Callable

from pydantic import Field

from .grader import build_task_grader as _build_simple_task_grader, grade_processed_ids
from .models import EmailSpec, StepRecord, VersionedModel

CANONICAL_BENCHMARK_TASK_NAMES = (
    "task_easy_classification",
    "task_medium_prioritization",
    "task_hard_thread_reasoning",
)


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
        EmailSpec(
            email_id="e-003",
            sender="pm@customer.example",
            subject="Migration checklist review",
            body="Please review the migration checklist and confirm missing dependencies.",
            thread_id="t-003",
            arrival_step=1,
            priority_hint="medium",
            noise_score=0.18,
            true_category="normal",
            response_template="request_info",
            requires_response=True,
            classification_deadline=3,
            response_deadline=5,
            escalation_deadline=6,
        ),
        EmailSpec(
            email_id="e-004",
            sender="events@vendor.example",
            subject="Webinar invite and newsletter bundle",
            body="Join our webinar and subscribe to the quarterly newsletter for updates.",
            thread_id="t-004",
            arrival_step=2,
            priority_hint="low",
            noise_score=0.89,
            true_category="spam",
            classification_deadline=4,
            response_deadline=6,
            escalation_deadline=6,
        ),
    ]
    return Task(
        name="task_easy_classification",
        description="Classify obvious spam, urgent operational mail, and routine requests while managing delayed follow-up consequences.",
        initial_state={"emails": [email.model_dump(mode="json") for email in emails]},
        success_criteria="Correctly classify the mixed inbox, respond to normal and urgent items, and avoid delayed follow-up incidents and SLA drift.",
        max_steps=28,
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
        EmailSpec(
            email_id="e-104",
            sender="security@customer.example",
            subject="Checkout timeout escalation request",
            body="Checkout timeout complaints are increasing and revenue is at risk.",
            thread_id="t-104",
            arrival_step=2,
            priority_hint="high",
            noise_score=0.08,
            true_category="escalation",
            response_template="escalate_notice",
            requires_response=True,
            requires_escalation=True,
            escalation_trigger_step=2,
            classification_deadline=3,
            response_deadline=4,
            escalation_deadline=4,
        ),
        EmailSpec(
            email_id="e-105",
            sender="marketing@vendor.example",
            subject="Newsletter and webinar sponsorship offer",
            body="We would love to include your team in our webinar and newsletter sponsorship package.",
            thread_id="t-105",
            arrival_step=3,
            priority_hint="low",
            noise_score=0.91,
            true_category="spam",
            classification_deadline=5,
            response_deadline=6,
            escalation_deadline=6,
        ),
    ]
    return Task(
        name="task_medium_prioritization",
        description="Prioritize urgent mail, classify routine work, ignore obvious spam, and manage the downstream impact of mistakes over time.",
        initial_state={"emails": [email.model_dump(mode="json") for email in emails]},
        success_criteria="Handle urgent and escalation requests first, clear routine follow-ups, avoid spam distractions, and keep system stress low over a longer horizon.",
        max_steps=34,
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
        EmailSpec(
            email_id="e-204",
            sender="ops@customer.example",
            subject="Migration review before release",
            body="Need a fast review of the migration plan before tonight's release window.",
            thread_id="t-204",
            arrival_step=3,
            priority_hint="medium",
            noise_score=0.17,
            true_category="normal",
            response_template="request_info",
            requires_response=True,
            classification_deadline=5,
            response_deadline=6,
            escalation_deadline=7,
        ),
        EmailSpec(
            email_id="e-205",
            sender="news@internal.example",
            subject="Quarterly webinar newsletter",
            body="Internal newsletter covering the next webinar calendar and community updates.",
            thread_id="t-205",
            arrival_step=4,
            priority_hint="low",
            noise_score=0.86,
            true_category="spam",
            classification_deadline=6,
            response_deadline=8,
            escalation_deadline=8,
        ),
    ]
    return Task(
        name="task_hard_thread_reasoning",
        description="Track a multi-step outage thread, routine side requests, and delayed fallout under sustained inbox pressure.",
        initial_state={"emails": [email.model_dump(mode="json") for email in emails]},
        success_criteria="Acknowledge the first outage note, manage side requests without distraction, absorb delayed consequences, and escalate when evidence justifies it.",
        max_steps=38,
        difficulty="hard",
        seed=303,
    )

def get_benchmark_tasks() -> list[Task]:
    return [_classification_task(), _prioritization_task(), _thread_reasoning_task()]


def get_benchmark_task_names() -> tuple[str, ...]:
    return CANONICAL_BENCHMARK_TASK_NAMES


def get_builtin_email_tasks() -> list[Task]:
    return get_benchmark_tasks()


def get_email_tasks(
    *,
    include_supplemental: bool = True,
) -> list[Task]:
    _ = include_supplemental
    return get_benchmark_tasks()


def grade_task(processed_ids, expected_ids):
    return grade_processed_ids(processed_ids, expected_ids)


def _build_task_grader(task: Task) -> Callable[[list[StepRecord]], float]:
    expected_ids = [str(email["email_id"]) for email in task.initial_state.get("emails", [])]
    return _build_simple_task_grader(expected_ids)


def get_benchmark_graders() -> dict[str, Callable[[list[StepRecord]], float]]:
    return {task.name: _build_task_grader(task) for task in get_benchmark_tasks()}


def get_graders() -> dict[str, Callable[[list[StepRecord]], float]]:
    return get_benchmark_graders()
