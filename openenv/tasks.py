from __future__ import annotations

from pathlib import Path
from typing import Callable

from pydantic import Field

from .models import Action, EmailSpec, StepRecord, VersionedModel

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


def _clone_task_sample(task: Task, sample_index: int, seed: int) -> Task:
    return task.model_copy(
        update={
            "name": f"{task.name}_sample_{sample_index:02d}",
            "description": f"{task.description} Seeded sample {sample_index:02d}.",
            "seed": seed,
        }
    )


def _sampled_task_variants() -> list[Task]:
    base_tasks = [_classification_task(), _prioritization_task(), _thread_reasoning_task()]
    variant_plan = [
        (base_tasks[0], 10, 1000),
        (base_tasks[1], 10, 2000),
        (base_tasks[2], 10, 3000),
    ]
    samples: list[Task] = []
    sample_index = 1
    for task, count, seed_offset in variant_plan:
        for local_index in range(count):
            samples.append(
                _clone_task_sample(
                    task=task,
                    sample_index=sample_index,
                    seed=task.seed + seed_offset + local_index,
                )
            )
            sample_index += 1
    return samples


def get_benchmark_tasks() -> list[Task]:
    return [_classification_task(), _prioritization_task(), _thread_reasoning_task()]


def get_benchmark_task_names() -> tuple[str, ...]:
    return CANONICAL_BENCHMARK_TASK_NAMES


def get_builtin_email_tasks() -> list[Task]:
    return get_benchmark_tasks() + _sampled_task_variants()


def get_supplemental_email_tasks(scenarios_path: str | Path | None = None) -> list[Task]:
    tasks = list(_sampled_task_variants())
    if scenarios_path is None:
        scenarios_path = Path("scenarios")
    from .task_loader import load_task_scenarios, log_task_load_report

    report = load_task_scenarios(scenarios_path)
    if report.issues:
        log_task_load_report(report)
    tasks.extend(report.loaded_tasks)
    return tasks


def get_email_tasks(
    scenarios_path: str | Path | None = None,
    *,
    include_supplemental: bool = True,
) -> list[Task]:
    tasks = list(get_benchmark_tasks())
    if include_supplemental:
        tasks.extend(get_supplemental_email_tasks(scenarios_path=scenarios_path))
    return tasks


def _task_one_grade(trajectory: list[StepRecord]) -> float:
    if not trajectory:
        return 0.0
    score = 0.0
    first_non_wait = next((step for step in trajectory if step.action.action_type != "wait"), None)
    if first_non_wait and first_non_wait.action.email_id == "e-002":
        score += 0.05
    if any(
        step.action.action_type == "classify"
        and step.action.email_id == "e-001"
        and step.action.category == "spam"
        for step in trajectory
    ):
        score += 0.15
    if any(
        step.action.action_type == "classify"
        and step.action.email_id == "e-002"
        and step.action.category == "urgent"
        for step in trajectory
    ):
        score += 0.25
    if any(
        step.action.action_type == "respond"
        and step.action.email_id == "e-002"
        and step.action.response_template == "acknowledge"
        for step in trajectory
    ):
        score += 0.2
    if any(
        step.action.action_type == "classify"
        and step.action.email_id == "e-003"
        and step.action.category == "normal"
        for step in trajectory
    ):
        score += 0.15
    if any(
        step.action.action_type == "respond"
        and step.action.email_id == "e-003"
        and step.action.response_template == "request_info"
        for step in trajectory
    ):
        score += 0.15
    if any(
        (
            step.action.action_type == "ignore"
            or (step.action.action_type == "classify" and step.action.category == "spam")
        )
        and step.action.email_id == "e-004"
        for step in trajectory
    ):
        score += 0.05
    loops = sum(1 for step in trajectory if step.info.get("loop_detected"))
    score -= min(loops * 0.05, 0.15)
    return max(0.0, min(1.0, score))


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
    if any(
        step.action.action_type == "classify"
        and step.action.email_id == "e-104"
        and step.action.category == "escalation"
        for step in trajectory
    ):
        score += 0.1
    if any(
        step.action.action_type == "escalate" and step.action.email_id == "e-104"
        for step in trajectory
    ):
        score += 0.1
    if any(
        step.action.action_type == "ignore" and step.action.email_id == "e-105"
        for step in trajectory
    ):
        score += 0.05
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
    if any(
        step.action.action_type == "respond"
        and step.action.email_id == "e-204"
        and step.action.response_template == "request_info"
        for step in trajectory
    ):
        score += 0.1
    if any(
        step.action.action_type == "ignore" and step.action.email_id == "e-205"
        for step in trajectory
    ):
        score += 0.1
    loops = sum(1 for step in trajectory if step.info.get("loop_detected"))
    score -= min(loops * 0.05, 0.2)
    return max(0.0, min(1.0, score))


def _generic_task_grade(trajectory: list[StepRecord]) -> float:
    if not trajectory:
        return 0.0
    final_step = trajectory[-1]
    unique_emails = {
        step.action.email_id
        for step in trajectory
        if step.action.email_id is not None
    }
    action_types = {step.action.action_type for step in trajectory}
    completion_ratio = min(len(final_step.observation.completed_email_ids) / max(len(unique_emails), 1), 1.0)
    progress_bonus = min(len(unique_emails) / 5.0, 0.3)
    action_bonus = 0.0
    if "classify" in action_types:
        action_bonus += 0.2
    if "respond" in action_types:
        action_bonus += 0.2
    if "escalate" in action_types:
        action_bonus += 0.1
    penalty = min(sum(1 for step in trajectory if step.info.get("loop_detected")) * 0.05, 0.2)
    if final_step.info.get("termination_reason") == "system_collapse":
        penalty += 0.25
    return max(0.0, min(1.0, completion_ratio * 0.4 + progress_bonus + action_bonus - penalty))


def get_benchmark_graders() -> dict[str, Callable[[list[StepRecord]], float]]:
    return {
        "task_easy_classification": _task_one_grade,
        "task_medium_prioritization": _task_two_grade,
        "task_hard_thread_reasoning": _task_three_grade,
    }


def get_graders() -> dict[str, Callable[[list[StepRecord]], float]]:
    graders = dict(get_benchmark_graders())
    for task in get_email_tasks(include_supplemental=True):
        if task.name in graders:
            continue
        if task.name.startswith("task_easy_classification"):
            graders[task.name] = _task_one_grade
        elif task.name.startswith("task_medium_prioritization"):
            graders[task.name] = _task_two_grade
        elif task.name.startswith("task_hard_thread_reasoning"):
            graders[task.name] = _task_three_grade
        else:
            graders[task.name] = _generic_task_grade
    return graders
