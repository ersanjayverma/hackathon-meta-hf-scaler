from __future__ import annotations

from typing import Iterable, List

from .models import EmailSpec, StepRecord


def grade_action_quality(trajectory: list[StepRecord], email_specs: List[EmailSpec]) -> float:
    if not email_specs:
        return 1.0
    if not trajectory:
        return 0.0
    completed_ids = set(trajectory[-1].observation.completed_email_ids)
    if not completed_ids:
        return 0.0
    spec_by_id = {spec.email_id: spec for spec in email_specs}
    actions_by_email: dict[str, list] = {}
    for record in trajectory:
        action = record.action
        if action.email_id and action.email_id in spec_by_id:
            actions_by_email.setdefault(action.email_id, []).append(action)
    total_quality = 0.0
    for email_id in completed_ids:
        if email_id not in spec_by_id:
            continue
        spec = spec_by_id[email_id]
        actions = actions_by_email.get(email_id, [])
        classify_score = 0.0
        for act in actions:
            if act.action_type == "classify" and act.category == spec.true_category:
                classify_score = 1.0
                break
        if spec.requires_response:
            response_score = 0.0
            for act in actions:
                if act.action_type == "respond" and act.response_template == spec.response_template:
                    response_score = 1.0
                    break
        else:
            response_score = 1.0
        if spec.requires_escalation:
            escalation_score = 0.0
            for act in actions:
                if act.action_type == "escalate":
                    escalation_score = 1.0
                    break
        else:
            escalation_score = 1.0
        total_quality += 0.4 * classify_score + 0.4 * response_score + 0.2 * escalation_score
    return total_quality / len(email_specs)


def grade_processed_ids(processed_ids: Iterable[str], expected_ids: Iterable[str]) -> float:
    processed = set(processed_ids)
    expected = set(expected_ids)
    if not expected:
        return 1.0
    return len(processed & expected) / len(expected)


def processed_ids_from_trajectory(trajectory: list[StepRecord]) -> list[str]:
    if not trajectory:
        return []
    return list(trajectory[-1].observation.completed_email_ids)


def build_task_grader(expected_ids: Iterable[str]):
    expected = list(expected_ids)

    def grade(trajectory: list[StepRecord]) -> float:
        return float(grade_processed_ids(processed_ids_from_trajectory(trajectory), expected))

    return grade
