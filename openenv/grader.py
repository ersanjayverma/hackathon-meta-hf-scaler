from __future__ import annotations

from typing import Iterable

from .models import StepRecord


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
