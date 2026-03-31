from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .tasks import Task

logger = logging.getLogger(__name__)


class TaskScenarioDocument(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = Field(default="1.0.0")
    tasks: list[dict[str, Any]]


@dataclass(slots=True)
class TaskLoadIssue:
    source: str
    message: str
    task_name: str | None = None
    task_index: int | None = None


@dataclass(slots=True)
class TaskLoadReport:
    loaded_tasks: list[Task] = field(default_factory=list)
    issues: list[TaskLoadIssue] = field(default_factory=list)

    @property
    def skipped_count(self) -> int:
        return len(self.issues)


def load_task_scenarios(path: str | Path) -> TaskLoadReport:
    scenario_path = Path(path)
    report = TaskLoadReport()
    if not scenario_path.exists():
        return report

    file_paths = sorted(scenario_path.glob("*.json")) if scenario_path.is_dir() else [scenario_path]
    seen_names = set()
    for file_path in file_paths:
        _load_single_file(file_path, report, seen_names)
    return report


def scenario_schema() -> dict[str, Any]:
    return TaskScenarioDocument.model_json_schema()


def _load_single_file(file_path: Path, report: TaskLoadReport, seen_names: set[str]) -> None:
    try:
        payload = json.loads(file_path.read_text(encoding="utf-8"))
    except Exception as exc:
        report.issues.append(TaskLoadIssue(source=str(file_path), message=f"invalid_json: {exc}"))
        return

    try:
        document = TaskScenarioDocument(**payload)
    except ValidationError as exc:
        report.issues.append(TaskLoadIssue(source=str(file_path), message=f"invalid_document: {exc.errors()}"))
        return

    for index, raw_task in enumerate(document.tasks):
        try:
            task = Task(**raw_task)
        except ValidationError as exc:
            report.issues.append(
                TaskLoadIssue(
                    source=str(file_path),
                    task_index=index,
                    task_name=raw_task.get("name") if isinstance(raw_task, dict) else None,
                    message=f"invalid_task: {exc.errors()}",
                )
            )
            continue

        if task.name in seen_names:
            report.issues.append(
                TaskLoadIssue(
                    source=str(file_path),
                    task_index=index,
                    task_name=task.name,
                    message="duplicate_task_name",
                )
            )
            continue

        seen_names.add(task.name)
        report.loaded_tasks.append(task)


def log_task_load_report(report: TaskLoadReport) -> None:
    for issue in report.issues:
        logger.warning(
            "task_loader_issue source=%s task_name=%s task_index=%s message=%s",
            issue.source,
            issue.task_name,
            issue.task_index,
            issue.message,
        )
