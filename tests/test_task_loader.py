from __future__ import annotations

import json
from pathlib import Path

from openenv.task_loader import load_task_scenarios, scenario_schema
from openenv.tasks import get_email_tasks


def test_load_task_scenarios_skips_invalid_and_keeps_valid(tmp_path: Path) -> None:
    scenario_file = tmp_path / "custom_tasks.json"
    scenario_file.write_text(
        json.dumps(
            {
                "schema_version": "1.0.0",
                "tasks": [
                    {
                        "name": "json_task_valid",
                        "description": "Loaded from JSON.",
                        "initial_state": {
                            "emails": [
                                {
                                    "email_id": "j-001",
                                    "sender": "ops@customer.example",
                                    "subject": "Migration review",
                                    "body": "Please review before release.",
                                    "thread_id": "j-thread",
                                    "arrival_step": 0,
                                    "priority_hint": "medium",
                                    "noise_score": 0.2,
                                    "true_category": "normal",
                                    "response_template": "request_info",
                                    "requires_response": True,
                                    "requires_escalation": False,
                                    "classification_deadline": 2,
                                    "response_deadline": 3,
                                    "escalation_deadline": 4
                                }
                            ]
                        },
                        "success_criteria": "Handle the task.",
                        "max_steps": 8,
                        "difficulty": "easy",
                        "seed": 9001
                    },
                    {
                        "name": "json_task_invalid",
                        "description": "Broken task.",
                        "initial_state": {},
                        "success_criteria": "Broken.",
                        "max_steps": 0,
                        "difficulty": "easy",
                        "seed": 9002
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    report = load_task_scenarios(tmp_path)

    assert [task.name for task in report.loaded_tasks] == ["json_task_valid"]
    assert report.skipped_count == 1
    assert report.issues[0].task_name == "json_task_invalid"


def test_get_email_tasks_includes_valid_json_tasks(tmp_path: Path) -> None:
    scenario_file = tmp_path / "tasks.json"
    scenario_file.write_text(
        json.dumps(
            {
                "schema_version": "1.0.0",
                "tasks": [
                    {
                        "name": "json_task_extra",
                        "description": "Extra JSON task.",
                        "initial_state": {"emails": []},
                        "success_criteria": "No-op.",
                        "max_steps": 3,
                        "difficulty": "easy",
                        "seed": 42
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    tasks = get_email_tasks(scenarios_path=tmp_path)
    assert any(task.name == "json_task_extra" for task in tasks)


def test_scenario_schema_has_tasks_root() -> None:
    schema = scenario_schema()
    assert "$ref" in schema or "properties" in schema
