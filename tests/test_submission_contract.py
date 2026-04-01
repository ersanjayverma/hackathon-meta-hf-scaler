from __future__ import annotations

from pathlib import Path

import yaml

from openenv.tasks import get_benchmark_graders, get_benchmark_task_names, get_benchmark_tasks


ROOT = Path(__file__).resolve().parents[1]


def test_canonical_benchmark_tasks_are_explicit() -> None:
    tasks = get_benchmark_tasks()
    assert [task.name for task in tasks] == list(get_benchmark_task_names())
    assert [task.difficulty for task in tasks] == ["easy", "medium", "hard"]


def test_root_manifest_lists_canonical_tasks() -> None:
    manifest = yaml.safe_load((ROOT / "openenv.yaml").read_text(encoding="utf-8"))
    task_definitions = manifest["task_definitions"]
    assert [task["name"] for task in task_definitions] == list(get_benchmark_task_names())
    assert manifest["service"]["app_file"] == "server/app.py"
    assert manifest["service"]["port"] == 7860


def test_canonical_graders_are_bounded() -> None:
    graders = get_benchmark_graders()
    assert set(graders) == set(get_benchmark_task_names())
    for grader in graders.values():
        assert 0.0 <= float(grader([])) <= 1.0
