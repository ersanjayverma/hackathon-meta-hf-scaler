from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from openenv.grader import grade_processed_ids
from openenv.tasks import get_benchmark_tasks


def main() -> None:
    parser = argparse.ArgumentParser(description="Grade processed email ids for a benchmark task.")
    parser.add_argument("--task-name", required=True, help="Benchmark task name.")
    parser.add_argument(
        "--processed-id",
        action="append",
        default=[],
        help="Processed email id. Repeat for multiple ids.",
    )
    args = parser.parse_args()

    tasks = {task.name: task for task in get_benchmark_tasks()}
    if args.task_name not in tasks:
        raise ValueError(f"unknown task: {args.task_name}")

    expected_ids = [email["email_id"] for email in tasks[args.task_name].initial_state["emails"]]
    score = grade_processed_ids(args.processed_id, expected_ids)
    print(f"score={score:.2f}")


if __name__ == "__main__":
    main()
