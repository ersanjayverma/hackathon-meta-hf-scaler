from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

from agents.heuristic_agent import HeuristicAgent
from baseline.run_baseline import BASELINE_SYSTEM_PROMPT
from environments.email_triage_env import EmailTriageEnv
from openenv.models import Action, Observation
from openenv.tasks import Task, get_benchmark_tasks


def render_llm_user_prompt(observation: Observation) -> str:
    return "\n\n".join(
        [
            "Choose the next email triage action.",
            "Return exactly one JSON object matching the action schema.",
            f"Observation JSON:\n{observation.model_dump_json(indent=2)}",
        ]
    )


def action_to_json(action: Action) -> str:
    return json.dumps(action.model_dump(mode="json", exclude_none=True), separators=(",", ":"))


def build_training_example(
    *,
    task: Task,
    episode_index: int,
    seed: int,
    observation: Observation,
    action: Action,
    reward: float,
    done: bool,
) -> dict[str, object]:
    user_prompt = render_llm_user_prompt(observation)
    assistant_output = action_to_json(action)
    return {
        "messages": [
            {"role": "system", "content": BASELINE_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": assistant_output},
        ],
        "metadata": {
            "task_name": task.name,
            "difficulty": task.difficulty,
            "episode_index": episode_index,
            "seed": seed,
            "step_index": observation.step_index,
            "reward": reward,
            "done": done,
        },
    }


def iter_training_examples(tasks: Iterable[Task], *, episodes_per_task: int = 1) -> Iterable[dict[str, object]]:
    teacher = HeuristicAgent()
    for task in tasks:
        for episode_index in range(episodes_per_task):
            seed = task.seed + episode_index
            env = EmailTriageEnv(task=task, seed=seed)
            try:
                observation = env.reset()
                done = False
                while not done:
                    action = teacher.act(observation)
                    next_observation, reward, done, _ = env.step(action)
                    yield build_training_example(
                        task=task,
                        episode_index=episode_index,
                        seed=seed,
                        observation=observation,
                        action=action,
                        reward=float(reward.total),
                        done=done,
                    )
                    observation = next_observation
            finally:
                env.close()


def export_sft_dataset(output_path: str | Path, *, episodes_per_task: int = 1) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for example in iter_training_examples(get_benchmark_tasks(), episodes_per_task=episodes_per_task):
            handle.write(json.dumps(example, sort_keys=True))
            handle.write("\n")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Export simple chat-style LLM training data from benchmark rollouts.")
    parser.add_argument(
        "--output",
        default="outputs/datasets/email_triage_sft.jsonl",
        help="Path to the JSONL dataset to write.",
    )
    parser.add_argument(
        "--episodes-per-task",
        type=int,
        default=1,
        help="Number of teacher rollouts to export for each benchmark task.",
    )
    args = parser.parse_args()
    if args.episodes_per_task <= 0:
        raise ValueError("episodes-per-task must be positive")

    output_path = export_sft_dataset(args.output, episodes_per_task=args.episodes_per_task)
    print(f"dataset_file={output_path}")


if __name__ == "__main__":
    main()
