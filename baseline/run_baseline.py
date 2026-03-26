from __future__ import annotations

import json
import os
from pathlib import Path

from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field

from environments.email_triage_env import EmailTriageEnv
from openenv.models import Action, Observation
from openenv.tasks import get_email_tasks, get_graders


class BaselineDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action_type: str = Field(pattern="^(classify|respond|escalate|ignore|wait)$")
    email_id: str | None = None
    category: str | None = None
    response_template: str | None = None
    priority: str | None = None


def choose_action(client: OpenAI, observation: Observation, model: str) -> Action:
    response = client.responses.parse(
        model=model,
        input=[
            {
                "role": "system",
                "content": (
                    "You are an email triage agent. Produce one valid JSON action that maximizes benchmark score. "
                    "Avoid redundant actions and prioritize urgent mail."
                ),
            },
            {
                "role": "user",
                "content": observation.model_dump_json(indent=2),
            },
        ],
        text_format=BaselineDecision,
    )
    decision = response.output_parsed
    return Action(
        action_type=decision.action_type,
        email_id=decision.email_id,
        category=decision.category,
        response_template=decision.response_template,
        priority=decision.priority,
    )


def run() -> None:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required")

    model = os.environ.get("OPENAI_MODEL", "gpt-5.2")
    client = OpenAI(api_key=api_key)
    results: dict[str, object] = {"model": model, "tasks": [], "average_score": 0.0}
    scores: list[float] = []

    for task in get_email_tasks():
        env = EmailTriageEnv(task=task, seed=task.seed)
        observation = env.reset()
        done = False
        total_reward = 0.0
        action_log: list[dict[str, object]] = []

        while not done:
            action = choose_action(client, observation, model=model)
            observation, reward, done, info = env.step(action)
            total_reward += reward.total
            action_log.append(
                {
                    "step": observation.step_index,
                    "action": action.model_dump(mode="json"),
                    "reward": reward.model_dump(mode="json"),
                    "info": info,
                }
            )
            print(
                f"task={task.name} step={observation.step_index} action={action.action_type} "
                f"reward={reward.total:.2f}"
            )

        score = get_graders()[task.name](env.trajectory)
        scores.append(score)
        task_result = {
            "task": task.name,
            "seed": task.seed,
            "score": score,
            "total_reward": total_reward,
            "actions": action_log,
        }
        results["tasks"].append(task_result)
        print(f"task={task.name} final_score={score:.3f} total_reward={total_reward:.2f}")
        env.close()

    results["average_score"] = sum(scores) / len(scores)
    output_dir = Path("baseline/results")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "baseline_results.json"
    output_path.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
    print(f"average_score={results['average_score']:.3f}")
    print(f"results_file={output_path}")


if __name__ == "__main__":
    run()
