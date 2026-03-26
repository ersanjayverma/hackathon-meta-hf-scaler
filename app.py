from __future__ import annotations

from fastapi import FastAPI

from agents.heuristic_agent import HeuristicAgent
from environments.email_triage_env import EmailTriageEnv
from openenv.tasks import get_email_tasks, get_graders

app = FastAPI(title="OpenEnv Email Triage Benchmark", version="1.0.0")


@app.get("/")
def root() -> dict[str, object]:
    return {
        "name": "OpenEnv Email Triage Benchmark",
        "version": "1.0.0",
        "tasks": [task.name for task in get_email_tasks()],
        "tag": "openenv",
    }


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/tasks")
def tasks() -> list[dict[str, object]]:
    return [task.model_dump(mode="json") for task in get_email_tasks()]


@app.post("/run/heuristic")
def run_heuristic() -> dict[str, object]:
    results: list[dict[str, object]] = []
    for task in get_email_tasks():
        env = EmailTriageEnv(task=task, seed=task.seed)
        observation = env.reset()
        agent = HeuristicAgent()
        done = False
        total_reward = 0.0
        while not done:
            action = agent.act(observation)
            observation, reward, done, _ = env.step(action)
            total_reward += reward.total
        score = get_graders()[task.name](env.trajectory)
        results.append(
            {
                "task": task.name,
                "seed": task.seed,
                "total_reward": total_reward,
                "score": score,
            }
        )
        env.close()
    average_score = sum(item["score"] for item in results) / len(results)
    return {"results": results, "average_score": average_score}
