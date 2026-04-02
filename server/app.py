from __future__ import annotations

import json
from dataclasses import dataclass, field
from json import JSONDecodeError
from threading import RLock
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict, ValidationError

from environments.email_triage_env import EmailTriageEnv
from openenv.models import Action, Observation, Reward
from openenv.runtime_config import DEFAULT_PORT, runtime_port
from openenv.tasks import Task, get_benchmark_task_names, get_benchmark_tasks

app = FastAPI(title="OpenEnv Email Triage Benchmark", version="1.0.0")
BENCHMARK_TASKS = get_benchmark_tasks()
TASKS_BY_NAME = {task.name: task for task in BENCHMARK_TASKS}


def _json_model(model: BaseModel) -> dict[str, Any]:
    return model.model_dump(mode="json")


def _task_summary(task: Task) -> dict[str, Any]:
    return {
        "name": task.name,
        "description": task.description,
        "difficulty": task.difficulty,
        "max_steps": task.max_steps,
        "seed": task.seed,
    }


class ResetRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    task_name: str | None = None
    seed: int | None = None


@dataclass(slots=True)
class EnvironmentSession:
    tasks: dict[str, Task] = field(default_factory=lambda: dict(TASKS_BY_NAME))
    current_task_name: str = field(default_factory=lambda: BENCHMARK_TASKS[0].name)
    env: EmailTriageEnv | None = None
    lock: RLock = field(default_factory=RLock)

    def reset(self, task_name: str | None = None, seed: int | None = None) -> Observation:
        with self.lock:
            selected_name = task_name or self.current_task_name
            task = self.tasks.get(selected_name)
            if task is None:
                raise KeyError(f"unknown task: {selected_name}")
            if self.env is not None:
                self.env.close()
            self.current_task_name = task.name
            self.env = EmailTriageEnv(task=task, seed=seed if seed is not None else task.seed)
            return self.env.reset()

    def ensure_env(self) -> EmailTriageEnv:
        with self.lock:
            if self.env is None:
                self.reset()
            assert self.env is not None
            return self.env

    def step(self, action: Action) -> tuple[Observation, Reward, bool, dict[str, Any]]:
        return self.ensure_env().step(action)

    def state(self) -> dict[str, Any]:
        return self.ensure_env().state()


session = EnvironmentSession()


@app.get("/")
def root() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/tasks")
def tasks() -> dict[str, Any]:
    return {
        "tasks": [_task_summary(task) for task in BENCHMARK_TASKS],
        "benchmark_task_names": list(get_benchmark_task_names()),
        "action_schema": Action.model_json_schema(),
        "observation_schema": Observation.model_json_schema(),
        "reward_schema": Reward.model_json_schema(),
    }


@app.post("/reset")
async def reset(request: Request) -> dict[str, Any]:
    try:
        raw_body = await request.body()
        if not raw_body or not raw_body.strip():
            payload = ResetRequest()
        else:
            parsed = json.loads(raw_body)
            payload = ResetRequest() if not isinstance(parsed, dict) else ResetRequest.model_validate(parsed)
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=exc.errors()) from exc
    except JSONDecodeError:
        payload = ResetRequest()
    try:
        observation = session.reset(task_name=payload.task_name, seed=payload.seed)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return _json_model(observation)


@app.post("/step")
def step(action: Action) -> dict[str, Any]:
    try:
        observation, reward, done, info = session.step(action)
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {
        "observation": _json_model(observation),
        "reward": _json_model(reward),
        "done": done,
        "info": info,
    }


@app.get("/state")
def state() -> dict[str, Any]:
    return session.state()


def main() -> None:
    uvicorn.run(app, host="0.0.0.0", port=runtime_port(DEFAULT_PORT))


if __name__ == "__main__":
    main()
