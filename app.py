from __future__ import annotations

import logging
from dataclasses import dataclass, field
from threading import RLock
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict

from baseline.run_baseline import run_baseline
from environments.email_triage_env import EmailTriageEnv
from openenv.models import Action, Observation, Reward, StepRecord
from openenv.tasks import Task, get_email_tasks, get_graders

logger = logging.getLogger(__name__)
app = FastAPI(title="OpenEnv Email Triage Benchmark", version="1.0.0")


class ResetRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_name: str | None = None
    seed: int | None = None


class BaselineRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str | None = None


class GraderRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_name: str
    trajectory: list[StepRecord]


@dataclass(slots=True)
class EnvironmentSession:
    tasks: dict[str, Task] = field(default_factory=lambda: {task.name: task for task in get_email_tasks()})
    current_task_name: str = field(default_factory=lambda: get_email_tasks()[0].name)
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
        env = self.ensure_env()
        return env.step(action)

    def state(self) -> dict[str, Any]:
        env = self.ensure_env()
        return env.state()


session = EnvironmentSession()


@app.exception_handler(HTTPException)
async def http_exception_handler(_: Request, exc: HTTPException) -> JSONResponse:
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})


@app.exception_handler(Exception)
async def unhandled_exception_handler(_: Request, exc: Exception) -> JSONResponse:
    logger.exception("unhandled_api_error")
    return JSONResponse(status_code=500, content={"detail": str(exc)})


@app.get("/")
def root() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/tasks")
def tasks() -> dict[str, Any]:
    return {
        "tasks": [
            {
                "name": task.name,
                "description": task.description,
                "difficulty": task.difficulty,
                "max_steps": task.max_steps,
                "seed": task.seed,
            }
            for task in get_email_tasks()
        ],
        "action_schema": Action.model_json_schema(),
        "observation_schema": Observation.model_json_schema(),
        "reward_schema": Reward.model_json_schema(),
    }


@app.post("/reset")
def reset(payload: ResetRequest | None = None) -> dict[str, Any]:
    request = payload or ResetRequest()
    try:
        observation = session.reset(task_name=request.task_name, seed=request.seed)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return observation.model_dump(mode="json")


@app.post("/step")
def step(action: Action) -> dict[str, Any]:
    try:
        observation, reward, done, info = session.step(action)
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {
        "observation": observation.model_dump(mode="json"),
        "reward": reward.model_dump(mode="json"),
        "done": done,
        "info": info,
    }


@app.get("/state")
def state() -> dict[str, Any]:
    return session.state()


@app.post("/baseline")
def baseline(payload: BaselineRequest | None = None) -> dict[str, Any]:
    request = payload or BaselineRequest()
    results = run_baseline(model=request.model)
    return {
        "tasks": results["tasks"],
        "average_score": results["average_score"],
        "metadata": {
            "model": results["model"],
            "backend": results["backend"],
            "api_failures": results["api_failures"],
            "fallback_actions": results["fallback_actions"],
        },
    }


@app.post("/grader")
def grader(payload: GraderRequest) -> dict[str, float | str]:
    graders = get_graders()
    if payload.task_name not in graders:
        raise HTTPException(status_code=404, detail=f"unknown task: {payload.task_name}")
    score = float(graders[payload.task_name](payload.trajectory))
    if not 0.0 <= score <= 1.0:
        raise HTTPException(status_code=500, detail=f"grader returned out-of-range score: {score}")
    return {"task_name": payload.task_name, "score": score}
