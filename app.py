from __future__ import annotations

import logging
import json
from json import JSONDecodeError
from dataclasses import dataclass, field
from threading import RLock
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, ValidationError

from baseline.run_baseline import run_baseline
from environments.email_triage_env import EmailTriageEnv
from openenv.models import Action, Observation, Reward, StepRecord
from openenv.tasks import Task, get_email_tasks, get_graders

logger = logging.getLogger(__name__)
request_logger = logging.getLogger("uvicorn.error")
app = FastAPI(title="OpenEnv Email Triage Benchmark", version="1.0.0")
TASKS = get_email_tasks()
TASKS_BY_NAME = {task.name: task for task in TASKS}
SPACE_PREFIX = "/spaces/{owner}/{space_name}"


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
    tasks: dict[str, Task] = field(default_factory=lambda: dict(TASKS_BY_NAME))
    current_task_name: str = field(default_factory=lambda: TASKS[0].name)
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


@app.middleware("http")
async def log_request_body(request: Request, call_next: Any) -> JSONResponse:
    body = await request.body()
    request_logger.info(
        "request method=%s path=%s content_type=%s body=%r",
        request.method,
        request.url.path,
        request.headers.get("content-type"),
        body.decode("utf-8", errors="replace"),
    )
    request._body = body
    return await call_next(request)


@app.exception_handler(HTTPException)
async def http_exception_handler(_: Request, exc: HTTPException) -> JSONResponse:
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})


@app.exception_handler(Exception)
async def unhandled_exception_handler(_: Request, exc: Exception) -> JSONResponse:
    request_logger.exception("unhandled_api_error")
    return JSONResponse(status_code=500, content={"detail": str(exc)})


@app.get("/")
@app.get(SPACE_PREFIX)
def root() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/health")
@app.get(f"{SPACE_PREFIX}/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/tasks")
@app.get(f"{SPACE_PREFIX}/tasks")
def tasks() -> dict[str, Any]:
    return {
        "tasks": [_task_summary(task) for task in TASKS],
        "action_schema": Action.model_json_schema(),
        "observation_schema": Observation.model_json_schema(),
        "reward_schema": Reward.model_json_schema(),
    }


@app.post("/reset")
@app.post(f"{SPACE_PREFIX}/reset")
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
    except JSONDecodeError as exc:
        payload = ResetRequest()
    try:
        observation = session.reset(task_name=payload.task_name, seed=payload.seed)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return _json_model(observation)


@app.post("/step")
@app.post(f"{SPACE_PREFIX}/step")
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
@app.get(f"{SPACE_PREFIX}/state")
def state() -> dict[str, Any]:
    return session.state()


@app.post("/baseline")
@app.post(f"{SPACE_PREFIX}/baseline")
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
@app.post(f"{SPACE_PREFIX}/grader")
def grader(payload: GraderRequest) -> dict[str, float | str]:
    graders = get_graders()
    if payload.task_name not in graders:
        raise HTTPException(status_code=404, detail=f"unknown task: {payload.task_name}")
    score = float(graders[payload.task_name](payload.trajectory))
    if not 0.0 <= score <= 1.0:
        raise HTTPException(status_code=500, detail=f"grader returned out-of-range score: {score}")
    return {"task_name": payload.task_name, "score": score}
