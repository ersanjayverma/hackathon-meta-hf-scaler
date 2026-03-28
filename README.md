---
title: OpenEnv Email Triage Benchmark
emoji: 📬
colorFrom: blue
colorTo: green
sdk: docker
sdk_version: "0.0.0"
app_file: server/app.py
pinned: false
app_port: 7860
---

# OpenEnv

OpenEnv is a benchmarkable real-world environment framework for agent evaluation. This repository follows a scaffold-style layout with `server/app.py` as the canonical service entrypoint and a typed email-triage benchmark underneath it.

## Environment description and motivation

Most agent benchmarks focus on puzzles or synthetic tool calls. Email triage is operationally real: support teams, SRE rotations, internal tooling desks, and managed service providers all need systems that can classify noise, respond correctly, prioritize urgent issues, and escalate only when warranted. This benchmark is built to evaluate those behaviors with deterministic tasks and inspectable scoring.

## Observation space

`Observation` is a versioned Pydantic model with:

- `task_name`, `step_index`, `max_steps`, `remaining_steps`, `seed`
- `inbox`: visible emails with sender, subject, body, thread, age, priority hint, and noise score
- `completed_email_ids`
- `action_history`

All observations are JSON-safe and schema-validated.

## Action space

`Action` is a versioned Pydantic model with:

- `action_type`: `classify`, `respond`, `escalate`, `ignore`, `wait`
- `email_id`: required for non-`wait`
- `category`: required only for `classify`
- `response_template`: required only for `respond`
- `priority`: required for `respond` and `escalate`

Validation forbids invalid field combinations.

## Reward model

`Reward` is a versioned Pydantic model with:

- `total`
- `components`
- `reason`

Reward is dense and trajectory-aware. It includes positive terms for correct classification, correct responses, timely actions, and correct escalation, plus negative terms for delays, loops, redundant actions, and unnecessary actions.

## What is included

- Strict OpenEnv API: `reset() -> Observation`, `step(Action) -> (Observation, Reward, done, info)`, `state() -> dict`
- Versioned Pydantic models for `Observation`, `Action`, `Reward`, task definitions, and recorded steps
- Real-world email triage environment with spam filtering, urgent handling, response selection, and escalation timing
- Three benchmark tasks with deterministic graders that score from `0.0` to `1.0`
- Dense reward shaping with step-level reward breakdowns
- `openenv validate ./environments` CLI for metadata, schema, determinism, and API checks
- Replay recording, JSON logs, action traces, and deterministic seeds in outputs
- Baseline script using the OpenAI Responses API with structured output parsing
- Unit tests, `pyproject.toml`, and a working `Dockerfile`

## Install

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

## Run the local demo

```bash
python main.py
```

## Run the server locally

```bash
uv run server
```

## Validate the environment

```bash
openenv validate ./environments
```

## Run tests

```bash
pytest
```

## Run the OpenAI baseline

```bash
export OPENAI_API_KEY=...
python baseline/run_baseline.py
```

The baseline writes `baseline/results/baseline_results.json` with per-task scores, the average score, action traces, rewards, and seeds.

## Tasks

- `task_easy_classification` (`easy`): classify obvious spam and urgent mail
- `task_medium_prioritization` (`medium`): choose the right order of operations and response templates
- `task_hard_thread_reasoning` (`hard`): track a thread across multiple arrivals and escalate only when evidence warrants it

## Baseline scores

The bundled heuristic policy is deterministic and serves as the reference local baseline for the current task set:

- `task_easy_classification`: `1.00`
- `task_medium_prioritization`: `0.85`
- `task_hard_thread_reasoning`: `1.00`
- average heuristic score: `0.95`

The OpenAI baseline script writes model-specific benchmark results to `baseline/results/baseline_results.json`.

## Repository layout

- `server/app.py`: canonical FastAPI application and `server` entrypoint
- `server/Dockerfile`: scaffold-style server container definition
- `server/requirements.txt`: runtime dependency snapshot for server deployments
- `openenv.yaml`: root environment manifest for validators running from repo root
- `environments/openenv.yaml`: environment-local manifest kept in sync with the root manifest
- `outputs/`: gitkept output directories for logs and eval artifacts

## Hugging Face Space / container usage

This repository is configured for a Docker-based Hugging Face Space tagged `openenv`.

- `README.md` contains HF Space front matter
- `Dockerfile` starts `server.app:app` on port `7860`
- `server/app.py` exposes `/`, `/health`, `/tasks`, `/reset`, `/step`, `/state`, `/baseline`, and `/grader`

Local container commands:

```bash
docker build -t openenv-email-triage .
docker run -p 7860:7860 openenv-email-triage
```

## Notes on OpenAI API usage

The baseline uses the official Python SDK with the Responses API and structured parsing via `client.responses.parse(...)`, consistent with current OpenAI docs:

- Responses API examples: https://platform.openai.com/docs/api-reference/responses/list?lang=python
- Text generation guide: https://platform.openai.com/docs/guides/text?lang=python
- Structured outputs guide: https://platform.openai.com/docs/guides/structured-outputs?api-mode=responses&lang=python
