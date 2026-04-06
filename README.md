---
title: OpenEnv Email Triage Benchmark
emoji: 📬
colorFrom: blue
colorTo: green
sdk: docker
app_file: server/app.py
pinned: false
app_port: 7860
---

# OpenEnv Email Triage Benchmark

A production-grade OpenEnv environment for **real-world email triage**, powered by LLM inference. It models operational inbox work handled by support, SRE, and internal service teams:

- identify spam and noise
- acknowledge urgent incidents quickly
- request information for routine work
- escalate only when evidence justifies it
- manage delayed consequences such as follow-ups, SLA breaches, overload, and system stress

## Official benchmark tasks

The **official scored benchmark** is exactly these three canonical tasks:

1. `task_easy_classification` (`easy`)
   Mixed inbox with obvious spam, one urgent production issue, and one routine migration request.
2. `task_medium_prioritization` (`medium`)
   Prioritize an enterprise outage, handle a true escalation request, reply to routine work, and avoid vendor noise.
3. `task_hard_thread_reasoning` (`hard`)
   Track an outage thread across multiple arrivals, react to the escalation trigger at the right time, and avoid distraction from lower-value work.

Only these three tasks are used by the benchmark runtime and graders.

## Architecture

This is an **LLM-only** system. All classification decisions are made by the configured language model (HF or OpenAI). There is no heuristic fallback in the inference path.

The LLM receives a filtered observation (only unhandled emails) and returns a single JSON action per step. Response parsing, normalization, and validation are handled inline in [inference.py](inference.py).

## OpenEnv implementation

This repo implements the required OpenEnv pieces:

- typed Pydantic models in [openenv/models.py](openenv/models.py)
- environment API in [environments/email_triage_env.py](environments/email_triage_env.py)
  - `reset() -> Observation`
  - `step(Action) -> (Observation, Reward, done, info)`
  - `state() -> dict`
- valid root manifest in [openenv.yaml](openenv.yaml)
- canonical server entrypoint in [server/app.py](server/app.py)

## Observation space

`Observation` is a versioned Pydantic model with:

- `task_name`
- `step_index`
- `max_steps`
- `remaining_steps`
- `seed`
- `inbox`
- `completed_email_ids`
- `action_history`

Each email view includes:

- `email_id`
- `sender`
- `subject`
- `body`
- `thread_id`
- `age`
- `priority_hint`
- `noise_score`
- `seen`

The environment is partially observable by design. The observation exposes useful hints, not perfect ground truth.

## Action space

`Action` is a versioned Pydantic model with these valid `action_type` values:

- `classify`
- `respond`
- `escalate`
- `ignore`
- `wait`

Action fields are strictly validated:

- `classify` requires `email_id` and `category`
- `respond` requires `email_id`, `response_template`, and `priority`
- `escalate` requires `email_id` and `priority`
- `ignore` requires `email_id`
- `wait` allows no extra fields

## Reward design

The reward is dense and multi-factor, not just a terminal score.

Positive signal:

- correct classification
- correct response template
- timely escalation
- step-level completion progress

Negative signal:

- looped or redundant actions
- waiting while urgent work is open
- missed classification/response/escalation deadlines
- delayed penalties from scheduled events
- SLA breaches
- accumulated system stress
- system collapse

## Delayed consequences and statefulness

The environment is intentionally long-running and pressure-driven:

- emails receive deadlines
- wrong actions schedule future penalties
- ignored or mishandled emails can trigger follow-up emails
- repeated misses increase `stress` and `sla_breaches`
- enough pressure triggers `system_overload`
- overload can spawn additional noisy urgent work
- excessive stress ends the episode with `system_collapse`

## Graders

Canonical benchmark graders are defined in [openenv/tasks.py](openenv/tasks.py).

Properties:

- one grader per canonical task
- deterministic
- bounded to `0.0–1.0`
- scores **action quality**, not just completion count

Each completed email is scored with a weighted breakdown:

| Component | Weight | Condition |
|---|---|---|
| Correct classification | 0.4 | `action.category == spec.true_category` |
| Correct response template | 0.4 | `action.response_template == spec.response_template` (when `requires_response`) |
| Correct escalation | 0.2 | escalation performed (when `requires_escalation`) |

The final score is the mean weighted quality across all emails in the task spec.

Use from Python:

```python
from openenv.grader import grade_action_quality, grade_processed_ids

# Quality-weighted scoring (used by benchmark graders)
score = grade_action_quality(trajectory, email_specs)

# Simple completion ratio (backward-compatible utility)
ratio = grade_processed_ids(["e-001", "e-002"], ["e-001", "e-002", "e-003"])
```

## Inference

`inference.py` is the submission runner. It is **LLM-only** - every action decision is made by the configured language model.

Features:

- auto-routes HF models (`org/name`) to HF router with `HF_TOKEN`
- auto-routes OpenAI models (`gpt-*`) to OpenAI API with `OPENAI_API_KEY`
- uses `max_tokens` for HF models, `max_completion_tokens` for OpenAI models
- filters observations to only show unhandled emails to the LLM
- stops early when all visible emails have been acted on
- uses official graders for canonical benchmark scoring

Environment variables:

| Variable | Purpose |
|---|---|
| `API_BASE_URL` | HF router URL (default: `https://router.huggingface.co/v1`) |
| `MODEL_NAME` | Model to use (e.g. `Qwen/Qwen2.5-72B-Instruct` or `gpt-5.4`) |
| `HF_TOKEN` | Hugging Face API token |
| `OPENAI_API_KEY` | OpenAI API key |
| `TASK_NAME` | Run a single canonical task (optional) |
| `MAX_STEPS` | Override step budget |
| `MAX_TOKENS` | Override max output tokens |
| `TEMPERATURE` | Override LLM temperature |
| `SUCCESS_SCORE_THRESHOLD` | Override success threshold (default: 0.95) |

Run:

```bash
python inference.py
```

Output protocol:

```text
[START] task=<task_name> env=<benchmark> model=<model_name>
[STEP] step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
[END] success=<true|false> steps=<n> score=<0.000> rewards=<r1,r2,...,rn>
```

## API

The canonical API is implemented in [server/app.py](server/app.py) and exposed on port `7860`.

Endpoints:

- `GET /` -> health status
- `GET /health` -> health status
- `GET /tasks` -> canonical benchmark tasks and schemas
- `POST /reset` -> initial observation
- `POST /step` -> next observation, reward, done, info
- `GET /state` -> current environment state

## Local setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

## Run locally

Canonical server:

```bash
uv run server
```

Submission entrypoint:

```bash
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
export HF_TOKEN="hf_..."
python inference.py
```

Or with OpenAI:

```bash
export MODEL_NAME="gpt-5.4"
export OPENAI_API_KEY="sk-..."
python inference.py
```

## Validation and tests

```bash
bash scripts/precheck.sh
```

This runs:

1. `pytest`
2. `python inference.py`
3. `docker build .`

You can also run the pieces directly:

```bash
pytest -q
python inference.py
docker build .
```

## Heuristic baseline

A deterministic heuristic agent (no LLM, no API key) is included at [baseline/run_baseline.py](baseline/run_baseline.py).

It picks the unfinished email with the lowest deadline at each step and applies the correct action sequence (classify → respond → escalate, or ignore for spam).

```bash
python baseline/run_baseline.py
```

Output:

```text
task=task_easy_classification score=0.800
task=task_medium_prioritization score=0.840
task=task_hard_thread_reasoning score=0.920
average=0.853
```

## Docker

Root container definition: [Dockerfile](Dockerfile)

Container behavior:

- installs pinned dependencies from `requirements.txt` first for stable resolution
- installs the package with `--no-deps`
- exposes port `7860`
- runs `uvicorn server.app:app --host 0.0.0.0 --port 7860`

```bash
docker build -t openenv-email-triage .
docker run -p 7860:7860 openenv-email-triage
```

## Repository structure

- [inference.py](inference.py): LLM inference entrypoint (self-contained with parsing, normalization, and API routing)
- [openenv/models.py](openenv/models.py): typed Pydantic schemas
- [openenv/tasks.py](openenv/tasks.py): canonical tasks and grader mapping
- [openenv/grader.py](openenv/grader.py): deterministic grader utilities
- [openenv/config.py](openenv/config.py): benchmark metadata and reward config
- [openenv/engine.py](openenv/engine.py): event queue and metrics engine
- [openenv/base_env.py](openenv/base_env.py): abstract environment contract
- [openenv/replay.py](openenv/replay.py): episode recording and replay
- [openenv/logger.py](openenv/logger.py): structured JSON logging
- [openenv/runtime_config.py](openenv/runtime_config.py): environment variable config helpers
- [environments/email_triage_env.py](environments/email_triage_env.py): core email triage environment
- [server/app.py](server/app.py): canonical FastAPI server
- [baseline/run_baseline.py](baseline/run_baseline.py): deterministic heuristic baseline (no LLM)
- [tests/](tests): environment, grader, inference, API, and contract tests
- [scripts/precheck.sh](scripts/precheck.sh): reviewer preflight script

## Quick submission checklist

- real-world task: email triage under SLA and overload pressure
- official benchmark tasks: exactly 3 canonical tasks
- OpenEnv API: `reset`, `step`, `state`
- typed models: yes
- valid manifest: yes
- deterministic graders in `0.0–1.0`: yes
- LLM-only inference: yes
- `inference.py` at repo root: yes
- Dockerfile at repo root: yes
- reviewer preflight script: yes

Additional reviewer notes are in [SUBMISSION_CHECKLIST.md](SUBMISSION_CHECKLIST.md).
