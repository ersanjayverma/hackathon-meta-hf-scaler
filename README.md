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

# OpenEnv Email Triage Benchmark

This repository is a complete OpenEnv environment for **real-world email triage**. It models the kind of inbox work handled by support, SRE, operations, and internal service teams:

- identify spam and noise
- acknowledge urgent incidents quickly
- request information for routine work
- escalate only when evidence justifies it
- manage delayed consequences such as follow-ups, SLA breaches, overload, and system stress

It is not a toy game. It is a deterministic, stateful benchmark for evaluating agents on operational triage behavior.

## Official benchmark tasks

The **official scored benchmark** is exactly these three canonical tasks:

1. `task_easy_classification` (`easy`)
   Mixed inbox with obvious spam, one urgent production issue, and one routine migration request.
2. `task_medium_prioritization` (`medium`)
   Prioritize an enterprise outage, handle a true escalation request, reply to routine work, and avoid vendor noise.
3. `task_hard_thread_reasoning` (`hard`)
   Track an outage thread across multiple arrivals, react to the escalation trigger at the right time, and avoid distraction from lower-value work.

Everything else in the repository is **supplemental**:

- seeded sample variants
- JSON-loaded scenarios from `scenarios/`

Supplemental scenarios are useful for experimentation, but they are **not** the canonical benchmark set.

## Why this is a real-world task

Email triage is a practical agent-evaluation problem:

- support desks must separate noise from actionable requests
- incident teams must acknowledge urgent issues fast
- escalation decisions should depend on evolving evidence, not static labels
- mistakes create delayed fallout through follow-ups, missed deadlines, and overload

That makes this environment useful for evaluating RL agents, tool-using agents, and deterministic baselines on a real operational workflow.

## OpenEnv implementation

This repo implements the required OpenEnv pieces:

- typed Pydantic models in [openenv/models.py](/home/sj/HACKATHON/openenv/models.py)
- environment API in [environments/email_triage_env.py](/home/sj/HACKATHON/environments/email_triage_env.py)
  - `reset() -> Observation`
  - `step(Action) -> (Observation, Reward, done, info)`
  - `state() -> dict`
- valid root manifest in [openenv.yaml](/home/sj/HACKATHON/openenv.yaml)
- canonical server entrypoint in [server/app.py](/home/sj/HACKATHON/server/app.py)

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

This gives agents partial-progress signal while still making long-term outcomes matter.

## Delayed consequences and statefulness

The environment is intentionally long-running and pressure-driven:

- emails receive deadlines
- wrong actions schedule future penalties
- ignored or mishandled emails can trigger follow-up emails
- repeated misses increase `stress` and `sla_breaches`
- enough pressure triggers `system_overload`
- overload can spawn additional noisy urgent work
- excessive stress ends the episode with `system_collapse`

This makes the environment suitable for RL and long-horizon agent evaluation rather than single-step grading.

## Graders

Canonical benchmark graders are defined in [openenv/tasks.py](/home/sj/HACKATHON/openenv/tasks.py).

Properties:

- one grader per canonical task
- deterministic
- bounded to `0.0–1.0`
- aligned with the official benchmark tasks only

`/tasks` exposes the canonical benchmark tasks separately from supplemental scenarios.

## Deterministic baseline

The **canonical baseline** is the deterministic heuristic agent in [agents/heuristic_agent.py](/home/sj/HACKATHON/agents/heuristic_agent.py).

Important:

- explicit `OPENENV_BASELINE_BACKEND` wins if set
- otherwise, complete external env config selects `openai`
  - `API_BASE_URL`
  - `MODEL_NAME`
  - `HF_TOKEN`
- if that env config is incomplete, the runner falls back to internal deterministic `heuristic`
- optional non-canonical backend: `openai`
- fixed seeds
- stable task ordering
- stable JSON output format

Expected canonical baseline scores:

- `task_easy_classification`: `1.00`
- `task_medium_prioritization`: `1.00`
- `task_hard_thread_reasoning`: `1.00`
- average canonical benchmark score: `1.00`

The baseline writes a stable result file to:

- [baseline/results/baseline_results.json](/home/sj/HACKATHON/baseline/results/baseline_results.json)

## API

The canonical API is served by [server/app.py](/home/sj/HACKATHON/server/app.py) on port `7860`.

Endpoints:

- `GET /` -> health status
- `GET /health` -> health status
- `GET /tasks` -> canonical benchmark tasks, supplemental tasks, schemas
- `POST /reset` -> initial observation
- `POST /step` -> next observation, reward, done, info
- `GET /state` -> current environment state
- `POST /baseline` -> canonical baseline results
- `POST /grader` -> bounded score for a trajectory

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
python inference.py
```

Optional OpenAI baseline mode:

```bash
export OPENENV_BASELINE_BACKEND=openai
export API_BASE_URL="..."
export MODEL_NAME="..."
export HF_TOKEN="..."
python inference.py
```

Default `inference.py` backend selection:

1. `OPENENV_BASELINE_BACKEND` if explicitly set
2. `openai` if `API_BASE_URL`, `MODEL_NAME`, and `HF_TOKEN` are all present
3. otherwise internal deterministic `heuristic`

## Validation and tests

Run the full reviewer path:

```bash
bash scripts/precheck.sh
```

This runs:

1. `pytest`
2. `openenv validate .`
3. `python inference.py`
4. `docker build .`

You can also run the pieces directly:

```bash
pytest -q
.venv/bin/openenv validate .
python inference.py
docker build .
```

## Docker

Root container definition:

- [Dockerfile](/home/sj/HACKATHON/Dockerfile)

Container behavior:

- installs the package
- exposes port `7860`
- runs `uvicorn server.app:app --host 0.0.0.0 --port 7860`

Example:

```bash
docker build -t openenv-email-triage .
docker run -p 7860:7860 openenv-email-triage
```

## Hugging Face Spaces deployment

This repository is configured for a **Docker Space**.

Reviewer-relevant files:

- [README.md](/home/sj/HACKATHON/README.md) front matter
- [Dockerfile](/home/sj/HACKATHON/Dockerfile)
- [server/app.py](/home/sj/HACKATHON/server/app.py)

Expected runtime behavior:

- app binds to port `7860`
- `GET /` returns `200`
- `POST /reset` returns a valid `Observation`

## Repository structure

- [openenv/models.py](/home/sj/HACKATHON/openenv/models.py): typed schemas
- [openenv/tasks.py](/home/sj/HACKATHON/openenv/tasks.py): canonical tasks, supplemental tasks, graders
- [environments/email_triage_env.py](/home/sj/HACKATHON/environments/email_triage_env.py): core environment
- [agents/heuristic_agent.py](/home/sj/HACKATHON/agents/heuristic_agent.py): deterministic canonical baseline
- [baseline/run_baseline.py](/home/sj/HACKATHON/baseline/run_baseline.py): benchmark runner
- [env/](/home/sj/HACKATHON/env): trainer-facing RL adapter
- [server/app.py](/home/sj/HACKATHON/server/app.py): canonical FastAPI app
- [scenarios/](/home/sj/HACKATHON/scenarios): supplemental JSON task scenarios
- [tests/](/home/sj/HACKATHON/tests): smoke, benchmark, RL, and contract tests
- [scripts/precheck.sh](/home/sj/HACKATHON/scripts/precheck.sh): reviewer preflight

## Quick submission checklist

- real-world task: email triage under SLA and overload pressure
- official benchmark tasks: exactly 3 canonical tasks
- OpenEnv API: `reset`, `step`, `state`
- typed models: yes
- valid manifest: yes
- deterministic graders in `0.0–1.0`: yes
- deterministic canonical baseline: yes
- `inference.py` at repo root: yes
- Dockerfile at repo root: yes
- Hugging Face Space entrypoint documented: yes
- reviewer preflight script: yes

Additional reviewer notes are in [SUBMISSION_CHECKLIST.md](/home/sj/HACKATHON/SUBMISSION_CHECKLIST.md).
