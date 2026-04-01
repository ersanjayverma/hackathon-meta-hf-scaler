# Submission Checklist

This file is optimized for a reviewer who wants to verify the submission quickly.

## Canonical benchmark

Official scored benchmark tasks:

1. `task_easy_classification`
2. `task_medium_prioritization`
3. `task_hard_thread_reasoning`

Source of truth:

- [openenv/tasks.py](/home/sj/HACKATHON/openenv/tasks.py)
- `get_benchmark_tasks()`
- `get_benchmark_graders()`

Supplemental scenarios exist, but they are not part of the canonical benchmark score.

## OpenEnv compliance

Required implementation:

- typed models in [openenv/models.py](/home/sj/HACKATHON/openenv/models.py)
- environment methods in [environments/email_triage_env.py](/home/sj/HACKATHON/environments/email_triage_env.py)
  - `reset()`
  - `step()`
  - `state()`
- manifest at [openenv.yaml](/home/sj/HACKATHON/openenv.yaml)

Validation command:

```bash
.venv/bin/openenv validate .
```

## Deterministic baseline

Canonical baseline:

- backend: `heuristic`
- entrypoint: [inference.py](/home/sj/HACKATHON/inference.py)
- runner: [baseline/run_baseline.py](/home/sj/HACKATHON/baseline/run_baseline.py)

Expected canonical scores:

- `task_easy_classification`: `1.00`
- `task_medium_prioritization`: `1.00`
- `task_hard_thread_reasoning`: `1.00`
- average: `1.00`

Result file:

- [baseline/results/baseline_results.json](/home/sj/HACKATHON/baseline/results/baseline_results.json)

Optional OpenAI comparison mode is available, but it is not the canonical submission baseline.

## API smoke path

Canonical app:

- [server/app.py](/home/sj/HACKATHON/server/app.py)

Important endpoints:

- `GET /`
- `GET /tasks`
- `POST /reset`
- `POST /step`
- `GET /state`
- `POST /baseline`
- `POST /grader`

## Docker and Spaces

Root Dockerfile:

- [Dockerfile](/home/sj/HACKATHON/Dockerfile)

Port:

- `7860`

Hugging Face Space front matter:

- [README.md](/home/sj/HACKATHON/README.md)

## One-command reviewer validation

```bash
bash scripts/precheck.sh
```

This runs:

1. `pytest`
2. `openenv validate .`
3. `python inference.py`
4. `docker build .`
