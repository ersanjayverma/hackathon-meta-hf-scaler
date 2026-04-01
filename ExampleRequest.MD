# Example Requests

Base URL:

```text
http://127.0.0.1:7860
```

## `GET /`

URL:

```text
http://127.0.0.1:7860/
```

Body:

```json
{}
```

Example:

```bash
curl -X GET "http://127.0.0.1:7860/"
```

## `GET /health`

URL:

```text
http://127.0.0.1:7860/health
```

Body:

```json
{}
```

Example:

```bash
curl -X GET "http://127.0.0.1:7860/health"
```

## `GET /tasks`

URL:

```text
http://127.0.0.1:7860/tasks
```

Body:

```json
{}
```

Example:

```bash
curl -X GET "http://127.0.0.1:7860/tasks"
```

## `POST /reset`

URL:

```text
http://127.0.0.1:7860/reset
```

### Default reset

Body:

```json
{}
```

Example:

```bash
curl -X POST "http://127.0.0.1:7860/reset" \
  -H "Content-Type: application/json" \
  -d '{}'
```

### Reset `task_easy_classification`

Body:

```json
{
  "task_name": "task_easy_classification",
  "seed": 101
}
```

Example:

```bash
curl -X POST "http://127.0.0.1:7860/reset" \
  -H "Content-Type: application/json" \
  -d '{
    "task_name": "task_easy_classification",
    "seed": 101
  }'
```

### Reset `task_medium_prioritization`

Body:

```json
{
  "task_name": "task_medium_prioritization",
  "seed": 202
}
```

Example:

```bash
curl -X POST "http://127.0.0.1:7860/reset" \
  -H "Content-Type: application/json" \
  -d '{
    "task_name": "task_medium_prioritization",
    "seed": 202
  }'
```

### Reset `task_hard_thread_reasoning`

Body:

```json
{
  "task_name": "task_hard_thread_reasoning",
  "seed": 303
}
```

Example:

```bash
curl -X POST "http://127.0.0.1:7860/reset" \
  -H "Content-Type: application/json" \
  -d '{
    "task_name": "task_hard_thread_reasoning",
    "seed": 303
  }'
```

## `POST /step`

URL:

```text
http://127.0.0.1:7860/step
```

### `wait`

Body:

```json
{
  "action_type": "wait"
}
```

Example:

```bash
curl -X POST "http://127.0.0.1:7860/step" \
  -H "Content-Type: application/json" \
  -d '{
    "action_type": "wait"
  }'
```

### `ignore`

Body:

```json
{
  "action_type": "ignore",
  "email_id": "e-001"
}
```

Example:

```bash
curl -X POST "http://127.0.0.1:7860/step" \
  -H "Content-Type: application/json" \
  -d '{
    "action_type": "ignore",
    "email_id": "e-001"
  }'
```

### `classify` spam

Body:

```json
{
  "action_type": "classify",
  "email_id": "e-001",
  "category": "spam"
}
```

Example:

```bash
curl -X POST "http://127.0.0.1:7860/step" \
  -H "Content-Type: application/json" \
  -d '{
    "action_type": "classify",
    "email_id": "e-001",
    "category": "spam"
  }'
```

### `classify` urgent

Body:

```json
{
  "action_type": "classify",
  "email_id": "e-002",
  "category": "urgent"
}
```

Example:

```bash
curl -X POST "http://127.0.0.1:7860/step" \
  -H "Content-Type: application/json" \
  -d '{
    "action_type": "classify",
    "email_id": "e-002",
    "category": "urgent"
  }'
```

### `classify` normal

Body:

```json
{
  "action_type": "classify",
  "email_id": "e-103",
  "category": "normal"
}
```

Example:

```bash
curl -X POST "http://127.0.0.1:7860/step" \
  -H "Content-Type: application/json" \
  -d '{
    "action_type": "classify",
    "email_id": "e-103",
    "category": "normal"
  }'
```

### `classify` escalation

Body:

```json
{
  "action_type": "classify",
  "email_id": "e-202",
  "category": "escalation"
}
```

Example:

```bash
curl -X POST "http://127.0.0.1:7860/step" \
  -H "Content-Type: application/json" \
  -d '{
    "action_type": "classify",
    "email_id": "e-202",
    "category": "escalation"
  }'
```

### `respond` acknowledge

Body:

```json
{
  "action_type": "respond",
  "email_id": "e-002",
  "response_template": "acknowledge",
  "priority": "critical"
}
```

Example:

```bash
curl -X POST "http://127.0.0.1:7860/step" \
  -H "Content-Type: application/json" \
  -d '{
    "action_type": "respond",
    "email_id": "e-002",
    "response_template": "acknowledge",
    "priority": "critical"
  }'
```

### `respond` resolve

Body:

```json
{
  "action_type": "respond",
  "email_id": "e-101",
  "response_template": "resolve",
  "priority": "critical"
}
```

Example:

```bash
curl -X POST "http://127.0.0.1:7860/step" \
  -H "Content-Type: application/json" \
  -d '{
    "action_type": "respond",
    "email_id": "e-101",
    "response_template": "resolve",
    "priority": "critical"
  }'
```

### `respond` request_info

Body:

```json
{
  "action_type": "respond",
  "email_id": "e-103",
  "response_template": "request_info",
  "priority": "medium"
}
```

Example:

```bash
curl -X POST "http://127.0.0.1:7860/step" \
  -H "Content-Type: application/json" \
  -d '{
    "action_type": "respond",
    "email_id": "e-103",
    "response_template": "request_info",
    "priority": "medium"
  }'
```

### `respond` escalate_notice

Body:

```json
{
  "action_type": "respond",
  "email_id": "e-202",
  "response_template": "escalate_notice",
  "priority": "critical"
}
```

Example:

```bash
curl -X POST "http://127.0.0.1:7860/step" \
  -H "Content-Type: application/json" \
  -d '{
    "action_type": "respond",
    "email_id": "e-202",
    "response_template": "escalate_notice",
    "priority": "critical"
  }'
```

### `escalate`

Body:

```json
{
  "action_type": "escalate",
  "email_id": "e-202",
  "priority": "critical"
}
```

Example:

```bash
curl -X POST "http://127.0.0.1:7860/step" \
  -H "Content-Type: application/json" \
  -d '{
    "action_type": "escalate",
    "email_id": "e-202",
    "priority": "critical"
  }'
```

## `GET /state`

URL:

```text
http://127.0.0.1:7860/state
```

Body:

```json
{}
```

Example:

```bash
curl -X GET "http://127.0.0.1:7860/state"
```

## `POST /baseline`

URL:

```text
http://127.0.0.1:7860/baseline
```

### Default baseline run

Body:

```json
{}
```

Example:

```bash
curl -X POST "http://127.0.0.1:7860/baseline" \
  -H "Content-Type: application/json" \
  -d '{}'
```

### Baseline with explicit model

Body:

```json
{
  "model": "gpt-5.2"
}
```

Example:

```bash
curl -X POST "http://127.0.0.1:7860/baseline" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-5.2"
  }'
```

## `POST /grader`

URL:

```text
http://127.0.0.1:7860/grader
```

### Grade `task_easy_classification`

Body:

```json
{
  "task_name": "task_easy_classification",
  "trajectory": [
    {
      "schema_version": "1.0.0",
      "step_index": 0,
      "action": {
        "schema_version": "1.0.0",
        "action_type": "classify",
        "email_id": "e-001",
        "category": "spam",
        "response_template": null,
        "priority": null
      },
      "reward": {
        "schema_version": "1.0.0",
        "total": 0.4,
        "components": {
          "classification": 0.3,
          "completion": 0.1
        },
        "reason": "dense trajectory-aware reward"
      },
      "observation": {
        "schema_version": "1.0.0",
        "task_name": "task_easy_classification",
        "step_index": 1,
        "max_steps": 4,
        "remaining_steps": 3,
        "seed": 101,
        "inbox": [],
        "completed_email_ids": [
          "e-001"
        ],
        "action_history": [
          {
            "schema_version": "1.0.0",
            "step_index": 0,
            "email_id": "e-001",
            "action_type": "classify",
            "summary": "classify:e-001:spam"
          }
        ]
      },
      "done": false,
      "info": {
        "task_name": "task_easy_classification",
        "reward_breakdown": {
          "classification": 0.3,
          "completion": 0.1
        },
        "pending_emails": 1,
        "backlog_size": 1,
        "seed": 101
      }
    }
  ]
}
```

Example:

```bash
curl -X POST "http://127.0.0.1:7860/grader" \
  -H "Content-Type: application/json" \
  -d '{
    "task_name": "task_easy_classification",
    "trajectory": [
      {
        "schema_version": "1.0.0",
        "step_index": 0,
        "action": {
          "schema_version": "1.0.0",
          "action_type": "classify",
          "email_id": "e-001",
          "category": "spam",
          "response_template": null,
          "priority": null
        },
        "reward": {
          "schema_version": "1.0.0",
          "total": 0.4,
          "components": {
            "classification": 0.3,
            "completion": 0.1
          },
          "reason": "dense trajectory-aware reward"
        },
        "observation": {
          "schema_version": "1.0.0",
          "task_name": "task_easy_classification",
          "step_index": 1,
          "max_steps": 4,
          "remaining_steps": 3,
          "seed": 101,
          "inbox": [],
          "completed_email_ids": [
            "e-001"
          ],
          "action_history": [
            {
              "schema_version": "1.0.0",
              "step_index": 0,
              "email_id": "e-001",
              "action_type": "classify",
              "summary": "classify:e-001:spam"
            }
          ]
        },
        "done": false,
        "info": {
          "task_name": "task_easy_classification",
          "reward_breakdown": {
            "classification": 0.3,
            "completion": 0.1
          },
          "pending_emails": 1,
          "backlog_size": 1,
          "seed": 101
        }
      }
    ]
  }'
```

### Grade `task_medium_prioritization`

Body:

```json
{
  "task_name": "task_medium_prioritization",
  "trajectory": []
}
```

Example:

```bash
curl -X POST "http://127.0.0.1:7860/grader" \
  -H "Content-Type: application/json" \
  -d '{
    "task_name": "task_medium_prioritization",
    "trajectory": []
  }'
```

### Grade `task_hard_thread_reasoning`

Body:

```json
{
  "task_name": "task_hard_thread_reasoning",
  "trajectory": []
}
```

Example:

```bash
curl -X POST "http://127.0.0.1:7860/grader" \
  -H "Content-Type: application/json" \
  -d '{
    "task_name": "task_hard_thread_reasoning",
    "trajectory": []
  }'
```
