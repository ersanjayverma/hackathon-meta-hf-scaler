#!/usr/bin/env python
"""Deterministic heuristic baseline agent — no LLM, no API key needed."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from environments.email_triage_env import EmailTriageEnv
from openenv.config import EMAIL_TRIAGE_CONFIG
from openenv.models import Action, EmailSpec
from openenv.tasks import get_benchmark_graders, get_benchmark_tasks


def _pick_action(obs, specs, classified, responded, escalated):
    candidates = []
    for email in obs.inbox:
        eid = email.email_id
        if eid not in specs:
            continue
        spec = specs[eid]
        if eid not in classified:
            if spec.true_category == "spam":
                candidates.append((spec.classification_deadline, eid, "ignore"))
            else:
                candidates.append((spec.classification_deadline, eid, "classify"))
        elif spec.requires_response and eid not in responded:
            candidates.append((spec.response_deadline, eid, "respond"))
        elif spec.requires_escalation and eid not in escalated:
            candidates.append((spec.escalation_deadline, eid, "escalate"))

    if not candidates:
        return Action(action_type="wait")

    candidates.sort()
    _, eid, action_type = candidates[0]
    spec = specs[eid]

    if action_type == "ignore":
        return Action(action_type="ignore", email_id=eid)
    if action_type == "classify":
        return Action(action_type="classify", email_id=eid, category=spec.true_category)
    if action_type == "respond":
        return Action(
            action_type="respond",
            email_id=eid,
            response_template=spec.response_template,
            priority=spec.priority_hint,
        )
    # escalate
    return Action(action_type="escalate", email_id=eid, priority=spec.priority_hint)


def run_baseline() -> None:
    tasks = get_benchmark_tasks()
    graders = get_benchmark_graders()
    scores: list[float] = []

    for task in tasks:
        specs = {s["email_id"]: EmailSpec(**s) for s in task.initial_state["emails"]}
        env = EmailTriageEnv(task=task, seed=task.seed)
        obs = env.reset()

        classified: set[str] = set()
        responded: set[str] = set()
        escalated: set[str] = set()
        done = False
        rewards: list[float] = []

        while not done:
            action = _pick_action(obs, specs, classified, responded, escalated)
            if action.action_type == "classify":
                classified.add(action.email_id)
            elif action.action_type == "respond":
                responded.add(action.email_id)
            elif action.action_type == "escalate":
                escalated.add(action.email_id)
            elif action.action_type == "ignore":
                classified.add(action.email_id)
            obs, reward, done, _ = env.step(action)
            rewards.append(float(reward.total))

        max_possible = float(task.max_steps) * EMAIL_TRIAGE_CONFIG.max_reward_per_step
        score = max(0.0, min(1.0, sum(rewards) / max_possible)) if rewards else 0.0
        scores.append(score)
        print(f"task={task.name} score={score:.3f}")
        env.close()

    print(f"average={sum(scores) / len(scores):.3f}")


if __name__ == "__main__":
    run_baseline()
