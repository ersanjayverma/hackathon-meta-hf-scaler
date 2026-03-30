from __future__ import annotations

from typing import Any, Callable

from pydantic import BaseModel, ConfigDict, Field

from openenv.models import Reward, StepRecord


class GradedReward(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reward: Reward
    grader_score: float | None = Field(default=None, ge=0.0, le=1.0)
    terminal_bonus: float = 0.0


def compute_reward(reward: Reward, done: bool, grader_score: float | None = None) -> GradedReward:
    terminal_bonus = 0.0
    if done and grader_score is not None:
        terminal_bonus = grader_score * 0.1
    total_reward = Reward(
        total=reward.total + terminal_bonus,
        components={**reward.components, "terminal_bonus": terminal_bonus} if terminal_bonus else dict(reward.components),
        reason=reward.reason,
    )
    return GradedReward(reward=total_reward, grader_score=grader_score, terminal_bonus=terminal_bonus)


def score_trajectory(
    trajectory: list[StepRecord],
    grader: Callable[[list[StepRecord]], float] | None = None,
) -> float | None:
    if grader is None or not trajectory:
        return None
    score = float(grader(trajectory))
    if not 0.0 <= score <= 1.0:
        raise ValueError(f"grader returned out-of-range score: {score}")
    return score
