from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from openenv.models import Action, Observation, Reward


class AgentAction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: Action
    policy_info: dict[str, Any] = Field(default_factory=dict)


class EnvironmentState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    observation: Observation
    history_length: int = Field(ge=0)
    metadata: dict[str, Any] = Field(default_factory=dict)


class EnvironmentStepResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    next_state: EnvironmentState
    reward: Reward
    done: bool
    info: dict[str, Any] = Field(default_factory=dict)
