from __future__ import annotations

from typing import Any

import numpy as np
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
    vector_observation: list[float] = Field(default_factory=list)
    action_mask: list[int] = Field(default_factory=list)
    text_observation: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def vector(self) -> np.ndarray:
        return np.asarray(self.vector_observation, dtype=np.float32)


class EnvironmentStepResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    next_state: EnvironmentState
    reward: Reward
    done: bool
    info: dict[str, Any] = Field(default_factory=dict)
