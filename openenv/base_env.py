from __future__ import annotations

import json
import threading
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Generic, Optional, Tuple, TypeVar

import numpy as np

from .engine import EnvironmentEngine
from .models import Action, Observation, Reward

ObservationT = TypeVar("ObservationT", bound=Observation)
ActionT = TypeVar("ActionT", bound=Action)
RewardT = TypeVar("RewardT", bound=Reward)


class BaseEnv(ABC, Generic[ObservationT, ActionT, RewardT]):
    """Strict OpenEnv contract with deterministic seeding and serializable state."""

    def __init__(self, engine: EnvironmentEngine, seed: Optional[int] = None) -> None:
        self._engine = engine
        self._lock = threading.RLock()
        self._seed = seed if seed is not None else 0
        self._rng = np.random.default_rng(self._seed)

    def set_seed(self, seed: int) -> None:
        with self._lock:
            self._seed = seed
            self._rng = np.random.default_rng(seed)

    @abstractmethod
    def reset(self) -> ObservationT:
        raise NotImplementedError

    @abstractmethod
    def step(self, action: ActionT) -> Tuple[ObservationT, RewardT, bool, dict]:
        raise NotImplementedError

    @abstractmethod
    def state(self) -> dict:
        raise NotImplementedError

    @abstractmethod
    def render(self, mode: str = "human") -> None:
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def snapshot(self) -> dict:
        raise NotImplementedError

    @abstractmethod
    def restore(self, snapshot: dict) -> None:
        raise NotImplementedError

    @abstractmethod
    def default_validation_action(self, observation: ObservationT) -> ActionT:
        raise NotImplementedError

    def save_state(self, path: str | Path) -> None:
        with self._lock:
            Path(path).write_text(
                json.dumps(self.snapshot(), indent=2, sort_keys=True),
                encoding="utf-8",
            )

    def load_state(self, path: str | Path) -> None:
        with self._lock:
            snapshot = json.loads(Path(path).read_text(encoding="utf-8"))
            self.restore(snapshot)
