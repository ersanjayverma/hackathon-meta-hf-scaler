from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class DiscreteSpace:
    n: int

    def sample(self, rng: np.random.Generator | None = None) -> int:
        generator = rng or np.random.default_rng()
        return int(generator.integers(0, self.n))


@dataclass(frozen=True, slots=True)
class BoxSpace:
    low: float
    high: float
    shape: tuple[int, ...]
    dtype: type[np.floating] = np.float32

    def sample(self, rng: np.random.Generator | None = None) -> np.ndarray:
        generator = rng or np.random.default_rng()
        return generator.uniform(self.low, self.high, size=self.shape).astype(self.dtype)

