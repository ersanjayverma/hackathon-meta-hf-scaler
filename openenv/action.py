from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Optional

import numpy as np


class ActionValidationError(ValueError):
    """Raised when an action is malformed for its action space."""


class ActionSpace(ABC):
    @abstractmethod
    def sample(self, rng: np.random.Generator) -> Any:
        raise NotImplementedError

    @abstractmethod
    def validate(self, action: Any) -> None:
        raise NotImplementedError


@dataclass(slots=True)
class DiscreteActionSpace(ActionSpace):
    n: int

    def sample(self, rng: np.random.Generator) -> int:
        return int(rng.integers(0, self.n))

    def validate(self, action: Any) -> None:
        if not isinstance(action, int) or not 0 <= action < self.n:
            raise ActionValidationError(f"expected integer action in [0, {self.n})")


@dataclass(slots=True)
class ContinuousActionSpace(ActionSpace):
    low: float
    high: float
    shape: tuple[int, ...]

    def sample(self, rng: np.random.Generator) -> np.ndarray:
        return rng.uniform(self.low, self.high, self.shape)

    def validate(self, action: Any) -> None:
        array = np.asarray(action, dtype=float)
        if array.shape != self.shape:
            raise ActionValidationError(f"expected shape {self.shape}, got {array.shape}")
        if np.any(array < self.low) or np.any(array > self.high):
            raise ActionValidationError("continuous action out of bounds")


@dataclass(slots=True)
class HybridActionSpace(ActionSpace):
    spaces: Dict[str, ActionSpace]

    def sample(self, rng: np.random.Generator) -> dict[str, Any]:
        return {name: space.sample(rng) for name, space in self.spaces.items()}

    def validate(self, action: Any) -> None:
        if not isinstance(action, dict):
            raise ActionValidationError("hybrid action must be a mapping")
        for name, space in self.spaces.items():
            if name not in action:
                raise ActionValidationError(f"missing action component: {name}")
            space.validate(action[name])


@dataclass(slots=True)
class OrderFulfillmentAction:
    """Hybrid action for order fulfillment."""

    prioritized_order_ids: list[str] = field(default_factory=list)
    allocations: Dict[str, Dict[str, int]] = field(default_factory=dict)
    reorder_quantities: Dict[str, float] = field(default_factory=dict)
    expedite_shipping: bool = False

    def validate(self, known_orders: Optional[Iterable[str]] = None) -> None:
        if not isinstance(self.prioritized_order_ids, list):
            raise ActionValidationError("prioritized_order_ids must be a list")
        if not isinstance(self.allocations, dict):
            raise ActionValidationError("allocations must be a dictionary")
        if not isinstance(self.reorder_quantities, dict):
            raise ActionValidationError("reorder_quantities must be a dictionary")
        if not isinstance(self.expedite_shipping, bool):
            raise ActionValidationError("expedite_shipping must be a boolean")

        valid_orders = set(known_orders or [])
        for order_id in self.prioritized_order_ids:
            if not isinstance(order_id, str):
                raise ActionValidationError("order ids must be strings")
            if valid_orders and order_id not in valid_orders:
                raise ActionValidationError(f"unknown order id: {order_id}")

        for order_id, sku_map in self.allocations.items():
            if valid_orders and order_id not in valid_orders:
                raise ActionValidationError(f"unknown allocation order id: {order_id}")
            if not isinstance(sku_map, dict):
                raise ActionValidationError("allocation entries must be dictionaries")
            for sku, quantity in sku_map.items():
                if not isinstance(sku, str):
                    raise ActionValidationError("SKU keys must be strings")
                if not isinstance(quantity, int) or quantity < 0:
                    raise ActionValidationError("allocation quantities must be non-negative ints")

        for sku, quantity in self.reorder_quantities.items():
            if not isinstance(sku, str):
                raise ActionValidationError("reorder SKU keys must be strings")
            if float(quantity) < 0:
                raise ActionValidationError("reorder quantities must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        return {
            "prioritized_order_ids": list(self.prioritized_order_ids),
            "allocations": {
                order_id: dict(sku_map) for order_id, sku_map in self.allocations.items()
            },
            "reorder_quantities": dict(self.reorder_quantities),
            "expedite_shipping": self.expedite_shipping,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "OrderFulfillmentAction":
        return cls(
            prioritized_order_ids=list(payload.get("prioritized_order_ids", [])),
            allocations={
                str(order_id): {str(sku): int(qty) for sku, qty in sku_map.items()}
                for order_id, sku_map in payload.get("allocations", {}).items()
            },
            reorder_quantities={
                str(sku): float(qty)
                for sku, qty in payload.get("reorder_quantities", {}).items()
            },
            expedite_shipping=bool(payload.get("expedite_shipping", False)),
        )
