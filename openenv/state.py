from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


def _restore_dataclass(cls: type[Any], value: Any) -> Any:
    if not is_dataclass(cls):
        return value
    kwargs: dict[str, Any] = {}
    for name, field_info in cls.__dataclass_fields__.items():  # type: ignore[attr-defined]
        if name not in value:
            continue
        field_value = value[name]
        field_type = field_info.type
        origin = getattr(field_type, "__origin__", None)
        if is_dataclass(field_type):
            kwargs[name] = _restore_dataclass(field_type, field_value)
        elif origin is list and getattr(field_type, "__args__", ()):
            inner = field_type.__args__[0]
            kwargs[name] = [
                _restore_dataclass(inner, item) if is_dataclass(inner) else item
                for item in field_value
            ]
        elif origin is dict:
            kwargs[name] = dict(field_value)
        else:
            kwargs[name] = field_value
    return cls(**kwargs)


@dataclass(slots=True)
class SerializableDataclass:
    """Mixin for dataclass-based JSON-serializable models."""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Any:
        return _restore_dataclass(cls, payload)

    @classmethod
    def from_json(cls, payload: str) -> Any:
        return cls.from_dict(json.loads(payload))


@dataclass(slots=True)
class InventoryItemState(SerializableDataclass):
    sku: str
    on_hand: int
    reserved: int
    incoming: int


@dataclass(slots=True)
class OrderRecord(SerializableDataclass):
    order_id: str
    sku: str
    quantity: int
    priority: int
    created_at: int
    due_at: int
    allocated: int = 0
    shipped: bool = False
    shipped_at: Optional[int] = None
    lead_time: int = 0

    @property
    def remaining(self) -> int:
        return max(self.quantity - self.allocated, 0)


@dataclass(slots=True)
class ScheduledReceipt(SerializableDataclass):
    sku: str
    quantity: int
    arrival_tick: int
    source: str = "reorder"


@dataclass(slots=True)
class OrderFulfillmentState(SerializableDataclass):
    tick: int
    inventory: Dict[str, InventoryItemState]
    pending_orders: list[OrderRecord]
    shipped_orders: list[OrderRecord]
    backlog_size: int
    total_reward: float
    capacity_remaining: int
    recent_events: list[dict[str, Any]] = field(default_factory=list)
    rng_state: Optional[dict[str, Any]] = None


@dataclass(slots=True)
class OrderDistributionConfig(SerializableDataclass):
    arrival_rate: float = 2.0
    quantity_low: int = 1
    quantity_high: int = 5
    priority_levels: int = 3
    due_window_low: int = 2
    due_window_high: int = 6
    delivery_delay_low: int = 1
    delivery_delay_high: int = 3


@dataclass(slots=True)
class OrderFulfillmentConfig(SerializableDataclass):
    episode_length: int = 50
    warehouse_capacity: int = 8
    skus: list[str] = field(default_factory=lambda: ["A", "B", "C"])
    initial_inventory: Dict[str, int] = field(
        default_factory=lambda: {"A": 20, "B": 20, "C": 20}
    )
    reorder_lead_time: int = 3
    reorder_cost: float = 0.7
    holding_cost: float = 0.05
    stockout_penalty: float = 1.2
    late_penalty: float = 1.5
    expedite_cost: float = 0.8
    fulfillment_reward: float = 4.0
    partial_allocation_reward: float = 0.3
    order_distribution: OrderDistributionConfig = field(
        default_factory=OrderDistributionConfig
    )

    @classmethod
    def from_file(cls, path: str | Path) -> "OrderFulfillmentConfig":
        raw = Path(path).read_text(encoding="utf-8")
        suffix = Path(path).suffix.lower()
        data = json.loads(raw) if suffix == ".json" else yaml.safe_load(raw)
        return cls.from_dict(data)
