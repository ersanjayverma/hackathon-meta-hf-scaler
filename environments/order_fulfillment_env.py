from __future__ import annotations

from dataclasses import replace
from typing import Any, Optional

from openenv.action import OrderFulfillmentAction
from openenv.base_env import BaseEnv
from openenv.engine import EnvironmentEngine, ScheduledEvent
from openenv.logger import StructuredLogger
from openenv.replay import EpisodeRecorder
from openenv.state import (
    InventoryItemState,
    OrderFulfillmentConfig,
    OrderFulfillmentState,
    OrderRecord,
)


class OrderFulfillmentEnv(BaseEnv[OrderFulfillmentState, OrderFulfillmentAction]):
    def __init__(
        self,
        config: OrderFulfillmentConfig,
        logger: Optional[StructuredLogger] = None,
    ) -> None:
        self.config = config
        self.logger = logger or StructuredLogger("openenv.order_fulfillment")
        super().__init__(engine=EnvironmentEngine(self.logger))
        self._state = self._initial_state()
        self._episode_recorder: Optional[EpisodeRecorder] = None
        self._order_counter = 0
        self._closed = False

    def _initial_state(self) -> OrderFulfillmentState:
        inventory = {
            sku: InventoryItemState(sku=sku, on_hand=qty, reserved=0, incoming=0)
            for sku, qty in self.config.initial_inventory.items()
        }
        return OrderFulfillmentState(
            tick=0,
            inventory=inventory,
            pending_orders=[],
            shipped_orders=[],
            backlog_size=0,
            total_reward=0.0,
            capacity_remaining=self.config.warehouse_capacity,
            recent_events=[],
        )

    def enable_recording(self) -> None:
        self._episode_recorder = EpisodeRecorder(
            environment_name=self.__class__.__name__,
            seed=self._seed,
            config=self.config.to_dict(),
        )

    @property
    def episode_recorder(self) -> Optional[EpisodeRecorder]:
        return self._episode_recorder

    def reset(self, seed: Optional[int] = None) -> OrderFulfillmentState:
        with self._lock:
            self.seed(seed)
            self._engine.event_queue.restore([])
            self._state = self._initial_state()
            self._state.rng_state = self._rng.bit_generator.state
            self._order_counter = 0
            self._closed = False
            if self._episode_recorder is not None:
                self.enable_recording()
            self._ingest_new_orders()
            return self.state()

    def step(self, action: OrderFulfillmentAction) -> tuple[OrderFulfillmentState, float, bool, dict[str, Any]]:
        with self._lock:
            action.validate(order.order_id for order in self._state.pending_orders)

            previous_state = self.state().to_dict()

            def _advance() -> tuple[OrderFulfillmentState, float, bool, dict[str, Any]]:
                if self._closed:
                    raise RuntimeError("environment is closed")
                self._state.tick += 1
                self._state.capacity_remaining = self.config.warehouse_capacity
                recent_events = self._engine.process_due_events(self._state.tick, self._handle_event)
                reward, info = self._apply_action(action)
                recent_events.extend(self._ingest_new_orders())
                lateness_penalty = self._apply_lateness_penalty()
                holding_penalty = self._apply_holding_cost()
                reward -= lateness_penalty + holding_penalty
                self._state.total_reward += reward
                self._state.backlog_size = len(self._state.pending_orders)
                self._state.recent_events = recent_events
                self._state.rng_state = self._rng.bit_generator.state
                done = self._state.tick >= self.config.episode_length
                info.update(
                    {
                        "tick": self._state.tick,
                        "backlog_size": self._state.backlog_size,
                        "lateness_penalty": lateness_penalty,
                        "holding_penalty": holding_penalty,
                    }
                )
                if self._episode_recorder is not None:
                    self._episode_recorder.record(
                        state=previous_state,
                        action=action.to_dict(),
                        reward=reward,
                        done=done,
                        info=info,
                        next_state=self.state().to_dict(),
                    )
                return self.state(), reward, done, info

            return self._engine.timed_step(_advance)

    def _handle_event(self, event: ScheduledEvent) -> None:
        if event.event_type == "inventory_arrival":
            sku = str(event.payload["sku"])
            quantity = int(event.payload["quantity"])
            item = self._state.inventory[sku]
            item.on_hand += quantity
            item.incoming = max(item.incoming - quantity, 0)

    def _ingest_new_orders(self) -> list[dict[str, Any]]:
        dist = self.config.order_distribution
        created: list[dict[str, Any]] = []
        arrivals = int(self._rng.poisson(dist.arrival_rate))
        for _ in range(arrivals):
            self._order_counter += 1
            sku = str(self._rng.choice(self.config.skus))
            quantity = int(self._rng.integers(dist.quantity_low, dist.quantity_high + 1))
            priority = int(self._rng.integers(1, dist.priority_levels + 1))
            due_at = self._state.tick + int(
                self._rng.integers(dist.due_window_low, dist.due_window_high + 1)
            )
            lead_time = int(
                self._rng.integers(dist.delivery_delay_low, dist.delivery_delay_high + 1)
            )
            order = OrderRecord(
                order_id=f"ORD-{self._order_counter}",
                sku=sku,
                quantity=quantity,
                priority=priority,
                created_at=self._state.tick,
                due_at=due_at,
                lead_time=lead_time,
            )
            self._state.pending_orders.append(order)
            created.append({"event_type": "order_created", "payload": order.to_dict()})
        return created

    def _ordered_pending_orders(self, action: OrderFulfillmentAction) -> list[OrderRecord]:
        explicit_priority = {
            order_id: index for index, order_id in enumerate(action.prioritized_order_ids)
        }
        return sorted(
            self._state.pending_orders,
            key=lambda order: (
                explicit_priority.get(order.order_id, len(explicit_priority) + order.due_at),
                order.due_at,
                -order.priority,
                order.created_at,
            ),
        )

    def _apply_action(self, action: OrderFulfillmentAction) -> tuple[float, dict[str, Any]]:
        reward = 0.0
        stockouts = 0
        fulfilled_orders = 0

        for sku, amount in action.reorder_quantities.items():
            quantity = int(round(amount))
            if quantity <= 0:
                continue
            if sku not in self._state.inventory:
                continue
            self._state.inventory[sku].incoming += quantity
            self._engine.schedule_in(
                current_tick=self._state.tick,
                delay=self.config.reorder_lead_time,
                event_type="inventory_arrival",
                payload={"sku": sku, "quantity": quantity},
            )
            reward -= quantity * self.config.reorder_cost

        processing_cost_multiplier = 1.0 + (0.5 if action.expedite_shipping else 0.0)

        for order in list(self._ordered_pending_orders(action)):
            if self._state.capacity_remaining <= 0:
                break
            item = self._state.inventory[order.sku]
            desired = action.allocations.get(order.order_id, {}).get(order.sku, order.remaining)
            allocated_now = min(desired, order.remaining, item.on_hand, self._state.capacity_remaining)
            if desired > allocated_now:
                stockouts += 1
            if allocated_now <= 0:
                continue
            item.on_hand -= allocated_now
            order.allocated += allocated_now
            self._state.capacity_remaining -= allocated_now
            reward += allocated_now * self.config.partial_allocation_reward
            if order.remaining == 0:
                order.shipped = True
                order.shipped_at = self._state.tick + max(order.lead_time - int(action.expedite_shipping), 0)
                self._state.pending_orders.remove(order)
                self._state.shipped_orders.append(replace(order))
                fulfilled_orders += 1
                reward += self.config.fulfillment_reward
                reward -= processing_cost_multiplier * self.config.expedite_cost * int(action.expedite_shipping)

        info = {
            "fulfilled_orders": fulfilled_orders,
            "stockouts": stockouts,
            "capacity_remaining": self._state.capacity_remaining,
        }
        return reward, info

    def _apply_lateness_penalty(self) -> float:
        penalty = 0.0
        for order in self._state.pending_orders:
            overdue = max(self._state.tick - order.due_at, 0)
            if overdue > 0:
                penalty += overdue * self.config.late_penalty * (1.0 + order.priority * 0.1)
        return penalty + len(self._state.pending_orders) * self.config.stockout_penalty * 0.02

    def _apply_holding_cost(self) -> float:
        holding_units = sum(item.on_hand + item.incoming for item in self._state.inventory.values())
        return holding_units * self.config.holding_cost

    def state(self) -> OrderFulfillmentState:
        with self._lock:
            return OrderFulfillmentState.from_dict(self._state.to_dict())

    def render(self, mode: str = "human") -> None:
        state = self.state()
        if mode == "human":
            print(
                f"tick={state.tick} backlog={state.backlog_size} "
                f"reward={state.total_reward:.2f} capacity={state.capacity_remaining}"
            )
        elif mode == "json":
            print(state.to_json())
        else:
            raise ValueError(f"unsupported render mode: {mode}")

    def close(self) -> None:
        with self._lock:
            self._closed = True

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "config": self.config.to_dict(),
                "state": self._state.to_dict(),
                "engine_events": self._engine.event_queue.snapshot(),
                "seed": self._seed,
                "order_counter": self._order_counter,
                "closed": self._closed,
            }

    def restore(self, snapshot: dict[str, Any]) -> None:
        with self._lock:
            self.config = OrderFulfillmentConfig.from_dict(snapshot["config"])
            self._state = OrderFulfillmentState.from_dict(snapshot["state"])
            self._seed = snapshot.get("seed")
            self._rng.bit_generator.state = self._state.rng_state or self._rng.bit_generator.state
            self._engine.event_queue.restore(snapshot.get("engine_events", []))
            self._order_counter = int(snapshot.get("order_counter", 0))
            self._closed = bool(snapshot.get("closed", False))
