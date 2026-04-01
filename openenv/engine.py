from __future__ import annotations

import heapq
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional

from .logger import StructuredLogger
from .models import Reward


@dataclass(order=True, slots=True)
class ScheduledEvent:
    tick: int
    priority: int
    event_type: str = field(compare=False)
    payload: dict[str, Any] = field(default_factory=dict, compare=False)


class EventQueue:
    def __init__(self) -> None:
        self._queue: list[ScheduledEvent] = []

    def schedule(self, event: ScheduledEvent) -> None:
        heapq.heappush(self._queue, event)

    def pop_due(self, tick: int) -> list[ScheduledEvent]:
        due: list[ScheduledEvent] = []
        while self._queue and self._queue[0].tick <= tick:
            due.append(heapq.heappop(self._queue))
        return due

    def snapshot(self) -> list[dict[str, Any]]:
        return [
            {
                "tick": event.tick,
                "priority": event.priority,
                "event_type": event.event_type,
                "payload": dict(event.payload),
            }
            for event in self._queue
        ]

    def restore(self, payload: list[dict[str, Any]]) -> None:
        self._queue = [
            ScheduledEvent(
                tick=item["tick"],
                priority=item["priority"],
                event_type=item["event_type"],
                payload=dict(item["payload"]),
            )
            for item in payload
        ]
        heapq.heapify(self._queue)


class PrometheusHooks:
    def __init__(self, namespace: str = "openenv") -> None:
        self.enabled = False
        try:
            from prometheus_client import CollectorRegistry, Gauge, Histogram
        except ImportError:
            return

        self.registry = CollectorRegistry()
        self.reward_gauge = Gauge(
            f"{namespace}_reward_total", "Accumulated reward from steps", registry=self.registry
        )
        self.step_latency = Histogram(
            f"{namespace}_step_latency_seconds", "Environment step latency", registry=self.registry
        )
        self.backlog_gauge = Gauge(f"{namespace}_backlog", "Current backlog size", registry=self.registry)
        self.enabled = True


@dataclass(slots=True)
class MetricsTracker:
    rewards: list[float] = field(default_factory=list)
    step_latencies: list[float] = field(default_factory=list)
    fulfilled_orders: int = 0
    stockouts: int = 0

    def record_step(self, reward: float, latency: float, info: Optional[dict[str, Any]] = None) -> None:
        self.rewards.append(float(reward))
        self.step_latencies.append(latency)
        if info:
            self.fulfilled_orders += int(info.get("fulfilled_orders", 0))
            self.stockouts += int(info.get("stockouts", 0))

    def summary(self) -> dict[str, float]:
        avg_latency = sum(self.step_latencies) / len(self.step_latencies) if self.step_latencies else 0.0
        total_reward = sum(self.rewards)
        throughput = self.fulfilled_orders / max(len(self.rewards), 1)
        return {
            "avg_step_latency": avg_latency,
            "total_reward": total_reward,
            "throughput": throughput,
            "stockouts": float(self.stockouts),
        }


class EnvironmentEngine:
    def __init__(
        self,
        logger: StructuredLogger,
        metrics: Optional[MetricsTracker] = None,
        prometheus: Optional[PrometheusHooks] = None,
    ) -> None:
        self.logger = logger
        self.metrics = metrics or MetricsTracker()
        self.prometheus = prometheus or PrometheusHooks()
        self.event_queue = EventQueue()

    def schedule_in(
        self, current_tick: int, delay: int, event_type: str, payload: dict[str, Any], priority: int = 0
    ) -> None:
        self.event_queue.schedule(
            ScheduledEvent(
                tick=current_tick + delay,
                priority=priority,
                event_type=event_type,
                payload=payload,
            )
        )

    def process_due_events(
        self, tick: int, handler: Callable[[ScheduledEvent], None]
    ) -> list[dict[str, Any]]:
        processed: list[dict[str, Any]] = []
        for event in self.event_queue.pop_due(tick):
            handler(event)
            processed.append(
                {
                    "tick": event.tick,
                    "event_type": event.event_type,
                    "payload": dict(event.payload),
                }
            )
        return processed

    def record_step(
        self, reward: float | Reward, latency: float, backlog: int, info: Optional[dict[str, Any]] = None
    ) -> None:
        reward_value = reward.total if isinstance(reward, Reward) else float(reward)
        self.metrics.record_step(reward_value, latency, info)
        self.logger.info(
            "environment_step",
            reward=reward_value,
            latency=latency,
            backlog=backlog,
            info=info or {},
        )
        if self.prometheus.enabled:
            self.prometheus.reward_gauge.inc(reward_value)
            self.prometheus.step_latency.observe(latency)
            self.prometheus.backlog_gauge.set(backlog)

    def timed_step(
        self, func: Callable[[], tuple[Any, Any, bool, dict[str, Any]]]
    ) -> tuple[Any, Any, bool, dict[str, Any]]:
        started = time.perf_counter()
        state, reward, done, info = func()
        latency = time.perf_counter() - started
        self.record_step(reward, latency, int(info.get("backlog_size", 0)), info)
        info = dict(info)
        info["step_latency"] = latency
        return state, reward, done, info
