from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


@dataclass(slots=True)
class EpisodeTransition:
    state: dict[str, Any]
    action: dict[str, Any]
    reward: float
    done: bool
    info: dict[str, Any]
    next_state: dict[str, Any]


@dataclass(slots=True)
class EpisodeRecorder:
    environment_name: str
    seed: Optional[int]
    config: dict[str, Any]
    transitions: list[EpisodeTransition] = field(default_factory=list)

    def record(
        self,
        state: dict[str, Any],
        action: dict[str, Any],
        reward: float,
        done: bool,
        info: dict[str, Any],
        next_state: dict[str, Any],
    ) -> None:
        self.transitions.append(
            EpisodeTransition(
                state=state,
                action=action,
                reward=reward,
                done=done,
                info=info,
                next_state=next_state,
            )
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "environment_name": self.environment_name,
            "seed": self.seed,
            "config": self.config,
            "transitions": [
                {
                    "state": item.state,
                    "action": item.action,
                    "reward": item.reward,
                    "done": item.done,
                    "info": item.info,
                    "next_state": item.next_state,
                }
                for item in self.transitions
            ],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "EpisodeRecorder":
        recorder = cls(
            environment_name=payload["environment_name"],
            seed=payload.get("seed"),
            config=dict(payload.get("config", {})),
        )
        recorder.transitions = [
            EpisodeTransition(
                state=dict(item["state"]),
                action=dict(item["action"]),
                reward=float(item["reward"]),
                done=bool(item["done"]),
                info=dict(item["info"]),
                next_state=dict(item["next_state"]),
            )
            for item in payload.get("transitions", [])
        ]
        return recorder


class ReplayStore:
    @staticmethod
    def save(recorder: EpisodeRecorder, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(recorder.to_dict(), indent=2, sort_keys=True),
            encoding="utf-8",
        )

    @staticmethod
    def load(path: str | Path) -> EpisodeRecorder:
        return EpisodeRecorder.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
