from __future__ import annotations

import multiprocessing as mp
from dataclasses import dataclass
from multiprocessing.connection import Connection
from typing import Any, Callable, Iterable, Optional


def _worker(remote: Connection, env_factory: Callable[[], Any]) -> None:
    env = env_factory()
    try:
        while True:
            command, payload = remote.recv()
            if command == "reset":
                remote.send(env.reset(seed=payload))
            elif command == "step":
                remote.send(env.step(payload))
            elif command == "state":
                remote.send(env.state())
            elif command == "close":
                env.close()
                remote.close()
                break
            elif command == "save_state":
                remote.send(env.snapshot())
            elif command == "load_state":
                env.restore(payload)
                remote.send(True)
            else:
                raise ValueError(f"unknown command: {command}")
    finally:
        env.close()


@dataclass(slots=True)
class VectorEnv:
    env_factories: list[Callable[[], Any]]
    start_method: str = ""

    def __post_init__(self) -> None:
        start_method = self.start_method or (
            "fork" if "fork" in mp.get_all_start_methods() else "spawn"
        )
        self._ctx = mp.get_context(start_method)
        self._processes: list[mp.Process] = []
        self._remotes: list[Connection] = []
        for factory in self.env_factories:
            parent, child = self._ctx.Pipe()
            process = self._ctx.Process(target=_worker, args=(child, factory), daemon=True)
            process.start()
            child.close()
            self._processes.append(process)
            self._remotes.append(parent)

    def reset(self, seeds: Optional[Iterable[Optional[int]]] = None) -> list[Any]:
        seed_list = list(seeds) if seeds is not None else [None] * len(self._remotes)
        for remote, seed in zip(self._remotes, seed_list, strict=True):
            remote.send(("reset", seed))
        return [remote.recv() for remote in self._remotes]

    def step(self, actions: Iterable[Any]) -> list[tuple[Any, float, bool, dict[str, Any]]]:
        for remote, action in zip(self._remotes, actions, strict=True):
            remote.send(("step", action))
        return [remote.recv() for remote in self._remotes]

    def state(self) -> list[Any]:
        for remote in self._remotes:
            remote.send(("state", None))
        return [remote.recv() for remote in self._remotes]

    def save_states(self) -> list[dict[str, Any]]:
        for remote in self._remotes:
            remote.send(("save_state", None))
        return [remote.recv() for remote in self._remotes]

    def load_states(self, snapshots: Iterable[dict[str, Any]]) -> None:
        for remote, snapshot in zip(self._remotes, snapshots, strict=True):
            remote.send(("load_state", snapshot))
        for remote in self._remotes:
            remote.recv()

    def close(self) -> None:
        for remote in self._remotes:
            remote.send(("close", None))
        for process in self._processes:
            process.join(timeout=5)
