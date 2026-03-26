from __future__ import annotations

from pathlib import Path

from agents.heuristic_agent import HeuristicAgent
from agents.random_agent import RandomAgent
from environments.email_triage_env import EmailTriageEnv
from openenv.replay import ReplayStore
from openenv.tasks import get_email_tasks, get_graders


def run_task(agent_name: str, task_name: str) -> tuple[float, float]:
    task = next(task for task in get_email_tasks() if task.name == task_name)
    agent = RandomAgent(seed=task.seed) if agent_name == "random" else HeuristicAgent()
    env = EmailTriageEnv(task=task, seed=task.seed)
    observation = env.reset()
    total_reward = 0.0
    done = False

    while not done:
        action = agent.act(observation)
        observation, reward, done, info = env.step(action)
        total_reward += reward.total
        print(
            f"agent={agent_name} task={task_name} step={observation.step_index} "
            f"reward={reward.total:.2f} pending={info['pending_emails']}"
        )

    score = get_graders()[task_name](env.trajectory)
    output = Path("artifacts")
    output.mkdir(exist_ok=True)
    ReplayStore.save(env.episode_recorder, output / f"{agent_name}_{task_name}.json")
    env.close()
    return total_reward, score


def main() -> None:
    for agent_name in ("random", "heuristic"):
        scores = []
        for task in get_email_tasks():
            total_reward, score = run_task(agent_name, task.name)
            scores.append(score)
            print(f"summary agent={agent_name} task={task.name} reward={total_reward:.2f} score={score:.2f}")
        print(f"average agent={agent_name} score={sum(scores) / len(scores):.2f}")


if __name__ == "__main__":
    main()
