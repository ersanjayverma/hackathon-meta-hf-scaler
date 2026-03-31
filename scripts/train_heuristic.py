from __future__ import annotations

from agents.heuristic_agent import HeuristicAgent
from env.environment import OpenEnvGymLikeEnv
import numpy as np


def main() -> None:
    env = OpenEnvGymLikeEnv(task_name="task_easy_classification")
    agent = HeuristicAgent()

    for episode in range(3):
        state = env.reset(seed=101 + episode)
        total_reward = 0.0
        done = False

        while not done:
            action = agent.act(state.observation)
            step_result = env.step(action)
            total_reward += step_result.reward.total
            state = step_result.next_state
            done = step_result.done

        print(
            f"episode={episode} task={state.metadata['task_name']} "
            f"history={state.history_length} total_reward={total_reward:.2f}"
        )

    env.close()


def rollout_random_policy(episodes: int = 3) -> None:
    env = OpenEnvGymLikeEnv(task_name="task_easy_classification")
    rng = np.random.default_rng(123)
    for episode in range(episodes):
        observation, info = env.reset_gym(seed=500 + episode)
        done = False
        total_reward = 0.0
        while not done:
            valid_actions = [index for index, allowed in enumerate(info["action_mask"]) if allowed]
            action_id = int(rng.choice(valid_actions))
            observation, reward, terminated, truncated, info = env.step_gym(action_id)
            total_reward += reward
            done = terminated or truncated
        print(f"random_episode={episode} total_reward={total_reward:.2f} features={len(observation)}")
    env.close()


if __name__ == "__main__":
    main()
