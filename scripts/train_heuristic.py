from __future__ import annotations

from agents.heuristic_agent import HeuristicAgent
from env.environment import OpenEnvGymLikeEnv


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


if __name__ == "__main__":
    main()
