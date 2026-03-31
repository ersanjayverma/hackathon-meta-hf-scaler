from __future__ import annotations

from pathlib import Path
import sys

from agents.heuristic_agent import HeuristicAgent

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from env.environment import OpenEnvGymLikeEnv


def test_rl_env_reset_and_step() -> None:
    env = OpenEnvGymLikeEnv(task_name="task_easy_classification", seed=101)
    state = env.reset()
    assert state.observation.task_name == "task_easy_classification"
    assert state.history_length == 0

    action = HeuristicAgent().act(state.observation)
    result = env.step(action)
    assert result.next_state.history_length == 1
    assert isinstance(result.done, bool)
    assert isinstance(result.info, dict)
    env.close()


def test_rl_env_done_includes_grader_score() -> None:
    env = OpenEnvGymLikeEnv(task_name="task_easy_classification", seed=101)
    agent = HeuristicAgent()
    state = env.reset()

    done = False
    final_info: dict[str, object] = {}
    while not done:
        result = env.step(agent.act(state.observation))
        state = result.next_state
        done = result.done
        final_info = result.info

    assert "grader_score" in final_info
    assert 0.0 <= float(final_info["grader_score"]) <= 1.0
    env.close()


def test_rl_env_exposes_vector_observation_and_mask() -> None:
    env = OpenEnvGymLikeEnv(task_name="task_easy_classification", seed=101)
    state = env.reset()
    assert len(state.vector_observation) == state.metadata["feature_count"]
    assert len(state.action_mask) == state.metadata["action_count"]
    assert state.action_mask[0] == 1
    env.close()


def test_rl_env_discrete_action_and_gym_api() -> None:
    env = OpenEnvGymLikeEnv(task_name="task_easy_classification", seed=101)
    observation, info = env.reset_gym(seed=101)
    valid_actions = [index for index, allowed in enumerate(info["action_mask"]) if allowed]
    next_observation, reward, terminated, truncated, next_info = env.step_gym(valid_actions[1])
    assert len(observation) == len(next_observation)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert "action_mask" in next_info
    env.close()
