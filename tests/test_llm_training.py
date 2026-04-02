from __future__ import annotations

import json

from environments.email_triage_env import EmailTriageEnv
from openenv.llm_training import action_to_json, build_training_example, export_sft_dataset, render_llm_user_prompt
from openenv.models import Action
from openenv.tasks import get_benchmark_tasks


def test_render_llm_user_prompt_includes_observation_json() -> None:
    task = get_benchmark_tasks()[0]
    env = EmailTriageEnv(task=task, seed=task.seed)
    observation = env.reset()
    prompt = render_llm_user_prompt(observation)
    assert "Choose the next email triage action." in prompt
    assert "Observation JSON:" in prompt
    assert observation.task_name in prompt
    env.close()


def test_build_training_example_has_simple_chat_shape() -> None:
    task = get_benchmark_tasks()[0]
    env = EmailTriageEnv(task=task, seed=task.seed)
    observation = env.reset()
    action = Action(action_type="classify", email_id=observation.inbox[0].email_id, category="spam")
    example = build_training_example(
        task=task,
        episode_index=0,
        seed=task.seed,
        observation=observation,
        action=action,
        reward=1.5,
        done=False,
    )
    assert [message["role"] for message in example["messages"]] == ["system", "user", "assistant"]
    assert example["metadata"]["task_name"] == task.name
    assert action_to_json(action) == example["messages"][2]["content"]
    env.close()


def test_export_sft_dataset_writes_chat_style_rows(tmp_path) -> None:
    output_path = export_sft_dataset(tmp_path / "email_triage_sft.jsonl")
    rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    assert rows
    first_row = rows[0]
    assert [message["role"] for message in first_row["messages"]] == ["system", "user", "assistant"]
    assert "task_name" in first_row["metadata"]
    assert isinstance(first_row["metadata"]["reward"], float)
    assert isinstance(first_row["metadata"]["done"], bool)
