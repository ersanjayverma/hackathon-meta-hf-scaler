from __future__ import annotations

import os

API_BASE_URL = "https://router.huggingface.co/v1"
MODEL_NAME = "Qwen/Qwen2.5-72B-Instruct"
TASK_NAME = "echo"
BENCHMARK = "my_env_v4"
MAX_STEPS = 8
TEMPERATURE = 0.7
MAX_TOKENS = 150
SUCCESS_SCORE_THRESHOLD = 0.1

DEFAULT_LOG_LEVEL = "WARNING"
DEFAULT_PORT = 7860

ENV_API_BASE_URL = "API_BASE_URL"
ENV_MODEL_NAME = "MODEL_NAME"
ENV_OPENAI_MODEL = "OPENAI_MODEL"
ENV_HF_TOKEN = "HF_TOKEN"
ENV_OPENAI_API_KEY = "OPENAI_API_KEY"
ENV_OPENENV_BASELINE_BACKEND = "OPENENV_BASELINE_BACKEND"
ENV_OPENENV_TASK = "OPENENV_TASK"
ENV_MY_ENV_V4_TASK = "MY_ENV_V4_TASK"
ENV_MY_ENV_V4_BENCHMARK = "MY_ENV_V4_BENCHMARK"
ENV_OPENENV_LOG_LEVEL = "OPENENV_LOG_LEVEL"
ENV_PORT = "PORT"


def runtime_api_base_url(default: str | None = None) -> str | None:
    return os.getenv(ENV_API_BASE_URL) or default


def runtime_model_name(default: str | None = None) -> str | None:
    return os.getenv(ENV_MODEL_NAME) or os.getenv(ENV_OPENAI_MODEL) or default


def runtime_hf_token() -> str | None:
    return os.getenv(ENV_HF_TOKEN)


def runtime_openai_api_key() -> str | None:
    return os.getenv(ENV_OPENAI_API_KEY)


def runtime_api_key() -> str | None:
    return runtime_hf_token() or runtime_openai_api_key()


def runtime_baseline_backend(default: str | None = None) -> str | None:
    return os.getenv(ENV_OPENENV_BASELINE_BACKEND) or default


def runtime_task_name(default: str | None = None) -> str | None:
    return os.getenv(ENV_OPENENV_TASK) or os.getenv(ENV_MY_ENV_V4_TASK) or default


def runtime_benchmark_name(default: str | None = None) -> str | None:
    return os.getenv(ENV_MY_ENV_V4_BENCHMARK) or default


def runtime_log_level(default: str = DEFAULT_LOG_LEVEL) -> str:
    return (os.getenv(ENV_OPENENV_LOG_LEVEL) or default).upper()


def runtime_port(default: int = DEFAULT_PORT) -> int:
    return int(os.getenv(ENV_PORT) or default)
