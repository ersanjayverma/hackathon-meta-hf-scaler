from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class EmailTriageConfig:
    loop_penalty: float = -0.2
    urgent_wait_penalty: float = -0.15
    missed_classification_penalty: float = -0.05
    missed_response_penalty: float = -0.08
    missed_escalation_penalty: float = -0.1
    stress_penalty_scale: float = 0.005
    sla_pressure_penalty_scale: float = 0.05
    system_collapse_stress: float = 30.0
    system_collapse_penalty: float = -0.5
    stable_resolution_ends_episode: bool = True
    reward_floor: float = -1.0
    reward_ceiling: float = 1.0
    max_reward_per_step: float = 1.0
    failure_collapse_window: int = 3
    cumulative_reward_floor: float = -3.0
    repetition_decay: float = 0.2


@dataclass(frozen=True, slots=True)
class BenchmarkMetadata:
    benchmark_name: str = "email_triage_benchmark"
    benchmark_version: str = "1.0.0"
    output_schema_version: str = "baseline_results/v2"
    default_model: str = "Qwen/Qwen2.5-72B-Instruct"
    deterministic_temperature: float = 0.0
    canonical_manifest: str = "openenv.yaml"
    generated_manifest: str = "environments/openenv.yaml"

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


EMAIL_TRIAGE_CONFIG = EmailTriageConfig()
BENCHMARK_METADATA = BenchmarkMetadata()
CANONICAL_MANIFEST_PATH = Path(BENCHMARK_METADATA.canonical_manifest)
GENERATED_MANIFEST_PATH = Path(BENCHMARK_METADATA.generated_manifest)
