from __future__ import annotations

import json
import os
from pathlib import Path

from baseline.run_baseline import run_baseline


def _resolve_backend() -> str:
    explicit_backend = os.environ.get("OPENENV_BASELINE_BACKEND")
    if explicit_backend:
        return explicit_backend
    if (
        os.environ.get("API_BASE_URL")
        and os.environ.get("MODEL_NAME")
        and os.environ.get("HF_TOKEN")
    ):
        return "openai"
    return "heuristic"


def main() -> None:
    results = run_baseline(
        model=os.environ.get("MODEL_NAME"),
        api_key=os.environ.get("HF_TOKEN"),
        base_url=os.environ.get("API_BASE_URL"),
        output_path=Path("baseline/results/baseline_results.json"),
        backend=_resolve_backend(),
        include_supplemental=os.environ.get("OPENENV_INCLUDE_SUPPLEMENTAL_TASKS", "0") == "1",
    )
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
