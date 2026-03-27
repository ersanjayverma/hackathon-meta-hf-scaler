from __future__ import annotations

import json
import os
from pathlib import Path

from baseline.run_baseline import run_baseline


def main() -> None:
    results = run_baseline(
        model=os.environ.get("MODEL_NAME"),
        api_key=os.environ.get("HF_TOKEN"),
        base_url=os.environ.get("API_BASE_URL"),
        output_path=Path("baseline/results/baseline_results.json"),
    )
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
