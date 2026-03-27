from __future__ import annotations

import importlib.util
import io
import json
import sys
from contextlib import redirect_stdout
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "inference.py"
MODULE_SPEC = importlib.util.spec_from_file_location("openenv_inference", MODULE_PATH)
assert MODULE_SPEC is not None and MODULE_SPEC.loader is not None
inference_module = importlib.util.module_from_spec(MODULE_SPEC)
sys.modules[MODULE_SPEC.name] = inference_module
MODULE_SPEC.loader.exec_module(inference_module)


def test_inference_main_prints_results(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        inference_module,
        "run_baseline",
        lambda **_: {
            "model": "test-model",
            "backend": "heuristic",
            "average_score": 1.0,
            "tasks": [{"task": "task_easy_classification", "score": 1.0}],
            "api_failures": 0,
            "fallback_actions": 0,
        },
    )
    output = io.StringIO()
    with redirect_stdout(output):
        inference_module.main()
    payload = json.loads(output.getvalue())
    assert payload["average_score"] == 1.0
    assert payload["backend"] == "heuristic"
