#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
PYTEST_BIN="${PYTEST_BIN:-pytest}"
OPENENV_BIN="${OPENENV_BIN:-openenv}"

if [[ -x ".venv/bin/python" ]]; then
  PYTHON_BIN=".venv/bin/python"
  PYTEST_BIN=".venv/bin/pytest"
  OPENENV_BIN=".venv/bin/openenv"
fi

"$PYTEST_BIN"
"$OPENENV_BIN" validate ./environments
"$PYTHON_BIN" baseline/run_baseline.py
docker build .
