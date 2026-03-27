#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
PYTEST_BIN="${PYTEST_BIN:-pytest}"

if [[ -x ".venv/bin/python" ]]; then
  PYTHON_BIN=".venv/bin/python"
  PYTEST_BIN=".venv/bin/pytest"
fi

"$PYTEST_BIN"
PYTHONPATH="${PWD}" "$PYTHON_BIN" -m openenv.cli validate ./environments
"$PYTHON_BIN" inference.py
docker build .
