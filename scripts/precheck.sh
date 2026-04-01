#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
PYTEST_BIN="${PYTEST_BIN:-pytest}"

if [[ -x ".venv/bin/python" ]]; then
  PYTHON_BIN=".venv/bin/python"
  PYTEST_BIN=".venv/bin/pytest"
fi

echo "[1/4] Running test suite"
"$PYTEST_BIN"

echo "[2/4] Validating OpenEnv manifest and environment"
PYTHONPATH="${PWD}" "$PYTHON_BIN" -m openenv.cli validate .

echo "[3/4] Running canonical deterministic inference baseline"
"$PYTHON_BIN" inference.py

echo "[4/4] Building Docker image"
docker build .
