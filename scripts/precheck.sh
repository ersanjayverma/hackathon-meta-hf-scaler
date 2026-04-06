#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
PYTEST_BIN="${PYTEST_BIN:-pytest}"

if [[ -x ".venv/bin/python" ]]; then
  PYTHON_BIN=".venv/bin/python"
  PYTEST_BIN=".venv/bin/pytest"
fi

echo "[1/3] Running test suite"
"$PYTEST_BIN"

echo "[2/3] Running canonical LLM inference"
"$PYTHON_BIN" inference.py

echo "[3/3] Building Docker image"
docker build .
