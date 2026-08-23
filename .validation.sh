#!/bin/bash
set -e

cd "$(dirname "$0")"

if [ ! -d ".venv" ]; then
    echo "Creating virtualenv..."
    uv venv --python 3.13 .venv
    uv pip install -e ".[dev]" -p .venv
fi
PY=".venv/bin/python"

echo "=== Validation Loop ==="

echo "[1/4] Building..."
$PY -m compileall -q src/rasch_per

echo "[2/4] Type checking..."
$PY -m mypy src/

echo "[3/4] Linting..."
$PY -m ruff check .
$PY -m ruff format --check .

echo "[4/4] Testing..."
$PY -m pytest --cov=rasch_per -q
# Coverage gate (85%) is enforced in CI (.github/workflows/ci.yml), not locally.

echo "=== All checks passed ==="
