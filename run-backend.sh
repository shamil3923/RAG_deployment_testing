#!/usr/bin/env bash
# Start the FastAPI backend on http://localhost:8000
set -e
cd "$(dirname "$0")/backend"
export TOKENIZERS_PARALLELISM=false

# If a virtualenv exists, use it; otherwise fall back to system python.
if [ -x "./.venv/bin/python" ]; then
	PY="./.venv/bin/python"
else
	PY="python3"
fi

exec "$PY" -m uvicorn main:app --reload --port 8000
