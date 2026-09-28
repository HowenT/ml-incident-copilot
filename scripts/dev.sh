#!/usr/bin/env bash
# Start API (:8000) and UI (:8501). Ctrl+C stops both. Run from the repo root.
set -euo pipefail
cd "$(dirname "$0")/.."
python -m uvicorn backend.app.main:app --port 8000 &
API_PID=$!
trap 'kill $API_PID 2>/dev/null' EXIT
python -m streamlit run frontend/app.py --server.port 8501
