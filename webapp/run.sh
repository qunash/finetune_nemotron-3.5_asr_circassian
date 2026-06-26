#!/usr/bin/env bash
# Launch the Circassian dictation web app.
#   ./run.sh                  -> http://127.0.0.1:8000
#   ./run.sh --port 9000
set -euo pipefail
cd "$(dirname "$0")/.."
exec .venv/bin/python -m webapp.server "$@"
