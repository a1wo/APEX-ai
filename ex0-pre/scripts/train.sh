#!/usr/bin/env bash
# Run training. All flags pass through, e.g.:  scripts/train.sh --reverse --pad --monitor
cd "$(dirname "$0")/.."
PY=.venv/bin/python; [ -x "$PY" ] || PY=python3
exec "$PY" train.py "$@"
