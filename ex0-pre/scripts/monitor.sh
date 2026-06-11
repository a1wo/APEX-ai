#!/usr/bin/env bash
# Live hardware metrics in the console. Flags pass through, e.g.:  scripts/monitor.sh --interval 0.5
# Run with sudo for GPU power/thermal metrics:  sudo scripts/monitor.sh
cd "$(dirname "$0")/.."
PY=.venv/bin/python; [ -x "$PY" ] || PY=python3
exec "$PY" monitor.py "$@"
