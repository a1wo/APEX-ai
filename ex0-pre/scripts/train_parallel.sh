#!/usr/bin/env bash
# Run two training configs in parallel — forward/unpadded (default) vs reversed/padded.
# Each process has its own --monitor, so sys/* metrics and checkpoint artifacts land
# in that run's own W&B/MLflow entry. Extra flags pass through to BOTH runs, e.g.:
#     scripts/train_parallel.sh --max_iters 20000
# Console output is split into logs/ so the two runs don't interleave.
cd "$(dirname "$0")/.."
PY=.venv/bin/python; [ -x "$PY" ] || PY=python3
mkdir -p logs

"$PY" train.py --monitor "$@" > logs/train_fwd_nopad.log 2>&1 &
A=$!
echo "run A (pid $A): forward, unpadded          → logs/train_fwd_nopad.log"

"$PY" train.py --monitor --reverse --pad "$@" > logs/train_rev_pad.log 2>&1 &
B=$!
echo "run B (pid $B): reversed, padded           → logs/train_rev_pad.log"

echo "watch with:  tail -f logs/train_*.log     (Ctrl-C stops both runs)"
trap 'kill $A $B 2>/dev/null' INT TERM
wait $A $B
