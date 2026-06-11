#!/usr/bin/env bash
# Train one run per operand size — ndigits 3..7 — in parallel.
# Extra flags pass through to EVERY run, e.g.:
#     scripts/train_sweep.sh --reverse --max_iters 20000
# Override the range:
#     FROM=2 TO=5 scripts/train_sweep.sh
# Each run gets its own checkpoint tag, W&B/MLflow run, and system/* monitor metrics.
# Exiting this script (Ctrl-C, kill, closing the terminal) stops every run; they shut
# down via SIGTERM, which train.py turns into a clean exit that closes the tracker runs.
cd "$(dirname "$0")/.."
PY=.venv/bin/python; [ -x "$PY" ] || PY=python3
mkdir -p logs

FROM="${FROM:-3}"
TO="${TO:-7}"

pids=()
for n in $(seq "$FROM" "$TO"); do
    log="logs/train_${n}digit.log"
    "$PY" -u train.py --monitor --ndigits "$n" "$@" > "$log" 2>&1 &
    pid=$!
    pids+=("$pid")
    echo "run (pid $pid): ndigits=$n → $log"
done

echo "watch with:  tail -f logs/train_*digit.log     (exiting this script stops all runs)"

cleanup() { kill "${pids[@]}" 2>/dev/null; }
trap cleanup INT TERM HUP EXIT
wait
