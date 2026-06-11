#!/usr/bin/env bash
# W&B helper: make sure you're logged in, then open the dashboard.
set -e
cd "$(dirname "$0")/.."
WANDB=.venv/bin/wandb; [ -x "$WANDB" ] || WANDB=wandb
"$WANDB" login --verify 2>/dev/null || "$WANDB" login
URL="https://wandb.ai/home"
echo "Project: gpt-addition  →  $URL"
open "$URL" 2>/dev/null || true
