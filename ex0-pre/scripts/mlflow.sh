#!/usr/bin/env bash
# Open the MLflow UI → http://127.0.0.1:5001
# Metrics live in ./mlflow.db (MLflow 3 default), artifacts in ./mlruns.
# Port 5001 because macOS AirPlay Receiver squats on localhost:5000 and
# answers with an empty 403 — looks exactly like "my runs are missing".
cd "$(dirname "$0")/.."
MLFLOW=.venv/bin/mlflow; [ -x "$MLFLOW" ] || MLFLOW=mlflow
PORT="${PORT:-5001}"
echo "MLflow UI → http://127.0.0.1:$PORT"
exec "$MLFLOW" ui --backend-store-uri sqlite:///mlflow.db --port "$PORT" "$@"
