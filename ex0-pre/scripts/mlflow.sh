#!/usr/bin/env bash
# Open the MLflow UI → http://localhost:5000
# Metrics live in ./mlflow.db (MLflow 3 default), artifacts in ./mlruns.
cd "$(dirname "$0")/.."
MLFLOW=.venv/bin/mlflow; [ -x "$MLFLOW" ] || MLFLOW=mlflow
exec "$MLFLOW" ui --backend-store-uri sqlite:///mlflow.db "$@"
