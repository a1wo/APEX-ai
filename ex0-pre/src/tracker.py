"""
Thin wrapper around W&B and MLflow so the training loop calls a single log()
regardless of which trackers are active.

Both are optional:
  pip install wandb  && wandb login   → W&B
  pip install mlflow                  → MLflow (local UI: mlflow ui)

Trackers that fail to import or initialise are silently skipped.
"""


class ExperimentTracker:

    def __init__(self, run_name: str, run_config: dict, use_wandb: bool, use_mlflow: bool):
        self._wb  = None
        self._mlf = None

        if use_wandb:
            try:
                import wandb
                wandb.init(
                    project="gpt-addition",
                    name=run_name,
                    config=run_config,
                    resume="allow",
                )
                self._wb = wandb
                print("wandb  ▸ logging to project 'gpt-addition'")
            except ImportError:
                print("wandb  ▸ not installed — skipping  (pip install wandb && wandb login)")
            except Exception as e:
                print(f"wandb  ▸ init failed ({e}) — skipping")

        if use_mlflow:
            try:
                import mlflow
                mlflow.set_experiment("gpt-addition")
                mlflow.start_run(run_name=run_name)
                mlflow.log_params(run_config)
                self._mlf = mlflow
                print(f"mlflow ▸ run '{run_name}' started  (mlflow ui → localhost:5000)")
            except ImportError:
                print("mlflow ▸ not installed — skipping  (pip install mlflow)")
            except Exception as e:
                print(f"mlflow ▸ init failed ({e}) — skipping")

    # ── Public API ────────────────────────────────────────────────────────────

    def log(self, metrics: dict, step: int) -> None:
        if self._wb:
            self._wb.log(metrics, step=step)
        if self._mlf:
            self._mlf.log_metrics(metrics, step=step)

    def log_artifact(self, path: str) -> None:
        """Upload a file (e.g. a checkpoint) to all active trackers."""
        if self._wb:
            try:
                self._wb.save(path, policy="now")
            except Exception as e:
                print(f"wandb  ▸ artifact upload failed ({e})")
        if self._mlf:
            try:
                self._mlf.log_artifact(path)
            except Exception as e:
                print(f"mlflow ▸ artifact upload failed ({e})")

    def finish(self) -> None:
        if self._wb:
            self._wb.finish()
        if self._mlf:
            self._mlf.end_run()
