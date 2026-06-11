"""
Thin wrapper around W&B and MLflow so the training loop calls a single log()
regardless of which trackers are active.

Both are optional:
  pip install wandb  && wandb login   → W&B
  pip install mlflow                  → MLflow (local UI: mlflow ui)

Trackers that fail to import or initialise are silently skipped.
"""


class ExperimentTracker:

    def __init__(self, run_name: str, run_config: dict, use_wandb: bool, use_mlflow: bool,
                 resume_ids: dict | None = None):
        """
        resume_ids: {"wandb": <id>, "mlflow": <id>} from a checkpoint — when given,
        logging continues in those existing runs instead of creating new ones.
        """
        resume_ids   = resume_ids or {}
        self._wb     = None
        self._mlf    = None
        self.run_ids = {}   # ids of the active runs, saved into checkpoints for --resume

        if use_wandb:
            try:
                import wandb
                if resume_ids.get("wandb"):
                    wandb.init(project="gpt-addition", id=resume_ids["wandb"],
                               resume="must", config=run_config)
                    print(f"wandb  ▸ resumed run {resume_ids['wandb']}")
                else:
                    wandb.init(project="gpt-addition", name=run_name, config=run_config)
                    print("wandb  ▸ logging to project 'gpt-addition'")
                self._wb = wandb
                self.run_ids["wandb"] = wandb.run.id
            except ImportError:
                print("wandb  ▸ not installed — skipping  (pip install wandb && wandb login)")
            except Exception as e:
                print(f"wandb  ▸ init failed ({e}) — skipping")

        if use_mlflow:
            try:
                import mlflow
                mlflow.set_experiment("gpt-addition")
                if resume_ids.get("mlflow"):
                    mlflow.start_run(run_id=resume_ids["mlflow"])
                    print(f"mlflow ▸ resumed run {resume_ids['mlflow']}")
                else:
                    mlflow.start_run(run_name=run_name)
                    mlflow.log_params(run_config)
                    print(f"mlflow ▸ run '{run_name}' started  (scripts/mlflow.sh → 127.0.0.1:5001)")
                self._mlf = mlflow
                self.run_ids["mlflow"] = mlflow.active_run().info.run_id
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
