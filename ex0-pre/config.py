from dataclasses import dataclass, field


@dataclass
class TrainConfig:
    # ── task ──────────────────────────────────────────────────────────────────
    ndigits:          int   = 3
    reverse_c:        bool  = True     # True = ones digit first; False = left-to-right

    # ── model ─────────────────────────────────────────────────────────────────
    n_layer:          int   = 4
    n_head:           int   = 4
    n_embd:           int   = 64

    # ── optimisation ──────────────────────────────────────────────────────────
    max_iters:        int   = 100_000
    batch_size:       int   = 128
    lr:               float = 3e-4
    warmup_steps:     int   = 1_000
    eval_every:       int   = 2_000

    # ── checkpointing ─────────────────────────────────────────────────────────
    ckpt_dir:         str   = "checkpoints"
    epoch_size:       int   = 5_000    # steps per epoch → one checkpoint per epoch
    keep_checkpoints: int   = 3        # prune older files, keep this many

    # ── experiment tracking ───────────────────────────────────────────────────
    use_wandb:        bool  = True     # silently skipped if wandb not installed
    use_mlflow:       bool  = True     # silently skipped if mlflow not installed

    # ── derived (read-only) ───────────────────────────────────────────────────
    @property
    def block_size(self) -> int:
        return 3 * self.ndigits + 2
