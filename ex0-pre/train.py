"""
Train a GPT to add two numbers: a + b = c.

    python train.py                        # default: reversed digits, all trackers auto
    python train.py --no_reverse           # forward digit order
    python train.py --no_wandb             # disable W&B
    python train.py --no_mlflow            # disable MLflow
    python train.py --ndigits 2            # easier task, trains faster
"""

import math
import os
import argparse
import torch
from datetime import datetime
from torch.utils.data import DataLoader

from config     import TrainConfig
from data       import AdditionDataset, VOCAB_SIZE
from evaluate   import evaluate
from monitor    import SystemMonitor
from tracker    import ExperimentTracker
from model      import GPT, GPTConfig
import checkpoint as ckpt


def _device() -> str:
    if torch.backends.mps.is_available():  return "mps"
    if torch.cuda.is_available():          return "cuda"
    return "cpu"


def _build_run_config(cfg: TrainConfig, gpt_cfg: GPTConfig, n_params: int,
                      device: str, started_at: str) -> dict:
    return {
        # task
        "ndigits":          cfg.ndigits,
        "reverse_c":        cfg.reverse_c,
        "c_order":          "reversed" if cfg.reverse_c else "forward",
        # model
        "n_layer":          gpt_cfg.n_layer,
        "n_head":           gpt_cfg.n_head,
        "n_embd":           gpt_cfg.n_embd,
        "block_size":       gpt_cfg.block_size,
        "vocab_size":       gpt_cfg.vocab_size,
        "n_params":         n_params,
        # optimisation
        "batch_size":       cfg.batch_size,
        "lr":               cfg.lr,
        "max_iters":        cfg.max_iters,
        "warmup_steps":     cfg.warmup_steps,
        "epoch_size":       cfg.epoch_size,
        # infra
        "device":           device,
        "started_at":       started_at,
    }


def train(cfg: TrainConfig) -> GPT:
    device = _device()
    os.makedirs(cfg.ckpt_dir, exist_ok=True)

    # ── Model & optimiser ─────────────────────────────────────────────────────
    gpt_cfg = GPTConfig(
        block_size = cfg.block_size,
        vocab_size  = VOCAB_SIZE,
        n_layer     = cfg.n_layer,
        n_head      = cfg.n_head,
        n_embd      = cfg.n_embd,
    )
    model     = GPT(gpt_cfg).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=0.1, betas=(0.9, 0.99)
    )

    # ── Resume ────────────────────────────────────────────────────────────────
    start_step = 0
    latest = ckpt.latest(cfg.ckpt_dir, cfg.ndigits, cfg.reverse_c)
    if latest:
        start_step = ckpt.load(latest, model, optimizer, device)
        print(f"Resumed from {latest}  (step {start_step})")
    else:
        print("Starting fresh training")

    # ── Run metadata ──────────────────────────────────────────────────────────
    started_at = datetime.now().isoformat(timespec="seconds")
    n_params   = sum(p.numel() for p in model.parameters())
    run_config = _build_run_config(cfg, gpt_cfg, n_params, device, started_at)
    run_name   = f"{ckpt.run_tag(cfg.ndigits, cfg.reverse_c)}_{started_at}"

    print(
        f"device={device}  ndigits={cfg.ndigits}  params={n_params:,}  "
        f"max_iters={cfg.max_iters}  epoch_size={cfg.epoch_size}  "
        f"c_order={run_config['c_order']}  started={started_at}"
    )

    # ── Trackers & monitor ────────────────────────────────────────────────────
    tracker = ExperimentTracker(run_name, run_config, cfg.use_wandb, cfg.use_mlflow)
    monitor = SystemMonitor(interval_ms=2_000)
    monitor.start()

    def log(metrics: dict, step: int) -> None:
        tracker.log({**metrics, **monitor.latest()}, step=step)

    # ── LR schedule ───────────────────────────────────────────────────────────
    def get_lr(it: int) -> float:
        if it < cfg.warmup_steps:
            return cfg.lr * it / cfg.warmup_steps
        t = (it - cfg.warmup_steps) / max(1, cfg.max_iters - cfg.warmup_steps)
        return cfg.lr * 0.5 * (1.0 + math.cos(math.pi * t))

    # ── Data ──────────────────────────────────────────────────────────────────
    remaining = cfg.max_iters - start_step
    dataset   = AdditionDataset(
        size      = remaining * cfg.batch_size,
        ndigits   = cfg.ndigits,
        reverse_c = cfg.reverse_c,
    )
    loader    = DataLoader(dataset, batch_size=cfg.batch_size, num_workers=0)
    data_iter = iter(loader)

    # ── Loop ──────────────────────────────────────────────────────────────────
    model.train()
    loss_accum = 0.0

    for step in range(start_step, cfg.max_iters):
        current_lr = get_lr(step)
        for g in optimizer.param_groups:
            g["lr"] = current_lr

        try:
            x, y = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            x, y     = next(data_iter)
        x, y = x.to(device), y.to(device)

        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        loss_accum += loss.item()

        log({"train/loss": loss.item(), "train/lr": current_lr}, step=step)

        # ── Eval ──────────────────────────────────────────────────────────────
        if step % cfg.eval_every == 0 or step == cfg.max_iters - 1:
            avg_loss = loss_accum / cfg.eval_every if step > start_step else loss.item()
            loss_accum = 0.0
            acc = evaluate(model, ndigits=cfg.ndigits, device=device, reverse_c=cfg.reverse_c)
            print(f"step {step:6d} | loss {avg_loss:.4f} | acc {acc:.3f}")
            log({"eval/loss": avg_loss, "eval/accuracy": acc}, step=step)

        # ── Checkpoint ────────────────────────────────────────────────────────
        if (step + 1) % cfg.epoch_size == 0 or step == cfg.max_iters - 1:
            epoch_num = (step + 1) // cfg.epoch_size
            path = ckpt.epoch_path(cfg.ckpt_dir, cfg.ndigits, cfg.reverse_c, epoch_num)
            ckpt.save(path, step=step, epoch=epoch_num, reverse_c=cfg.reverse_c,
                      run_config=run_config, model=model, optimizer=optimizer, gpt_cfg=gpt_cfg)
            ckpt.prune(cfg.ckpt_dir, cfg.ndigits, cfg.reverse_c, keep=cfg.keep_checkpoints)
            print(f"  → checkpoint: {path}")

    # ── Cleanup ───────────────────────────────────────────────────────────────
    monitor.stop()
    tracker.finish()
    print("\nDone.")
    return model


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Train GPT to add two numbers")

    # task
    p.add_argument("--ndigits",          type=int,   default=TrainConfig.ndigits)
    p.add_argument("--no_reverse",       action="store_true",
                   help="predict c left-to-right instead of reversed")
    # model
    p.add_argument("--n_layer",          type=int,   default=TrainConfig.n_layer)
    p.add_argument("--n_head",           type=int,   default=TrainConfig.n_head)
    p.add_argument("--n_embd",           type=int,   default=TrainConfig.n_embd)
    # optimisation
    p.add_argument("--max_iters",        type=int,   default=TrainConfig.max_iters)
    p.add_argument("--batch_size",       type=int,   default=TrainConfig.batch_size)
    p.add_argument("--lr",               type=float, default=TrainConfig.lr)
    # checkpointing
    p.add_argument("--ckpt_dir",         type=str,   default=TrainConfig.ckpt_dir)
    p.add_argument("--epoch_size",       type=int,   default=TrainConfig.epoch_size)
    p.add_argument("--keep_checkpoints", type=int,   default=TrainConfig.keep_checkpoints)
    # tracking
    p.add_argument("--no_wandb",         action="store_true")
    p.add_argument("--no_mlflow",        action="store_true")

    args = p.parse_args()

    train(TrainConfig(
        ndigits          = args.ndigits,
        reverse_c        = not args.no_reverse,
        n_layer          = args.n_layer,
        n_head           = args.n_head,
        n_embd           = args.n_embd,
        max_iters        = args.max_iters,
        batch_size       = args.batch_size,
        lr               = args.lr,
        ckpt_dir         = args.ckpt_dir,
        epoch_size       = args.epoch_size,
        keep_checkpoints = args.keep_checkpoints,
        use_wandb        = not args.no_wandb,
        use_mlflow       = not args.no_mlflow,
    ))
