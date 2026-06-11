"""
Train a GPT to add two numbers: a + b = c.

Key design choices:
  - Answer digits are predicted in REVERSE order (ones digit first), mirroring
    the right-to-left carry propagation of pencil-and-paper addition.
  - Loss is MASKED on the question portion (a+b=) using ignore_index=-1 in
    CrossEntropyLoss, so the model is only graded on the answer.
  - Data is generated on-the-fly — no train.bin / val.bin needed.

Run:
    python train_addition.py                        # default (reversed digits)
    python train_addition.py --no_reverse           # forward digit order
    python train_addition.py --wandb                # log to Weights & Biases
    python train_addition.py --mlflow               # log to MLflow
    python train_addition.py --wandb --mlflow       # log to both
"""

import math
import os
import glob
import random
import argparse
from datetime import datetime
import torch
from torch.utils.data import Dataset, DataLoader
from model import GPT, GPTConfig

# ── Vocabulary ────────────────────────────────────────────────────────────────
VOCAB = "0123456789+="   # 12 tokens, indices 0-11
stoi  = {c: i for i, c in enumerate(VOCAB)}
itos  = {i: c for c, i in stoi.items()}
VOCAB_SIZE = len(VOCAB)

NDIGITS = 3  # default; controls problem difficulty


def encode(s: str) -> list[int]:
    return [stoi[c] for c in s]


def decode(tokens: list[int]) -> str:
    return "".join(itos[t] for t in tokens)


# ── Data ──────────────────────────────────────────────────────────────────────
def make_problem(ndigits: int = NDIGITS, reverse_c: bool = True) -> tuple[str, int, int, int]:
    """
    Return (problem_string, a, b, c) where c = a + b.

    Format: "A+B=CCCC"
      - a and b are written without leading zeros (variable length)
      - c is zero-padded to ndigits+1 digits; reversed when reverse_c=True
        (reversed = ones digit first, matching right-to-left addition)

    reverse_c=True  (default): "12+34=640"   ← ones digit of 46 is 6, then 4, then 0
    reverse_c=False           : "12+34=0046"  ← standard left-to-right
    """
    a = random.randint(0, 10**ndigits - 1)
    b = random.randint(0, 10**ndigits - 1)
    c = a + b
    c_str = str(c).zfill(ndigits + 1)
    c_out = c_str[::-1] if reverse_c else c_str
    return f"{a}+{b}={c_out}", a, b, c


class AdditionDataset(Dataset):
    """
    Each __getitem__ call generates a fresh random problem, so we never
    repeat examples and don't need a pre-built dataset file.

    Sequences are padded to block_size (= 3*ndigits+2, the maximum possible
    x-length) so the DataLoader can batch them without a custom collate_fn.
    Padded positions have y=-1 so they contribute nothing to the loss.
    """

    def __init__(self, size: int = 200_000, ndigits: int = NDIGITS, reverse_c: bool = True):
        self.size       = size
        self.ndigits    = ndigits
        self.reverse_c  = reverse_c
        self.block_size = 3 * ndigits + 2  # max x length (both operands at full width)

    def __len__(self):
        return self.size

    def __getitem__(self, _idx):
        problem, _, _, _ = make_problem(self.ndigits, self.reverse_c)
        tokens  = encode(problem)
        seq_len = len(tokens) - 1  # length of x for this problem

        # Find '=' to know where the answer starts in the shifted targets.
        eq_pos = tokens.index(stoi["="])  # position of '=' in tokens

        raw_x = tokens[:-1]
        raw_y = list(tokens[1:])
        for i in range(eq_pos):   # mask question part in y
            raw_y[i] = -1

        # Pad to block_size (padding positions stay -1 in y → ignored by loss)
        x = torch.zeros(self.block_size, dtype=torch.long)
        y = torch.full((self.block_size,), -1, dtype=torch.long)
        x[:seq_len] = torch.tensor(raw_x, dtype=torch.long)
        y[:seq_len] = torch.tensor(raw_y, dtype=torch.long)
        return x, y


# ── Evaluation ────────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(
    model: GPT,
    ndigits: int = NDIGITS,
    n_samples: int = 500,
    device: str = "cpu",
    reverse_c: bool = True,
) -> float:
    """Return fraction of problems answered exactly correctly."""
    model.eval()
    correct = 0

    for _ in range(n_samples):
        problem, a, b, c = make_problem(ndigits, reverse_c)
        question     = f"{a}+{b}="
        question_len = len(question)
        x = torch.tensor(encode(question), dtype=torch.long, device=device).unsqueeze(0)
        out = model.generate(x, max_new_tokens=ndigits + 1, greedy=True)
        answer_tokens = out[0, question_len:].tolist()
        if len(answer_tokens) == ndigits + 1:
            try:
                ans = decode(answer_tokens)
                c_pred = int(ans[::-1] if reverse_c else ans)
                correct += int(c_pred == c)
            except ValueError:
                pass

    model.train()
    return correct / n_samples


# ── Checkpointing helpers ─────────────────────────────────────────────────────
def _run_tag(ndigits: int, reverse_c: bool) -> str:
    return f"addition_{ndigits}digit_{'rev' if reverse_c else 'fwd'}"


def _ckpt_path(ckpt_dir: str, ndigits: int, reverse_c: bool, epoch: int) -> str:
    return os.path.join(ckpt_dir, f"{_run_tag(ndigits, reverse_c)}_epoch{epoch:04d}.pt")


def _latest_checkpoint(ckpt_dir: str, ndigits: int, reverse_c: bool) -> str | None:
    pattern = os.path.join(ckpt_dir, f"{_run_tag(ndigits, reverse_c)}_epoch*.pt")
    files = sorted(glob.glob(pattern))
    return files[-1] if files else None


def _prune_checkpoints(ckpt_dir: str, ndigits: int, reverse_c: bool, keep: int) -> None:
    pattern = os.path.join(ckpt_dir, f"{_run_tag(ndigits, reverse_c)}_epoch*.pt")
    files = sorted(glob.glob(pattern))
    for old in files[:-keep]:
        os.remove(old)


# ── Training ──────────────────────────────────────────────────────────────────
def train(ndigits: int = NDIGITS, max_iters: int = 100_000,
          epoch_size: int = 5_000, keep_checkpoints: int = 3,
          reverse_c: bool = True,
          use_wandb: bool = True, use_mlflow: bool = True):
    """
    reverse_c        — True: predict c's digits right-to-left (ones first)
                       False: predict left-to-right (standard order)
    epoch_size       — steps per epoch; a checkpoint is saved at the end of each epoch
    keep_checkpoints — how many recent checkpoint files to keep on disk
    use_wandb        — log to W&B if installed; silently skipped otherwise
    use_mlflow       — log to MLflow if installed; silently skipped otherwise
    """
    batch_size = 128
    eval_every = 2_000
    lr         = 3e-4
    ckpt_dir   = "checkpoints"
    os.makedirs(ckpt_dir, exist_ok=True)

    device = (
        "mps"  if torch.backends.mps.is_available() else
        "cuda" if torch.cuda.is_available()          else
        "cpu"
    )

    block_size = 3 * ndigits + 2
    cfg = GPTConfig(
        block_size=block_size,
        vocab_size=VOCAB_SIZE,
        n_layer=4,
        n_head=4,
        n_embd=64,
    )
    model = GPT(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1, betas=(0.9, 0.99))

    # ── Resume from latest checkpoint if one exists ──
    start_step = 0
    latest = _latest_checkpoint(ckpt_dir, ndigits, reverse_c)
    if latest:
        ckpt = torch.load(latest, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_step = ckpt["step"] + 1
        print(f"Resumed from {latest}  (step {start_step})")
    else:
        print("Starting fresh training")

    started_at = datetime.now().isoformat(timespec="seconds")
    direction  = "reversed" if reverse_c else "forward"
    n_params   = sum(p.numel() for p in model.parameters())

    run_config = {
        # data / task
        "ndigits":    ndigits,
        "reverse_c":  reverse_c,
        "c_order":    direction,
        # model
        "n_layer":    cfg.n_layer,
        "n_head":     cfg.n_head,
        "n_embd":     cfg.n_embd,
        "block_size": block_size,
        "vocab_size": VOCAB_SIZE,
        "n_params":   n_params,
        # training
        "batch_size": batch_size,
        "lr":         lr,
        "max_iters":  max_iters,
        "epoch_size": epoch_size,
        "device":     device,
        "started_at": started_at,
    }

    print(f"device={device}  ndigits={ndigits}  params={n_params:,}  "
          f"max_iters={max_iters}  epoch_size={epoch_size}  c_order={direction}  "
          f"started_at={started_at}")

    # ── Experiment trackers ───────────────────────────────────────────────────
    tag = _run_tag(ndigits, reverse_c)
    run_name = f"{tag}_{started_at}"

    _wb = None
    if use_wandb:
        try:
            import wandb as _wandb_mod
            _wb = _wandb_mod
            _wb.init(project="gpt-addition", name=run_name, config=run_config, resume="allow")
            print("wandb: logging to project 'gpt-addition'")
        except ImportError:
            print("wandb not found — skipping  (pip install wandb && wandb login)")
        except Exception as e:
            print(f"wandb init failed ({e}) — skipping")

    _mlf = None
    if use_mlflow:
        try:
            import mlflow as _mlflow_mod
            _mlf = _mlflow_mod
            _mlf.set_experiment("gpt-addition")
            _mlf.start_run(run_name=run_name)
            _mlf.log_params(run_config)
            print(f"mlflow: run '{run_name}' started  (mlflow ui to explore)")
        except ImportError:
            print("mlflow not found — skipping  (pip install mlflow)")
        except Exception as e:
            print(f"mlflow init failed ({e}) — skipping")

    def log_metrics(metrics: dict, step: int) -> None:
        if _wb:
            _wb.log(metrics, step=step)
        if _mlf:
            _mlf.log_metrics(metrics, step=step)

    # ── Training loop ─────────────────────────────────────────────────────────
    def get_lr(it: int) -> float:
        warmup = 1_000
        if it < warmup:
            return lr * it / warmup
        t = (it - warmup) / max(1, max_iters - warmup)
        return lr * 0.5 * (1.0 + math.cos(math.pi * t))

    remaining = max_iters - start_step
    dataset   = AdditionDataset(size=remaining * batch_size, ndigits=ndigits, reverse_c=reverse_c)
    loader    = DataLoader(dataset, batch_size=batch_size, num_workers=0)
    data_iter = iter(loader)

    model.train()
    loss_accum = 0.0

    for step in range(start_step, max_iters):
        current_lr = get_lr(step)
        for g in optimizer.param_groups:
            g["lr"] = current_lr

        try:
            x, y = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            x, y = next(data_iter)
        x, y = x.to(device), y.to(device)

        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        loss_accum += loss.item()

        log_metrics({"train/loss": loss.item(), "train/lr": current_lr}, step=step)

        # ── Periodic eval ──
        if step % eval_every == 0 or step == max_iters - 1:
            avg_loss = loss_accum / eval_every if step > start_step else loss.item()
            loss_accum = 0.0
            acc = evaluate(model, ndigits=ndigits, device=device, reverse_c=reverse_c)
            print(f"step {step:6d} | loss {avg_loss:.4f} | acc {acc:.3f}")
            log_metrics({"eval/loss": avg_loss, "eval/accuracy": acc}, step=step)

        # ── End-of-epoch checkpoint ──
        if (step + 1) % epoch_size == 0 or step == max_iters - 1:
            epoch_num = (step + 1) // epoch_size
            path = _ckpt_path(ckpt_dir, ndigits, reverse_c, epoch_num)
            torch.save({
                "step":       step,
                "epoch":      epoch_num,
                "reverse_c":  reverse_c,
                "run_config": run_config,
                "model":      model.state_dict(),
                "optimizer":  optimizer.state_dict(),
                "config":     cfg,
            }, path)
            _prune_checkpoints(ckpt_dir, ndigits, reverse_c, keep=keep_checkpoints)
            print(f"  → checkpoint saved: {path}")

    if _wb:
        _wb.finish()
    if _mlf:
        _mlf.end_run()

    print("\nDone.")
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ndigits",          type=int,  default=3)
    parser.add_argument("--max_iters",        type=int,  default=100_000)
    parser.add_argument("--epoch_size",       type=int,  default=5_000,
                        help="steps per epoch (checkpoint saved after each)")
    parser.add_argument("--keep_checkpoints", type=int,  default=3,
                        help="number of recent checkpoints to keep on disk")
    parser.add_argument("--no_reverse",       action="store_true",
                        help="predict c left-to-right instead of reversed")
    parser.add_argument("--no_wandb",         action="store_true",
                        help="disable Weights & Biases logging")
    parser.add_argument("--no_mlflow",        action="store_true",
                        help="disable MLflow logging")
    args = parser.parse_args()
    train(ndigits=args.ndigits, max_iters=args.max_iters,
          epoch_size=args.epoch_size, keep_checkpoints=args.keep_checkpoints,
          reverse_c=not args.no_reverse,
          use_wandb=not args.no_wandb, use_mlflow=not args.no_mlflow)
