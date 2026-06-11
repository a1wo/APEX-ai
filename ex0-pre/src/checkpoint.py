import os
import glob
import torch
from .model import GPT, GPTConfig


# ── Run identity ──────────────────────────────────────────────────────────────

def run_tag(ndigits: int, reverse_c: bool, pad_c: bool) -> str:
    # pad_c=True omits the suffix so pre-existing checkpoint names stay valid
    suffix = "" if pad_c else "_nopad"
    return f"addition_{ndigits}digit_{'rev' if reverse_c else 'fwd'}{suffix}"


def epoch_path(ckpt_dir: str, ndigits: int, reverse_c: bool, pad_c: bool, epoch: int) -> str:
    return os.path.join(ckpt_dir, f"{run_tag(ndigits, reverse_c, pad_c)}_epoch{epoch:04d}.pt")


# ── Disk helpers ──────────────────────────────────────────────────────────────

def latest(ckpt_dir: str, ndigits: int, reverse_c: bool, pad_c: bool) -> str | None:
    """Return path of the most-recent checkpoint, or None if none exist."""
    files = sorted(glob.glob(os.path.join(ckpt_dir, f"{run_tag(ndigits, reverse_c, pad_c)}_epoch*.pt")))
    return files[-1] if files else None


def prune(ckpt_dir: str, ndigits: int, reverse_c: bool, pad_c: bool, keep: int) -> None:
    """Delete all but the `keep` most-recent checkpoints for this run tag."""
    files = sorted(glob.glob(os.path.join(ckpt_dir, f"{run_tag(ndigits, reverse_c, pad_c)}_epoch*.pt")))
    for old in files[:-keep]:
        os.remove(old)


# ── Save / load ───────────────────────────────────────────────────────────────

def save(
    path:       str,
    *,
    step:       int,
    epoch:      int,
    reverse_c:  bool,
    run_config: dict,
    model:      GPT,
    optimizer:  torch.optim.Optimizer,
    gpt_cfg:    GPTConfig,
) -> None:
    torch.save({
        "step":       step,
        "epoch":      epoch,
        "reverse_c":  reverse_c,
        "run_config": run_config,
        "model":      model.state_dict(),
        "optimizer":  optimizer.state_dict(),
        "gpt_cfg":    gpt_cfg,
    }, path)


def load(
    path:      str,
    model:     GPT,
    optimizer: torch.optim.Optimizer,
    device:    str,
) -> int:
    """Load model + optimizer weights in-place. Returns the next step to run."""
    # GPTConfig is stored in the checkpoint; allowlist it for weights_only loading
    with torch.serialization.safe_globals([GPTConfig]):
        ckpt = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt["step"] + 1
