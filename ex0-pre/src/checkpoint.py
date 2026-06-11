import os
import glob
import torch
from .model import GPTConfig


# ── Run identity ──────────────────────────────────────────────────────────────

def run_tag(ndigits: int, reverse_c: bool, pad_c: bool, model: str = "nano") -> str:
    # model="nano" omits its prefix (and pad_c=True the suffix) so pre-existing
    # checkpoint names stay valid
    prefix = "" if model == "nano" else f"{model}_"
    suffix = "" if pad_c else "_nopad"
    return f"{prefix}addition_{ndigits}digit_{'rev' if reverse_c else 'fwd'}{suffix}"


def epoch_path(ckpt_dir: str, ndigits: int, reverse_c: bool, pad_c: bool, epoch: int,
               model: str = "nano") -> str:
    return os.path.join(ckpt_dir, f"{run_tag(ndigits, reverse_c, pad_c, model)}_epoch{epoch:04d}.pt")


# ── Disk helpers ──────────────────────────────────────────────────────────────

def latest(ckpt_dir: str, ndigits: int, reverse_c: bool, pad_c: bool,
           model: str = "nano") -> str | None:
    """Return path of the most-recent checkpoint, or None if none exist."""
    files = sorted(glob.glob(os.path.join(ckpt_dir, f"{run_tag(ndigits, reverse_c, pad_c, model)}_epoch*.pt")))
    return files[-1] if files else None


def prune(ckpt_dir: str, ndigits: int, reverse_c: bool, pad_c: bool, keep: int,
          model: str = "nano") -> None:
    """Delete all but the `keep` most-recent checkpoints for this run tag."""
    files = sorted(glob.glob(os.path.join(ckpt_dir, f"{run_tag(ndigits, reverse_c, pad_c, model)}_epoch*.pt")))
    for old in files[:-keep]:
        os.remove(old)


def clear(ckpt_dir: str, ndigits: int, reverse_c: bool, pad_c: bool,
          model: str = "nano") -> int:
    """Delete every checkpoint for this run tag. Returns how many were removed."""
    files = sorted(glob.glob(os.path.join(ckpt_dir, f"{run_tag(ndigits, reverse_c, pad_c, model)}_epoch*.pt")))
    for f in files:
        os.remove(f)
    return len(files)


# ── Save / load ───────────────────────────────────────────────────────────────

def save(
    path:        str,
    *,
    step:        int,
    epoch:       int,
    reverse_c:   bool,
    run_config:  dict,
    model:       torch.nn.Module,
    optimizer:   torch.optim.Optimizer,
    gpt_cfg:     GPTConfig | None,   # None for Hugging Face models
    tracker_ids: dict | None = None,   # {"wandb": ..., "mlflow": ...} for --resume
) -> None:
    torch.save({
        "step":        step,
        "epoch":       epoch,
        "reverse_c":   reverse_c,
        "run_config":  run_config,
        "model":       model.state_dict(),
        "optimizer":   optimizer.state_dict(),
        "gpt_cfg":     gpt_cfg,
        "tracker_ids": tracker_ids or {},
    }, path)


def load(
    path:      str,
    model:     torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device:    str,
) -> tuple[int, dict]:
    """
    Load model + optimizer weights in-place.
    Returns (next step to run, tracker run ids saved with the checkpoint).
    """
    # GPTConfig is stored in the checkpoint; allowlist it for weights_only loading
    with torch.serialization.safe_globals([GPTConfig]):
        ckpt = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt["step"] + 1, ckpt.get("tracker_ids") or {}
