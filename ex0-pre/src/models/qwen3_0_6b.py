"""Qwen3-0.6B — most capable option here; slowest to fine-tune, biggest checkpoints."""

from .hf import HFCharLM

HF_ID      = "Qwen/Qwen3-0.6B"
DEFAULT_LR = 2e-5


def build(cfg) -> HFCharLM:
    return HFCharLM(HF_ID)
