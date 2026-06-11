"""Pythia-70M — EleutherAI's smallest research model. Fastest HF option to fine-tune."""

from .hf import HFCharLM

HF_ID      = "EleutherAI/pythia-70m"
DEFAULT_LR = 1e-4


def build(cfg) -> HFCharLM:
    return HFCharLM(HF_ID)
