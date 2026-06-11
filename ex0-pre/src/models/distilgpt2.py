"""DistilGPT-2 — 82M-param distilled GPT-2; same architecture family as src/model.py."""

from .hf import HFCharLM

HF_ID      = "distilbert/distilgpt2"
DEFAULT_LR = 1e-4


def build(cfg) -> HFCharLM:
    return HFCharLM(HF_ID)
