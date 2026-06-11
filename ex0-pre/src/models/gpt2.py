"""GPT-2 (124M) — the classic baseline.

Note: GPT-2's BPE normally merges digit runs into single tokens; the
char-level mapping in hf.py feeds digits one token at a time instead,
which is a distribution shift from pretraining but better for arithmetic.
"""

from .hf import HFCharLM

HF_ID      = "openai-community/gpt2"
DEFAULT_LR = 1e-4


def build(cfg) -> HFCharLM:
    return HFCharLM(HF_ID)
