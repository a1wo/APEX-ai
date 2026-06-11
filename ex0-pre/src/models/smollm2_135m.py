"""SmolLM2-135M — modern small model (2024); tokenizes digits individually."""

from .hf import HFCharLM

HF_ID      = "HuggingFaceTB/SmolLM2-135M"
DEFAULT_LR = 1e-4


def build(cfg) -> HFCharLM:
    return HFCharLM(HF_ID)
