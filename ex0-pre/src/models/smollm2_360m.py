"""SmolLM2-360M — sweet spot of capability vs fine-tune speed on a 32GB Mac."""

from .hf import HFCharLM

HF_ID      = "HuggingFaceTB/SmolLM2-360M"
DEFAULT_LR = 5e-5


def build(cfg) -> HFCharLM:
    return HFCharLM(HF_ID)
