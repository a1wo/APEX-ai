"""The from-scratch GPT in src/model.py — ~200K params, trains from random init."""

from ..data import VOCAB_SIZE
from ..model import GPT, GPTConfig

DEFAULT_LR = 3e-4


def build(cfg) -> GPT:
    model = GPT(GPTConfig(
        block_size = cfg.block_size,
        vocab_size = VOCAB_SIZE,
        n_layer    = cfg.n_layer,
        n_head     = cfg.n_head,
        n_embd     = cfg.n_embd,
    ))
    model.meta = {
        "n_layer":    cfg.n_layer,
        "n_head":     cfg.n_head,
        "n_embd":     cfg.n_embd,
        "vocab_size": VOCAB_SIZE,
    }
    return model
