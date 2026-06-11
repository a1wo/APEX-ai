"""Model registry — pick one with `python train.py --model <name>`.

Each option lives in its own module and exposes:
    build(cfg)  -> nn.Module   compatible with train.py / evaluate.py
    DEFAULT_LR  -> float       used when --lr is not given

Modules are imported lazily so `transformers` is only required when a
Hugging Face model is actually selected.
"""

from importlib import import_module

MODELS = {
    "nano":         ".nano",          # from-scratch GPT in src/model.py (default)
    "pythia-70m":   ".pythia_70m",
    "distilgpt2":   ".distilgpt2",
    "gpt2":         ".gpt2",
    "smollm2-135m": ".smollm2_135m",
    "smollm2-360m": ".smollm2_360m",
    "qwen3-0.6b":   ".qwen3_0_6b",
}


def build_model(name: str, cfg) -> tuple:
    """Return (model, default_lr) for the named option."""
    if name not in MODELS:
        raise ValueError(f"Unknown model {name!r}. Options: {', '.join(MODELS)}")
    mod = import_module(MODELS[name], __package__)
    return mod.build(cfg), mod.DEFAULT_LR
