"""Shared adapter for pretrained Hugging Face causal LMs.

The dataset and evaluator speak the 12-char vocab in src/data.py ('0'-'9',
'+', '='). This wrapper maps each char to the matching single token in the
HF tokenizer on the way in, and slices the output logits back down to those
12 token columns on the way out — so train.py and evaluate.py work unchanged
for any pretrained model, and the loss/generation are restricted to the
task alphabet.

Requires:  pip install transformers
"""

import torch
import torch.nn as nn
from torch.nn import functional as F

from ..data import VOCAB


class HFCharLM(nn.Module):
    def __init__(self, hf_id: str):
        super().__init__()
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as e:
            raise ImportError(
                "Hugging Face models need the transformers package:\n"
                "    .venv/bin/pip install transformers"
            ) from e

        self.hf_id = hf_id
        # transformers 5.x defaults to the checkpoint dtype (often fp16), which
        # NaNs under plain AdamW — train in fp32
        self.lm = AutoModelForCausalLM.from_pretrained(hf_id, dtype=torch.float32)

        tokenizer = AutoTokenizer.from_pretrained(hf_id)
        ids = []
        for ch in VOCAB:
            enc = tokenizer.encode(ch, add_special_tokens=False)
            assert len(enc) == 1, (
                f"{hf_id} tokenizer encodes {ch!r} as {len(enc)} tokens; "
                "the char-level mapping needs exactly one"
            )
            ids.append(enc[0])
        # char-vocab id (0..11) -> HF token id; moves with the model via .to(device)
        self.register_buffer("char_to_hf", torch.tensor(ids, dtype=torch.long))

        self.meta = {"hf_id": hf_id}

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        out = self.lm(input_ids=self.char_to_hf[idx])
        logits = out.logits[..., self.char_to_hf]  # (B, T, 12) in char-vocab order
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=-1,
            )
        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, greedy: bool = True):
        for _ in range(max_new_tokens):
            logits, _ = self(idx)
            logits = logits[:, -1, :]
            if greedy:
                idx_next = logits.argmax(dim=-1, keepdim=True)
            else:
                idx_next = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
