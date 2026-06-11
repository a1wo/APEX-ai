import random
import torch
from torch.utils.data import Dataset

# ── Vocabulary ────────────────────────────────────────────────────────────────
VOCAB      = "0123456789+="
stoi       = {c: i for i, c in enumerate(VOCAB)}
itos       = {i: c for c, i in stoi.items()}
VOCAB_SIZE = len(VOCAB)
NDIGITS    = 3  # default digit width


def encode(s: str) -> list[int]:
    return [stoi[c] for c in s]


def decode(tokens: list[int]) -> str:
    return "".join(itos[t] for t in tokens)


# ── Problem generation ────────────────────────────────────────────────────────
def make_problem(ndigits: int = NDIGITS, reverse_c: bool = True) -> tuple[str, int, int, int]:
    """
    Return (problem_string, a, b, c) where c = a + b.

    Format:  "A+B=CCCC"
      - a, b  written without leading zeros (variable length)
      - c     zero-padded to ndigits+1 digits; reversed when reverse_c=True
              so the model outputs the least-significant digit first

    reverse_c=True  → "12+34=640"    ones digit of 46 is 6, then 4, then 0
    reverse_c=False → "12+34=0046"   standard left-to-right
    """
    a = random.randint(0, 10**ndigits - 1)
    b = random.randint(0, 10**ndigits - 1)
    c = a + b
    c_str = str(c).zfill(ndigits + 1)
    c_out = c_str[::-1] if reverse_c else c_str
    return f"{a}+{b}={c_out}", a, b, c


# ── Dataset ───────────────────────────────────────────────────────────────────
class AdditionDataset(Dataset):
    """
    Each __getitem__ generates a fresh random problem — no train.bin / val.bin.

    Sequences are right-padded to block_size so the DataLoader can batch them
    without a custom collate_fn. Padded positions carry y=-1 (ignored by loss).
    """

    def __init__(self, size: int = 200_000, ndigits: int = NDIGITS, reverse_c: bool = True):
        self.size       = size
        self.ndigits    = ndigits
        self.reverse_c  = reverse_c
        self.block_size = 3 * ndigits + 2  # max x-length when both operands are full-width

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, _idx) -> tuple[torch.Tensor, torch.Tensor]:
        problem, _, _, _ = make_problem(self.ndigits, self.reverse_c)
        tokens  = encode(problem)
        seq_len = len(tokens) - 1

        eq_pos = tokens.index(stoi["="])  # mask everything before '=' in targets
        raw_y  = list(tokens[1:])
        for i in range(eq_pos):
            raw_y[i] = -1

        x = torch.zeros(self.block_size, dtype=torch.long)
        y = torch.full((self.block_size,), -1, dtype=torch.long)
        x[:seq_len] = torch.tensor(tokens[:-1], dtype=torch.long)
        y[:seq_len] = torch.tensor(raw_y,       dtype=torch.long)
        return x, y
