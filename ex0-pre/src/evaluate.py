import torch
from .model import GPT
from .data  import make_problem, encode, decode


@torch.no_grad()
def evaluate(
    model:     GPT,
    ndigits:   int  = 3,
    n_samples: int  = 500,
    device:    str  = "cpu",
    reverse_c: bool = False,
    pad_c:     bool = False,
) -> float:
    """
    Return fraction of addition problems answered with exact match.

    With pad_c=False the answer is variable-length and the vocab has no EOS
    token, so we generate exactly len(true answer) tokens — the model is never
    asked to decide when to stop.
    """
    model.eval()
    correct = 0

    for _ in range(n_samples):
        _, a, b, c = make_problem(ndigits, reverse_c, pad_c)
        question     = f"{a}+{b}="
        question_len = len(question)

        expected = str(c).zfill(ndigits + 1) if pad_c else str(c)
        if reverse_c:
            expected = expected[::-1]

        x   = torch.tensor(encode(question), dtype=torch.long, device=device).unsqueeze(0)
        out = model.generate(x, max_new_tokens=len(expected), greedy=True)
        ans_tokens = out[0, question_len:].tolist()

        correct += int(decode(ans_tokens) == expected)

    model.train()
    return correct / n_samples
