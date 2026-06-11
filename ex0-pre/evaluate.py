import torch
from model import GPT
from data  import make_problem, encode, decode


@torch.no_grad()
def evaluate(
    model:     GPT,
    ndigits:   int  = 3,
    n_samples: int  = 500,
    device:    str  = "cpu",
    reverse_c: bool = True,
) -> float:
    """Return fraction of addition problems answered with exact integer match."""
    model.eval()
    correct = 0

    for _ in range(n_samples):
        _, a, b, c = make_problem(ndigits, reverse_c)
        question     = f"{a}+{b}="
        question_len = len(question)

        x   = torch.tensor(encode(question), dtype=torch.long, device=device).unsqueeze(0)
        out = model.generate(x, max_new_tokens=ndigits + 1, greedy=True)
        ans_tokens = out[0, question_len:].tolist()

        if len(ans_tokens) == ndigits + 1:
            try:
                ans    = decode(ans_tokens)
                c_pred = int(ans[::-1] if reverse_c else ans)
                correct += int(c_pred == c)
            except ValueError:
                pass

    model.train()
    return correct / n_samples
