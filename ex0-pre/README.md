# GPT Addition

> **Visual reference:** [bbycroft.net/llm](https://bbycroft.net/llm) — see a GPT's internals animated in 3D as you build.

Train a small Transformer from scratch to add two numbers (`a + b = c`) and watch it learn arithmetic.

---

## The task

Given the string `"123+456="`, the model autoregressively generates the digits of `c = 579`.

**Key design choices:**

### 1. Reversed digit output (default)
The model predicts the digits of `c` in **right-to-left** order — ones digit first:

```
123 + 456 = 579
                → output: "9", "7", "5", "0"   (reversed, zero-padded to ndigits+1)
```

This mirrors pencil-and-paper addition: carry information flows upward, so each output token only needs to know about carries from lower digits already generated. Left-to-right would require the model to "look ahead" and compute all carries internally before writing the first digit.

Try `--no_reverse` to see how much harder the forward-order task is.

### 2. Loss masking
The question portion `"NNN+NNN="` is masked out of the loss using `ignore_index=-1` in `CrossEntropyLoss`. The model is only penalised on the answer digits, which encourages it to attend to the operands rather than memorise question patterns.

### 3. On-the-fly data
There is no `train.bin` or `val.bin`. Each training step samples fresh random problems. With `10^ndigits × 10^ndigits` possible pairs the space is too large to overfit; a random batch is better than any fixed split.

---

## Files

```
ex0-pre/
├── model.py            # Minimal GPT: CausalSelfAttention → MLP → Block → GPT
├── train_addition.py   # Data generation, training loop, checkpointing, trackers
├── addition.ipynb      # Interactive notebook: data viz, training, eval, attention maps
└── README.md           # This file
```

---

## Setup

```bash
pip install torch
pip install wandb   # optional — for W&B logging
pip install mlflow  # optional — for MLflow logging
```

If using W&B, log in once:
```bash
wandb login
```

---

## Running experiments

### Basic training
```bash
python train_addition.py
```

### Compare reversed vs forward digit order
```bash
# Run both — they write to separate checkpoint files (_rev vs _fwd)
python train_addition.py                 # reversed (default, easier for the model)
python train_addition.py --no_reverse    # forward  (harder — requires implicit carry lookahead)
```

### Key flags
| Flag | Default | Meaning |
|------|---------|---------|
| `--ndigits N` | `3` | operand size; try 1, 2, 3 |
| `--max_iters N` | `100_000` | total training steps |
| `--epoch_size N` | `5_000` | steps between checkpoints |
| `--keep_checkpoints N` | `3` | how many recent checkpoints to keep |
| `--no_reverse` | off | predict c left-to-right |
| `--no_wandb` | off | disable W&B logging |
| `--no_mlflow` | off | disable MLflow logging |

---

## Experiment tracking

Both trackers are **on by default** and activated automatically when installed. If a library is missing or login fails it silently skips — training still runs.

### Weights & Biases

```bash
pip install wandb && wandb login
python train_addition.py          # logs to project "gpt-addition"
```

Open [wandb.ai](https://wandb.ai) → project **gpt-addition** to see:
- `train/loss` and `train/lr` every step
- `eval/loss` and `eval/accuracy` every 2 000 steps
- Full run config (ndigits, reverse_c, architecture, lr, …) attached to each run

### MLflow

```bash
pip install mlflow
python train_addition.py          # logs to local ./mlruns/
mlflow ui                         # open http://localhost:5000
```

The MLflow UI lets you:
- Filter runs by `reverse_c`, `ndigits`, etc.
- Plot `eval/accuracy` curves side-by-side across runs
- Diff hyperparameters between any two runs

### Run naming

Every run is tagged `addition_{ndigits}digit_{rev|fwd}_{ISO-timestamp}`, e.g.:
```
addition_3digit_rev_2025-06-11T14:30:00
addition_3digit_fwd_2025-06-11T14:35:00
```

Checkpoint files follow the same tag so the two experiments never overwrite each other:
```
checkpoints/addition_3digit_rev_epoch0001.pt
checkpoints/addition_3digit_fwd_epoch0001.pt
```

---

## Model architecture

| Hyperparameter | Value |
|---|---|
| Layers | 4 |
| Heads | 4 |
| Embedding dim | 64 |
| Block size | `3 × ndigits + 2` (e.g. 11 for ndigits=3) |
| Vocab | 12 tokens: `0-9`, `+`, `=` |
| Parameters | ~200 K |

The block size is exactly the maximum sequence length needed — no wasted capacity.

---

## Expected results

| ndigits | Mode | Steps to ~99% accuracy |
|---------|------|------------------------|
| 2 | reversed | ~10 000 |
| 3 | reversed | ~50 000 |
| 3 | forward | significantly more (if at all) |

The gap between reversed and forward is the main thing to observe — it demonstrates that output ordering is a meaningful inductive bias, not just a data formatting quirk.

---

## Notebook

Open `addition.ipynb` for an interactive version:

1. **Data** — visualise problems, tokenisation, and mask positions
2. **Model** — inspect architecture and parameter count
3. **Training** — quick 5 000-step run with live loss curve (or load a checkpoint)
4. **Evaluation** — exact-match accuracy + breakdown by operand magnitude
5. **Attention patterns** — heatmaps of what each layer/head attends to when predicting answer digits
