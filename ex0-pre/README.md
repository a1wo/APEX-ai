# GPT Addition

> **Visual reference:** [bbycroft.net/llm](https://bbycroft.net/llm) — see a GPT's internals animated in 3D as you build.

Train a small Transformer from scratch to add two numbers (`a + b = c`) and watch it learn arithmetic.

---

## The task

Given the string `"123+456="`, the model autoregressively generates the digits of `c = 579`.

**Key design choices:**

### 1. Digit order and padding (both opt-in)
By default the answer is written naturally: forward order, no padding (`123+456=579`).
Two flags make the task easier for the model:

- `--reverse` — predict the digits of `c` **right-to-left**, ones digit first:

  ```
  123 + 456 = 579
                  → output: "9", "7", "5"
  ```

  This mirrors pencil-and-paper addition: carry information flows upward, so each output token only needs to know about carries from lower digits already generated. Forward order requires the model to "look ahead" and resolve all carries internally before writing the first digit.

- `--pad` — zero-pad `c` to a fixed `ndigits+1` digits (`123+456=0579`), so the model never has to decide how many digits to emit. (The vocab has no end-of-sequence token.)

Comparing default vs `--reverse --pad` shows how much these inductive biases matter.

### 2. Loss masking
The question portion `"NNN+NNN="` is masked out of the loss using `ignore_index=-1` in `CrossEntropyLoss`. The model is only penalised on the answer digits, which encourages it to attend to the operands rather than memorise question patterns.

### 3. On-the-fly data
There is no `train.bin` or `val.bin`. Each training step samples fresh random problems. With `10^ndigits × 10^ndigits` possible pairs the space is too large to overfit; a random batch is better than any fixed split.

---

## Files

```
ex0-pre/
├── train.py            # ── executable: training run (CLI flags below)
├── monitor.py          # ── executable: live hardware metrics in the console
├── src/                # library modules
│   ├── config.py       #   TrainConfig dataclass (all defaults)
│   ├── data.py         #   vocab, problem generation, AdditionDataset
│   ├── model.py        #   minimal GPT: CausalSelfAttention → MLP → Block → GPT
│   ├── evaluate.py     #   exact-match accuracy
│   ├── checkpoint.py   #   run tags, save/load/prune
│   ├── tracker.py      #   W&B + MLflow wrapper
│   └── monitor.py      #   SystemMonitor (psutil + powermetrics)
├── scripts/            # one-liners
│   ├── train.sh        #   scripts/train.sh --reverse --pad …
│   ├── mlflow.sh       #   open MLflow UI on ./mlruns
│   ├── wandb.sh        #   verify W&B login, open dashboard
│   └── clear_checkpoints.sh  # delete all checkpoints (with confirmation)
├── explore.ipynb       # notebook: explore trained models (data, eval, attention maps)
└── README.md           # this file
```

---

## Setup

```bash
python3.13 -m venv .venv
.venv/bin/pip install -r requirements.txt
source .venv/bin/activate
```

`requirements.txt` covers torch, psutil (monitor), wandb + mlflow (tracking), and jupyter + matplotlib (notebook). The `scripts/*.sh` helpers find `.venv` automatically — no need to activate first.

If using W&B, log in once:
```bash
wandb login
```

---

## Running experiments

### Basic training
```bash
python train.py
```

### Compare task formats
```bash
# Each format writes to its own checkpoint tag — they never overwrite each other
python train.py                  # forward, unpadded (default — hardest)
python train.py --reverse --pad  # reversed, fixed-length (easiest for the model)
```

> Every run starts **fresh** and clears that tag's old checkpoints. Pass `--resume`
> to continue from the latest checkpoint instead — it also reattaches to the same
> W&B and MLflow runs, so the curves continue where they left off.

### Key flags
| Flag | Default | Meaning |
|------|---------|---------|
| `--ndigits N` | `3` | operand size; try 1, 2, 3 |
| `--max_iters N` | `100_000` | total training steps |
| `--batch_size N` | `128` | batch size |
| `--lr F` | `3e-4` | learning rate |
| `--epoch_size N` | `5_000` | steps between checkpoints |
| `--keep_checkpoints N` | `3` | how many recent checkpoints to keep |
| `--reverse` | off | predict c ones-digit-first |
| `--pad` | off | zero-pad c to ndigits+1 digits |
| `--model NAME` | `nano` | `nano` (from scratch) or a pretrained HF model (see `src/models/`) |
| `--resume` | off | continue the latest checkpoint **and** the same W&B/MLflow runs |
| `--monitor` | off | log hardware metrics (`system/*`) |
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
python train.py                   # metrics → ./mlflow.db, artifacts → ./mlruns/
scripts/mlflow.sh                 # open http://127.0.0.1:5001
```

> ⚠️ Don't use `localhost:5000` — macOS AirPlay Receiver squats on that port and serves an empty 403, which looks exactly like "my runs are missing". The script uses port 5001.

The MLflow UI lets you:
- Filter runs by `reverse_c`, `ndigits`, etc.
- Plot `eval/accuracy` curves side-by-side across runs
- Diff hyperparameters between any two runs

### Run naming

Every run is tagged `addition_{ndigits}digit_{rev|fwd}[_nopad]_{ISO-timestamp}` (the `_nopad` suffix appears when `--pad` is off), e.g.:
```
addition_3digit_fwd_nopad_2025-06-11T14:30:00     # default
addition_3digit_rev_2025-06-11T14:35:00           # --reverse --pad
```

Checkpoint files follow the same tag so experiments never overwrite each other:
```
checkpoints/addition_3digit_fwd_nopad_epoch0001.pt
checkpoints/addition_3digit_rev_epoch0001.pt
```

---

## Hardware monitoring

**Off by default.** Two ways to use it:

```bash
python train.py --monitor    # log system/* metrics to W&B/MLflow during training
python monitor.py            # standalone: print live metrics to the console
sudo python monitor.py       # + GPU power/thermal (powermetrics needs root)
```

The monitor announces its status on startup (psutil active, powermetrics available or not) instead of failing silently.

### What gets logged

| Metric | Source | Requires |
|--------|--------|---------|
| `system/cpu_pct` | psutil | `pip install psutil` |
| `system/ram_used_gb`, `system/ram_pct` | psutil | `pip install psutil` |
| `system/gpu_mem_mb` | `torch.mps` | nothing extra |
| `system/gpu_active_pct` | powermetrics | sudo (see below) |
| `system/gpu_freq_mhz` | powermetrics | sudo |
| `system/gpu_power_mw` | powermetrics | sudo |
| `system/cpu_power_mw` | powermetrics | sudo |
| `system/total_power_mw` | powermetrics | sudo |
| `system/thermal` | powermetrics | sudo (0=Nominal … 4=Critical) |

### Enabling powermetrics without a password prompt

`powermetrics` requires root. Add a one-time sudoers entry so it runs silently:

```bash
echo "$(whoami) ALL = NOPASSWD: /usr/bin/powermetrics" | sudo tee /etc/sudoers.d/powermetrics
```

Alternatively just run the standalone monitor with `sudo`. Without root, powermetrics is skipped (with a console notice) and you still get CPU/RAM/GPU-memory metrics.

### Reading the numbers

- **GPU active residency 100%** — GPU fully utilised; good.
- **GPU power ≫ CPU power** — GPU-bound; ideal.
- **CPU power ≫ GPU power** — CPU-bound; try increasing `batch_size` or `num_workers`.
- **thermal > 0** — chip is throttling; check cooling.
- **ANE power = 0** — expected; PyTorch MPS does not use the Apple Neural Engine.

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

Open `explore.ipynb` to explore a trained model (training happens via `scripts/train.sh`,
not in the notebook). Set `ndigits` / `reverse_c` / `pad_c` in the setup cell to pick
which run's checkpoint to load.

1. **Data** — visualise problems, tokenisation, and mask positions
2. **Load a trained model** — newest checkpoint matching the chosen config
3. **Evaluation** — exact-match accuracy, hand-picked examples, breakdown by operand magnitude
4. **Attention patterns** — heatmaps of what each layer/head attends to when predicting answer digits
