"""Train the MNIST row-mean classifiers (digit 1 vs. rest).

Each image is the sequence of its 28 row-mean intensities. The saved
TorchScript models keep the interface of robustness_check.predict_mnist:
raw (unnormalised) values of shape (batch, length, 1), any length up to 51,
output the probability of "digit 1". Training data: src/mnist_train.csv;
test data: src/mnist_test.csv, the file the PAC pipeline samples from.

    python3 scripts/train_mnist.py --model transformer --out nn_models_retrained
"""
import argparse
import csv
import json
import os
import sys
import time

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(__file__))
from net_accuracy import PAD, WIDTH  # noqa: E402
from train_networks import LSTMClassifier, TransformerClassifier  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
SCALE = 10.0  # row means lie in [0, 20]


class RowMeanClassifier(nn.Module):
    """Adapts (batch, length, 1) raw input to the padded sequence models."""

    def __init__(self, inner):
        super().__init__()
        self.inner = inner

    def forward(self, x):
        x = x.squeeze(-1) / SCALE
        pad = torch.full((x.size(0), WIDTH - x.size(1)), PAD, dtype=x.dtype)
        return self.inner(torch.cat([x, pad], dim=1))


def load(path):
    xs, ys = [], []
    with open(path) as f:
        rows = csv.reader(f)
        for row in rows:
            if not row or not row[0].strip().lstrip("-").isdigit():
                continue  # header
            vals = row[1].split(",") if len(row) == 2 else row[1:29]
            xs.append([[float(v)] for v in vals])
            ys.append(1.0 if int(row[0]) == 1 else 0.0)
    return torch.tensor(xs), torch.tensor(ys)


def accuracy(model, x, y):
    model.eval()
    with torch.no_grad():
        return ((model(x) >= 0.5).float() == y).float().mean().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["lstm", "transformer"], required=True)
    ap.add_argument("--out", default="nn_models_retrained")
    ap.add_argument("--max_steps", type=int, default=50000)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--eval_interval", type=int, default=500)
    ap.add_argument("--patience", type=int, default=10,
                    help="stop after this many evaluations without improvement")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed)
    t0 = time.time()
    x, y = load(os.path.join(ROOT, "src", "mnist_train.csv"))
    xtest, ytest = load(os.path.join(ROOT, "src", "mnist_test.csv"))
    perm = torch.randperm(len(y))
    x, y = x[perm], y[perm]
    n_val = len(y) // 10
    xv, yv, xt, yt = x[:n_val], y[:n_val], x[n_val:], y[n_val:]

    inner = LSTMClassifier() if a.model == "lstm" else TransformerClassifier()
    model = RowMeanClassifier(inner)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.max_steps)
    loss_fn = nn.BCELoss()
    best, best_state, stale, step = -1.0, None, 0, 0
    while step < a.max_steps and stale < a.patience:
        model.train()
        idx = torch.randint(0, len(yt), (a.batch_size,))
        opt.zero_grad()
        loss = loss_fn(model(xt[idx]).clamp(1e-6, 1 - 1e-6), yt[idx])
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        step += 1
        if step % a.eval_interval == 0:
            acc = accuracy(model, xv, yv)
            print(f"step {step} loss {loss.item():.4f} val {100 * acc:.2f}", flush=True)
            if acc > best:
                best, stale = acc, 0
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                stale += 1

    model.load_state_dict(best_state)
    model.eval()
    test_acc = accuracy(model, xtest, ytest)
    traced = torch.jit.trace(model, xtest[:1])
    # The traced model must agree with the eager one for other batch sizes
    # and sequence lengths (edit/DTW witnesses change the length).
    with torch.no_grad():
        for length in (5, 27, 28, 29, 40, WIDTH):
            probe = torch.rand(7, length, 1) * 20
            assert torch.allclose(traced(probe), model(probe), atol=1e-5), length
    os.makedirs(os.path.join(a.out, a.model), exist_ok=True)
    traced.save(os.path.join(a.out, a.model, "mnist.pt"))
    info = dict(language="mnist", model=a.model, steps=step, val_acc=best,
                test_acc=test_acc, seconds=round(time.time() - t0, 1), seed=a.seed)
    with open(os.path.join(a.out, a.model, "mnist.json"), "w") as f:
        json.dump(info, f, indent=1)
    print(json.dumps(info))


if __name__ == "__main__":
    main()
