"""Train the LSTM / transformer sequence classifiers used as black boxes.

Data are drawn from the benchmark generators exactly as the PAC pipeline
draws them (benchmark_gen, lengths 1..max_len, balanced) and labelled by the
ground-truth membership function. The saved TorchScript models keep the
interface of robustness_check.predict_with_{lstm,transformer}: input of shape
(batch, 51) holding the z-scored sequence padded with -999, output the
probability of acceptance.

    python3 scripts/train_networks.py --language S11 --model lstm --out nn_models_retrained
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(__file__))
from net_accuracy import PAD, WIDTH, encode, sample  # noqa: E402


class LSTMClassifier(nn.Module):
    """LSTM read up to the last real position (padding is not fed through)."""

    def __init__(self, hidden=128, layers=2, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(2, hidden, layers, batch_first=True, dropout=dropout)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, 1))

    def forward(self, x):
        valid = (x != PAD).float()
        feats = torch.stack([x * valid, valid], dim=-1)
        out, _ = self.lstm(feats)
        last = (valid.sum(1).long() - 1).clamp(min=0)
        idx = last.view(-1, 1, 1).expand(-1, 1, out.size(-1))
        h = out.gather(1, idx).squeeze(1)
        return torch.sigmoid(self.head(h).squeeze(-1))


class EncoderLayer(nn.Module):
    """Pre-norm transformer encoder layer with explicit key-padding mask, so
    that tracing yields a model that works for any batch size."""

    def __init__(self, d, heads, ff, dropout):
        super().__init__()
        self.h, self.dk = heads, d // heads
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.ff = nn.Sequential(nn.Linear(d, ff), nn.ReLU(), nn.Dropout(dropout), nn.Linear(ff, d))
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, keep):
        b, n, d = x.shape
        q, k, v = self.qkv(self.n1(x)).view(b, n, 3, self.h, self.dk).permute(2, 0, 3, 1, 4)
        att = q @ k.transpose(-1, -2) / self.dk ** 0.5
        att = att.masked_fill(~keep[:, None, None, :], -1e9).softmax(-1)
        y = (att @ v).transpose(1, 2).reshape(b, n, d)
        x = x + self.drop(self.proj(y))
        return x + self.drop(self.ff(self.n2(x)))


class TransformerClassifier(nn.Module):
    """Transformer encoder with learned positions and masked mean pooling."""

    def __init__(self, d=64, heads=4, layers=3, ff=256, dropout=0.1):
        super().__init__()
        self.inp = nn.Linear(2, d)
        self.pos = nn.Parameter(torch.randn(1, WIDTH, d) * 0.02)
        self.layers = nn.ModuleList(EncoderLayer(d, heads, ff, dropout) for _ in range(layers))
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, 1)

    def forward(self, x):
        keep = x != PAD
        valid = keep.float()
        h = self.inp(torch.stack([x * valid, valid], dim=-1)) + self.pos
        for layer in self.layers:
            h = layer(h, keep)
        h = self.norm(h)
        pooled = (h * valid.unsqueeze(-1)).sum(1) / valid.sum(1, keepdim=True).clamp(min=1)
        return torch.sigmoid(self.head(pooled).squeeze(-1))


def dataset(language, n, max_len, seed):
    pos, neg = sample(language, n, max_len, seed)
    x = encode(pos + neg)
    y = torch.tensor([1.0] * len(pos) + [0.0] * len(neg))
    return x, y


def accuracy(model, x, y):
    model.eval()
    with torch.no_grad():
        return ((model(x) >= 0.5).float() == y).float().mean().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--language", required=True)
    ap.add_argument("--model", choices=["lstm", "transformer"], required=True)
    ap.add_argument("--out", default="nn_models_retrained")
    ap.add_argument("--n_train", type=int, default=40000)
    ap.add_argument("--n_test", type=int, default=4000)
    ap.add_argument("--max_len", type=int, default=10)
    ap.add_argument("--max_steps", type=int, default=50000)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--eval_interval", type=int, default=500)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    t0 = time.time()
    x, y = dataset(a.language, a.n_train, a.max_len, a.seed)
    perm = torch.randperm(len(y))
    x, y = x[perm], y[perm]
    n_val = len(y) // 10
    xv, yv, xt, yt = x[:n_val], y[:n_val], x[n_val:], y[n_val:]
    # Independent test set: different generator seed.
    xtest, ytest = dataset(a.language, a.n_test, a.max_len, a.seed + 1000)

    model = LSTMClassifier() if a.model == "lstm" else TransformerClassifier()
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.max_steps)
    loss_fn = nn.BCELoss()
    best, best_state, hits, step = -1.0, None, 0, 0
    while step < a.max_steps:
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
                best, best_state = acc, {k: v.clone() for k, v in model.state_dict().items()}
            hits = hits + 1 if acc >= 1.0 else 0
            if hits >= 3:  # early stop: 100% validation accuracy three times in a row
                break

    model.load_state_dict(best_state)
    model.eval()
    test_acc = accuracy(model, xtest, ytest)
    traced = torch.jit.trace(model, xtest[:1])
    # The traced model must agree with the eager one for any batch size.
    with torch.no_grad():
        assert torch.allclose(traced(xtest[:257]), model(xtest[:257]), atol=1e-5)
    os.makedirs(os.path.join(a.out, a.model), exist_ok=True)
    path = os.path.join(a.out, a.model, f"{a.language}.pt")
    traced.save(path)
    info = dict(language=a.language, model=a.model, steps=step, val_acc=best,
                test_acc=test_acc, seconds=round(time.time() - t0, 1),
                n_train=a.n_train, max_len=a.max_len, seed=a.seed)
    with open(os.path.join(a.out, a.model, f"{a.language}.json"), "w") as f:
        json.dump(info, f, indent=1)
    print(json.dumps(info))


if __name__ == "__main__":
    main()
