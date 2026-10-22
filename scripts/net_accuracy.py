"""Accuracy of the sequence classifiers against the ground-truth languages.

Samples are drawn like the PAC pipeline does (balanced, benchmark_gen
generators, lengths 1..max_len) and fed through the same normalisation and
padding as robustness_check.predict_with_{lstm,transformer}.

    python3 scripts/net_accuracy.py --models_dir nn_models --n 2000
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from benchmark_gen import Language  # noqa: E402

LANGS = [f"L{i}" for i in range(1, 8)] + [f"S{i}" for i in range(1, 12)]
PAD, WIDTH = -999.0, 51


def language(name):
    return {c.__name__: c for c in Language.__subclasses__()}[name](0, 0, 0)


def sample(name, n, max_len, rng_seed):
    """n/2 positive and n/2 negative sequences (rejection sampling, no dedup)."""
    np.random.seed(rng_seed)
    lang = language(name)
    pos, neg = [], []
    tries = 0
    while (len(pos) < n // 2 or len(neg) < n // 2) and tries < 200 * n:
        tries += 1
        s, _ = lang.generate_sequence(max_len)
        if not s:
            continue
        s = [float(x) for x in s]
        (pos if lang.membership(s) else neg).append(s)
    return pos[: n // 2], neg[: n // 2]


def encode(seqs):
    rows = []
    for s in seqs:
        m, sd = np.mean(s), np.std(s)
        z = [0.0] * len(s) if sd == 0 else [(x - m) / sd for x in s]
        rows.append(z + [PAD] * (WIDTH - len(z)))
    return torch.tensor(rows, dtype=torch.float)


def predict(model, seqs):
    """One sequence per call: the original transformers were traced with
    batch size 1, exactly as robustness_check queries them."""
    x = encode(seqs)
    with torch.no_grad():
        return [model(x[i:i + 1]).item() >= 0.5 for i in range(len(seqs))]


def accuracy(model, pos, neg):
    hits = sum(predict(model, pos)) + sum(not y for y in predict(model, neg))
    return hits / (len(pos) + len(neg))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models_dir", default="nn_models")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--max_len", type=int, default=10)
    ap.add_argument("--seed", type=int, default=12345)
    a = ap.parse_args()
    print("language,model,accuracy")
    for name in LANGS:
        pos, neg = sample(name, a.n, a.max_len, a.seed)
        for kind in ("lstm", "transformer"):
            path = os.path.join(a.models_dir, kind, f"{name}.pt")
            if not os.path.exists(path):
                print(f"{name},{kind},missing")
                continue
            model = torch.jit.load(path, map_location="cpu").eval()
            print(f"{name},{kind},{100 * accuracy(model, pos, neg):.2f}", flush=True)
