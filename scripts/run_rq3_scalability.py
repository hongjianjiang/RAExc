#!/usr/bin/env python3
"""RQ3 scalability: learning and robustness checking vs. states, registers,
and input length.

For each point we learn a DRA with local search (same settings as RQ1/RQ2),
measure its accuracy on fresh balanced test samples, and then run the exact
checker (last-letter distance) on fresh test sequences.

Families (as in the paper):
  states    : Sn  -- strictly increasing sequences of length n; n+1 states, 1 register
  registers : DistinctK -- length-k sequences of pairwise distinct values;
              k+2 states, k registers
  length    : S1  -- strictly increasing; training max length varies

Run from the repository root:
  python3 scripts/run_rq3_scalability.py --out results/rq3_scalability_<date>
"""
import argparse
import csv
import json
import multiprocessing as mp
import os
import random
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from benchmark_gen import Sn, S1, DistinctK  # noqa: E402
from local_serach_synthesis import dra_learner  # noqa: E402
from register_automata.ra import load_double_ls_dra, load_integer_ls_dra  # noqa: E402
from register_automata.robustness import RobustnessChecker  # noqa: E402
from robustness_check import get_metric_raa  # noqa: E402

FIELDS = [
    "family", "param", "language", "states", "registers", "max_len",
    "train_pos", "train_neg", "learning_time", "train_score", "test_acc",
    "num_checked", "robust", "non_robust", "check_timeouts",
    "check_total_time", "avg_check_time", "max_check_time", "seed", "error",
]


def make_language(family, param):
    if family == "states":
        lang = Sn(0, 0, 0, n=param)
        return lang, "Sn", param + 1, 1, param + 10, "Double"
    if family == "registers":
        lang = DistinctK(0, 0, 0)
        lang.k = param
        lang.regs = [f"r{i}" for i in range(param)]
        lang.states = param + 2
        return lang, "DistinctK", param + 2, param, param + 10, "Integer"
    if family == "length":
        lang = S1(0, 0, 0)
        return lang, "S1", lang.states, len(lang.regs), param, "Double"
    raise ValueError(family)


def run_point(task):
    family, param, a = task
    row = {k: "" for k in FIELDS}
    row.update(family=family, param=param, seed=a["seed"])
    try:
        random.seed(a["seed"])
        np.random.seed(a["seed"])
        lang, name, states, regs, max_len, theory = make_language(family, param)
        row.update(language=name, states=states, registers=regs, max_len=max_len)

        pos, neg = lang.generate_samples(a["train"], a["train"], max_len)
        row.update(train_pos=len(pos), train_neg=len(neg))

        t = time.monotonic()
        learner = dra_learner.DRAlearner(
            num_states=states, num_registers=regs, constants=[],
            pos_samples=pos, neg_samples=neg, golden_pos_samples=[],
            auto=None, debug=False,
            exactly_one=(lang.edge_limit_type == "exactly"), theory=theory)
        learner.simulated_annealing(initial_temperature=1.0, cooling_rate=0.995,
                                    max_iterations=a["max_iter"],
                                    timeout=a["learn_timeout"])
        auto = learner.best_dra.translate_to_initial_zero()
        os.makedirs(a["out"] / "dra", exist_ok=True)
        path = str(a["out"] / "dra" / f"{family}_{param}.json")
        auto.to_json(path, theory)
        dra = load_double_ls_dra(path) if theory == "Double" else load_integer_ls_dra(path)
        row["learning_time"] = f"{time.monotonic() - t:.2f}"
        row["train_score"] = f"{learner.best_score:.4f}"

        checker = RobustnessChecker(dra, get_metric_raa(a["metric"]))
        tpos, tneg = lang.generate_samples(a["test"], a["test"], max_len)
        correct = sum(checker.accepts(s) for s in tpos) + \
            sum(not checker.accepts(s) for s in tneg)
        row["test_acc"] = f"{correct / (len(tpos) + len(tneg)):.4f}"

        vecs = tpos[: a["check"] // 2] + tneg[: a["check"] // 2]
        times, robust, nonrobust, timeouts = [], 0, 0, 0
        deadline = time.monotonic() + a["check_timeout"]
        for v in vecs:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                timeouts += 1
                continue
            t = time.monotonic()
            try:
                res = checker.find_witness(v, a["delta"], accepted=checker.accepts(v),
                                           timeout=remaining)
            except TimeoutError:
                timeouts += 1
                times.append(time.monotonic() - t)
                continue
            times.append(time.monotonic() - t)
            if res.robust:
                robust += 1
            else:
                nonrobust += 1
        row.update(num_checked=len(times), robust=robust, non_robust=nonrobust,
                   check_timeouts=timeouts,
                   check_total_time=f"{sum(times):.3f}",
                   avg_check_time=f"{sum(times) / len(times):.4f}" if times else "",
                   max_check_time=f"{max(times):.4f}" if times else "")
    except Exception as e:  # keep the sweep going
        row["error"] = repr(e)
    print(f"[{family} {param}] {json.dumps(row)}", flush=True)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--train", type=int, default=369, help="samples per class")
    ap.add_argument("--test", type=int, default=500, help="test samples per class")
    ap.add_argument("--check", type=int, default=100, help="sequences checked")
    ap.add_argument("--learn_timeout", type=float, default=1800)
    ap.add_argument("--check_timeout", type=float, default=3600)
    ap.add_argument("--max_iter", type=int, default=50000)
    ap.add_argument("--metric", default="last_letter")
    ap.add_argument("--delta", type=float, default=1.0)
    ap.add_argument("--states", type=int, nargs="*", default=[20, 40, 60, 80, 100])
    ap.add_argument("--registers", type=int, nargs="*", default=[2, 4, 6, 8])
    ap.add_argument("--lengths", type=int, nargs="*", default=[20, 40, 60, 80, 100])
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    a = vars(args).copy()
    with open(args.out / "manifest.json", "w") as f:
        json.dump({**{k: (str(v) if isinstance(v, Path) else v) for k, v in a.items()},
                   "argv": sys.argv,
                   "started_at": time.strftime("%Y-%m-%dT%H:%M:%S")}, f, indent=2)
    tasks = [("states", n, a) for n in args.states] + \
            [("registers", k, a) for k in args.registers] + \
            [("length", L, a) for L in args.lengths]
    # longest points first so the pool finishes evenly
    tasks.sort(key=lambda t: -t[1] * (3 if t[0] == "registers" else 1))
    with mp.get_context("spawn").Pool(args.workers) as pool:
        rows = pool.map(run_point, tasks, chunksize=1)
    order = {"states": 0, "registers": 1, "length": 2}
    rows.sort(key=lambda r: (order[r["family"]], int(r["param"])))
    with open(args.out / "rq3_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print("wrote", args.out / "rq3_results.csv")


if __name__ == "__main__":
    main()
